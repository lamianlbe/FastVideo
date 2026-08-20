#!/usr/bin/env python3
"""Convert one ComfyUI LTX-2.3 DiT export into a diffusers transformer dir.

Input: a ComfyUI ``ModelSave`` (or ModelSaveKJ) export of the LoRA-merged
diffusion model — every key under ``model.diffusion_model.*``, fp8-scaled
mixed precision (fp8 e4m3 ``.weight`` payloads + 0-d f32 ``.weight_scale``
siblings + ``.comfy_quant`` descriptors for a learned subset of linears,
bf16 for everything else).

Output: ``<component>/model.safetensors`` + ``config.json`` in the diffusers
key layout FastVideo loads. By default the fp8 payload/scale pairs pass
through VERBATIM (FastVideo's loader runs them quantized, exactly the
values ComfyUI ran); ``--dequant`` emits plain bf16 instead. The
``.comfy_quant`` descriptors are ComfyUI loader hints and are dropped
either way.

Why transformer-only: the two-stage workflow merges the distilled LoRA at
different strengths per stage, so stage 1 and stage 2 are DIFFERENT DiTs —
but everything outside the DiT blocks (including the text-encoder-routed
``*_embeddings_connector`` tensors inside the export) is identical across
stages, so the rest of an existing converted repo is reused as-is. The
connector tensors are therefore EXCLUDED here (they live in the repo's
text_encoder/, not the transformer); pass ``--check-connectors-against``
with the other stage's export to prove that assumption on your files.

Typical use against an existing repo:

    # stage 1 (replaces the bf16 transformer with the fp8 one)
    python convert_ltx23_transformer.py --dit stage1_00001_.safetensors \\
        --repo /path/10eros_v1 --component transformer

    # stage 2 (new directory + model_index.json pointer)
    python convert_ltx23_transformer.py --dit stage2_00001_.safetensors \\
        --repo /path/10eros_v1 --component transformer_stage2 --set-refine-path \\
        --check-connectors-against stage1_00001_.safetensors

Streaming: peak RAM is one tensor, not the ~23 GB file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_ltx23_reference_configs", _HERE / "_ltx23_reference_configs.py")
assert _spec is not None and _spec.loader is not None
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

DIT_PREFIX = "model.diffusion_model."
# Routed into text_encoder/ by the full converter; identical across the two
# stage exports (the LoRA strength differences only touch the DiT proper).
CONNECTOR_PREFIXES = (
    DIT_PREFIX + "video_embeddings_connector.",
    DIT_PREFIX + "audio_embeddings_connector.",
)
DTYPE_SIZE = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2,
              "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta


def nbytes(dtype: str, shape) -> int:
    size = DTYPE_SIZE[dtype]
    for d in shape:
        size *= d
    return size


def check_connectors(dit_path: Path, other_path: Path) -> None:
    """Byte-compare the connector tensors of two exports; abort on mismatch."""
    ha, _ = read_header(dit_path)
    hb, _ = read_header(other_path)
    keys = sorted(k for k in ha if k.startswith(CONNECTOR_PREFIXES))
    if not keys:
        raise SystemExit("connector check: no connector tensors found in the export")
    missing = [k for k in keys if k not in hb]
    if missing:
        raise SystemExit(f"connector check: {len(missing)} connector key(s) absent from {other_path.name}")
    mismatched = []
    with safe_open(str(dit_path), framework="pt", device="cpu") as fa, \
         safe_open(str(other_path), framework="pt", device="cpu") as fb:
        for key in keys:
            ta = fa.get_tensor(key)
            tb = fb.get_tensor(key)
            if ta.shape != tb.shape or not torch.equal(ta.view(torch.uint8), tb.view(torch.uint8)):
                mismatched.append(key)
    if mismatched:
        print(f"connector check FAILED: {len(mismatched)} tensor(s) differ between the two exports:")
        for key in mismatched[:10]:
            print(f"   {key}")
        raise SystemExit("The connectors are NOT shared between stages — a transformer-only conversion "
                         "is insufficient; the text_encoder side would need per-stage weights too.")
    print(f"connector check OK: {len(keys)} tensors bit-identical to {other_path.name}")


def resolve_config(config_from: str | None, repo: Path | None) -> dict:
    if config_from:
        src = Path(config_from)
        if src.is_dir():
            src = src / "config.json"
        return json.loads(src.read_text())
    if repo is not None and (repo / "transformer" / "config.json").is_file():
        return json.loads((repo / "transformer" / "config.json").read_text())
    return {"_class_name": "LTX2Transformer3DModel", **_ref.TRANSFORMER_REFERENCE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", required=True, help="ComfyUI export of the merged DiT (.safetensors)")
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="write <output>/model.safetensors + config.json")
    out.add_argument("--repo", help="existing converted repo; writes <repo>/<component>/")
    ap.add_argument("--component", default="transformer",
                    help="component directory name under --repo (default: transformer; "
                         "use transformer_stage2 for the refine DiT)")
    ap.add_argument("--set-refine-path", action="store_true",
                    help="point model_index.json fastvideo_refine_transformer_path at --component")
    ap.add_argument("--dequant", action="store_true",
                    help="dequantize fp8 payloads to bf16 instead of passing them through")
    ap.add_argument("--config-from", default=None,
                    help="config.json (or a transformer dir) to copy; default: the repo's "
                         "existing transformer/config.json, else the embedded 2.3 reference")
    ap.add_argument("--check-connectors-against", default=None,
                    help="another stage's export; abort unless the connector tensors are bit-identical")
    args = ap.parse_args()

    dit_path = Path(args.dit)
    if not dit_path.is_file():
        sys.exit(f"--dit must be a safetensors file: {dit_path}")
    repo = Path(args.repo) if args.repo else None
    if repo is not None and not repo.is_dir():
        sys.exit(f"--repo is not a directory: {repo}")
    if args.set_refine_path and repo is None:
        sys.exit("--set-refine-path needs --repo")
    if args.set_refine_path and args.component == "transformer":
        sys.exit("--set-refine-path with --component transformer makes no sense: the refine "
                 "path must name a SEPARATE stage-2 directory")
    component_dir = Path(args.output) if args.output else repo / args.component

    if args.check_connectors_against:
        check_connectors(dit_path, Path(args.check_connectors_against))

    header, _meta = read_header(dit_path)
    payload_keys = sorted(k for k in header if not k.endswith(".comfy_quant"))
    stray = [k for k in payload_keys if not k.startswith(DIT_PREFIX)]
    if stray:
        sys.exit(f"{len(stray)} key(s) outside {DIT_PREFIX!r} — this is not a DiT-only export: {stray[:5]}")
    connectors = [k for k in payload_keys if k.startswith(CONNECTOR_PREFIXES)]
    kept = [k for k in payload_keys if not k.startswith(CONNECTOR_PREFIXES)]
    dropped_quant_meta = sum(1 for k in header if k.endswith(".comfy_quant"))

    # Build the output header: strip the prefix; either keep fp8+scale pairs
    # verbatim or fold the scale in and emit bf16.
    plan: list[tuple[str, str, str, bool]] = []  # (src_key, out_key, out_dtype, dequant)
    fp8_kept = fp8_dequantized = 0
    for key in kept:
        entry = header[key]
        out_key = key[len(DIT_PREFIX):]
        if key.endswith(".weight_scale"):
            if args.dequant:
                continue  # folded into its payload below
            plan.append((key, out_key, entry["dtype"], False))
            continue
        if entry["dtype"].startswith("F8_"):
            if f"{key}_scale" not in header:
                sys.exit(f"fp8 payload without a weight_scale sibling: {key}")
            if args.dequant:
                plan.append((key, out_key, "BF16", True))
                fp8_dequantized += 1
            else:
                plan.append((key, out_key, entry["dtype"], False))
                fp8_kept += 1
            continue
        plan.append((key, out_key, entry["dtype"], False))

    out_header: dict[str, dict] = {}
    offset = 0
    for _src, out_key, dtype, _deq in plan:
        shape = header[_src]["shape"]
        size = nbytes(dtype, shape)
        out_header[out_key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size

    meta = {"converted_from": dit_path.name,
            "format": "bf16" if args.dequant else ("fp8_scaled" if fp8_kept else "bf16")}
    blob = json.dumps({"__metadata__": meta, **out_header}, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    component_dir.mkdir(parents=True, exist_ok=True)
    out_path = component_dir / "model.safetensors"
    print(f"input : {dit_path}  ({len(payload_keys)} tensors + {dropped_quant_meta} comfy_quant dropped)")
    print(f"        {len(connectors)} connector tensor(s) excluded (they live in text_encoder/)")
    print(f"output: {out_path}  ({len(plan)} tensors, {offset / 1e9:.1f} GB, "
          f"{'bf16 (dequantized)' if args.dequant else f'{fp8_kept} fp8-scaled pairs kept verbatim'})")

    with safe_open(str(dit_path), framework="pt", device="cpu") as handle, out_path.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, (src_key, _out_key, dtype, dequant) in enumerate(plan):
            tensor = handle.get_tensor(src_key)
            if dequant:
                scale = handle.get_tensor(f"{src_key}_scale").float()
                tensor = (tensor.float() * scale).to(torch.bfloat16)
            out.write(tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes())
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)

    config = resolve_config(args.config_from, repo)
    (component_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {component_dir / 'config.json'}")

    if args.set_refine_path:
        index_path = repo / "model_index.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {}
        backup = index_path.with_suffix(".json.bak")
        if index_path.is_file() and not backup.exists():
            shutil.copy2(index_path, backup)
        index["fastvideo_refine_transformer_path"] = args.component
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        print(f"model_index.json: fastvideo_refine_transformer_path -> {args.component!r}")

    print("done")


if __name__ == "__main__":
    main()
