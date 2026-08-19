#!/usr/bin/env python3
"""Fingerprint LTX-2.3 weights so two machines' checkpoints can be compared.

The reference (a ComfyUI ``ModelSaveKJ`` export of the LoRA-merged model) and
the converted FastVideo repo normally live on different pods, and each is
tens of GB. So instead of moving weights, each side runs ``summarize`` to
emit a small JSON fingerprint, and ``diff`` compares the two fingerprints.

Per tensor the fingerprint stores shape/dtype plus rms/absmax/mean and the
values at a handful of deterministic flat indices. Statistics alone would
miss a permutation (e.g. a wrong NVFP4 nibble order leaves the distribution
intact), which is exactly why the sampled values are included.

Handles the two layouts transparently:
  * ComfyUI single file — keys like ``model.diffusion_model.<name>``, fp8
    weights paired with an F32 ``<name>.weight_scale`` sibling;
  * converted diffusers repo — per-component directories, bf16, and the
    connector tensors routed into ``text_encoder/``.
Both are canonicalized to a bare ``<name>`` so the two sides line up.

    # on the ComfyUI pod
    python ltx23_weight_fingerprint.py summarize \\
        --input /workspace/ComfyUI/output/diffusion_models/ComfyUI_00001_.safetensors \\
        --out comfy.json
    # on the FastVideo pod
    python ltx23_weight_fingerprint.py summarize \\
        --input /workspace/My-LTX-2.3-Diffusers --out fastvideo.json
    # anywhere
    python ltx23_weight_fingerprint.py diff --a comfy.json --b fastvideo.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

SAMPLE_FRACTIONS = (0.0, 1 / 7, 1 / 3, 1 / 2, 2 / 3, 6 / 7)

# e4m3 has a 3-bit mantissa, so a value re-quantized to it lands within
# ~2^-4 (6.25%) relative of the original, worst case. ComfyUI stores its
# LoRA-merged weights back in fp8, so a correct bf16-side merge still differs
# from the reference by up to that much — anything beyond it is a real defect.
E4M3_RELATIVE_ULP = 0.0625


def canonical_key(key: str) -> str:
    """Strip layout-specific prefixes so both sides share one namespace."""
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _sample_values(tensor: torch.Tensor) -> list[float]:
    flat = tensor.reshape(-1)
    n = flat.numel()
    if n == 0:
        return []
    idx = sorted({min(n - 1, int(f * n)) for f in SAMPLE_FRACTIONS})
    return [float(flat[i]) for i in idx]


def _iter_files(path: Path):
    if path.is_file():
        yield path
    else:
        # A converted repo: every component, so connector tensors that were
        # routed into text_encoder/ are picked up alongside the transformer.
        yield from sorted(path.glob("*/*.safetensors"))
        yield from sorted(path.glob("*.safetensors"))


def summarize(path: Path) -> dict:
    entries: dict[str, dict] = {}
    skipped_scales = 0
    for shard in _iter_files(path):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for key in sorted(keys):
                if key.endswith(".weight_scale") or key.endswith(".comfy_quant"):
                    skipped_scales += 1
                    continue
                tensor = handle.get_tensor(key)
                # ComfyUI keeps fp8 payloads next to an F32 scale; dequantize
                # so both sides are compared in the same (real) value space.
                scale_key = f"{key}_scale" if key.endswith(".weight") else None
                if scale_key in keys:
                    tensor = tensor.float() * handle.get_tensor(scale_key).float()
                    quantized = True
                else:
                    tensor = tensor.float()
                    quantized = False
                if not torch.isfinite(tensor).all():
                    entries[canonical_key(key)] = {"nonfinite": True}
                    continue
                entries[canonical_key(key)] = {
                    "shape": list(tensor.shape),
                    "rms": float(tensor.pow(2).mean().sqrt()),
                    "absmax": float(tensor.abs().max()),
                    "mean": float(tensor.mean()),
                    "samples": _sample_values(tensor),
                    "quantized_source": quantized,
                }
    return {"source": str(path), "tensor_count": len(entries), "skipped_sidecars": skipped_scales,
            "tensors": entries}


def _relative(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def diff(a: dict, b: dict, tolerance: float, top: int) -> bool:
    ta, tb = a["tensors"], b["tensors"]
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    shared = sorted(set(ta) & set(tb))

    print(f"A: {a['source']}  ({a['tensor_count']} tensors)")
    print(f"B: {b['source']}  ({b['tensor_count']} tensors)")
    print(f"shared={len(shared)}  only-in-A={len(only_a)}  only-in-B={len(only_b)}\n")
    failed = False

    if only_a:
        failed = True
        print(f"[FAIL] {len(only_a)} tensor(s) present only in A — missing from the converted side:")
        for key in only_a[:top]:
            print(f"       {key}")
    if only_b:
        # Extra tensors are usually benign (converters emit config-shaped
        # buffers), so report without failing.
        print(f"[note] {len(only_b)} tensor(s) present only in B:")
        for key in only_b[:top]:
            print(f"       {key}")

    shape_mismatch, worst = [], []
    for key in shared:
        ea, eb = ta[key], tb[key]
        if ea.get("nonfinite") or eb.get("nonfinite"):
            shape_mismatch.append((key, "non-finite values"))
            continue
        if ea["shape"] != eb["shape"]:
            shape_mismatch.append((key, f"{ea['shape']} vs {eb['shape']}"))
            continue
        errs = [_relative(x, y) for x, y in zip(ea["samples"], eb["samples"], strict=False)]
        errs.append(_relative(ea["rms"], eb["rms"]))
        worst.append((max(errs) if errs else 0.0, key))

    if shape_mismatch:
        failed = True
        print(f"\n[FAIL] {len(shape_mismatch)} shape/validity mismatch(es):")
        for key, detail in shape_mismatch[:top]:
            print(f"       {key}: {detail}")

    worst.sort(reverse=True)
    over = [(e, k) for e, k in worst if e > tolerance]
    print(f"\nvalue agreement over {len(worst)} shared tensors (tolerance {tolerance:.3f}):")
    if worst:
        median = worst[len(worst) // 2][0]
        print(f"       median relative error {median:.5f}, worst {worst[0][0]:.5f} ({worst[0][1]})")
    if over:
        failed = True
        print(f"[FAIL] {len(over)} tensor(s) exceed the tolerance:")
        for err, key in over[:top]:
            print(f"       {err:.5f}  {key}")
    else:
        print("[PASS] every shared tensor agrees within tolerance")
    return not failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("summarize", help="emit a JSON fingerprint of a checkpoint or repo")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)

    d = sub.add_parser("diff", help="compare two fingerprints")
    d.add_argument("--a", required=True, help="reference fingerprint (e.g. the ComfyUI export)")
    d.add_argument("--b", required=True, help="fingerprint under test (the converted repo)")
    d.add_argument("--tolerance", type=float, default=E4M3_RELATIVE_ULP,
                   help="max relative error; defaults to one e4m3 ULP because the ComfyUI "
                        "reference re-quantizes its LoRA-merged weights to fp8")
    d.add_argument("--top", type=int, default=15)

    args = parser.parse_args()
    if args.command == "summarize":
        result = summarize(Path(args.input))
        Path(args.out).write_text(json.dumps(result))
        print(f"wrote {args.out}: {result['tensor_count']} tensors "
              f"({result['skipped_sidecars']} scale/quant sidecars skipped)")
        return

    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    raise SystemExit(0 if diff(a, b, args.tolerance, args.top) else 1)


if __name__ == "__main__":
    main()
