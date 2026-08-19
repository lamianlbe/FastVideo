"""Assemble a complete LTX-2.3 source checkpoint for convert_ltx23_weights.py.

The ComfyUI ModelSaveKJ export contains ONLY the LoRA-merged DiT
(model.diffusion_model.*). Everything else the converter needs — vae,
audio_vae, vocoder, text_embedding_projection — is untouched by the LoRAs
(verified: all 1660 LoRA targets live under model.diffusion_model.*), so it
comes verbatim from the base checkpoint.

Emits bf16: convert_ltx2_weights.py has NO fp8 handling, so any
F8_E4M3 payload left in the source would be written straight through into
the converted repo, where FastVideo's DiT loader (which expects bf16 and
quantizes at load time) cannot use it. Every fp8 weight is therefore
dequantized here with its `weight_scale` sibling, and the `weight_scale` /
`comfy_quant` sidecars are dropped. The base's safetensors metadata is kept
because the converter reads its `config` JSON.

Streaming: peak RAM is one tensor, not the ~29 GB total.
"""
import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

DIT_PREFIX = "model.diffusion_model."
SIDECAR_SUFFIXES = (".weight_scale", ".comfy_quant")
DTYPE_SIZE = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2,
              "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta


def nbytes(entry) -> int:
    size = DTYPE_SIZE[entry["dtype"]]
    for d in entry["shape"]:
        size *= d
    return size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", required=True, help="ComfyUI ModelSaveKJ export (merged DiT)")
    ap.add_argument("--base", required=True, help="original ComfyUI checkpoint (non-DiT parts + metadata)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dit_path, base_path, out_path = Path(args.dit), Path(args.base), Path(args.output)
    dit_header, dit_meta = read_header(dit_path)
    base_header, base_meta = read_header(base_path)

    def payload_keys(header, keep):
        return sorted(k for k in header if keep(k) and not k.endswith(SIDECAR_SUFFIXES))

    dit_keys = payload_keys(dit_header, lambda k: k.startswith(DIT_PREFIX))
    stray = payload_keys(dit_header, lambda k: not k.startswith(DIT_PREFIX))
    rest_keys = payload_keys(base_header, lambda k: not k.startswith(DIT_PREFIX))
    base_dit = len(payload_keys(base_header, lambda k: k.startswith(DIT_PREFIX)))

    print(f"merged DiT export : {len(dit_keys)} tensors" + (f" (+{len(stray)} non-DiT ignored)" if stray else ""))
    print(f"base non-DiT parts: {len(rest_keys)} tensors")
    if len(dit_keys) != base_dit:
        raise SystemExit(f"DiT tensor count mismatch: export has {len(dit_keys)}, base has {base_dit}. "
                         "The export must be a complete DiT.")

    plan = [(k, dit_header[k], "dit") for k in dit_keys] + [(k, base_header[k], "base") for k in rest_keys]
    out_header, offset, dequantized = {}, 0, 0
    for key, entry, src in plan:
        header = dit_header if src == "dit" else base_header
        # fp8 payloads become bf16; everything else keeps its stored dtype.
        if entry["dtype"].startswith("F8_") and f"{key}_scale" in header:
            dtype = "BF16"
            dequantized += 1
        else:
            dtype = entry["dtype"]
        size = nbytes({"dtype": dtype, "shape": entry["shape"]})
        out_header[key] = {"dtype": dtype, "shape": entry["shape"],
                           "data_offsets": [offset, offset + size]}
        offset += size
    print(f"dequantizing {dequantized} fp8 tensors to bf16")

    meta = dict(base_meta)  # carries the `config` JSON the converter parses
    meta["assembled_from"] = json.dumps({"dit": dit_path.name, "base": base_path.name})
    blob = json.dumps({"__metadata__": meta, **out_header}, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    print(f"writing {out_path} ({offset / 1e9:.1f} GB payload)")
    with safe_open(str(dit_path), framework="pt", device="cpu") as df, \
         safe_open(str(base_path), framework="pt", device="cpu") as bf, \
         out_path.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, (key, _, src) in enumerate(plan):
            handle = df if src == "dit" else bf
            tensor = handle.get_tensor(key)
            if out_header[key]["dtype"] == "BF16" and tensor.dtype != torch.bfloat16:
                scale = handle.get_tensor(f"{key}_scale").float()
                tensor = (tensor.float() * scale).to(torch.bfloat16)
            # view(uint8) reinterprets the raw bytes and works for fp8 dtypes,
            # which numpy cannot represent at all.
            out.write(tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes())
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)
    print(f"done: {len(plan)} tensors")


if __name__ == "__main__":
    main()
