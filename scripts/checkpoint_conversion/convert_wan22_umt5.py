#!/usr/bin/env python3
"""Dequantize a ComfyUI scaled-fp8 UMT5-XXL text encoder to bf16 for the
diffusers/FastVideo text_encoder layout.

Handles both formats seen in the Wan2.2 stack (per inspect_wan22_stack.py):
- Official ``umt5_xxl_fp8_e4m3fn_scaled``: fp8 weights + per-key f32
  ``.scale_weight`` companions -> dequant = weight * scale.
- NSFW finetune ``nsfw_wan_umt5-xxl_fp8_scaled``: fp8 weights + a single
  ``scaled_fp8`` marker tensor and NO per-key scales -> direct cast
  (scale = 1), which is ComfyUI's fallback for that layout.

The embedded ``spiece_model`` tokenizer blob and the marker are dropped —
use the tokenizer/ and config.json from the official
Wan-AI/Wan2.2-I2V-A14B-Diffusers repo; only the encoder weights differ.

  python convert_wan22_umt5.py --source nsfw_wan_umt5-xxl_fp8_scaled.safetensors \
      --out out/nsfw/text_encoder

Then copy the official text_encoder/config.json next to the shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DROP_KEYS = {"scaled_fp8", "spiece_model"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="ComfyUI scaled-fp8 UMT5 safetensors")
    ap.add_argument("--out", required=True, help="Output text_encoder directory")
    ap.add_argument("--shard-gb", type=float, default=9.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"dequant_scaled": 0, "dequant_cast": 0, "plain": 0, "dropped": 0}
    shards: list[dict[str, torch.Tensor]] = [{}]
    shard_bytes = [0]
    limit = int(args.shard_gb * 1e9)

    with safe_open(args.source, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        scale_keys = {k for k in keys if k.endswith(".scale_weight")}
        has_marker = "scaled_fp8" in keys

        for key in sorted(keys):
            if key in DROP_KEYS or key.endswith(".scale_weight"):
                stats["dropped"] += 1
                continue
            t = f.get_tensor(key)
            if t.dtype == torch.float8_e4m3fn:
                sk = f"{key[:-len('.weight')]}.scale_weight" if key.endswith(".weight") else None
                if sk in scale_keys:
                    t = t.to(torch.float32) * f.get_tensor(sk).to(torch.float32)
                    stats["dequant_scaled"] += 1
                elif has_marker:
                    t = t.to(torch.float32)  # direct-cast layout (scale = 1)
                    stats["dequant_cast"] += 1
                else:
                    raise RuntimeError(f"fp8 tensor without scale or scaled_fp8 marker: {key}")
            else:
                t = t.to(torch.float32)
                stats["plain"] += 1

            t = t.to(torch.bfloat16).contiguous()
            nbytes = t.numel() * t.element_size()
            if shard_bytes[-1] + nbytes > limit and shards[-1]:
                shards.append({})
                shard_bytes.append(0)
            shards[-1][key] = t
            shard_bytes[-1] += nbytes

    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    for i, shard in enumerate(shards, 1):
        fname = (f"model-{i:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors")
        save_file(shard, str(out_dir / fname))
        for k, v in shard.items():
            weight_map[k] = fname
            total += v.numel() * v.element_size()
        print(f"  wrote {fname} ({sum(v.numel() * v.element_size() for v in shard.values()) / 1e9:.2f} GB)")
    if n > 1:
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))

    print(f"  stats: {stats}")
    # Post-hoc sanity: dequantized weight magnitudes should be O(0.01..50).
    biggest = max(shards[0].items(), key=lambda kv: kv[1].numel())
    print(f"  sanity: {biggest[0]} absmax={biggest[1].float().abs().max().item():.3f}")
    print("  copy config.json (+ tokenizer/ at repo level) from the official "
          "Wan2.2 diffusers text_encoder next to these shards")


if __name__ == "__main__":
    main()
