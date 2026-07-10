#!/usr/bin/env python3
"""Rebuild an HF-format Gemma3 directory from a ComfyUI fp8mixed text encoder.

Input: a ComfyUI-layout gemma safetensors (key_layout=comfy_gemma3_12b), e.g.
    gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors
where 48 layers x 7 projections are stored as F8_E4M3 `.weight` plus a scalar
F32 `.weight_scale` sibling, and everything else (embed_tokens, norms,
layernorms) is bf16. `.comfy_quant` descriptors and the embedded
`spiece_model` tokenizer blob are ignored (tokenizer comes from the donor).

Donor: a local snapshot of google/gemma-3-12b-it (HF multimodal layout,
Gemma3ForConditionalGeneration). The donor supplies:
  - config.json / generation_config.json / tokenizer files / index structure
  - vision_tower + multi_modal_projector weights (unused by LTX text encoding
    but required so `Gemma3ForConditionalGeneration.from_pretrained` loads
    without missing keys — this is exactly how FastVideo loads it, see
    fastvideo/models/encoders/gemma.py)

Output: a directory that drop-in replaces the donor, with all language-model
weights swapped for the dequantized abliterated ones:
    W_bf16 = W_fp8.float() * weight_scale

Key mapping (both transformers layouts handled):
    language_model.model.<X>  /  model.language_model.<X>   ->  model.<X>

Streams shard by shard; peak RAM ~= largest single tensor (~2 GB embed).

Usage:
    python dequant_gemma_fp8mixed_to_hf.py \
        --source gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors \
        --donor  /path/to/google/gemma-3-12b-it \
        --output /path/to/gemma-3-12b-it-ablit-hf
Then pass `--gemma-path <output>` to FastVideo's convert_ltx2_weights.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DONOR_LM_PREFIXES = ("language_model.model.", "model.language_model.")
DONOR_LM_HEAD_PREFIXES = ("language_model.", "model.language_model.")


def donor_to_comfy_key(donor_key: str) -> str | None:
    """Map a donor (HF multimodal) key to the comfy text-backbone key."""
    for p in DONOR_LM_PREFIXES:
        if donor_key.startswith(p):
            return "model." + donor_key[len(p):]
    # lm_head lives outside .model in HF layout; comfy file has no lm_head
    # (Gemma ties it to embed_tokens) so this returns the tied candidate.
    for p in DONOR_LM_HEAD_PREFIXES:
        if donor_key.startswith(p) and donor_key[len(p):] == "lm_head.weight":
            return "model.embed_tokens.weight"
    return None


COPIED_SIDE_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="comfy fp8mixed gemma safetensors file")
    ap.add_argument("--donor", required=True,
                    help="local google/gemma-3-12b-it snapshot directory")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    donor = Path(args.donor)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    index_path = donor / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index["weight_map"]
    else:
        single = donor / "model.safetensors"
        if not single.exists():
            sys.exit(f"donor has neither model.safetensors.index.json nor "
                     f"model.safetensors: {donor}")
        with safe_open(single, framework="pt", device="cpu") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
        index = None

    # shard -> [donor keys]
    shards: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        shards.setdefault(shard, []).append(k)

    stats = {"dequantized": 0, "bf16_from_source": 0, "donor_passthrough": 0,
             "tied_lm_head": 0}
    used_source_keys: set[str] = set()

    with safe_open(args.source, framework="pt", device="cpu") as src:
        src_keys = set(src.keys())

        def fetch_source(comfy_key: str) -> torch.Tensor | None:
            """Return the bf16 tensor for a comfy key, dequantizing if fp8."""
            if comfy_key in src_keys:
                t = src.get_tensor(comfy_key)
                used_source_keys.add(comfy_key)
                if t.dtype == torch.float8_e4m3fn:
                    scale_key = comfy_key[:-len(".weight")] + ".weight_scale"
                    if scale_key not in src_keys:
                        sys.exit(f"fp8 tensor {comfy_key} has no {scale_key}")
                    scale = src.get_tensor(scale_key).to(torch.float32)
                    used_source_keys.add(scale_key)
                    stats["dequantized"] += 1
                    return (t.to(torch.float32) * scale).to(torch.bfloat16)
                stats["bf16_from_source"] += 1
                return t
            return None

        for shard_name, keys in sorted(shards.items()):
            out_tensors: dict[str, torch.Tensor] = {}
            with safe_open(donor / shard_name, framework="pt",
                           device="cpu") as df:
                for dk in keys:
                    ck = donor_to_comfy_key(dk)
                    t = fetch_source(ck) if ck is not None else None
                    if t is None:
                        t = df.get_tensor(dk)
                        stats["donor_passthrough"] += 1
                    else:
                        donor_shape = tuple(df.get_slice(dk).get_shape())
                        if tuple(t.shape) != donor_shape:
                            sys.exit(f"shape mismatch for {dk} <- {ck}: "
                                     f"source {tuple(t.shape)} vs donor "
                                     f"{donor_shape}")
                        if dk.endswith("lm_head.weight"):
                            stats["tied_lm_head"] += 1
                    out_tensors[dk] = t
            save_file(out_tensors, str(output / shard_name),
                      metadata={"format": "pt"})
            print(f"wrote {shard_name} ({len(out_tensors)} tensors)")

    if index is not None:
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2))

    for name in COPIED_SIDE_FILES:
        p = donor / name
        if p.exists():
            shutil.copy2(p, output / name)

    leftovers = {k for k in src_keys - used_source_keys
                 if not k.endswith(".comfy_quant") and k != "spiece_model"}
    print(f"\nstats: {stats}")
    if leftovers:
        print(f"WARNING: {len(leftovers)} source tensors were never used, "
              f"e.g.: {sorted(leftovers)[:8]}")
        sys.exit(1)
    expected_lm = stats["dequantized"] + stats["bf16_from_source"]
    print(f"language-model tensors from ablit source: {expected_lm} "
          f"(expect 626 for gemma3-12b: 336 dequantized + 290 bf16)")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
