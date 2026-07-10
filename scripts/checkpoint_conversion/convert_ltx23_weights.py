#!/usr/bin/env python3
"""One-step LTX-2.3 -> FastVideo diffusers conversion (convert + 2.3 patch).

Wraps two steps that must always run together for LTX-2.3 checkpoints:
  1. convert_ltx2_weights.py (component split, model_index.json, Gemma copy).
  2. patch_ltx23_configs.py — re-adds the LTX-2.3 architecture fields that
     the stock converter's 2.0-era allow-list drops (apply_gated_attention,
     cross_attention_adaln, connector geometry, ...). Without the patch,
     FastVideo builds a 2.0-shaped model and 2.3 weights fail to load.
     Patched output is key-identical to the official
     FastVideo/LTX-2.3-Distilled-Diffusers repo configs.

CPU-only; peak RAM ~= checkpoint size (~44 GB for 22B bf16) because the
upstream converter loads the whole file.

Usage:
    python scripts/checkpoint_conversion/convert_ltx23_weights.py \
        --source my_merged_ltx23_bf16.safetensors \
        --gemma-path /path/to/gemma-hf-dir \
        --output My-LTX-2.3-Distilled-Diffusers
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Sibling modules in scripts/checkpoint_conversion/ (the script directory is
# on sys.path when run directly).
import convert_ltx2_weights as upstream
from patch_ltx23_configs import (
    patch_text_encoder,
    patch_transformer,
    read_metadata_config,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="merged single-file LTX-2.3 checkpoint (.safetensors)")
    ap.add_argument("--gemma-path", required=True,
                    help="HF-format Gemma directory (e.g. output of "
                         "dequant_gemma_fp8mixed_to_hf.py)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--class-name", default="LTX2Transformer3DModel")
    ap.add_argument("--pipeline-class-name", default="LTX2Pipeline")
    ap.add_argument("--diffusers-version", default="0.33.0.dev0")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_file():
        sys.exit(f"--source must be a single safetensors file: {source}")
    gemma_src = Path(args.gemma_path)
    if not gemma_src.is_dir():
        sys.exit(f"--gemma-path must be a directory: {gemma_src}")
    output_dir = Path(args.output)

    # -- sanity: this really is a 2.3 checkpoint with config metadata --------
    metadata_config = upstream._read_metadata_config(source)
    if not metadata_config:
        sys.exit(f"{source} has no `config` safetensors metadata — the "
                 f"converter cannot emit component configs without it")
    tcfg = metadata_config.get("transformer", {})
    for flag in ("apply_gated_attention", "cross_attention_adaln",
                 "caption_proj_before_connector"):
        if not tcfg.get(flag):
            print(f"WARNING: transformer metadata has {flag}="
                  f"{tcfg.get(flag)!r} — expected True for LTX-2.3")

    # -- step A: Gemma copy (mirrors upstream main()) -------------------------
    gemma_dest = output_dir / "text_encoder" / "gemma"
    if gemma_dest.exists():
        shutil.rmtree(gemma_dest)
    gemma_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(gemma_src, gemma_dest)
    upstream.copy_gemma_tokenizer(gemma_src, output_dir / "tokenizer")

    # -- step B: component split + diffusers repo ----------------------------
    upstream.convert_components(
        source,
        output_dir,
        metadata_config,
        args.class_name,
        components_to_write=None,
        emit_diffusers_repo=True,
        pipeline_class_name=args.pipeline_class_name,
        diffusers_version=args.diffusers_version,
        gemma_model_path="gemma",
    )

    # -- step C: LTX-2.3 config patch -----------------------------------------
    patch_cfg = read_metadata_config(str(source)).get("transformer", {})
    patch_transformer(output_dir, patch_cfg)
    patch_text_encoder(output_dir, patch_cfg)

    # -- final structure check ------------------------------------------------
    required = [
        "model_index.json",
        "transformer/model.safetensors", "transformer/config.json",
        "vae/model.safetensors", "vae/config.json",
        "audio_vae/model.safetensors",
        "vocoder/model.safetensors",
        "text_encoder/model.safetensors", "text_encoder/config.json",
        "text_encoder/gemma",
        "tokenizer",
    ]
    missing = [r for r in required if not (output_dir / r).exists()]
    if missing:
        sys.exit(f"FAILED — output incomplete, missing: {missing}")

    index = json.loads((output_dir / "model_index.json").read_text())
    print(f"\nOK: {output_dir}")
    print(f"model_index components: "
          f"{[k for k in index if not k.startswith('_')]}")


if __name__ == "__main__":
    main()
