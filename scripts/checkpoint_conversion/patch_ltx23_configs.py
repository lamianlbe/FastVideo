#!/usr/bin/env python3
"""Patch a convert_ltx2_weights.py output repo with LTX-2.3 config fields.

FastVideo's stock scripts/checkpoint_conversion/convert_ltx2_weights.py was
written for LTX-2.0: its transformer-config allow-list drops the 2.3
architecture flags (apply_gated_attention, cross_attention_adaln,
caption_proj_before_connector, connector_* geometry) and its text-encoder
config hardcodes 2.0 connector values (30 heads / 2 layers). Loading such a
repo builds an LTX-2.0-shaped model that cannot accept 2.3 weights
(to_gate_logits, cross-attn AdaLN tables, 8-layer gated connectors).

This script re-reads the source checkpoint's safetensors metadata (the same
`config` JSON the converter used) and patches, in place:
  - <repo>/transformer/config.json
  - <repo>/text_encoder/config.json
  - <repo>/text_embedding_projection/config.json

The emitted key set mirrors FastVideo's official
FastVideo/LTX-2.3-Distilled-Diffusers repo configs.

Usage:
    python patch_ltx23_configs.py \
        --repo 10Eros-LTX-2.3-Distilled-Diffusers \
        --checkpoint 10Eros_v1.4_DMD_merged_bf16.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# transformer/config.json: 2.3 keys the stock converter drops, copied verbatim
# from the checkpoint metadata's transformer config when present.
TRANSFORMER_23_KEYS = (
    "apply_gated_attention",
    "cross_attention_adaln",
    "caption_proj_before_connector",
    "caption_proj_input_norm",
    "caption_projection_first_linear",
    "caption_projection_second_linear",
    "connector_num_attention_heads",
    "connector_attention_head_dim",
    "connector_num_layers",
    "audio_connector_num_attention_heads",
    "audio_connector_attention_head_dim",
)


def read_metadata_config(path: str) -> dict:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    meta = header.get("__metadata__", {})
    if "config" not in meta:
        sys.exit(f"checkpoint has no `config` metadata: {path}")
    return json.loads(meta["config"])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def patch_transformer(repo: Path, tcfg: dict) -> None:
    path = repo / "transformer" / "config.json"
    cfg = load_json(path)
    added = {}
    for key in TRANSFORMER_23_KEYS:
        if key in tcfg:
            added[key] = tcfg[key]
    missing = [k for k in TRANSFORMER_23_KEYS if k not in tcfg]
    cfg.update(added)
    save_json(path, cfg)
    print(f"patched {path}: set {len(added)} keys {sorted(added)}")
    if missing:
        print(f"  note: not present in checkpoint metadata (left at FastVideo "
              f"defaults): {missing}")


def patch_text_encoder(repo: Path, tcfg: dict) -> None:
    """Rewrite the connector geometry in the text-encoder config to match the
    checkpoint (the stock converter hardcodes LTX-2.0 values)."""
    connector_layers = tcfg.get("connector_num_layers", 8)
    updates = {
        "caption_proj_before_connector": tcfg.get("caption_proj_before_connector", True),
        "caption_projection_first_linear": tcfg.get("caption_projection_first_linear", False),
        "caption_proj_input_norm": tcfg.get("caption_proj_input_norm", False),
        "caption_projection_second_linear": tcfg.get("caption_projection_second_linear", False),
        "connector_num_attention_heads": tcfg.get("connector_num_attention_heads", 32),
        "connector_attention_head_dim": tcfg.get("connector_attention_head_dim", 128),
        "connector_num_layers": connector_layers,
        "audio_connector_num_attention_heads": tcfg.get("audio_connector_num_attention_heads", 32),
        "audio_connector_attention_head_dim": tcfg.get("audio_connector_attention_head_dim", 64),
        "audio_connector_num_layers": tcfg.get("audio_connector_num_layers", connector_layers),
        "connector_positional_embedding_theta": tcfg.get("connector_positional_embedding_theta", 10000.0),
        "connector_positional_embedding_max_pos": tcfg.get("connector_positional_embedding_max_pos", [4096]),
        "connector_rope_type": "split",
        "connector_double_precision_rope": True,
        "connector_apply_gated_attention": tcfg.get("connector_apply_gated_attention", True),
        "connector_num_learnable_registers": tcfg.get("connector_num_learnable_registers", 128),
        # LTX-2.3 dual text projections (video 4096-d, audio 2048-d)
        "video_feature_extractor_out_features": tcfg.get("cross_attention_dim", 4096),
        "audio_feature_extractor_out_features": tcfg.get("audio_cross_attention_dim", 2048),
    }
    for sub in ("text_encoder", "text_embedding_projection"):
        path = repo / sub / "config.json"
        if not path.exists():
            print(f"  skip missing {path}")
            continue
        cfg = load_json(path)
        cfg.update(updates)  # gemma_model_path etc. preserved from converter
        save_json(path, cfg)
        print(f"patched {path}: connector "
              f"{updates['connector_num_attention_heads']}h/"
              f"{updates['connector_num_layers']}L gated="
              f"{updates['connector_apply_gated_attention']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="output dir of convert_ltx2_weights.py")
    ap.add_argument("--checkpoint", required=True,
                    help="merged single-file checkpoint (metadata source)")
    args = ap.parse_args()

    repo = Path(args.repo)
    cfg = read_metadata_config(args.checkpoint)
    tcfg = cfg.get("transformer", {})
    if not tcfg:
        sys.exit("checkpoint metadata has no transformer config")

    for flag in ("apply_gated_attention", "cross_attention_adaln",
                 "caption_proj_before_connector"):
        if not tcfg.get(flag):
            print(f"WARNING: metadata says {flag}={tcfg.get(flag)} — this "
                  f"does not look like an LTX-2.3 checkpoint; patching anyway")

    patch_transformer(repo, tcfg)
    patch_text_encoder(repo, tcfg)
    print("done")


if __name__ == "__main__":
    main()
