# SPDX-License-Identifier: Apache-2.0
"""
Convert LTX-2 weights to FastVideo naming conventions and split by component.

LTX 2 conversion requires two huggingface models:
- LTX 2 model
- Gemma model

Example usage:
    python scripts/checkpoint_conversion/convert_ltx2_weights.py \\
        --source "<PATH_TO_LOCAL_REPO>/Lightricks/LTX-2/ltx-2-19b-dev.safetensors" \\
        --output "converted_weights/ltx2-base" \\
        --class-name "LTX2Transformer3DModel" \\
        --pipeline-class-name "LTX2Pipeline" \\
        --diffusers-version "0.33.0.dev0" \\
        --gemma-path "<PATH_TO_LOCAL_REPO>/google/gemma-3-12b-it"

LTX-2.5 uses separate official comfy-style component files. Convert that layout with:

    python scripts/checkpoint_conversion/convert_ltx2_weights.py \
        --transformer-source diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
        --text-encoder-source text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
        --vae-source vae/ltx-2.5-video-vae-conv-bf16.safetensors \
        --audio-vae-source vae/ltx-2.5-audio-vae-bf16.safetensors \
        --spatial-upscaler-source latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
        --distilled-lora-source loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
        --variant dev \
        --output converted_weights/ltx2-5-dev

Every ``--*-source`` argument is independent: passing a subset converts (or replaces) just those
components inside ``--output``, so individual components can be swapped without re-converting the
rest. Examples:

    # Swap the conv video VAE for the diffusion/HQ one in an existing directory:
    python scripts/checkpoint_conversion/convert_ltx2_weights.py \
        --vae-source vae/ltx-2.5-video-vae-bf16.safetensors \
        --output converted_weights/ltx2-5-dev

    # Swap the dev transformer for the distilled one (also refreshes the connectors
    # that live inside text_encoder/model.safetensors):
    python scripts/checkpoint_conversion/convert_ltx2_weights.py \
        --transformer-source diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
        --variant distilled \
        --output converted_weights/ltx2-5-dev

Cross-component notes:

- The transformer file also carries the video/audio embeddings connectors, which belong to
  ``text_encoder/model.safetensors``; converting the transformer refreshes exactly those keys
  in-place (or stages them for a later ``--text-encoder-source`` run on a fresh directory).
- Converting only the text encoder needs the transformer's architecture metadata for the
  text-encoder config; it is read from an already-converted ``<output>/transformer/config.json``
  when ``--transformer-source`` is not given in the same run.
- ``model_index.json`` is (re)written whenever the output directory contains a complete
  component set, preserving previously recorded refine/variant fields unless overridden.

Only the official ``-bf16`` files are supported; quantized variants (``-comfy-int8-convrot``,
``-nvfp4``) are rejected with an error.

--vae-source accepts either LTX-2.5 video VAE file; the decoder flavor is detected from the
checkpoint's safetensors metadata (``config.vae._class_name``):

- ``vae/ltx-2.5-video-vae-conv-bf16.safetensors``  -> convolutional decoder (CausalVideoAutoencoder)
- ``vae/ltx-2.5-video-vae-bf16.safetensors``       -> diffusion/HQ decoder (CausalDiffusionVAE)

Which decoder runs at inference follows the converted vae/ directory the pipeline loads.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None

PARAM_NAME_MAP: dict[str, str] = {
    r"^model\.diffusion_model\.(.*)$": r"\1",
}

COMPONENT_PREFIXES: dict[str, tuple[str, ...]] = {
    "transformer": ("model.diffusion_model.", ),
    "vae": ("vae.", ),
    "audio_vae": ("audio_vae.", ),
    "vocoder": ("vocoder.", ),
    "text_embedding_projection": ("text_embedding_projection.", "model.text_embedding_projection."),
}

PACKED_GEMMA_CONFIG_METADATA_KEY = "gemma_config"
PACKED_GEMMA_TOKENIZER_KEY = "tokenizer_json"
PACKED_GEMMA_ASSET_PREFIX = "hf_asset__"

SPLIT_SOURCE_ARGUMENTS = (
    "transformer_source",
    "text_encoder_source",
    "vae_source",
    "audio_vae_source",
    "spatial_upscaler_source",
)

TEXT_PROJECTION_PREFIXES: dict[str, str] = {
    "text_embedding_projection.aggregate_embed.": "feature_extractor_linear.aggregate_embed.",
    "text_embedding_projection.video_aggregate_embed.": "video_feature_extractor_linear.",
    "text_embedding_projection.audio_aggregate_embed.": "audio_feature_extractor_linear.",
    "model.text_embedding_projection.aggregate_embed.": "feature_extractor_linear.aggregate_embed.",
    "model.text_embedding_projection.video_aggregate_embed.": "video_feature_extractor_linear.",
    "model.text_embedding_projection.audio_aggregate_embed.": "audio_feature_extractor_linear.",
}

TEXT_CONNECTOR_PREFIXES: dict[str, str] = {
    "model.diffusion_model.video_embeddings_connector.": "embeddings_connector.",
    "model.diffusion_model.audio_embeddings_connector.": "audio_embeddings_connector.",
    "model.diffusion_model.embeddings_connector.": "embeddings_connector.",
    "video_embeddings_connector.": "embeddings_connector.",
    "audio_embeddings_connector.": "audio_embeddings_connector.",
}

_CONV_VAE_CLASS_NAME = "CausalVideoAutoencoder"
_DIFFUSION_VAE_CLASS_NAME = "CausalDiffusionVAE"

# LTX-2.5 diffusion decoder remapping (mirrors the official
# ltx_core.model.video_vae.model_configurator SDOps / _strip_diffusion_decoder_prefix chain).
DIFFUSION_DECODER_RENAMES: dict[str, str] = {
    # The official t_embedder *is* PixArtAlphaCombinedTimestepSizeEmbeddings saved under
    # shorter names; only its two Linears need renaming.
    "t_embedder.mlp.0.": "t_embedder.timestep_embedder.linear_1.",
    "t_embedder.mlp.2.": "t_embedder.timestep_embedder.linear_2.",
}

_DIFFUSION_GATE_SUFFIXES = (".gate_msa", ".gate_mlp", ".gate_ctx")

# Intentionally skipped diffusion-decoder keys (key after the ``decoder.`` strip -> reason).
# Verified against the real ltx-2.5-video-vae-bf16.safetensors header (2026-08).
DIFFUSION_DECODER_SKIPPED_KEYS: dict[str, str] = {
    # Shipped in the released 2.5 HQ checkpoint but consumed by no released decoder:
    # the official ltx-core NADiffusionDecoder has no such attribute (its non-strict
    # load ignores the tensor) and the diffusers LTX2VideoDiffusionDecoderModel has
    # no such parameter either. Dropped explicitly so strict loads stay clean.
    "type_emb": "training-only type embedding; unused by every released decoder implementation",
}

# Static AdaLN gates carried by distilled checkpoints are folded into the Linear they gate
# (W <- g * W, b <- g * b) and dropped: the decoder's residuals are ungated. Post-rename leaf
# suffix -> sibling gate key suffix (official _GATE_FOLD_TARGETS).
DIFFUSION_GATE_FOLD_TARGETS: tuple[tuple[str, str], ...] = (
    (".attn.proj.weight", ".gate_msa"),
    (".attn.proj.bias", ".gate_msa"),
    (".mlp.w_down.weight", ".gate_mlp"),
    (".mlp.w_down.bias", ".gate_mlp"),
    (".context_proj.weight", ".gate_ctx"),
    (".context_proj.bias", ".gate_ctx"),
)


def _fold_gate(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fold a static gate into a Linear weight (rank 2) or bias (rank 1) in fp32."""
    gate_f = gate.to(torch.float32)
    value_f = value.to(torch.float32)
    if value.ndim == 2:
        folded = gate_f.unsqueeze(1) * value_f
    elif value.ndim == 1:
        folded = gate_f * value_f
    else:
        raise ValueError(f"Unsupported param rank {value.ndim} for gate fold")
    return folded.to(value.dtype)


def convert_diffusion_vae_weights(weights: dict[str, torch.Tensor]) -> OrderedDict:
    """Remap an official LTX-2.5 diffusion-VAE state dict onto FastVideo's native surface.

    Input keys have any ``vae.`` prefix already stripped: ``encoder.*`` (conv encoder,
    passthrough), ``decoder.*`` (diffusion decoder), and ``per_channel_statistics.*`` (shared
    latent stats, mirrored under both halves). Per the official loader chain this
    - drops bundled preview heads (``coarse_*``),
    - drops the documented :data:`DIFFUSION_DECODER_SKIPPED_KEYS` (e.g. ``type_emb``),
    - folds static AdaLN gates into the Linear they gate and drops the gates,
    - renames ``t_embedder.mlp.{0,2}`` to the timestep-embedder Linears,
    - splits fused ``qkv.{weight,bias}`` into ``qkv.to_q`` / ``qkv.to_k`` / ``qkv.to_v``.
    """
    gates = {
        key: value
        for key, value in weights.items()
        if key.startswith("decoder.") and key.endswith(_DIFFUSION_GATE_SUFFIXES)
    }

    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in weights.items():
        if key.startswith("encoder."):
            converted[key] = value
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics."):]
            converted[f"encoder.per_channel_statistics.{suffix}"] = value
            # Clone: safetensors refuses to serialize aliased tensors.
            converted[f"decoder.per_channel_statistics.{suffix}"] = value.detach().clone()
            continue
        if not key.startswith("decoder."):
            print(f"Skipping unexpected diffusion-VAE key: {key}")
            continue
        if key in gates:
            continue
        stripped = key[len("decoder."):]
        if stripped.startswith("coarse_") or ".coarse_" in stripped:
            continue
        if stripped in DIFFUSION_DECODER_SKIPPED_KEYS:
            print(f"Skipping diffusion-VAE key decoder.{stripped}: "
                  f"{DIFFUSION_DECODER_SKIPPED_KEYS[stripped]}")
            continue

        new_key = stripped
        for old, new in DIFFUSION_DECODER_RENAMES.items():
            new_key = new_key.replace(old, new)

        for leaf, gate_suffix in DIFFUSION_GATE_FOLD_TARGETS:
            if new_key.endswith(leaf):
                # Gates are siblings of the Linear they gate; renames never touch gated paths.
                gate = gates.get("decoder." + new_key[:-len(leaf)] + gate_suffix)
                if gate is not None:
                    value = _fold_gate(value, gate)
                break

        if new_key.endswith((".qkv.weight", ".qkv.bias")):
            leaf = "weight" if new_key.endswith(".weight") else "bias"
            prefix = new_key[:-len(leaf)]  # "...qkv."
            if value.shape[0] % 3 != 0:
                raise ValueError(f"Fused QKV param {key!r} leading dim {value.shape[0]} is not divisible by 3")
            chunk = value.shape[0] // 3
            converted[f"decoder.{prefix}to_q.{leaf}"] = value[:chunk].detach().clone()
            converted[f"decoder.{prefix}to_k.{leaf}"] = value[chunk:2 * chunk].detach().clone()
            converted[f"decoder.{prefix}to_v.{leaf}"] = value[2 * chunk:].detach().clone()
            continue

        converted[f"decoder.{new_key}"] = value

    if not any(key.startswith("decoder.diff_blocks.") for key in converted):
        raise ValueError("Diffusion VAE source contained no decoder.diff_blocks.* weights; "
                         "is this really the LTX-2.5 diffusion video VAE?")
    return converted


# Official quantized releases this converter refuses (bf16 files only).
_QUANTIZED_NAME_MARKERS = ("int8", "nvfp4", "fp8")
_QUANTIZED_HEADER_DTYPES = {"F8_E4M3", "F8_E5M2", "I8"}


def _read_safetensors_header_index(path: Path) -> dict[str, dict]:
    """Read only the safetensors JSON header (tensor name -> {dtype, shape, offsets})."""
    with path.open("rb") as f:
        (header_len, ) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _reject_quantized_source(source: Path, flag: str) -> None:
    """Raise for the official quantized component variants (comfy-int8-convrot / nvfp4).

    Detection is header-only: the filename markers plus any quantized tensor payload
    (fp8 / int8 anywhere, or uint8 ``*.weight`` — the packed-nvfp4 layout). The packed
    Gemma text encoder legitimately stores tokenizer assets as uint8 tensors, so plain
    uint8 non-weight tensors pass.
    """
    for shard in _find_shards(source):
        lowered = shard.name.lower()
        marker = next((m for m in _QUANTIZED_NAME_MARKERS if m in lowered), None)
        if marker is None:
            for key, info in _read_safetensors_header_index(shard).items():
                dtype = info.get("dtype")
                if dtype in _QUANTIZED_HEADER_DTYPES or (dtype == "U8" and key.endswith(".weight")):
                    marker = f"{dtype} tensor {key!r}"
                    break
        if marker is not None:
            raise ValueError(
                f"{flag}: {shard.name} is a quantized checkpoint ({marker}); quantized sources "
                "are not supported by this converter. Pass the matching -bf16 file instead "
                "(quantized deployment happens after conversion, not through it).")


def _find_shards(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]

    index_files = list(model_path.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as f:
            index = json.load(f)
        return sorted({model_path / shard for shard in index["weight_map"].values()})
    return sorted(Path(p) for p in glob.glob(str(model_path / "*.safetensors")))


def _apply_mapping(key: str) -> str:
    for pattern, replacement in PARAM_NAME_MAP.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def _load_weights(shards: list[Path]) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        weights.update(load_file(str(shard)))
    return weights


def _read_metadata_config(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    if not metadata or "config" not in metadata:
        return {}
    return json.loads(metadata["config"])


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def _metadata_config_for_component(metadata_config: dict, component_name: str) -> dict:
    component_config = metadata_config.get(component_name)
    if isinstance(component_config, dict):
        return component_config
    return metadata_config


def _filter_transformer_config(config: dict) -> dict:
    transformer = config.get("transformer", {})
    allowed = {
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "cross_attention_dim",
        "caption_channels",
        "norm_eps",
        "attention_type",
        "positional_embedding_theta",
        "positional_embedding_max_pos",
        "timestep_scale_multiplier",
        "use_middle_indices_grid",
        "rope_type",
        "frequencies_precision",
        "in_channels",
        "out_channels",
        "audio_num_attention_heads",
        "audio_attention_head_dim",
        "audio_in_channels",
        "audio_out_channels",
        "audio_cross_attention_dim",
        "audio_positional_embedding_max_pos",
        "av_ca_timestep_scale_multiplier",
        # LTX-2.3 architecture extensions.
        "cross_attention_adaln",
        "caption_proj_before_connector",
        "apply_gated_attention",
        "caption_projection_first_linear",
        "caption_proj_input_norm",
        "caption_projection_second_linear",
        "connector_num_attention_heads",
        "connector_attention_head_dim",
        "connector_num_layers",
        "audio_connector_num_attention_heads",
        "audio_connector_attention_head_dim",
        "audio_connector_num_layers",
        "connector_positional_embedding_max_pos",
        "connector_apply_gated_attention",
        "connector_ff_bias",
        # LTX-2.5 architecture extensions. Defaults live in the native model
        # config, so preserving an explicit False is important.
        "use_prompt_adaln_single",
        "ff_bias",
        "audio_ff_bias",
        "use_keyframes_abs_pos_embedding",
    }
    filtered = {k: v for k, v in transformer.items() if k in allowed}
    if "frequencies_precision" in filtered:
        filtered["double_precision_rope"] = filtered["frequencies_precision"] == "float64"
        del filtered["frequencies_precision"]
    return filtered


def _build_text_embedding_projection_config(gemma_model_path: str = "", ) -> dict:
    return {
        "architectures": ["LTX2GemmaTextEncoderModel"],
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 30,
        "text_len": 1024,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "gemma_model_path": gemma_model_path,
        "gemma_dtype": "bfloat16",
        "padding_side": "left",
        "feature_extractor_in_features": 3840 * 49,
        "feature_extractor_out_features": 3840,
        "connector_num_attention_heads": 30,
        "connector_attention_head_dim": 128,
        "connector_num_layers": 2,
        "connector_positional_embedding_theta": 10000.0,
        "connector_positional_embedding_max_pos": [4096],
        "connector_rope_type": "split",
        "connector_double_precision_rope": True,
        "connector_num_learnable_registers": 128,
    }


def _gemma_text_config(gemma_config: dict) -> dict:
    text_config = gemma_config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return gemma_config


def _build_split_text_encoder_config(transformer_config: dict, gemma_config: dict) -> dict:
    text_config = _gemma_text_config(gemma_config)
    hidden_size = int(text_config.get("hidden_size", 3840))
    num_hidden_layers = int(text_config.get("num_hidden_layers", 48))
    video_inner_dim = int(transformer_config.get("num_attention_heads", 30)) * int(
        transformer_config.get("attention_head_dim", 128))
    audio_inner_dim = int(transformer_config.get("audio_num_attention_heads", 30)) * int(
        transformer_config.get("audio_attention_head_dim", 128))

    eos_token_id = gemma_config.get("eos_token_id", text_config.get("eos_token_id", 2))
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else 2
    config = _build_text_embedding_projection_config(gemma_model_path="gemma")
    config.update({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(text_config.get("num_attention_heads", 30)),
        # The packed LTX tokenizer always pads/truncates to 1024 regardless of
        # the Gemma backbone's much larger context window.
        "text_len": 1024,
        "pad_token_id": int(gemma_config.get("pad_token_id", text_config.get("pad_token_id", 0)) or 0),
        "bos_token_id": int(gemma_config.get("bos_token_id", text_config.get("bos_token_id", 2)) or 2),
        "eos_token_id": int(eos_token_id),
        "feature_extractor_in_features": hidden_size * (num_hidden_layers + 1),
        "feature_extractor_out_features": video_inner_dim,
        "video_feature_extractor_out_features": video_inner_dim,
        "audio_feature_extractor_out_features": audio_inner_dim,
        "caption_proj_before_connector": bool(transformer_config.get("caption_proj_before_connector", False)),
        "caption_projection_first_linear": bool(transformer_config.get("caption_projection_first_linear", True)),
        "caption_proj_input_norm": bool(transformer_config.get("caption_proj_input_norm", True)),
        "caption_projection_second_linear": bool(transformer_config.get("caption_projection_second_linear", True)),
        "connector_num_attention_heads": int(transformer_config.get("connector_num_attention_heads", 30)),
        "connector_attention_head_dim": int(transformer_config.get("connector_attention_head_dim", 128)),
        "connector_num_layers": int(transformer_config.get("connector_num_layers", 2)),
        "audio_connector_num_attention_heads": int(
            transformer_config.get(
                "audio_connector_num_attention_heads",
                transformer_config.get("connector_num_attention_heads", 30),
            )),
        "audio_connector_attention_head_dim": int(
            transformer_config.get(
                "audio_connector_attention_head_dim",
                transformer_config.get("connector_attention_head_dim", 128),
            )),
        "audio_connector_num_layers": int(
            transformer_config.get(
                "audio_connector_num_layers",
                transformer_config.get("connector_num_layers", 2),
            )),
        "connector_positional_embedding_max_pos": transformer_config.get(
            "connector_positional_embedding_max_pos", [4096]),
        "connector_rope_type": transformer_config.get("rope_type", "split"),
        "connector_apply_gated_attention": bool(transformer_config.get("connector_apply_gated_attention", False)),
        "connector_ff_bias": bool(transformer_config.get("connector_ff_bias", True)),
        # Official metadata carries frequencies_precision; an already-converted
        # transformer/config.json carries the normalized double_precision_rope.
        "connector_double_precision_rope": bool(
            transformer_config.get(
                "double_precision_rope",
                transformer_config.get("frequencies_precision") == "float64",
            )),
    })
    return config


def _tensor_to_bytes(tensor: torch.Tensor, key: str) -> bytes:
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"Packed Gemma asset {key!r} must be a 1-D uint8 tensor, got {tensor.dtype} {tensor.shape}.")
    return bytes(tensor.cpu().tolist())


def _map_packed_gemma_weight_key(key: str) -> str:
    """Map the official Comfy-flat Gemma 4 pack back to HF key names.

    The official packer flattens the language tower and renames vision/audio
    components. Already-HF keys pass through unchanged so repacked checkpoints
    are also accepted.
    """
    if key.startswith("model.language_model.") or key.startswith("model.vision_embedder."):
        return key
    if key.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return "model.language_model." + key.removeprefix("model.")
    if key.startswith("vision_model."):
        return "model.vision_embedder." + key.removeprefix("vision_model.")
    if key.startswith("multi_modal_projector."):
        return "model.embed_vision." + key.removeprefix("multi_modal_projector.")
    if key.startswith("audio_projector."):
        return "model.embed_audio." + key.removeprefix("audio_projector.")
    return key


def _route_text_projection_key(key: str) -> str | None:
    for prefix, replacement in TEXT_PROJECTION_PREFIXES.items():
        if key.startswith(prefix):
            return replacement + key[len(prefix):]
    for prefix, replacement in TEXT_CONNECTOR_PREFIXES.items():
        if key.startswith(prefix):
            return replacement + key[len(prefix):]
    return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _unpack_packed_gemma(
    source_path: Path,
    output_dir: Path,
) -> tuple[OrderedDict, dict]:
    shards = _find_shards(source_path)
    if len(shards) != 1:
        raise ValueError("Packed LTX-2.5 text encoder must resolve to exactly one safetensors file.")
    source_file = shards[0]
    metadata = _read_safetensors_metadata(source_file)
    raw_gemma_config = metadata.get(PACKED_GEMMA_CONFIG_METADATA_KEY)
    if raw_gemma_config is None:
        raise ValueError(
            f"Packed text encoder {source_file} is missing {PACKED_GEMMA_CONFIG_METADATA_KEY!r} metadata.")
    gemma_config = json.loads(raw_gemma_config)

    gemma_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
    projection_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
    tokenizer_assets: dict[str, bytes] = {}
    for key, tensor in _load_weights(shards).items():
        projection_key = _route_text_projection_key(key)
        if projection_key is not None:
            projection_weights[projection_key] = tensor
        elif key == PACKED_GEMMA_TOKENIZER_KEY:
            tokenizer_assets["tokenizer.json"] = _tensor_to_bytes(tensor, key)
        elif key.startswith(PACKED_GEMMA_ASSET_PREFIX):
            asset_name = key.removeprefix(PACKED_GEMMA_ASSET_PREFIX)
            tokenizer_assets[asset_name] = _tensor_to_bytes(tensor, key)
        else:
            gemma_weights[_map_packed_gemma_weight_key(key)] = tensor

    if not gemma_weights:
        raise ValueError(f"Packed text encoder {source_file} did not contain Gemma model weights.")
    if "tokenizer.json" not in tokenizer_assets:
        raise ValueError(f"Packed text encoder {source_file} did not contain tokenizer_json.")

    gemma_dir = output_dir / "text_encoder" / "gemma"
    gemma_dir.mkdir(parents=True, exist_ok=True)
    save_file(gemma_weights, str(gemma_dir / "model.safetensors"))
    _write_json(gemma_dir / "config.json", gemma_config)
    for asset_name, asset_bytes in tokenizer_assets.items():
        _write_bytes(gemma_dir / asset_name, asset_bytes)
        _write_bytes(output_dir / "tokenizer" / asset_name, asset_bytes)
    return projection_weights, gemma_config


def _wrap_component_config(
    component_name: str,
    component_config: dict | None,
    class_name: str | None = None,
) -> dict | None:
    if component_config is None:
        return None
    wrapped = {component_name: component_config}
    if class_name is not None:
        wrapped["_class_name"] = class_name
    return wrapped


def _split_component_weights(weights: dict[str, torch.Tensor]) -> dict[str, OrderedDict]:
    components: dict[str, OrderedDict] = {name: OrderedDict() for name in COMPONENT_PREFIXES}
    for key, value in weights.items():
        if key.startswith("model.diffusion_model.audio_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
            components["text_embedding_projection"][new_key] = value
            continue
        if key.startswith("model.diffusion_model.video_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
            components["text_embedding_projection"][new_key] = value
            continue

        matched = False
        for component, prefixes in COMPONENT_PREFIXES.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    components[component][new_key] = value
                    matched = True
                    break
            if matched:
                break
    return {name: weights for name, weights in components.items() if weights}


def _strip_first_matching_prefix(key: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _split_transformer_and_connectors(
    weights: dict[str, torch.Tensor],
) -> tuple[OrderedDict, OrderedDict]:
    transformer: OrderedDict[str, torch.Tensor] = OrderedDict()
    text_encoder: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in weights.items():
        text_key = _route_text_projection_key(key)
        if text_key is not None:
            text_encoder[text_key] = tensor
            continue
        transformer_key = _strip_first_matching_prefix(
            key,
            ("model.diffusion_model.", "diffusion_model."),
        )
        transformer[transformer_key] = tensor
    return transformer, text_encoder


def _split_audio_vae_source(weights: dict[str, torch.Tensor]) -> tuple[OrderedDict, OrderedDict]:
    audio_vae: OrderedDict[str, torch.Tensor] = OrderedDict()
    vocoder: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in weights.items():
        if key.startswith("audio_vae."):
            audio_vae[key.removeprefix("audio_vae.")] = tensor
        elif key.startswith("vocoder."):
            # Strip exactly one prefix. LTX-2.3+ BWE keys intentionally keep
            # the second "vocoder." segment.
            vocoder[key.removeprefix("vocoder.")] = tensor
    if not audio_vae:
        raise ValueError("Audio VAE source contains no keys with the 'audio_vae.' prefix.")
    if not vocoder:
        raise ValueError("Audio VAE source contains no keys with the 'vocoder.' prefix.")
    return audio_vae, vocoder


def _component_weights(
    source_path: Path,
    prefixes: tuple[str, ...],
) -> OrderedDict:
    shards = _find_shards(source_path)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_path}")
    return OrderedDict(
        (_strip_first_matching_prefix(key, prefixes), tensor) for key, tensor in _load_weights(shards).items())


def _write_component(
    output_dir: Path,
    name: str,
    weights: OrderedDict,
    config: dict | None,
    dir_name: str | None = None,
) -> None:
    component_dir = output_dir / (dir_name or name)
    component_dir.mkdir(parents=True, exist_ok=True)
    output_file = component_dir / "model.safetensors"
    save_file(weights, str(output_file))
    print(f"Saved {name} weights to {output_file}")

    if config is not None:
        config_path = component_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Saved {name} config to {config_path}")


def _build_model_index(
    transformer_class_name: str,
    vae_class_name: str,
    pipeline_class_name: str,
    diffusers_version: str,
) -> dict:
    return {
        "_class_name": pipeline_class_name,
        "_diffusers_version": diffusers_version,
        "transformer": ["diffusers", transformer_class_name],
        "vae": ["diffusers", vae_class_name],
        "text_encoder": ["transformers", "LTX2GemmaTextEncoderModel"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "audio_vae": ["diffusers", "LTX2AudioDecoder"],
        "vocoder": ["diffusers", "LTX2Vocoder"],
    }


def _build_split_model_index(
    transformer_class_name: str,
    pipeline_class_name: str,
    diffusers_version: str,
    variant: str,
    distilled_lora: bool,
    vae_class_name: str = _CONV_VAE_CLASS_NAME,
) -> dict:
    model_index = _build_model_index(
        transformer_class_name=transformer_class_name,
        vae_class_name=vae_class_name,
        pipeline_class_name=pipeline_class_name,
        diffusers_version=diffusers_version,
    )
    model_index.update({
        "spatial_upsampler": ["diffusers", "LTX2LatentUpsampler"],
        "fastvideo_ltx2_variant": f"ltx2.5-{variant}",
        "fastvideo_refine_enabled": variant == "distilled" or distilled_lora,
        "fastvideo_refine_upsampler_path": "spatial_upsampler",
        "fastvideo_refine_num_inference_steps": 3,
        "fastvideo_refine_guidance_scale": 1.0,
        "fastvideo_refine_add_noise": True,
    })
    if distilled_lora:
        model_index["fastvideo_refine_lora_path"] = "distilled_lora/model.safetensors"
    return model_index


def _write_model_index(output_dir: Path, model_index: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_index_path = output_dir / "model_index.json"
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")
    print(f"Saved model_index.json to {model_index_path}")


def convert_components(
    source_path: Path,
    output_dir: Path,
    metadata_config: dict,
    transformer_class_name: str,
    components_to_write: set[str] | None = None,
    emit_diffusers_repo: bool = True,
    pipeline_class_name: str = "LTX2Pipeline",
    diffusers_version: str = "0.33.0.dev0",
    gemma_model_path: str = "",
) -> None:
    shards = _find_shards(source_path)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_path}")

    weights = _load_weights(shards)
    split_weights = _split_component_weights(weights)
    if components_to_write is not None:
        split_weights = {name: weights for name, weights in split_weights.items() if name in components_to_write}

    transformer_weights = split_weights.get("transformer", OrderedDict())
    converted_transformer = OrderedDict()
    for key, value in transformer_weights.items():
        new_key = _apply_mapping(f"model.diffusion_model.{key}")
        converted_transformer[new_key] = value
    split_weights["transformer"] = converted_transformer

    transformer_config = _filter_transformer_config(metadata_config)
    if transformer_config:
        transformer_config["_class_name"] = transformer_class_name

    component_configs: dict[str, dict | None] = {
        "transformer": transformer_config or None,
        "vae": _wrap_component_config(
            "vae",
            metadata_config.get("vae"),
            class_name="CausalVideoAutoencoder",
        ),
        "audio_vae": _wrap_component_config(
            "audio_vae",
            metadata_config.get("audio_vae"),
            class_name="LTX2AudioDecoder",
        ),
        "vocoder": _wrap_component_config(
            "vocoder",
            metadata_config.get("vocoder"),
            class_name="LTX2Vocoder",
        ),
        "text_embedding_projection": _build_text_embedding_projection_config(gemma_model_path=gemma_model_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, component_weights in split_weights.items():
        _write_component(output_dir, name, component_weights, component_configs.get(name))
        if emit_diffusers_repo and name == "text_embedding_projection":
            _write_component(
                output_dir,
                name,
                component_weights,
                component_configs.get(name),
                dir_name="text_encoder",
            )
    if emit_diffusers_repo:
        required_for_index = {
            "transformer",
            "vae",
            "audio_vae",
            "vocoder",
            "text_embedding_projection",
        }
        if components_to_write is not None and not required_for_index.issubset(components_to_write):
            print("Skipping model_index.json; not all diffusers components were written.")
            return
        if not required_for_index.issubset(split_weights.keys()):
            print("Skipping model_index.json; missing diffusers components in weights.")
            return
        vae_class_name = (component_configs.get("vae") or {}).get("_class_name", "CausalVideoAutoencoder")
        model_index = _build_model_index(
            transformer_class_name=transformer_class_name,
            vae_class_name=vae_class_name,
            pipeline_class_name=pipeline_class_name,
            diffusers_version=diffusers_version,
        )
        _write_model_index(output_dir, model_index)


def _source_metadata_config(source_path: Path) -> dict:
    shards = _find_shards(source_path)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_path}")
    return _read_metadata_config(shards[0])


# text_encoder/model.safetensors is assembled from two sources: the packed Gemma file
# contributes the aggregate projections, the transformer file contributes the embeddings
# connectors. These target-key prefixes identify the connector half for in-place swaps.
_TEXT_CONNECTOR_TARGET_PREFIXES = ("embeddings_connector.", "audio_embeddings_connector.")


def _merge_text_encoder_weights(
    output_dir: Path,
    new_weights: OrderedDict,
    *,
    replace: str,
) -> None:
    """Merge one half of text_encoder/model.safetensors in place.

    ``replace`` is ``"connectors"`` (drop existing connector keys, keep projections)
    or ``"projections"`` (the inverse); ``new_weights`` supplies the fresh half.
    This is what lets a transformer swap refresh its connectors without touching a
    previously converted packed-Gemma projection set, and vice versa.
    """
    if replace not in ("connectors", "projections"):
        raise ValueError(f"Unknown text-encoder merge mode: {replace!r}")
    target = output_dir / "text_encoder" / "model.safetensors"
    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    if target.exists():
        for key, tensor in load_file(str(target)).items():
            is_connector = key.startswith(_TEXT_CONNECTOR_TARGET_PREFIXES)
            if (replace == "connectors") == is_connector:
                continue
            merged[key] = tensor
    merged.update(new_weights)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(target))
    print(f"Saved text_encoder weights ({replace} refreshed) to {target}")


def _converted_transformer_metadata(output_dir: Path) -> dict | None:
    """Transformer architecture dict from an already-converted transformer/config.json."""
    config_path = output_dir / "transformer" / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.pop("_class_name", None)
    return config


def _convert_split_transformer(
    source: Path,
    output_dir: Path,
    transformer_class_name: str,
) -> dict:
    """Convert one official transformer file; returns its full metadata config.

    Also refreshes the connector half of ``text_encoder/model.safetensors`` — the
    connectors ship inside the transformer file but belong to the text encoder
    component, so a transformer swap must carry them along.
    """
    transformer_metadata = _source_metadata_config(source)
    full_transformer_config = _metadata_config_for_component(transformer_metadata, "transformer")
    transformer_config = _filter_transformer_config(transformer_metadata)
    if not transformer_config:
        raise ValueError("Transformer source is missing config.transformer metadata.")
    transformer_config["_class_name"] = transformer_class_name

    transformer_weights, connector_weights = _split_transformer_and_connectors(_load_weights(_find_shards(source)))
    if not transformer_weights:
        raise ValueError("Transformer source did not contain transformer weights.")

    _write_component(output_dir, "transformer", transformer_weights, transformer_config)
    if connector_weights:
        _merge_text_encoder_weights(output_dir, connector_weights, replace="connectors")
        if not (output_dir / "text_encoder" / "config.json").exists():
            print("text_encoder/ holds only the transformer's connectors so far; also run "
                  "--text-encoder-source to add the Gemma weights, projections, tokenizer, and config.")
    return full_transformer_config


def _convert_split_text_encoder(
    source: Path,
    output_dir: Path,
    transformer_metadata: dict | None,
) -> None:
    """Convert the packed Gemma 4 text encoder file.

    The text-encoder config derives from the transformer architecture metadata:
    from a ``--transformer-source`` converted in the same run when available,
    otherwise from the already-converted ``<output>/transformer/config.json``.
    """
    if transformer_metadata is None:
        transformer_metadata = _converted_transformer_metadata(output_dir)
    if transformer_metadata is None:
        raise ValueError(
            "Converting the text encoder requires transformer architecture metadata: pass "
            "--transformer-source in the same run, or convert the transformer into this "
            "--output first (text_encoder/config.json derives from transformer/config.json).")

    projection_weights, gemma_config = _unpack_packed_gemma(source, output_dir)
    if not projection_weights:
        raise ValueError("Text encoder source did not contain LTX text projections.")
    text_encoder_config = _build_split_text_encoder_config(transformer_metadata, gemma_config)
    _merge_text_encoder_weights(output_dir, projection_weights, replace="projections")
    _write_json(output_dir / "text_encoder" / "config.json", text_encoder_config)
    print(f"Saved text_encoder config to {output_dir / 'text_encoder' / 'config.json'}")


def _convert_split_vae(source: Path, output_dir: Path) -> str:
    """Convert either LTX-2.5 video VAE file; returns the written vae class name."""
    vae_metadata = _source_metadata_config(source)
    vae_config = _metadata_config_for_component(vae_metadata, "vae")
    # Metadata-driven decoder selection (mirrors the official is_diffusion_video_vae): the conv
    # VAE declares CausalVideoAutoencoder; anything else is the LTX-2.5 diffusion (HQ) decoder
    # VAE (ltx-2.5-video-vae-bf16.safetensors), normalized to CausalDiffusionVAE for dispatch.
    metadata_vae_class = vae_config.get("_class_name", _CONV_VAE_CLASS_NAME)
    if metadata_vae_class == _CONV_VAE_CLASS_NAME:
        vae_class_name = _CONV_VAE_CLASS_NAME
        vae_weights = _component_weights(source, ("vae.", "model.vae."))
    else:
        if metadata_vae_class != _DIFFUSION_VAE_CLASS_NAME:
            print(f"VAE metadata declares {metadata_vae_class!r}; converting it as the "
                  f"diffusion video VAE ({_DIFFUSION_VAE_CLASS_NAME}).")
        vae_class_name = _DIFFUSION_VAE_CLASS_NAME
        vae_weights = convert_diffusion_vae_weights(_component_weights(source, ("vae.", "model.vae.")))
    _write_component(
        output_dir,
        "vae",
        vae_weights,
        _wrap_component_config("vae", vae_config, class_name=vae_class_name),
    )
    return vae_class_name


def _convert_split_audio_vae(source: Path, output_dir: Path) -> None:
    audio_metadata = _source_metadata_config(source)
    audio_weights, vocoder_weights = _split_audio_vae_source(_load_weights(_find_shards(source)))
    audio_vae_config = _metadata_config_for_component(audio_metadata, "audio_vae")
    vocoder_config = _metadata_config_for_component(audio_metadata, "vocoder")
    if audio_vae_config is audio_metadata or vocoder_config is audio_metadata:
        raise ValueError("Audio VAE source metadata must contain both config.audio_vae and config.vocoder.")
    _write_component(
        output_dir,
        "audio_vae",
        audio_weights,
        _wrap_component_config("audio_vae", audio_vae_config, class_name="LTX2AudioDecoder"),
    )
    _write_component(
        output_dir,
        "vocoder",
        vocoder_weights,
        _wrap_component_config("vocoder", vocoder_config, class_name="LTX2Vocoder"),
    )


def _convert_split_upscaler(source: Path, output_dir: Path) -> None:
    upscaler_metadata = _source_metadata_config(source)
    upscaler_config = dict(upscaler_metadata)
    if "upsampler" in upscaler_config and isinstance(upscaler_config["upsampler"], dict):
        # Official upsampler checkpoints nest the architecture under "upsampler";
        # flatten so the loader-side configurator sees the plain field names.
        upscaler_config = dict(upscaler_config["upsampler"])
    upscaler_config["_class_name"] = "LTX2LatentUpsampler"
    upscaler_weights = _component_weights(
        source,
        ("spatial_upscaler.", "spatial_upsampler.", "upsampler.", "model."),
    )
    upscaler_weights = OrderedDict((f"upsampler.{key}" if key.split(".", 1)[0].isdigit() else key, value)
                                   for key, value in upscaler_weights.items())
    _write_component(output_dir, "spatial_upsampler", upscaler_weights, upscaler_config)


def _convert_split_distilled_lora(source: Path, output_dir: Path) -> None:
    lora_weights = _component_weights(source, ())
    lora_dir = output_dir / "distilled_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    # Preserve source metadata (lora_rank / lora_alpha / model_version) — the runtime
    # LoRA loader falls back to alpha == rank only when no alpha is recorded anywhere.
    metadata = {
        key: value
        for key, value in _read_safetensors_metadata(_find_shards(source)[0]).items()
        if key in ("lora_rank", "lora_alpha", "model_version", "license")
    }
    output_file = lora_dir / "model.safetensors"
    save_file(lora_weights, str(output_file), metadata=metadata or None)
    print(f"Saved distilled_lora weights to {output_file}")


_SPLIT_INDEX_REQUIRED_DIRS = (
    "transformer",
    "text_encoder",
    "vae",
    "audio_vae",
    "vocoder",
    "spatial_upsampler",
)


def _refresh_split_model_index(
    output_dir: Path,
    *,
    transformer_class_name: str,
    pipeline_class_name: str,
    diffusers_version: str,
    variant: str | None,
) -> None:
    """(Re)write model_index.json from the output directory's current contents.

    Runs after every split conversion, so swapping a single component keeps the
    index consistent (e.g. the vae class after a conv <-> HQ swap). Fields not
    derivable from the directory (variant) fall back to the existing index, so
    partial re-runs don't silently reset refine defaults.
    """
    missing = [
        name for name in _SPLIT_INDEX_REQUIRED_DIRS if not (output_dir / name / "config.json").exists()
    ]
    if missing:
        print("Skipping model_index.json; convert the remaining components first: " + ", ".join(missing))
        return

    existing: dict = {}
    model_index_path = output_dir / "model_index.json"
    if model_index_path.exists():
        with model_index_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    if variant is None:
        recorded = str(existing.get("fastvideo_ltx2_variant", ""))
        variant = recorded.removeprefix("ltx2.5-") if recorded.startswith("ltx2.5-") else "dev"

    with (output_dir / "vae" / "config.json").open("r", encoding="utf-8") as f:
        vae_class_name = json.load(f).get("_class_name", _CONV_VAE_CLASS_NAME)

    model_index = _build_split_model_index(
        transformer_class_name=transformer_class_name,
        pipeline_class_name=pipeline_class_name,
        diffusers_version=diffusers_version,
        variant=variant,
        distilled_lora=(output_dir / "distilled_lora" / "model.safetensors").exists(),
        vae_class_name=vae_class_name,
    )
    _write_model_index(output_dir, model_index)


def convert_split_components(
    *,
    output_dir: Path,
    transformer_source: Path | None = None,
    text_encoder_source: Path | None = None,
    vae_source: Path | None = None,
    audio_vae_source: Path | None = None,
    spatial_upscaler_source: Path | None = None,
    distilled_lora_source: Path | None = None,
    transformer_class_name: str = "LTX2Transformer3DModel",
    variant: str | None = None,
    emit_diffusers_repo: bool = True,
    pipeline_class_name: str = "LTX2Pipeline",
    diffusers_version: str = "0.33.0.dev0",
) -> None:
    """Convert the official LTX-2.5 separate-component (comfy-style) checkpoint layout.

    Every source is optional and converted independently into ``output_dir``, so
    components can be swapped one at a time (e.g. conv vs HQ video VAE, dev vs
    distilled transformer) without re-converting the rest. All routing is explicit
    so key drift fails visibly rather than being dropped. Quantized official
    variants are rejected — bf16 files only.
    """
    sources = {
        "--transformer-source": transformer_source,
        "--text-encoder-source": text_encoder_source,
        "--vae-source": vae_source,
        "--audio-vae-source": audio_vae_source,
        "--spatial-upscaler-source": spatial_upscaler_source,
        "--distilled-lora-source": distilled_lora_source,
    }
    if not any(source is not None for source in sources.values()):
        raise ValueError("LTX-2.5 split conversion needs at least one --*-source argument.")
    for flag, source in sources.items():
        if source is not None:
            _reject_quantized_source(source, flag)

    output_dir.mkdir(parents=True, exist_ok=True)

    transformer_metadata: dict | None = None
    if transformer_source is not None:
        transformer_metadata = _convert_split_transformer(transformer_source, output_dir, transformer_class_name)
    if text_encoder_source is not None:
        _convert_split_text_encoder(text_encoder_source, output_dir, transformer_metadata)
    if vae_source is not None:
        _convert_split_vae(vae_source, output_dir)
    if audio_vae_source is not None:
        _convert_split_audio_vae(audio_vae_source, output_dir)
    if spatial_upscaler_source is not None:
        _convert_split_upscaler(spatial_upscaler_source, output_dir)
    if distilled_lora_source is not None:
        _convert_split_distilled_lora(distilled_lora_source, output_dir)

    if emit_diffusers_repo:
        _refresh_split_model_index(
            output_dir,
            transformer_class_name=transformer_class_name,
            pipeline_class_name=pipeline_class_name,
            diffusers_version=diffusers_version,
            variant=variant,
        )


def update_transformer_config(config_path: Path, class_name: str) -> None:
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["_class_name"] = class_name
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Updated _class_name in {config_path} -> {class_name}")


def maybe_download(repo_id: str, target_dir: Path, token: str | None, allow_patterns: str | None) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for --download")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
    )
    return target_dir


def copy_gemma_tokenizer(gemma_src: Path, tokenizer_dest: Path) -> None:
    tokenizer_dest.mkdir(parents=True, exist_ok=True)
    tokenizer_file_names = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    ]
    copied = 0
    for file_name in tokenizer_file_names:
        src_path = gemma_src / file_name
        if src_path.is_file():
            shutil.copy2(src_path, tokenizer_dest / file_name)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No tokenizer files found in {gemma_src}. Expected at least one tokenizer file.")
    print(f"Copied {copied} tokenizer files to {tokenizer_dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LTX-2 weights to FastVideo format")
    parser.add_argument("--source", type=str, help="Path to transformer weights directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for converted weights")
    parser.add_argument("--download", type=str, help="HF repo id to download before conversion")
    parser.add_argument("--allow-patterns", type=str, help="Limit HF download to matching files")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    parser.add_argument("--update-config", action="store_true", help="Update source config.json _class_name")
    parser.add_argument("--class-name", type=str, default="LTX2Transformer3DModel")
    parser.add_argument(
        "--diffusers-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit a diffusers-style repo layout with model_index.json.",
    )
    parser.add_argument(
        "--pipeline-class-name",
        type=str,
        default="LTX2Pipeline",
        help="Pipeline class name for model_index.json.",
    )
    parser.add_argument(
        "--diffusers-version",
        type=str,
        default="0.33.0.dev0",
        help="Diffusers version for model_index.json.",
    )
    parser.add_argument(
        "--transformer-only",
        action="store_true",
        help="Only convert transformer weights (no component split).",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="",
        help=("Comma-separated component list to write "
              "(transformer,vae,audio_vae,vocoder,text_embedding_projection)."),
    )
    parser.add_argument(
        "--gemma-path",
        type=str,
        default="",
        help="Optional local Gemma model path to copy into the output repo.",
    )
    parser.add_argument(
        "--transformer-source",
        type=str,
        help="LTX-2.5 split transformer safetensors file or sharded directory.",
    )
    parser.add_argument(
        "--text-encoder-source",
        type=str,
        help="LTX-2.5 packed Gemma 4 text-encoder safetensors file.",
    )
    parser.add_argument(
        "--vae-source",
        type=str,
        help=("LTX-2.5 video VAE safetensors file or sharded directory. Accepts the convolutional "
              "VAE (ltx-2.5-video-vae-conv-bf16) or the diffusion/HQ decoder VAE "
              "(ltx-2.5-video-vae-bf16); the flavor is detected from checkpoint metadata."),
    )
    parser.add_argument(
        "--audio-vae-source",
        type=str,
        help="LTX-2.5 audio VAE safetensors containing audio_vae.* and vocoder.* keys.",
    )
    parser.add_argument(
        "--spatial-upscaler-source",
        type=str,
        help="LTX-2.5 spatial latent upscaler safetensors file or sharded directory.",
    )
    parser.add_argument(
        "--distilled-lora-source",
        type=str,
        help="LTX-2.5 distilled LoRA safetensors (two-stage refine / distilled-recipe adapter).",
    )
    parser.add_argument(
        "--variant",
        choices=("dev", "distilled"),
        default=None,
        help=("LTX-2.5 split transformer variant; controls bundled refine defaults. When omitted "
              "on a partial re-run, the existing model_index.json's recorded variant is kept "
              "(fresh directories default to dev)."),
    )

    args = parser.parse_args()

    split_values = {name: getattr(args, name) for name in SPLIT_SOURCE_ARGUMENTS}
    split_values["distilled_lora_source"] = args.distilled_lora_source
    split_mode = any(value is not None for value in split_values.values())
    if split_mode:
        # Each LTX-2.5 split source converts independently; pass any subset to
        # swap just those components inside --output.
        incompatible = []
        if args.source:
            incompatible.append("--source")
        if args.download:
            incompatible.append("--download")
        if args.gemma_path:
            incompatible.append("--gemma-path")
        if args.transformer_only:
            incompatible.append("--transformer-only")
        if args.components:
            incompatible.append("--components")
        if args.update_config:
            incompatible.append("--update-config")
        if incompatible:
            raise ValueError("LTX-2.5 split source arguments cannot be combined with " + ", ".join(incompatible))
        convert_split_components(
            output_dir=Path(args.output),
            transformer_source=(Path(args.transformer_source) if args.transformer_source else None),
            text_encoder_source=(Path(args.text_encoder_source) if args.text_encoder_source else None),
            vae_source=(Path(args.vae_source) if args.vae_source else None),
            audio_vae_source=(Path(args.audio_vae_source) if args.audio_vae_source else None),
            spatial_upscaler_source=(Path(args.spatial_upscaler_source) if args.spatial_upscaler_source else None),
            distilled_lora_source=(Path(args.distilled_lora_source) if args.distilled_lora_source else None),
            transformer_class_name=args.class_name,
            variant=args.variant,
            emit_diffusers_repo=args.diffusers_repo,
            pipeline_class_name=args.pipeline_class_name,
            diffusers_version=args.diffusers_version,
        )
        return

    if args.download:
        if args.source:
            raise ValueError("Use either --download or --source, not both.")
        source_dir = maybe_download(args.download, Path(args.output) / "download", args.token, args.allow_patterns)
    else:
        if not args.source:
            raise ValueError("--source is required when not using --download")
        source_dir = Path(args.source)

    output_dir = Path(args.output)
    shards = _find_shards(source_dir)
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {source_dir}")
    metadata_path = shards[0]
    metadata_config = _read_metadata_config(metadata_path)
    components_to_write: set[str] | None = None
    if args.transformer_only:
        components_to_write = {"transformer"}
    elif args.components:
        components_to_write = {component.strip() for component in args.components.split(",") if component.strip()}

    gemma_model_path = ""
    if args.gemma_path:
        gemma_src = Path(args.gemma_path)
        if not gemma_src.is_dir():
            raise ValueError(f"--gemma-path must be a directory: {gemma_src}")
        gemma_dest = output_dir / "text_encoder" / "gemma"
        if gemma_dest.exists():
            shutil.rmtree(gemma_dest)
        gemma_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(gemma_src, gemma_dest)
        copy_gemma_tokenizer(gemma_src, output_dir / "tokenizer")
        gemma_model_path = "gemma"

    convert_components(
        source_dir,
        output_dir,
        metadata_config,
        args.class_name,
        components_to_write=components_to_write,
        emit_diffusers_repo=args.diffusers_repo,
        pipeline_class_name=args.pipeline_class_name,
        diffusers_version=args.diffusers_version,
        gemma_model_path=gemma_model_path,
    )

    if args.update_config:
        if source_dir.is_dir():
            update_transformer_config(source_dir / "config.json", args.class_name)


if __name__ == "__main__":
    main()
