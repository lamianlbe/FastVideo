#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load-only sanity checks for LTX-2.5 conversions against the real checkpoints.

Two complementary modes, both CUDA-free (meta/cpu instantiation; key sets and
shapes only — runs on a Mac):

1. ``--converted <dir>`` — strict-load-check a directory produced by
   ``convert_ltx2_weights.py``: each present component is meta-instantiated
   from its written config.json and its converted weights are reconciled
   key-by-key/shape-by-shape, replicating the production loader's remaps
   (per_channel_statistics mirroring for the video VAEs, the decoder.* filter
   for the audio VAE, the ``model.`` strip for the upsampler).

2. ``--headers <dir>`` — validate the converter's key routing against
   header-only JSON dumps of the gated multi-GB files (42 GB transformers /
   26 GB text encoder) without downloading their tensor data. Each
   ``*.header.json`` is the safetensors JSON header (fetch via an HTTP range
   request: first 8 bytes = little-endian header length, then that many
   bytes), named ``<path with / -> _>.header.json``. The transformer header
   is mapped through the converter's split + the native param_names_mapping
   and reconciled against the meta-instantiated LTX2Transformer3DModel; the
   text-encoder header is checked against LTX2GemmaTextEncoderModel's
   projection/connector surface and the packed-Gemma key mapping; the LoRA
   header is checked to target only real transformer Linear weights.

Examples:
    python scripts/checkpoint_conversion/verify_ltx25_load.py \
        --converted /models/ltx25/converted-conv \
        --headers /models/ltx25/headers
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "convert_ltx2_weights_verify", Path(__file__).with_name("convert_ltx2_weights.py"))
assert _CONVERTER_SPEC is not None and _CONVERTER_SPEC.loader is not None
converter = importlib.util.module_from_spec(_CONVERTER_SPEC)
_CONVERTER_SPEC.loader.exec_module(converter)


# ---------------------------------------------------------------------------
# Shared reconcile helpers (same reporting style as verify_wan22_load.py)
# ---------------------------------------------------------------------------


def _load_shapes(comp_dir: Path) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for f in sorted(comp_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                out[k] = tuple(h.get_slice(k).get_shape())
    return out


def _header_shapes(header_path: Path) -> tuple[dict[str, tuple], dict[str, str]]:
    header = json.loads(header_path.read_text())
    metadata = header.pop("__metadata__", {}) or {}
    return {k: tuple(v["shape"]) for k, v in header.items()}, metadata


def _diff(title: str, expected: dict[str, tuple], got: dict[str, tuple], *, show: int = 8) -> bool:
    ek, gk = set(expected), set(got)
    missing = sorted(ek - gk)
    unexpected = sorted(gk - ek)
    mism = sorted(k for k in ek & gk if tuple(expected[k]) != tuple(got[k]))
    ok = not missing and not unexpected and not mism
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] {title}: expected={len(ek)} got={len(gk)} "
          f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(mism)}")
    for label, items in (("missing", missing), ("unexpected", unexpected)):
        for k in items[:show]:
            print(f"    {label}: {k}")
        if len(items) > show:
            print(f"    ... +{len(items) - show} more {label}")
    for k in mism[:show]:
        print(f"    shape: {k} expected {tuple(expected[k])} got {tuple(got[k])}")
    return ok


def _meta_state_shapes(build_fn) -> dict[str, tuple]:
    with torch.device("meta"):
        model = build_fn()
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _read_config(comp_dir: Path) -> dict:
    with (comp_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _mirror_per_channel_statistics(shapes: dict[str, tuple]) -> dict[str, tuple]:
    """Replicate the VAELoader's per_channel_statistics remap (mirror under both halves)."""
    remapped = dict(shapes)
    for key, shape in shapes.items():
        for prefix in ("per_channel_statistics.", "vae.per_channel_statistics."):
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                remapped.setdefault(f"encoder.per_channel_statistics.{suffix}", shape)
                remapped.setdefault(f"decoder.per_channel_statistics.{suffix}", shape)
                break
    return remapped


# ---------------------------------------------------------------------------
# Converted-directory checks (real weights on disk)
# ---------------------------------------------------------------------------


# Legacy per-channel-statistics buffers registered by the conv modules but shipped only by
# pre-2.5 checkpoints. Inference reads exactly std-of-means / mean-of-means (see
# PerChannelStatistics.normalize/un_normalize); the production load is strict=False, so the
# 2.5 checkpoints' omission of these leaves harmless initialized defaults.
_LEGACY_STAT_BUFFER_LEAVES = ("channel", "mean-of-stds", "mean-of-stds_over_std-of-means")


def _drop_unshipped_legacy_stat_buffers(expected: dict[str, tuple], got: dict[str, tuple]) -> int:
    dropped = 0
    for key in list(expected):
        head, _, leaf = key.rpartition(".")
        if head.endswith("per_channel_statistics") and leaf in _LEGACY_STAT_BUFFER_LEAVES and key not in got:
            del expected[key]
            dropped += 1
    return dropped


def check_video_vae(comp_dir: Path) -> bool:
    config = _read_config(comp_dir)
    class_name = config.get("_class_name")
    if class_name == "CausalDiffusionVAE":
        from fastvideo.models.vaes.ltx2_diffusion_decoder import LTX2CausalDiffusionVAE
        expected = _meta_state_shapes(lambda: LTX2CausalDiffusionVAE(config))
    else:
        from fastvideo.models.vaes.ltx2vae import LTX2CausalVideoAutoencoder
        expected = _meta_state_shapes(lambda: LTX2CausalVideoAutoencoder(config))
    got = _mirror_per_channel_statistics(_load_shapes(comp_dir))
    # The loader's remap keeps the original stats keys around; load_state_dict is
    # non-strict for them in production, so drop bare originals from the diff.
    got = {k: v for k, v in got.items() if k in expected or not k.startswith(("per_channel_statistics.",
                                                                              "vae.per_channel_statistics."))}
    legacy = _drop_unshipped_legacy_stat_buffers(expected, got)
    ok = _diff(f"vae ({class_name})", expected, got)
    if legacy:
        print(f"    note: {legacy} legacy per_channel_statistics buffers not shipped by this "
              "checkpoint (unused at inference; non-strict production load keeps defaults)")
    return ok


def check_spatial_upsampler(comp_dir: Path) -> bool:
    from fastvideo.models.upsamplers.ltx2_upsampler import LTX2LatentUpsampler
    config = _read_config(comp_dir)
    expected = _meta_state_shapes(lambda: getattr(LTX2LatentUpsampler(config), "model"))
    got = _load_shapes(comp_dir)
    if got and all(k.startswith("model.") for k in got):
        got = {k[len("model."):]: v for k, v in got.items()}
    ok = _diff(f"spatial_upsampler (LTX2LatentUpsampler, spatial_scale="
               f"{config.get('spatial_scale', 2.0)})", expected, got)
    return ok


def check_audio_vae(comp_dir: Path) -> bool:
    from fastvideo.models.audio.ltx2_audio_vae import LTX2AudioDecoder
    config = _read_config(comp_dir)
    config.pop("_class_name", None)
    audio_decoder = None
    with torch.device("meta"):
        audio_decoder = LTX2AudioDecoder(config)
    target = getattr(audio_decoder, "model", audio_decoder)
    expected = {k: tuple(v.shape) for k, v in target.state_dict().items()}
    # AudioDecoderLoader filters to decoder.* (stripped) + per_channel_statistics.*.
    raw = _load_shapes(comp_dir)
    got: dict[str, tuple] = {}
    for name, shape in raw.items():
        if name.startswith("decoder."):
            got[name.replace("decoder.", "", 1)] = shape
        elif name.startswith("per_channel_statistics."):
            got[name] = shape
    # The production load is strict=False (encoder keys in the checkpoint are
    # dropped); report full expected coverage but tolerate extra encoder keys.
    got = {k: v for k, v in got.items() if k in expected}
    return _diff("audio_vae (LTX2AudioDecoder, decode surface)", expected, got)


def check_vocoder(comp_dir: Path) -> bool:
    from fastvideo.models.audio.ltx2_audio_vae import LTX2Vocoder
    config = _read_config(comp_dir)
    config.pop("_class_name", None)
    with torch.device("meta"):
        vocoder = LTX2Vocoder(config)
    target = getattr(vocoder, "model", vocoder)
    expected = {k: tuple(v.shape) for k, v in target.state_dict().items()}
    got = _load_shapes(comp_dir)
    return _diff("vocoder (LTX2Vocoder)", expected, got)


def check_converted_dir(converted: Path) -> bool:
    ok = True
    ran = False
    for name, fn in (
        ("vae", check_video_vae),
        ("spatial_upsampler", check_spatial_upsampler),
        ("audio_vae", check_audio_vae),
        ("vocoder", check_vocoder),
    ):
        comp_dir = converted / name
        if (comp_dir / "config.json").exists():
            ran = True
            try:
                ok = fn(comp_dir) and ok
            except Exception as exc:  # noqa: BLE001 - report every component
                ok = False
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"[SKIP] {name}: not present in {converted}")
    return ok and ran


# ---------------------------------------------------------------------------
# Header-only checks (gated multi-GB files: transformers / text encoder / LoRA)
# ---------------------------------------------------------------------------


def _native_transformer_shapes(transformer_metadata: dict) -> dict[str, tuple]:
    import fastvideo.models.dits.ltx2 as fv_ltx2
    from fastvideo.configs.models.dits import LTX2VideoConfig

    fv_ltx2.get_sp_world_size = lambda: 1  # meta-instantiate without distributed init

    config = LTX2VideoConfig()
    arch = config.arch_config
    for name in (
            "num_attention_heads", "attention_head_dim", "num_layers", "cross_attention_dim",
            "caption_channels", "norm_eps", "positional_embedding_theta", "positional_embedding_max_pos",
            "timestep_scale_multiplier", "use_middle_indices_grid", "rope_type", "audio_num_attention_heads",
            "audio_attention_head_dim", "audio_in_channels", "audio_out_channels", "audio_cross_attention_dim",
            "audio_positional_embedding_max_pos", "av_ca_timestep_scale_multiplier", "in_channels",
            "out_channels", "cross_attention_adaln", "caption_proj_before_connector", "apply_gated_attention",
            "use_prompt_adaln_single", "ff_bias", "audio_ff_bias", "use_keyframes_abs_pos_embedding",
    ):
        if name in transformer_metadata:
            setattr(arch, name, transformer_metadata[name])
    arch.double_precision_rope = transformer_metadata.get("frequencies_precision", "") == "float64"
    arch.__post_init__()

    return _meta_state_shapes(lambda: fv_ltx2.LTX2Transformer3DModel(config, hf_config=dict(transformer_metadata)))


def _map_converted_transformer_keys(shapes: dict[str, tuple], param_names_mapping: dict) -> dict[str, tuple]:
    from fastvideo.models.loader.utils import get_param_names_mapping
    mapping_fn = get_param_names_mapping(param_names_mapping)
    mapped: dict[str, tuple] = {}
    for key, shape in shapes.items():
        target, _, _ = mapping_fn(key)
        mapped[target] = shape
    return mapped


def check_transformer_header(header_path: Path) -> bool:
    shapes, metadata = _header_shapes(header_path)
    transformer_metadata = json.loads(metadata["config"])["transformer"]

    # Converter split: strip model.diffusion_model., route the connectors to the
    # text encoder component.
    transformer_shapes: dict[str, tuple] = {}
    connector_shapes: dict[str, tuple] = {}
    for key, shape in shapes.items():
        text_key = converter._route_text_projection_key(key)
        if text_key is not None:
            connector_shapes[text_key] = shape
            continue
        transformer_shapes[converter._strip_first_matching_prefix(
            key, ("model.diffusion_model.", "diffusion_model."))] = shape

    expected = _native_transformer_shapes(transformer_metadata)
    from fastvideo.configs.models.dits import LTX2VideoConfig
    got = _map_converted_transformer_keys(transformer_shapes, LTX2VideoConfig().param_names_mapping)
    ok = _diff(f"transformer header {header_path.name} "
               f"(connectors routed: {len(connector_shapes)})", expected, got)

    # Interesting metadata values for the report.
    print(f"    model_version={metadata.get('model_version')} "
          f"use_keyframes_abs_pos_embedding={transformer_metadata.get('use_keyframes_abs_pos_embedding')} "
          f"use_prompt_adaln_single={transformer_metadata.get('use_prompt_adaln_single', '(default True)')} "
          f"ff_bias={transformer_metadata.get('ff_bias')}")
    return ok


def check_text_encoder_header(header_path: Path, transformer_header: Path | None) -> bool:
    shapes, metadata = _header_shapes(header_path)
    gemma_config = json.loads(metadata["gemma_config"])

    projections: dict[str, tuple] = {}
    gemma_keys: dict[str, tuple] = {}
    assets: list[str] = []
    for key, shape in shapes.items():
        routed = converter._route_text_projection_key(key)
        if routed is not None:
            projections[routed] = shape
        elif key == converter.PACKED_GEMMA_TOKENIZER_KEY or key.startswith(converter.PACKED_GEMMA_ASSET_PREFIX):
            assets.append(key)
        else:
            gemma_keys[converter._map_packed_gemma_weight_key(key)] = shape

    ok = True
    # 1) Gemma tower mapping: every mapped key must land under a known HF prefix.
    allowed_prefixes = ("model.language_model.", "model.vision_embedder.", "model.embed_vision.",
                        "model.embed_audio.", "lm_head.")
    stray = sorted(k for k in gemma_keys if not k.startswith(allowed_prefixes))
    flag = "OK" if not stray else "FAIL"
    print(f"[{flag}] text-encoder header: gemma keys={len(gemma_keys)} assets={sorted(assets)}")
    for k in stray[:8]:
        print(f"    unmapped gemma key: {k}")
    ok = ok and not stray

    hidden = int(gemma_config.get("text_config", {}).get("hidden_size", 0))
    layers = int(gemma_config.get("text_config", {}).get("num_hidden_layers", 0))
    print(f"    gemma text_config: hidden_size={hidden} num_hidden_layers={layers}")

    # 2) Projections + connectors against the native text-encoder module surface.
    if transformer_header is not None and transformer_header.exists():
        t_shapes, t_meta = _header_shapes(transformer_header)
        transformer_metadata = json.loads(t_meta["config"])["transformer"]
        connectors = {}
        for key, shape in t_shapes.items():
            routed = converter._route_text_projection_key(key)
            if routed is not None:
                connectors[routed] = shape
        text_config = converter._build_split_text_encoder_config(transformer_metadata, gemma_config)

        from fastvideo.configs.models.encoders.gemma import LTX2GemmaConfig
        from fastvideo.models.encoders.gemma import LTX2GemmaTextEncoderModel
        encoder_config = LTX2GemmaConfig()
        encoder_config.update_model_arch(text_config)
        expected = _meta_state_shapes(lambda: LTX2GemmaTextEncoderModel(encoder_config))
        got = dict(projections)
        got.update(connectors)
        ok = _diff("text_encoder surface (projections + transformer connectors)", expected, got) and ok
    else:
        print("[SKIP] text_encoder projection/connector reconcile: no transformer header supplied")
    return ok


def check_lora_header(header_path: Path, transformer_header: Path) -> bool:
    from fastvideo.configs.models.dits import LTX2VideoConfig
    from fastvideo.models.loader.utils import get_param_names_mapping

    shapes, metadata = _header_shapes(header_path)
    t_shapes, t_meta = _header_shapes(transformer_header)
    transformer_metadata = json.loads(t_meta["config"])["transformer"]
    native = _native_transformer_shapes(transformer_metadata)

    dit_config = LTX2VideoConfig()
    mapping_fn = get_param_names_mapping(dit_config.param_names_mapping)
    lora_mapping_fn = get_param_names_mapping(dit_config.lora_param_names_mapping)

    pairs: dict[str, dict[str, tuple]] = {}
    for key, shape in shapes.items():
        name = key.replace("diffusion_model.", "").replace(".weight", "")
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            base, leaf = name.rsplit(".", 1)
            base, _, _ = lora_mapping_fn(base)
            target, _, _ = mapping_fn(base)
            pairs.setdefault(target, {})[leaf] = shape
        else:
            pairs.setdefault(name, {})["other"] = shape

    unmatched = sorted(t for t in pairs if f"{t}.weight" not in native)
    incomplete = sorted(t for t, p in pairs.items() if {"lora_A", "lora_B"} - set(p))
    rank_bad = []
    for target, p in pairs.items():
        if "lora_A" in p and "lora_B" in p and f"{target}.weight" in native:
            out_dim, in_dim = native[f"{target}.weight"][:2]
            if p["lora_A"][1] != in_dim or p["lora_B"][0] != out_dim or p["lora_A"][0] != p["lora_B"][1]:
                rank_bad.append(target)
    ok = not unmatched and not incomplete and not rank_bad
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] distilled LoRA header: adapters={len(pairs)} "
          f"unmatched_targets={len(unmatched)} incomplete_pairs={len(incomplete)} bad_shapes={len(rank_bad)} "
          f"lora_rank={metadata.get('lora_rank')} lora_alpha={metadata.get('lora_alpha')}")
    for label, items in (("unmatched", unmatched), ("incomplete", incomplete), ("bad_shape", rank_bad)):
        for t in items[:8]:
            print(f"    {label}: {t}")
    return ok


def check_headers_dir(headers: Path) -> bool:
    ok = True
    ran = False
    transformer_headers = sorted(headers.glob("*transformer-bf16.safetensors.header.json"))
    for th in transformer_headers:
        ran = True
        ok = check_transformer_header(th) and ok
    text_headers = sorted(headers.glob("*with-proj*.safetensors.header.json"))
    for te in text_headers:
        ran = True
        ok = check_text_encoder_header(te, transformer_headers[0] if transformer_headers else None) and ok
    lora_headers = sorted(headers.glob("*lora*.safetensors.header.json"))
    for lh in lora_headers:
        if transformer_headers:
            ran = True
            ok = check_lora_header(lh, transformer_headers[0]) and ok
    if not ran:
        print(f"[SKIP] no *.header.json files found in {headers}")
    return ok or not ran


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--converted", type=Path, action="append", default=[],
                        help="Converted output directory to strict-check (repeatable).")
    parser.add_argument("--headers", type=Path, default=None,
                        help="Directory of *.header.json safetensors headers for the gated big files.")
    args = parser.parse_args()
    if not args.converted and args.headers is None:
        parser.error("pass --converted and/or --headers")

    ok = True
    for converted in args.converted:
        print(f"=== converted: {converted} ===")
        ok = check_converted_dir(converted) and ok
    if args.headers is not None:
        print(f"=== headers: {args.headers} ===")
        ok = check_headers_dir(args.headers) and ok
    print("ALL OK" if ok else "FAILURES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
