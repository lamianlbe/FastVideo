# SPDX-License-Identifier: Apache-2.0
"""Offline transformer LoRA merging in the LTX-2.5 split converter.

Synthetic-only coverage (no downloads, CPU-only) for
``scripts/checkpoint_conversion/convert_ltx2_weights.py --transformer-lora``:

* exact merge math ``W += strength * (alpha/rank) * (B @ A)`` — fp32
  accumulation, bf16 (source-dtype) output — through the real converter path,
  including LoRA modules that target the embeddings connectors routed into
  ``text_encoder/model.safetensors``;
* chained multi-LoRA application (PEFT ``lora_A/lora_B`` and comfy
  ``lora_down/lora_up`` dialects, alpha present and absent);
* ``PATH[:STRENGTH]`` flag parsing edge cases (strength omitted -> 1.0);
* model_index.json provenance: ``fastvideo_transformer_merged_loras`` recorded,
  refine implied, runtime ``fastvideo_refine_lora_path`` auto-wiring suppressed
  so a pre-merged transformer never gets an adapter double-applied.

Run directly (``python3 tests/local_tests/ltx2_5/test_ltx2_5_lora_merge.py``)
when pytest is not installed; the __main__ guard executes every test function.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # direct __main__ runs need the repo root importable
    sys.path.insert(0, str(REPO_ROOT))
CONVERTER_PATH = REPO_ROOT / "scripts" / "checkpoint_conversion" / "convert_ltx2_weights.py"
SPEC = importlib.util.spec_from_file_location("convert_ltx2_weights_lora_test", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

_TO_Q = "transformer_blocks.0.attn1.to_q"
_FF = "transformer_blocks.0.ff.net.0.proj"
_CONNECTOR = "video_embeddings_connector.transformer_1d_blocks.0.attn.to_q"


def _uint8(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode()), dtype=torch.uint8)


def _save(path: Path, tensors: dict[str, torch.Tensor], *, config: dict | None = None,
          metadata: dict[str, str] | None = None) -> None:
    safetensors_metadata = dict(metadata or {})
    if config is not None:
        safetensors_metadata["config"] = json.dumps(config)
    save_file(tensors, str(path), metadata=safetensors_metadata)


def _tiny_transformer_source(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """bf16 transformer source with a core Linear, an untouched Linear + bias, and a connector Linear."""
    generator = torch.Generator().manual_seed(7)
    weights = {
        f"model.diffusion_model.{_TO_Q}.weight": torch.randn(4, 3, generator=generator).to(torch.bfloat16),
        f"model.diffusion_model.{_TO_Q}.bias": torch.randn(4, generator=generator).to(torch.bfloat16),
        f"model.diffusion_model.{_FF}.weight": torch.randn(2, 2, generator=generator).to(torch.bfloat16),
        f"model.diffusion_model.{_CONNECTOR}.weight": torch.randn(4, 3, generator=generator).to(torch.bfloat16),
    }
    source = tmp_path / "transformer.safetensors"
    _save(source, weights, config={"transformer": {"num_attention_heads": 2, "attention_head_dim": 4}})
    return source, weights


def _tiny_lora(tmp_path: Path, name: str, *, rank: int = 2, alpha: float | None = None,
               dialect: str = "peft", modules: tuple[str, ...] = (_TO_Q, _CONNECTOR),
               orphan: bool = True, seed: int = 21) -> tuple[Path, dict[str, torch.Tensor]]:
    """A tiny LoRA in the official key dialect: diffusion_model.<module>.{lora_A,lora_B}.weight."""
    down_leaf, up_leaf = ("lora_A", "lora_B") if dialect == "peft" else ("lora_down", "lora_up")
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for module in modules:
        tensors[f"diffusion_model.{module}.{down_leaf}.weight"] = (
            torch.randn(rank, 3, generator=generator).to(torch.bfloat16))
        tensors[f"diffusion_model.{module}.{up_leaf}.weight"] = (
            torch.randn(4, rank, generator=generator).to(torch.bfloat16))
        if alpha is not None:
            tensors[f"diffusion_model.{module}.alpha"] = torch.tensor(alpha)
    if orphan:
        # A module absent from the base must be skipped, not fatal (ComfyUI behavior).
        tensors[f"diffusion_model.transformer_blocks.99.attn1.to_q.{down_leaf}.weight"] = (
            torch.zeros(rank, 3, dtype=torch.bfloat16))
        tensors[f"diffusion_model.transformer_blocks.99.attn1.to_q.{up_leaf}.weight"] = (
            torch.zeros(4, rank, dtype=torch.bfloat16))
    path = tmp_path / name
    _save(path, tensors)
    return path, tensors


def _expected_merge(base: torch.Tensor, deltas: list[tuple[torch.Tensor, torch.Tensor, float]]) -> torch.Tensor:
    """Reference math mirroring the converter: fp32 accumulate, cast back to bf16."""
    total: torch.Tensor | None = None
    for down, up, scale in deltas:
        delta = scale * (up.to(torch.float32) @ down.to(torch.float32))
        total = delta if total is None else total + delta
    assert total is not None
    return (base.to(torch.float32) + total).to(base.dtype)


def test_transformer_lora_merge_exact_math(tmp_path: Path) -> None:
    """One LoRA at 0.7 with alpha=3 (rank 2): W' == (W + 0.7*(3/2)*B@A).bf16 exactly."""
    source, base = _tiny_transformer_source(tmp_path)
    lora_path, lora = _tiny_lora(tmp_path, "lora.safetensors", rank=2, alpha=3.0)
    output = tmp_path / "converted"

    converter.convert_split_components(
        transformer_source=source,
        transformer_loras=[(lora_path, 0.7)],
        output_dir=output,
    )

    merged = load_file(str(output / "transformer" / "model.safetensors"))
    scale = 0.7 * (3.0 / 2)
    expected_to_q = _expected_merge(
        base[f"model.diffusion_model.{_TO_Q}.weight"],
        [(lora[f"diffusion_model.{_TO_Q}.lora_A.weight"], lora[f"diffusion_model.{_TO_Q}.lora_B.weight"], scale)],
    )
    assert merged[f"{_TO_Q}.weight"].dtype == torch.bfloat16
    assert torch.equal(merged[f"{_TO_Q}.weight"], expected_to_q)
    assert not torch.equal(merged[f"{_TO_Q}.weight"], base[f"model.diffusion_model.{_TO_Q}.weight"])
    # Untouched tensors pass through byte-identically.
    assert torch.equal(merged[f"{_TO_Q}.bias"], base[f"model.diffusion_model.{_TO_Q}.bias"])
    assert torch.equal(merged[f"{_FF}.weight"], base[f"model.diffusion_model.{_FF}.weight"])

    # Connector modules merge BEFORE the transformer/text-encoder split, so the
    # routed text_encoder half carries the merged tensor.
    text_encoder = load_file(str(output / "text_encoder" / "model.safetensors"))
    expected_connector = _expected_merge(
        base[f"model.diffusion_model.{_CONNECTOR}.weight"],
        [(lora[f"diffusion_model.{_CONNECTOR}.lora_A.weight"],
          lora[f"diffusion_model.{_CONNECTOR}.lora_B.weight"], scale)],
    )
    connector_key = _CONNECTOR.replace("video_embeddings_connector.", "embeddings_connector.")
    assert torch.equal(text_encoder[f"{connector_key}.weight"], expected_connector)


def test_transformer_lora_merge_chains_multiple_loras(tmp_path: Path) -> None:
    """Two LoRAs accumulate in order: alpha-less files scale by strength alone (alpha==rank)."""
    source, base = _tiny_transformer_source(tmp_path)
    lora_a_path, lora_a = _tiny_lora(tmp_path, "a.safetensors", rank=2, alpha=3.0, seed=21)
    # Second file: comfy lora_down/lora_up naming, no alpha, rank 1, to_q only.
    lora_b_path, lora_b = _tiny_lora(tmp_path, "b.safetensors", rank=1, alpha=None,
                                     dialect="comfy", modules=(_TO_Q, ), orphan=False, seed=42)
    output = tmp_path / "converted_chain"

    converter.convert_split_components(
        transformer_source=source,
        transformer_loras=[(lora_a_path, 0.7), converter.parse_lora_spec(str(lora_b_path))],
        output_dir=output,
    )

    merged = load_file(str(output / "transformer" / "model.safetensors"))
    expected_to_q = _expected_merge(
        base[f"model.diffusion_model.{_TO_Q}.weight"],
        [
            (lora_a[f"diffusion_model.{_TO_Q}.lora_A.weight"],
             lora_a[f"diffusion_model.{_TO_Q}.lora_B.weight"], 0.7 * (3.0 / 2)),
            # strength omitted in the spec -> 1.0; no alpha key -> factor 1.0.
            (lora_b[f"diffusion_model.{_TO_Q}.lora_down.weight"],
             lora_b[f"diffusion_model.{_TO_Q}.lora_up.weight"], 1.0),
        ],
    )
    assert torch.equal(merged[f"{_TO_Q}.weight"], expected_to_q)


def test_parse_lora_spec_edge_cases(tmp_path: Path) -> None:
    parse = converter.parse_lora_spec
    assert parse("lora.safetensors:0.7") == (Path("lora.safetensors"), 0.7)
    # Strength omitted -> 1.0.
    assert parse("lora.safetensors") == (Path("lora.safetensors"), 1.0)
    assert parse("/abs/dir/lora.safetensors:1") == (Path("/abs/dir/lora.safetensors"), 1.0)
    assert parse("lora.safetensors:1e-1") == (Path("lora.safetensors"), 0.1)
    assert parse("lora.safetensors:-0.5") == (Path("lora.safetensors"), -0.5)
    # A trailing segment that is not a float belongs to the path.
    assert parse("odd:name.safetensors") == (Path("odd:name.safetensors"), 1.0)
    # Colon in a parent dir plus an explicit strength still splits on the last colon.
    assert parse("dir:v2/lora.safetensors:0.5") == (Path("dir:v2/lora.safetensors"), 0.5)
    for bad in (":0.7", "lora.safetensors:nan", "lora.safetensors:inf"):
        try:
            parse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_lora_spec({bad!r}) must fail")


def _full_split_sources(tmp_path: Path) -> dict[str, Path]:
    """Minimal full component set so model_index.json gets written."""
    transformer, _ = _tiny_transformer_source(tmp_path)

    gemma_config = {
        "model_type": "gemma4_unified",
        "pad_token_id": 0,
        "eos_token_id": 2,
        "text_config": {"hidden_size": 6, "num_hidden_layers": 2, "num_attention_heads": 2},
    }
    text_encoder = tmp_path / "text_encoder.safetensors"
    _save(
        text_encoder,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "text_embedding_projection.video_aggregate_embed.weight": torch.ones(8, 18),
            "text_embedding_projection.audio_aggregate_embed.weight": torch.ones(3, 18),
            "tokenizer_json": _uint8("{}"),
        },
        metadata={"gemma_config": json.dumps(gemma_config)},
    )

    vae = tmp_path / "vae.safetensors"
    _save(vae, {"vae.encoder.conv.weight": torch.ones(1)},
          config={"vae": {"_class_name": "CausalVideoAutoencoder", "dims": 3}})

    audio_vae = tmp_path / "audio_vae.safetensors"
    _save(
        audio_vae,
        {
            "audio_vae.decoder.conv_in.weight": torch.ones(1),
            "vocoder.vocoder.conv_pre.weight": torch.ones(1),
        },
        config={"audio_vae": {"model": {}}, "vocoder": {"vocoder": {}}},
    )

    spatial_upscaler = tmp_path / "spatial_upscaler.safetensors"
    _save(spatial_upscaler, {"model.initial_conv.weight": torch.ones(1)},
          config={"in_channels": 128, "spatial_upsample": True})

    distilled_lora = tmp_path / "distilled_lora.safetensors"
    _save(distilled_lora, {"diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones(1, 3)})
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": vae,
        "audio_vae": audio_vae,
        "spatial_upscaler": spatial_upscaler,
        "distilled_lora": distilled_lora,
    }


def test_merged_lora_model_index_disables_runtime_wiring(tmp_path: Path) -> None:
    """Pre-merged transformers record provenance and never auto-wire a runtime refine LoRA."""
    sources = _full_split_sources(tmp_path)
    lora_path, _ = _tiny_lora(tmp_path, "merge.safetensors", rank=2, alpha=3.0)
    output = tmp_path / "converted_index"

    converter.convert_split_components(
        transformer_source=sources["transformer"],
        text_encoder_source=sources["text_encoder"],
        vae_source=sources["vae"],
        audio_vae_source=sources["audio_vae"],
        spatial_upscaler_source=sources["spatial_upscaler"],
        distilled_lora_source=sources["distilled_lora"],
        transformer_loras=[(lora_path, 0.7)],
        output_dir=output,
        variant="dev",
    )

    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index["fastvideo_transformer_merged_loras"] == [
        {"file": "merge.safetensors", "strength": 0.7, "applied": 2},
    ]
    # A merged transformer behaves like the distilled recipe: refine implied ...
    assert model_index["fastvideo_refine_enabled"] is True
    # ... but the runtime adapter must NOT be auto-wired on top (double-apply).
    assert "fastvideo_refine_lora_path" not in model_index

    # A partial re-run (component swap) preserves the merged-LoRA record.
    converter.convert_split_components(vae_source=sources["vae"], output_dir=output)
    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index["fastvideo_transformer_merged_loras"][0]["file"] == "merge.safetensors"
    assert "fastvideo_refine_lora_path" not in model_index

    # Re-converting the transformer WITHOUT LoRAs replaces the weights, clears the
    # marker, and restores the bundled distilled_lora runtime wiring.
    converter.convert_split_components(transformer_source=sources["transformer"], output_dir=output)
    model_index = json.loads((output / "model_index.json").read_text())
    assert "fastvideo_transformer_merged_loras" not in model_index
    assert model_index["fastvideo_refine_lora_path"] == "distilled_lora/model.safetensors"


def test_lora_flag_validation(tmp_path: Path) -> None:
    """--transformer-lora needs --transformer-source; wrong-dialect files fail loudly."""
    source, _ = _tiny_transformer_source(tmp_path)
    lora_path, _ = _tiny_lora(tmp_path, "ok.safetensors", rank=2, alpha=None)

    try:
        converter.convert_split_components(
            vae_source=None,
            transformer_loras=[(lora_path, 1.0)],
            output_dir=tmp_path / "no_source",
            distilled_lora_source=None,
            spatial_upscaler_source=source,  # some source so split validation proceeds
        )
    except ValueError as exc:
        assert "--transformer-source" in str(exc)
    else:
        raise AssertionError("--transformer-lora without --transformer-source must fail")

    # Nothing in this file resolves against the transformer (wrong model dialect).
    bogus = tmp_path / "bogus.safetensors"
    _save(bogus, {
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.zeros(2, 3, dtype=torch.bfloat16),
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": torch.zeros(4, 2, dtype=torch.bfloat16),
    })
    try:
        converter.convert_split_components(
            transformer_source=source,
            transformer_loras=[(bogus, 1.0)],
            output_dir=tmp_path / "bogus_out",
        )
    except ValueError as exc:
        assert "no LoRA module resolved" in str(exc)
    else:
        raise AssertionError("wrong-dialect LoRA must be rejected")


def test_runtime_lora_params_default_to_noop() -> None:
    """Pre-merged deployments rely on the per-stage runtime LoRA being off by default.

    The pipeline only builds LoRA stages when ltx2_refine_lora_path resolves to
    something (model_index fastvideo_refine_lora_path or an explicit arg); the
    per-stage strengths must stay unset/no-op out of the box, and the two-stage
    preset must not smuggle runtime strengths in through its defaults.
    """
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.pipelines.basic.ltx2.presets import LTX2_5_DISTILLED_TWO_STAGE_I2V

    assert FastVideoArgs.ltx2_stage1_lora_strength is None
    assert FastVideoArgs.ltx2_refine_lora_path is None
    assert FastVideoArgs.refine_lora_path is None
    assert FastVideoArgs.refine_lora_strength is None

    lora_keys = [key for key in LTX2_5_DISTILLED_TWO_STAGE_I2V.defaults if "lora" in key]
    assert lora_keys == [], f"two-stage i2v preset must not hardcode runtime LoRA params: {lora_keys}"
    for stage_defaults in LTX2_5_DISTILLED_TWO_STAGE_I2V.stage_defaults.values():
        stage_lora_keys = [key for key in stage_defaults if "lora" in key]
        assert stage_lora_keys == [], f"stage defaults must not hardcode runtime LoRA params: {stage_lora_keys}"


ALL_TESTS = [
    test_transformer_lora_merge_exact_math,
    test_transformer_lora_merge_chains_multiple_loras,
    test_parse_lora_spec_edge_cases,
    test_merged_lora_model_index_disables_runtime_wiring,
    test_lora_flag_validation,
    test_runtime_lora_params_default_to_noop,
]

if __name__ == "__main__":
    import tempfile
    import traceback

    failures = 0
    for test in ALL_TESTS:
        name = test.__name__
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                kwargs = {}
                if "tmp_path" in inspect.signature(test).parameters:
                    kwargs["tmp_path"] = Path(tmp_dir)
                test(**kwargs)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    if failures:
        sys.exit(1)
    print("all tests passed")
