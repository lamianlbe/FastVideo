
# SPDX-License-Identifier: Apache-2.0
"""Gemma config resolution for text encoders without ``gemma_config`` metadata.

Community finetunes of the official LTX-2.5 encoder (e.g. the Heretic
uncensored Gemma 4) ship the identical packed layout but drop the header
metadata the converter reads the architecture from. These cover the three
ways that config is resolved and the guard that keeps the built-in official
config from being applied to a genuinely different architecture.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERTER_PATH = REPO_ROOT / "scripts" / "checkpoint_conversion" / "convert_ltx2_weights.py"
SPEC = importlib.util.spec_from_file_location("convert_ltx2_weights", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _official_like_header() -> dict[str, tuple[int, ...]]:
    """Header shapes matching the real official/Heretic encoder layout."""
    text_config = converter.OFFICIAL_GEMMA4_CONFIG["text_config"]
    hidden = int(text_config["hidden_size"])
    header: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (int(text_config["vocab_size"]), hidden),
        "model.norm.weight": (hidden, ),
        "text_embedding_projection.video_aggregate_embed.weight": (hidden, hidden),
        "tokenizer_json": (1024, ),
    }
    for layer in range(int(text_config["num_hidden_layers"])):
        header[f"model.layers.{layer}.input_layernorm.weight"] = (hidden, )
    return header


def test_builtin_config_matches_official_metadata_shape() -> None:
    config = converter.OFFICIAL_GEMMA4_CONFIG
    assert config["architectures"] == ["Gemma4UnifiedForConditionalGeneration"]
    assert config["gemma_version"] == "gemma4-12b-ltx-v1"
    assert config["text_config"]["hidden_size"] == 3840
    assert config["text_config"]["num_hidden_layers"] == 48
    assert config["text_config"]["vocab_size"] == 262144
    # The constant must stay valid JSON that round-trips (it is copied verbatim
    # from the official header, so a hand edit that breaks it should fail here).
    assert json.loads(converter.OFFICIAL_GEMMA4_CONFIG_JSON) == config


def test_official_layout_passes_the_fingerprint() -> None:
    assert converter._gemma_layout_mismatches(_official_like_header()) == []


def test_layout_differences_are_reported() -> None:
    truncated = {k: v for k, v in _official_like_header().items() if k != "model.layers.47.input_layernorm.weight"}
    assert any("found 47" in problem for problem in converter._gemma_layout_mismatches(truncated))

    rewidened = dict(_official_like_header())
    rewidened["model.embed_tokens.weight"] = (262144, 2048)
    assert any("model.embed_tokens.weight" in problem for problem in converter._gemma_layout_mismatches(rewidened))

    no_projection = {k: v for k, v in _official_like_header().items() if not k.startswith("text_embedding_projection.")}
    assert any("text_embedding_projection" in problem for problem in converter._gemma_layout_mismatches(no_projection))


def test_config_override_from_json(tmp_path: Path) -> None:
    config = {"model_type": "gemma4_unified", "text_config": {"hidden_size": 6}}
    path = tmp_path / "gemma_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert converter._load_gemma_config_override(path) == config

    # A wrapper object holding the config under its metadata key also works.
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"gemma_config": config}), encoding="utf-8")
    assert converter._load_gemma_config_override(wrapped) == config


def test_config_override_borrowed_from_safetensors(tmp_path: Path) -> None:
    config = {"model_type": "gemma4_unified", "text_config": {"hidden_size": 6}}
    donor = tmp_path / "official.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(2, 2)},
              str(donor),
              metadata={"gemma_config": json.dumps(config)})
    assert converter._load_gemma_config_override(donor) == config

    bare = tmp_path / "bare.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(2, 2)}, str(bare))
    try:
        converter._load_gemma_config_override(bare)
    except ValueError as err:
        assert "no 'gemma_config' metadata to borrow" in str(err)
    else:
        raise AssertionError("borrowing from a metadata-less file must fail")


def test_resolution_order_and_mismatch_guard(tmp_path: Path) -> None:
    embedded = {"model_type": "gemma4_unified", "text_config": {"hidden_size": 6}}
    override = {"model_type": "gemma4_unified", "text_config": {"hidden_size": 8}}
    override_path = tmp_path / "override.json"
    override_path.write_text(json.dumps(override), encoding="utf-8")

    with_metadata = tmp_path / "with_metadata.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(2, 2)},
              str(with_metadata),
              metadata={"gemma_config": json.dumps(embedded)})
    assert converter._resolve_gemma_config(with_metadata, None) == embedded
    # Explicit override wins over embedded metadata.
    assert converter._resolve_gemma_config(with_metadata, override_path) == override

    # No metadata and a layout that is not the official one: refuse to guess.
    tiny = tmp_path / "tiny.safetensors"
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2)}, str(tiny))
    try:
        converter._resolve_gemma_config(tiny, None)
    except ValueError as err:
        message = str(err)
        assert "--text-encoder-config" in message
        assert "transformer layers" in message
    else:
        raise AssertionError("a metadata-less non-official layout must not fall back silently")
    # ... but the same file converts once the config is supplied explicitly.
    assert converter._resolve_gemma_config(tiny, override_path) == override


if __name__ == "__main__":
    failures = 0
    import tempfile

    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if fn.__code__.co_argcount:
                    fn(Path(tmp))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as err:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(err).__name__}: {err}")
    raise SystemExit(1 if failures else 0)
