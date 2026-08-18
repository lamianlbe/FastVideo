# SPDX-License-Identifier: Apache-2.0
"""model_index.json metadata must not be mistaken for a component directory.

``verify_model_config_and_directory`` requires a subfolder for every declared
component. Diffusers spells a component as ``["library", "Class"]``, so a
metadata entry that merely happens to be a list — the converter's record of
the LoRAs merged into the transformer — must not trigger that requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastvideo.utils import verify_model_config_and_directory

MERGED_LORA_RECORD = [
    {"file": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors", "strength": 0.7, "applied": 1660},
    {"file": "sulphur_experimental_lora_v1.safetensors", "strength": 1.0, "applied": 1056},
]


def _write_model_index(root: Path, extra: dict) -> None:
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "vae").mkdir(parents=True, exist_ok=True)
    config = {
        "_class_name": "LTX2Pipeline",
        "_diffusers_version": "0.33.0.dev0",
        "transformer": ["fastvideo", "LTX2Transformer3DModel"],
        "vae": ["fastvideo", "CausalVideoAutoencoder"],
        **extra,
    }
    (root / "model_index.json").write_text(json.dumps(config), encoding="utf-8")


def test_merged_lora_record_is_metadata_not_a_component(tmp_path: Path) -> None:
    for key in ("_fastvideo_transformer_merged_loras", "fastvideo_transformer_merged_loras"):
        root = tmp_path / key
        _write_model_index(root, {key: MERGED_LORA_RECORD})
        config = verify_model_config_and_directory(str(root))
        assert config[key] == MERGED_LORA_RECORD


def test_declared_component_still_requires_its_directory(tmp_path: Path) -> None:
    root = tmp_path / "missing_component"
    _write_model_index(root, {"text_encoder": ["fastvideo", "LTX2GemmaTextEncoderModel"]})
    try:
        verify_model_config_and_directory(str(root))
    except ValueError as err:
        assert "text_encoder/ subfolder" in str(err)
    else:
        raise AssertionError("a declared component with no directory must still fail")


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS {name}")
            except Exception as err:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(err).__name__}: {err}")
    raise SystemExit(1 if failures else 0)
