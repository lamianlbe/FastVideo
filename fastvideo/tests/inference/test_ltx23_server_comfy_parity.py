# SPDX-License-Identifier: Apache-2.0
"""ComfyUI-parity knobs of the LTX-2.3 production server engine.

Covers the CPU-only logic of examples/inference/ltx23_server/ltx23_engine.py:
the per-step CFG derivation (ComfyUI's sigma lookup, not a step-index zip),
config validation of the parity fields, and the two guide-image geometries.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ENGINE_PATH = (Path(__file__).resolve().parents[3] / "examples" / "inference" / "ltx23_server" / "ltx23_engine.py")


def _load_engine():
    spec = importlib.util.spec_from_file_location("ltx23_engine_under_test", _ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load_engine()

# The production stage-1 schedule = the optimized workflow's own sampling
# schedule: its raw ManualSigmas (WORKFLOW_CFG_SIGMA_LIST) through the
# Sigmas Easing node (cubic in_out, strength 0.7), 10 steps.
PRODUCTION_STAGE1_SIGMAS = [
    1.0, 0.99987238, 0.99820748, 0.99001548, 0.96332988, 0.89394948, 0.744596, 0.47298248, 0.20186216, 0.04708576, 0.0
]


def _base_config(**overrides):
    cfg = engine.Ltx23ServerConfig(
        model_path="/does/not/exist",
        modes=[engine.Ltx23Mode(width=1344, height=768, num_frames=241, fps=24)],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# --- Per-step CFG derivation -----------------------------------------------


def test_cfg_derivation_matches_workflow_on_production_schedule():
    """The load-bearing assertion: the optimized workflow's guider lists
    mapped onto the shipped 10-step schedule. Steps 1-4 keep cfg 2 because
    their eased sigmas (0.99987 .. 0.96333) are still above the guider's
    second entry (0.9550) — a step-index zip would give 1.5/1.0 there."""
    derived = engine.derive_stage1_cfg_values(PRODUCTION_STAGE1_SIGMAS, engine.WORKFLOW_CFG_SIGMA_LIST,
                                              engine.WORKFLOW_CFG_VALUES)
    assert derived == [2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0]


def test_cfg_derivation_lookup_rules():
    sigma_list = [1.0, 0.8, 0.5, 0.0]
    cfg_values = [4.0, 3.0, 2.0]
    # 0.9 -> smallest listed sigma >= 0.9 is 1.0 (index 0).
    # 0.8 -> exact hit at index 1. 0.5 -> exact hit at index 2.
    derived = engine.derive_stage1_cfg_values([0.9, 0.8, 0.5, 0.0], sigma_list, cfg_values)
    assert derived == [4.0, 3.0, 2.0]
    # One value per STEP, so the trailing 0.0 endpoint is not mapped.
    assert len(derived) == 3
    # A sigma above every listed entry falls through to the last cfg
    # (comfy's `closest_idx = -1` branch).
    assert engine.derive_stage1_cfg_values([2.0, 0.0], [1.0, 0.0], [4.0, 3.0]) == [3.0]
    # An index past the end of a short cfg list is clamped, not an error.
    assert engine.derive_stage1_cfg_values([0.5, 0.0], [1.0, 0.9, 0.5, 0.0], [4.0]) == [4.0]


def test_resolve_stage1_cfg_values_off_by_default():
    assert engine.resolve_stage1_cfg_values(_base_config()) is None
    cfg = _base_config(
        stage1_sigmas=PRODUCTION_STAGE1_SIGMAS,
        stage1_cfg_sigma_list=engine.WORKFLOW_CFG_SIGMA_LIST,
        stage1_cfg_values_by_sigma=engine.WORKFLOW_CFG_VALUES,
    )
    assert engine.resolve_stage1_cfg_values(cfg) == [2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0]


# --- Config validation ------------------------------------------------------


def test_shipped_defaults_keep_todays_behaviour():
    cfg = _base_config()
    engine.validate_parity_config(cfg)
    assert cfg.stage1_conditioning == "inplace_and_reference"
    assert cfg.stage1_cfg_sigma_list == [] and cfg.stage1_cfg_values_by_sigma == []
    assert cfg.guide_resize == "cover_crop"
    assert cfg.stage1_sigmas == engine.DEFAULT_STAGE1_SIGMAS
    assert cfg.stage2_sigmas == engine.DEFAULT_STAGE2_SIGMAS
    assert cfg.image_crf == engine.DEFAULT_IMAGE_CRF


def test_full_parity_config_validates():
    cfg = _base_config(
        stage1_conditioning="guide_only",
        stage1_guide_strength=0.8,
        stage1_cfg_sigma_list=engine.WORKFLOW_CFG_SIGMA_LIST,
        stage1_cfg_values_by_sigma=engine.WORKFLOW_CFG_VALUES,
        guide_resize="comfy_lanczos_stretch",
    )
    engine.validate_parity_config(cfg)


@pytest.mark.parametrize(
    "overrides",
    [
        {"stage1_conditioning": "guide"},                       # unknown mode
        {"stage1_guide_strength": 0.0},                         # must be > 0
        {"stage1_guide_strength": 1.5},                         # must be <= 1
        {"stage1_cfg_sigma_list": [1.0, 0.0]},                  # cfg values missing
        {"stage1_cfg_values_by_sigma": [2.0]},                  # sigma list missing
        {"stage1_cfg_sigma_list": [0.5, 1.0], "stage1_cfg_values_by_sigma": [2.0]},   # not decreasing
        {"stage1_cfg_sigma_list": [1.0, 0.0], "stage1_cfg_values_by_sigma": [0.5]},   # cfg < 1
        {"stage1_cfg_sigma_list": [1.0, 0.5, 0.0], "stage1_cfg_values_by_sigma": []},  # empty pair half
        {"stage1_cfg_sigma_list": [1.0, 0.5, 0.2, 0.0], "stage1_cfg_values_by_sigma": [2.0, 1.5]},  # too few values
        {"guide_resize": "letterbox"},                          # unknown mode
        {"guide_longer_size": 32},                              # too small
    ],
)
def test_parity_config_rejects_bad_values(overrides):
    with pytest.raises(ValueError):
        engine.validate_parity_config(_base_config(**overrides))


def test_cfg_values_may_outnumber_sigmas():
    # The optimized workflow's node ships 13 cfg values for 11 sigmas; the
    # unreachable tail entries must not fail validation.
    engine.validate_parity_config(_base_config(
        stage1_cfg_sigma_list=engine.WORKFLOW_CFG_SIGMA_LIST,
        stage1_cfg_values_by_sigma=engine.WORKFLOW_CFG_VALUES,
    ))


# --- Guide-image geometry ----------------------------------------------------


def test_scale_longer_dimension_matches_comfy():
    assert engine.scale_longer_dimension((1000, 500), 1536) == (1536, 768)
    assert engine.scale_longer_dimension((500, 1000), 1536) == (768, 1536)
    assert engine.scale_longer_dimension((640, 640), 1536) == (1536, 1536)
    # Rounding follows comfy's round(), not floor.
    assert engine.scale_longer_dimension((1000, 333), 1536) == (1536, 511)


def _checker(width, height, block=16):
    np = pytest.importorskip("numpy")
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = (((ys // block) + (xs // block)) % 2 * 255).astype("uint8")
    return np.stack([pattern, pattern, pattern], axis=-1)


def test_guide_resize_stretches_without_cropping(tmp_path):
    """comfy_lanczos_stretch must SQUASH a mismatched aspect (bilinear with
    crop='disabled'), never drop content at the edges."""
    np = pytest.importorskip("numpy")
    from PIL import Image

    src = tmp_path / "src.png"
    array = _checker(1200, 600)
    # Distinctive corner markers survive a stretch but not a crop.
    array[:8, :8] = [255, 0, 0]
    array[:8, -8:] = [0, 255, 0]
    Image.fromarray(array).save(src)

    out = engine.preprocess_guide_image(src, tmp_path / "guide.png", 1344, 768, 1536)
    result = np.asarray(Image.open(out).convert("RGB"))
    assert result.shape == (768, 1344, 3)
    assert result[0, 0, 0] > result[0, 0, 1]  # red marker still in the corner
    assert result[0, -1, 1] > result[0, -1, 0]  # green marker still in the corner


def test_guide_resize_is_identity_for_matching_aspect(tmp_path):
    """A 1536-long-edge lanczos pass followed by the bilinear stretch must
    land exactly on the mode resolution for an already-matching aspect, so
    the pipeline's own resize + center crop becomes a no-op."""
    from PIL import Image

    src = tmp_path / "src.png"
    Image.fromarray(_checker(2688, 1536)).save(src)
    out = engine.preprocess_guide_image(src, tmp_path / "guide.png", 1344, 768, 1536)
    assert Image.open(out).size == (1344, 768)
