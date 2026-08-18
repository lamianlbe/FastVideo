# SPDX-License-Identifier: Apache-2.0
"""Config, mode-matching and recipe-drift coverage for examples/inference/ltx25_server.

The server is a self-contained fork of the LTX-2.3 one, so nothing here can
be shared with that server's code. What it MUST stay in sync with is the
validated recipe in ``examples/inference/basic/basic_ltx2_5_i2av_two_stage.py``
— every sigma, strength, CRF and sampler name below is asserted equal between
the two files. That comparison is done by parsing both modules with ``ast``
so the guard runs on any machine, with no torch/fastvideo/PIL import; the
handful of checks that genuinely need fastvideo skip cleanly without it.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_DIR = _REPO_ROOT / "examples" / "inference" / "ltx25_server"
_ENGINE_PATH = _SERVER_DIR / "ltx25_engine.py"
_EXAMPLE_PATH = _REPO_ROOT / "examples" / "inference" / "basic" / "basic_ltx2_5_i2av_two_stage.py"
_CONFIG_EXAMPLE_PATH = _SERVER_DIR / "config.example.yaml"


def _load_engine_module() -> Any:
    """Import ltx25_engine.py by path. It only needs pyyaml at import time —
    every torch/fastvideo import inside it is function-local by design."""
    spec = importlib.util.spec_from_file_location("ltx25_engine_under_test", _ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load_engine_module()


# --------------------------------------------------------------------------
# ast helpers: read constants and call kwargs without importing the modules.
# --------------------------------------------------------------------------
def _literal(node: ast.AST, constants: dict[str, Any] | None = None) -> Any:
    """Value of a constant-ish AST node: literals, ``dict(...)`` calls, and
    (when ``constants`` is given) references to already-known module names."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        return {kw.arg: _literal(kw.value, constants) for kw in node.keywords if kw.arg}
    if constants is not None and isinstance(node, ast.Name):
        if node.id not in constants:
            raise ValueError(f"unresolved name {node.id}")
        return constants[node.id]
    return ast.literal_eval(node)


def _module_constants(path: Path) -> dict[str, Any]:
    """Module-level ``NAME = <constant>`` assignments (annotated or not)."""
    tree = ast.parse(path.read_text())
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, node.value
        else:
            continue
        try:
            out[target] = _literal(value, out)
        except (ValueError, TypeError, SyntaxError):
            continue  # not a constant (function call, f-string, ...) — ignore
    return out


def _call_kwargs(path: Path, dotted_name: str, constants: dict[str, Any]) -> dict[str, Any]:
    """Constant keyword arguments of the first ``<owner>.<attr>(...)`` call in
    the module, e.g. ``VideoGenerator.from_pretrained`` (the owner matters:
    both files also call ``PipelineConfig.from_pretrained``). Keywords whose
    value is not resolvable to a constant are omitted."""
    owner, attr_name = dotted_name.split(".")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != attr_name:
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == owner):
            continue
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = _literal(kw.value, constants)
            except (ValueError, TypeError, SyntaxError):
                continue
        return kwargs
    raise AssertionError(f"no {dotted_name}(...) call found in {path}")


ENGINE_CONSTANTS = _module_constants(_ENGINE_PATH)
EXAMPLE_CONSTANTS = _module_constants(_EXAMPLE_PATH)


# --------------------------------------------------------------------------
# Config loading and validation.
# --------------------------------------------------------------------------
_MINIMAL_CONFIG = """
model_path: /models/LTX-2.5-Dev-Merged-Diffusers
modes:
  - {width: 2048, height: 1152, num_frames: 121, fps: 24}
"""


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_shipped_config_example_loads() -> None:
    """config.example.yaml is the copy-paste starting point; it must parse
    and validate exactly as handed out."""
    cfg = engine.load_config(_CONFIG_EXAMPLE_PATH)

    assert cfg.modes
    assert cfg.quant == "none"  # bf16: quantized 2.5 deployment is follow-up work
    assert cfg.image_crf == engine.DEFAULT_IMAGE_CRF
    assert cfg.stage2_sigmas == engine.DEFAULT_STAGE2_SIGMAS
    for mode in cfg.modes:
        mode.validate()


def test_load_config_defaults_match_the_recipe(tmp_path: Path) -> None:
    cfg = engine.load_config(_write_config(tmp_path, _MINIMAL_CONFIG))

    assert cfg.stage1_steps == engine.DEFAULT_STAGE1_STEPS
    assert cfg.stage1_max_shift == engine.STAGE1_SCHEDULER_KWARGS["max_shift"]
    assert cfg.stage1_base_shift == engine.STAGE1_SCHEDULER_KWARGS["base_shift"]
    assert cfg.stage1_stretch is engine.STAGE1_SCHEDULER_KWARGS["stretch"]
    assert cfg.stage1_terminal == engine.STAGE1_SCHEDULER_KWARGS["terminal"]
    assert cfg.first_frame_strength_stage1 == engine.DEFAULT_FIRST_FRAME_STRENGTH_STAGE1
    assert cfg.first_frame_strength_stage2 == engine.DEFAULT_FIRST_FRAME_STRENGTH_STAGE2
    assert cfg.last_frame_strength == engine.DEFAULT_LAST_FRAME_STRENGTH
    assert cfg.stage1_lora_strength == engine.DEFAULT_STAGE1_LORA_STRENGTH
    assert cfg.refine_lora_strength == engine.DEFAULT_REFINE_LORA_STRENGTH
    assert cfg.negative_prompt == engine.DEFAULT_NEGATIVE_PROMPT


@pytest.mark.parametrize(
    "mode_line, expected_message",
    [
        # Stage 1 renders at HALF the configured size and still needs /32
        # dims, so the FINAL dims must be multiples of 64.
        ("{width: 2016, height: 1152, num_frames: 121, fps: 24}", "divisible by 64"),
        ("{width: 2048, height: 1120, num_frames: 121, fps: 24}", "divisible by 64"),
        ("{width: 2048, height: 1152, num_frames: 120, fps: 24}", r"8\*k\+1"),
        ("{width: 2048, height: 1152, num_frames: 121, fps: 0}", "must be positive"),
    ],
)
def test_load_config_rejects_bad_modes(tmp_path: Path, mode_line: str, expected_message: str) -> None:
    text = f"model_path: /models/x\nmodes:\n  - {mode_line}\n"
    with pytest.raises(ValueError, match=expected_message):
        engine.load_config(_write_config(tmp_path, text))


def test_load_config_requires_at_least_one_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="modes"):
        engine.load_config(_write_config(tmp_path, "model_path: /models/x\nmodes: []\n"))


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        engine.load_config(_write_config(tmp_path, _MINIMAL_CONFIG + "typo_key: 1\n"))


def test_load_config_rejects_pre_merged_with_runtime_lora(tmp_path: Path) -> None:
    """A pre-merged transformer must never get a runtime adapter stacked on
    top — the same contradiction the example script rejects."""
    text = _MINIMAL_CONFIG + "pre_merged: true\ndistilled_lora_path: /loras/distilled.safetensors\n"
    with pytest.raises(ValueError, match="contradict"):
        engine.load_config(_write_config(tmp_path, text))


@pytest.mark.parametrize(
    "override, expected_message",
    [
        ("quant: int4\n", "quant must be"),
        ("na_backend: cudnn\n", "na_backend must be"),
        ("stage1_steps: 0\n", "stage1_steps"),
        ("stage2_sigmas: [0.85, 0.4219, 0.1]\n", "must end at 0.0"),
        ("stage2_sigmas: [0.85, 0.85, 0.0]\n", "strictly decreasing"),
        ("first_frame_strength_stage1: 1.5\n", "must be in \\[0, 1\\]"),
        ("api_keys: ['']\n", "api_keys"),
        ("cuda_visible_devices: 'gpu0'\n", "cuda_visible_devices"),
    ],
)
def test_load_config_rejects_bad_values(tmp_path: Path, override: str, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        engine.load_config(_write_config(tmp_path, _MINIMAL_CONFIG + override))


# --------------------------------------------------------------------------
# Mode geometry and matching.
# --------------------------------------------------------------------------
def test_mode_stage1_geometry() -> None:
    """(width, height) is the FINAL size; stage 1 denoises half of it, and
    the scheduler shift follows that half-resolution latent's token count."""
    mode = engine.Ltx25Mode(width=2048, height=1152, num_frames=121, fps=24)

    assert mode.stage1_size() == (1024, 576)
    expected_tokens = ((121 - 1) // 8 + 1) * (1152 // 2 // 32) * (2048 // 2 // 32)
    assert mode.stage1_tokens() == expected_tokens
    assert mode.shape_key() == (2048, 1152, 121, 24)


def test_match_mode_exact() -> None:
    modes = [
        engine.Ltx25Mode(2048, 1152, 121, 24),
        engine.Ltx25Mode(1152, 2048, 121, 24),
    ]

    mode, exact = engine.match_mode(modes, 1152, 2048, 121, 24)

    assert exact is True
    assert mode is modes[1]


def test_match_mode_falls_back_to_closest_resolution() -> None:
    """Only configured shapes have compiled kernels, so a near-miss request
    is served by the closest-resolution mode (aspect-aware), not rejected."""
    landscape = engine.Ltx25Mode(2048, 1152, 121, 24)
    portrait = engine.Ltx25Mode(1152, 2048, 121, 24)
    modes = [landscape, portrait]

    mode, exact = engine.match_mode(modes, 1920, 1080, 121, 24)
    assert exact is False
    assert mode is landscape

    mode, exact = engine.match_mode(modes, 1080, 1920, 121, 24)
    assert exact is False
    assert mode is portrait


def test_match_mode_breaks_resolution_ties_on_frames_then_fps() -> None:
    short = engine.Ltx25Mode(2048, 1152, 121, 24)
    long = engine.Ltx25Mode(2048, 1152, 241, 24)

    mode, exact = engine.match_mode([short, long], 2048, 1152, 200, 24)

    assert exact is False
    assert mode is long


# --------------------------------------------------------------------------
# Pre-merged vs runtime LoRA detection (mirrors the example's logic).
# --------------------------------------------------------------------------
def _cfg(**overrides: Any) -> Any:
    base = dict(model_path="/models/x", modes=[engine.Ltx25Mode(2048, 1152, 121, 24)])
    base.update(overrides)
    return engine.Ltx25ServerConfig(**base)


@pytest.mark.parametrize("merged_key", ["_fastvideo_transformer_merged_loras", "fastvideo_transformer_merged_loras"])
def test_detect_lora_mode_pre_merged_directory(merged_key: str) -> None:
    """Both the current and the legacy (unprefixed) metadata spellings mark a
    pre-merged transformer, which must run with NO runtime adapter."""
    model_index = {merged_key: [{"path": "distilled.safetensors", "strength": 0.7}]}

    use_runtime_lora, note = engine.detect_lora_mode(model_index, _cfg())

    assert use_runtime_lora is False
    assert "offline-merged" in note


def test_detect_lora_mode_runtime_lora_directory() -> None:
    model_index = {"fastvideo_refine_lora_path": "distilled_lora/model.safetensors"}

    use_runtime_lora, note = engine.detect_lora_mode(model_index, _cfg())

    assert use_runtime_lora is True
    assert str(engine.DEFAULT_STAGE1_LORA_STRENGTH) in note
    assert str(engine.DEFAULT_REFINE_LORA_STRENGTH) in note


def test_detect_lora_mode_pre_merged_flag_wins_over_bundled_lora() -> None:
    model_index = {"fastvideo_refine_lora_path": "distilled_lora/model.safetensors"}

    use_runtime_lora, _ = engine.detect_lora_mode(model_index, _cfg(pre_merged=True))

    assert use_runtime_lora is False


def test_detect_lora_mode_plain_directory_has_no_lora() -> None:
    use_runtime_lora, note = engine.detect_lora_mode({}, _cfg())

    assert use_runtime_lora is False
    assert "no distilled LoRA is wired" in note


# --------------------------------------------------------------------------
# THE drift guard: the server's recipe == the validated example's recipe.
# --------------------------------------------------------------------------
def test_recipe_constants_match_the_validated_example() -> None:
    """basic_ltx2_5_i2av_two_stage.py is the source of truth for the recipe.
    If this fails, one of the two files changed a number the other did not."""
    pairs = [
        ("STAGE1_SCHEDULER_KWARGS", "STAGE1_SCHEDULER_KWARGS"),
        ("STAGE1_STEPS", "DEFAULT_STAGE1_STEPS"),
        ("STAGE2_SIGMAS", "DEFAULT_STAGE2_SIGMAS"),
        ("FIRST_FRAME_STRENGTH_STAGE1", "DEFAULT_FIRST_FRAME_STRENGTH_STAGE1"),
        ("FIRST_FRAME_STRENGTH_STAGE2", "DEFAULT_FIRST_FRAME_STRENGTH_STAGE2"),
        ("LAST_FRAME_STRENGTH_STAGE1", "DEFAULT_LAST_FRAME_STRENGTH"),
        ("IMAGE_CRF", "DEFAULT_IMAGE_CRF"),
        ("STAGE1_LORA_STRENGTH", "DEFAULT_STAGE1_LORA_STRENGTH"),
        ("REFINE_LORA_STRENGTH", "DEFAULT_REFINE_LORA_STRENGTH"),
        # The example copies the production server compile recipe verbatim
        # for its --warmup benchmark, so this guards both directions.
        ("SERVER_COMPILE_KWARGS", "COMPILE_KWARGS"),
    ]
    for example_name, engine_name in pairs:
        assert example_name in EXAMPLE_CONSTANTS, f"{example_name} vanished from the example"
        assert engine_name in ENGINE_CONSTANTS, f"{engine_name} vanished from the engine"
        assert EXAMPLE_CONSTANTS[example_name] == ENGINE_CONSTANTS[engine_name], (
            f"recipe drift: example {example_name}={EXAMPLE_CONSTANTS[example_name]!r} != "
            f"engine {engine_name}={ENGINE_CONSTANTS[engine_name]!r}")


def test_sampler_wiring_matches_the_validated_example() -> None:
    example_kwargs = _call_kwargs(_EXAMPLE_PATH, "VideoGenerator.from_pretrained", EXAMPLE_CONSTANTS)
    engine_kwargs = _call_kwargs(_ENGINE_PATH, "VideoGenerator.from_pretrained", ENGINE_CONSTANTS)

    # Both stages run ComfyUI's CFG++ ancestral sampler at cfg=1.
    assert example_kwargs["ltx2_sampler"] == "euler_ancestral_cfg_pp"
    assert example_kwargs["ltx2_refine_sampler"] == "euler_ancestral_cfg_pp"
    for key in ("ltx2_sampler", "ltx2_refine_sampler", "ltx2_refine_enabled",
                "ltx2_refine_guidance_scale", "ltx2_refine_add_noise"):
        assert engine_kwargs[key] == example_kwargs[key], f"from_pretrained drift on {key}"
    assert engine_kwargs["ltx2_sampler"] == ENGINE_CONSTANTS["DEFAULT_SAMPLER"]
    assert engine_kwargs["ltx2_refine_sampler"] == ENGINE_CONSTANTS["DEFAULT_REFINE_SAMPLER"]
    # STAGE2_SIGMAS reaches the example's engine args directly; the server
    # routes it through a config field, so compare the constant instead.
    assert example_kwargs["ltx2_stage2_sigmas"] == ENGINE_CONSTANTS["DEFAULT_STAGE2_SIGMAS"]


def test_guider_wiring_matches_the_validated_example() -> None:
    """Plain CFG++ guider: no STG, no modality isolation, no rescale, and the
    official-2.5 ancestral path stays off in both files."""
    example_kwargs = _call_kwargs(_EXAMPLE_PATH, "generator.generate_video", EXAMPLE_CONSTANTS)
    engine_kwargs = _call_kwargs(_ENGINE_PATH, "generator.generate_video", ENGINE_CONSTANTS)

    shared = ("guidance_scale", "ltx2_use_ancestral_sampler", "ltx2_cfg_scale_video", "ltx2_cfg_scale_audio",
              "ltx2_modality_scale_video", "ltx2_modality_scale_audio", "ltx2_rescale_scale",
              "ltx2_stg_scale_video", "ltx2_stg_scale_audio")
    for key in shared:
        assert key in example_kwargs, f"{key} vanished from the example's generate_video call"
        assert engine_kwargs[key] == example_kwargs[key], f"generate_video drift on {key}"
    assert engine_kwargs["ltx2_use_ancestral_sampler"] is False


def test_example_module_matches_engine_when_fastvideo_is_available() -> None:
    """Same guard as above, but through real imports — catches anything the
    ast reader could mis-parse. Skipped where fastvideo/PIL aren't installed."""
    pytest.importorskip("fastvideo", reason="fastvideo not installed on this host")
    pytest.importorskip("PIL", reason="Pillow not installed on this host")

    spec = importlib.util.spec_from_file_location("ltx2_5_two_stage_example", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)

    assert example.STAGE1_STEPS == engine.DEFAULT_STAGE1_STEPS
    assert example.STAGE2_SIGMAS == engine.DEFAULT_STAGE2_SIGMAS
    assert example.STAGE1_SCHEDULER_KWARGS == engine.STAGE1_SCHEDULER_KWARGS
    assert example.IMAGE_CRF == engine.DEFAULT_IMAGE_CRF
    assert example.FIRST_FRAME_STRENGTH_STAGE1 == engine.DEFAULT_FIRST_FRAME_STRENGTH_STAGE1
    assert example.FIRST_FRAME_STRENGTH_STAGE2 == engine.DEFAULT_FIRST_FRAME_STRENGTH_STAGE2
    assert example.LAST_FRAME_STRENGTH_STAGE1 == engine.DEFAULT_LAST_FRAME_STRENGTH
    assert example.STAGE1_LORA_STRENGTH == engine.DEFAULT_STAGE1_LORA_STRENGTH
    assert example.REFINE_LORA_STRENGTH == engine.DEFAULT_REFINE_LORA_STRENGTH
    assert example.SERVER_COMPILE_KWARGS == engine.COMPILE_KWARGS


def test_stage1_sigmas_follow_the_example_schedule() -> None:
    """The per-mode stage-1 schedule must equal what the example computes for
    the same geometry (LTXVScheduler shifted by the stage-1 token count)."""
    ltx2_stages = pytest.importorskip(
        "fastvideo.pipelines.basic.ltx2.stages",
        reason="fastvideo not installed on this host",
    )

    mode = engine.Ltx25Mode(width=2048, height=1152, num_frames=121, fps=24)
    cfg = _cfg(modes=[mode])

    expected = ltx2_stages.compute_ltxv_scheduler_sigmas(
        EXAMPLE_CONSTANTS["STAGE1_STEPS"],
        tokens=mode.stage1_tokens(),
        **EXAMPLE_CONSTANTS["STAGE1_SCHEDULER_KWARGS"],
    ).tolist()
    actual = engine.stage1_sigmas_for_mode(cfg, mode)

    assert actual == pytest.approx(expected)
    assert len(actual) == engine.DEFAULT_STAGE1_STEPS + 1
    assert actual[-1] == 0.0
    # The detached tokens=4096 anchor is a different schedule; the default
    # (attached latent) is what the reference workflow uses.
    anchored = engine.stage1_sigmas_for_mode(_cfg(modes=[mode], sigmas_token_anchor=True), mode)
    assert anchored != pytest.approx(actual)
