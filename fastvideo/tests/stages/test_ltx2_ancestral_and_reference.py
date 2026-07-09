# SPDX-License-Identifier: Apache-2.0
"""Tests for the LTX-2 ancestral samplers, per-stage sigma overrides, and
reference-token prefix conditioning.

The sampler step functions are checked element-wise against verbatim ports
of ComfyUI's reference implementations (comfy/k_diffusion/sampling.py:
``sample_euler_ancestral_RF`` and ``sample_euler_ancestral_cfg_pp`` with the
CONST/rectified-flow half-logSNR reduction ``alpha = 1 - sigma``).
"""

from __future__ import annotations

import os

import pytest
import torch

from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import (
    euler_ancestral_cfg_pp_step,
    euler_ancestral_rf_step,
    get_ancestral_step,
)

SEED = 20260709
WORKFLOW_STAGE1_SIGMAS = [1.000, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
WORKFLOW_STAGE2_SIGMAS = [0.92, 0.725, 0.421875, 0.0]


# --- Reference implementations (verbatim ComfyUI math) ---------------------


def _ref_euler_ancestral_rf(x, sigmas, model_fn, noises, eta=1.0, s_noise=1.0):
    """Verbatim loop of comfy sample_euler_ancestral_RF (CONST models)."""
    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i])
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1]**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2)**0.5
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noises[i] * s_noise * renoise_coeff
    return x


def _ref_euler_ancestral_cfg_pp(x, sigmas, cond_fn, uncond_fn, noises, eta=1.0, s_noise=1.0):
    """Verbatim loop of comfy sample_euler_ancestral_cfg_pp for CONST models
    (half-logSNR: ``lambda(sigma).exp() == (1 - sigma) / sigma`` so
    ``alpha = 1 - sigma``). ``denoised`` is the post-CFG conditional x0 and
    ``uncond_denoised`` the raw unconditional x0."""

    def _ref_get_ancestral_step(sigma_from, sigma_to, eta=1.0):
        if not eta:
            return sigma_to, 0.0
        sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)**0.5)
        sigma_down = (sigma_to**2 - sigma_up**2)**0.5
        return sigma_down, sigma_up

    for i in range(len(sigmas) - 1):
        denoised = cond_fn(x, sigmas[i])
        uncond_denoised = uncond_fn(x, sigmas[i])
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            alpha_s = 1 - sigmas[i]
            alpha_t = 1 - sigmas[i + 1]
            d = (x - alpha_s * uncond_denoised) / sigmas[i]
            sigma_down, sigma_up = _ref_get_ancestral_step(sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t, eta=eta)
            sigma_down = alpha_t * sigma_down
            x = alpha_t * denoised + sigma_down * d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noises[i] * s_noise * sigma_up
    return x


def _synthetic_model(shift: float):
    """Deterministic synthetic x0 predictor: contract toward a constant."""

    def model_fn(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x * 0.9 + shift * (1.0 - sigma)

    return model_fn


# --- Sampler math ------------------------------------------------------------


def test_get_ancestral_step_matches_reference():
    for sigma_from, sigma_to in [(0.955, 0.893), (0.5, 0.25), (0.241, 0.121)]:
        for eta in (0.0, 0.5, 1.0):
            got = get_ancestral_step(sigma_from, sigma_to, eta=eta)
            if not eta:
                expected = (sigma_to, 0.0)
            else:
                sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)**0.5)
                expected = ((sigma_to**2 - sigma_up**2)**0.5, sigma_up)
            assert got == pytest.approx(expected)


@pytest.mark.parametrize("sigmas", [WORKFLOW_STAGE1_SIGMAS, WORKFLOW_STAGE2_SIGMAS])
@pytest.mark.parametrize("eta,s_noise", [(1.0, 1.0), (0.7, 0.9), (0.0, 1.0)])
def test_euler_ancestral_rf_matches_comfy_reference(sigmas, eta, s_noise):
    torch.manual_seed(SEED)
    x0 = torch.randn(2, 4, 3, 5, dtype=torch.float32)
    noises = [torch.randn_like(x0) for _ in range(len(sigmas) - 1)]
    model_fn = _synthetic_model(shift=0.3)

    expected = _ref_euler_ancestral_rf(x0.clone(), sigmas, model_fn, noises, eta=eta, s_noise=s_noise)

    x = x0.clone()
    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i])
        x = euler_ancestral_rf_step(
            x,
            denoised,
            sigmas[i],
            sigmas[i + 1],
            eta=eta,
            s_noise=s_noise,
            noise=noises[i],
        )

    torch.testing.assert_close(x, expected, rtol=0.0, atol=1e-6)


# NOTE: cfg_pp is undefined at sigma == 1.0 for rectified flow (alpha_s = 0),
# so it is only tested with sub-1.0 start schedules — matching the workflow,
# which uses cfg_pp exclusively for the stage-2 refine pass.
@pytest.mark.parametrize("sigmas", [WORKFLOW_STAGE2_SIGMAS, [0.99, 0.7, 0.35, 0.0]])
@pytest.mark.parametrize("eta,s_noise", [(1.0, 1.0), (0.5, 0.8), (0.0, 0.0)])
def test_euler_ancestral_cfg_pp_matches_comfy_reference(sigmas, eta, s_noise):
    torch.manual_seed(SEED + 1)
    x0 = torch.randn(2, 4, 3, 5, dtype=torch.float32)
    noises = [torch.randn_like(x0) for _ in range(len(sigmas) - 1)]
    cond_fn = _synthetic_model(shift=0.3)
    uncond_fn = _synthetic_model(shift=-0.2)

    expected = _ref_euler_ancestral_cfg_pp(x0.clone(), sigmas, cond_fn, uncond_fn, noises, eta=eta, s_noise=s_noise)

    x = x0.clone()
    for i in range(len(sigmas) - 1):
        denoised = cond_fn(x, sigmas[i])
        uncond = uncond_fn(x, sigmas[i])
        x = euler_ancestral_cfg_pp_step(
            x,
            denoised,
            uncond,
            sigmas[i],
            sigmas[i + 1],
            eta=eta,
            s_noise=s_noise,
            noise=noises[i],
        )

    torch.testing.assert_close(x, expected, rtol=0.0, atol=1e-6)


def test_cfg_pp_differs_from_plain_ancestral_even_at_cfg1():
    """CFG++ uses the unconditional x0 for its direction term, so it must
    not degenerate to plain euler_ancestral when cond != uncond."""
    torch.manual_seed(SEED + 2)
    x = torch.randn(1, 4, 2, 2, dtype=torch.float32)
    noise = torch.randn_like(x)
    cond = x * 0.9 + 0.3
    uncond = x * 0.9 - 0.2
    a = euler_ancestral_rf_step(x, cond, 0.92, 0.725, eta=1.0, s_noise=1.0, noise=noise)
    b = euler_ancestral_cfg_pp_step(x, cond, uncond, 0.92, 0.725, eta=1.0, s_noise=1.0, noise=noise)
    assert not torch.allclose(a, b)


def test_cfg_pp_rejects_sigma_one():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="sigma >= 1.0"):
        euler_ancestral_cfg_pp_step(x, x, x, 1.0, 0.725, eta=1.0, s_noise=1.0, noise=x)


# --- FastVideoArgs validation -------------------------------------------------


def _make_args(**kwargs):
    from fastvideo.fastvideo_args import FastVideoArgs
    return FastVideoArgs(model_path="FastVideo/LTX2-Distilled-Diffusers", **kwargs)


def test_stage_sigmas_validation_accepts_workflow_schedules():
    args = _make_args(
        ltx2_stage1_sigmas=WORKFLOW_STAGE1_SIGMAS,
        ltx2_stage2_sigmas=WORKFLOW_STAGE2_SIGMAS,
        ltx2_sampler="euler_ancestral",
        ltx2_refine_sampler="euler_ancestral_cfg_pp",
    )
    assert args.ltx2_stage1_sigmas[-1] == 0.0
    assert args.ltx2_stage2_sigmas == [0.92, 0.725, 0.421875, 0.0]


@pytest.mark.parametrize(
    "bad_sigmas",
    [
        [0.5, 0.7, 0.0],        # not decreasing
        [1.0, 0.5, 0.1],        # does not end at 0
        [1.5, 0.5, 0.0],        # > 1.0
        [0.0],                  # too short
    ],
)
def test_stage_sigmas_validation_rejects_bad_schedules(bad_sigmas):
    with pytest.raises(ValueError):
        _make_args(ltx2_stage1_sigmas=bad_sigmas)


def test_sampler_name_validation():
    with pytest.raises(ValueError):
        _make_args(ltx2_sampler="euler_ancestral_cfg_pp")  # stage 1 disallows cfg_pp
    with pytest.raises(ValueError):
        _make_args(ltx2_refine_sampler="ddim")
    with pytest.raises(ValueError):
        _make_args(ltx2_reference_position_mode="bogus")


# --- Reference token prefix (tiny DiT forward) --------------------------------


def _init_single_process_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"


def _build_tiny_ltx2_model():
    from fastvideo.configs.models.dits.ltx2 import (
        LTX2VideoArchConfig,
        LTX2VideoConfig,
    )
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel, )
    from fastvideo.models.dits.ltx2 import LTX2Transformer3DModel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    arch_config = LTX2VideoArchConfig(
        num_attention_heads=4,
        attention_head_dim=8,
        num_layers=2,
        cross_attention_dim=32,
        caption_channels=12,
        norm_eps=1e-6,
        attention_type="default",
        rope_type="split",
        double_precision_rope=False,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[8, 32, 32],
        timestep_scale_multiplier=1000,
        use_middle_indices_grid=True,
        patch_size=(1, 1, 1),
        num_channels_latents=4,
        in_channels=4,
        out_channels=4,
        audio_num_attention_heads=4,
        audio_attention_head_dim=4,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_cross_attention_dim=16,
        audio_positional_embedding_max_pos=[8],
        av_ca_timestep_scale_multiplier=1,
    )
    config = LTX2VideoConfig(arch_config=arch_config)
    model = LTX2Transformer3DModel(config=config, hf_config={})
    model = model.to(dtype=torch.float32)
    torch.manual_seed(SEED + 3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim <= 1:
                if name.endswith("weight") and "norm" in name:
                    param.fill_(1.0)
                else:
                    param.zero_()
                continue
            torch.nn.init.xavier_uniform_(param)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_ltx2_model():
    _init_single_process_env()
    try:
        model = _build_tiny_ltx2_model()
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"tiny LTX2 model unavailable in this environment: {exc}")
    return model


def _forward_tiny(model, ref_latent=None, **ref_kwargs):
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    torch.manual_seed(SEED + 4)
    hidden_states = torch.randn(1, 4, 2, 2, 2, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 4, 12, dtype=torch.float32)
    seq_len = 2 * 2 * 2
    timestep = torch.full((1, seq_len), 0.7, dtype=torch.float32)
    batch = ForwardBatch(data_type="dummy")
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=batch):
        return model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            ref_latent=ref_latent,
            **ref_kwargs,
        )


def test_reference_prefix_preserves_output_shape(tiny_ltx2_model):
    base = _forward_tiny(tiny_ltx2_model)
    ref_latent = torch.randn(1, 4, 1, 2, 2, dtype=torch.float32)
    with_ref = _forward_tiny(tiny_ltx2_model, ref_latent=ref_latent)
    assert with_ref.shape == base.shape
    # The prefix must influence the target tokens through self-attention.
    assert not torch.allclose(with_ref, base)


def test_reference_prefix_position_modes(tiny_ltx2_model):
    ref_latent = torch.randn(1, 4, 1, 2, 2, dtype=torch.float32)
    out_ref = _forward_tiny(tiny_ltx2_model, ref_latent=ref_latent, ref_position_mode="reference")
    out_prefix = _forward_tiny(tiny_ltx2_model, ref_latent=ref_latent, ref_position_mode="prefix_continuous")
    assert out_ref.shape == out_prefix.shape
    with pytest.raises(ValueError):
        _forward_tiny(tiny_ltx2_model, ref_latent=ref_latent, ref_position_mode="bogus")


def test_reference_prefix_zero_timesteps_changes_output(tiny_ltx2_model):
    ref_latent = torch.randn(1, 4, 1, 2, 2, dtype=torch.float32)
    inherited = _forward_tiny(tiny_ltx2_model, ref_latent=ref_latent, ref_zero_timesteps=False)
    zeroed = _forward_tiny(tiny_ltx2_model, ref_latent=ref_latent, ref_zero_timesteps=True)
    assert not torch.allclose(inherited, zeroed)
