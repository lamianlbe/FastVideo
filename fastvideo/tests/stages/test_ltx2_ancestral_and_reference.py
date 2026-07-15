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


# --- Refine init stage: scale-aware stage-1 resolution -------------------------


def _refine_args(spatial_ratio=32):
    from types import SimpleNamespace
    return SimpleNamespace(
        ltx2_refine_enabled=True,
        pipeline_config=SimpleNamespace(vae_config=SimpleNamespace(arch_config=SimpleNamespace(
            spatial_compression_ratio=spatial_ratio))),
    )


def _refine_batch(height, width):
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    batch = ForwardBatch(data_type="dummy")
    batch.height = height
    batch.width = width
    return batch


@pytest.mark.parametrize("scale,target,expected_stage1", [
    (2.0, (1344, 1024), (672, 512)),
    (1.5, (1344, 1056), (896, 704)),
    (1.5, (1344, 960), (896, 640)),
])
def test_refine_init_stage_scales(scale, target, expected_stage1):
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_refine import LTX2RefineInitStage
    batch = _refine_batch(*target)
    LTX2RefineInitStage(spatial_scale=scale).forward(batch, _refine_args())
    assert (batch.height, batch.width) == expected_stage1
    assert batch.extra["ltx2_refine_target_height"] == target[0]
    assert batch.extra["ltx2_refine_target_width"] == target[1]


@pytest.mark.parametrize("scale,target", [
    (1.5, (1344, 1024)),   # 1024/1.5 is not an integer
    (1.5, (1344, 1008)),   # 1008 not divisible by 32
    (2.0, (1376, 1024)),   # 688 not divisible by 32
])
def test_refine_init_stage_rejects_bad_dims(scale, target):
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_refine import LTX2RefineInitStage
    with pytest.raises(ValueError):
        LTX2RefineInitStage(spatial_scale=scale).forward(_refine_batch(*target), _refine_args())


# --- All-in-one workflow ports: block ranges, text amp, latent anchor ---------


def test_parse_block_range():
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import parse_block_range
    assert parse_block_range("36-48", 48) == list(range(36, 48))  # clamped to 47
    assert parse_block_range("10-30", 48) == list(range(10, 31))
    assert parse_block_range("1,3,5", 48) == [1, 3, 5]
    with pytest.raises(ValueError):
        parse_block_range("50-60", 48)


def test_build_text_amp_weight_matches_comfy_math():
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import build_text_amp_weight
    frames, h, w = 2, 5, 7
    # Uniform path.
    uni = build_text_amp_weight(scale=1.3, spatial_focus=0.0, frames=frames, height_tokens=h, width_tokens=w,
                                device=torch.device("cpu"), dtype=torch.float32)
    assert uni.shape == (1, frames * h * w, 1)
    assert torch.allclose(uni, torch.full_like(uni, 1.3))
    # Spatial path: reference math inline (comfy _build_spatial_weight).
    amp = build_text_amp_weight(scale=1.3, spatial_focus=0.15, frames=frames, height_tokens=h, width_tokens=w,
                                device=torch.device("cpu"), dtype=torch.float32)
    sigma_g = max(0.3, 1.0 - 0.7 * 0.15) * min(h, w)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dy = torch.arange(h, dtype=torch.float32) - cy
    dx = torch.arange(w, dtype=torch.float32) - cx
    dist_sq = dy[:, None]**2 + dx[None, :]**2
    g = torch.exp(-dist_sq / (2 * sigma_g * sigma_g))
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    expected = (1.0 + 0.3 * g).reshape(-1).repeat(frames).reshape(1, -1, 1)
    torch.testing.assert_close(amp, expected)
    # Center gets full amplification, farthest corner none, all frames equal.
    grid0 = amp[0, :h * w, 0].reshape(h, w)
    grid1 = amp[0, h * w:, 0].reshape(h, w)
    torch.testing.assert_close(grid0, grid1)
    assert grid0[h // 2, w // 2] == pytest.approx(1.3, abs=1e-5)
    assert grid0[0, 0] == pytest.approx(1.0, abs=1e-5)


def _ref_anchor_pull(x_grid, anchor_flat, anchor_mean, *, strength, sim_thr, decay, anchor_idx, energy=None,
                     energy_thr=0.0):
    """Inline reference of the ComfyUI anchor math for a [1,F,H,W,D] grid."""
    b, f, h, w, d = x_grid.shape
    n = f * h * w
    frame_mean = x_grid.mean(dim=(2, 3), keepdim=True)
    centered_all = (x_grid - frame_mean).reshape(b, n, d)
    centered_anchor = anchor_flat - anchor_mean
    import torch.nn.functional as TF
    sim = torch.bmm(TF.normalize(centered_all, dim=-1, eps=1e-6),
                    TF.normalize(centered_anchor, dim=-1, eps=1e-6).transpose(1, 2))
    best_sim, best_idx = sim.max(dim=-1)
    gathered = torch.gather(anchor_flat, 1, best_idx.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.sigmoid((best_sim - sim_thr) * 8.0).reshape(b, f, h, w, 1)
    if energy is not None and energy_thr > 0:
        mask = mask * torch.sigmoid((energy - energy_thr) * 16.0).reshape(1, 1, h, w, 1)
    dist = (torch.arange(f, dtype=torch.float32) - anchor_idx).abs() / max(1, f - 1)
    fs = strength * (1.0 - decay * dist).clamp(min=0.0)
    fs[anchor_idx] = 0.0
    diff = gathered.reshape(b, f, h, w, d) - x_grid
    return x_grid + fs.view(1, f, 1, 1, 1) * mask * diff


def test_latent_anchor_matches_reference_math():
    from fastvideo.models.dits.ltx2_anchor import LatentAnchorContext, apply_latent_anchor
    torch.manual_seed(SEED + 10)
    f, h, w, d = 4, 3, 5, 8
    x = torch.randn(1, f * h * w, d, dtype=torch.float32)
    energy = torch.rand(h, w)
    ctx = LatentAnchorContext(strength=0.11, blocks=[0], frames=f, height_tokens=h, width_tokens=w,
                              energy_grid=energy, energy_threshold=0.3)
    out = apply_latent_anchor(x, ctx, block_idx=0)

    grid = x.reshape(1, f, h, w, d)
    anchor_flat = grid[:, 0].reshape(1, h * w, d)
    anchor_mean = anchor_flat.mean(dim=1, keepdim=True)
    expected = _ref_anchor_pull(grid, anchor_flat, anchor_mean, strength=0.11, sim_thr=0.5, decay=0.15, anchor_idx=0,
                                energy=energy, energy_thr=0.3).reshape(1, -1, d)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    # Anchor frame itself is never pulled.
    torch.testing.assert_close(out[0, :h * w], x[0, :h * w])


def test_latent_anchor_snapshot_cache():
    from fastvideo.models.dits.ltx2_anchor import LatentAnchorContext, apply_latent_anchor
    torch.manual_seed(SEED + 11)
    f, h, w, d = 3, 2, 2, 4
    k = h * w
    # Compile-safe cache: preallocated buffers + 0-d bool tensor flags.
    ctx = LatentAnchorContext(strength=0.2, blocks=[5], frames=f, height_tokens=h, width_tokens=w,
                              energy_threshold=0.0, slot_of={5: 0},
                              anchor_buf=torch.zeros(1, k, d), anchor_mean_buf=torch.zeros(1, 1, d))
    # Capture step (use_cache False, capture True): freeze x1's anchor frame.
    x1 = torch.randn(1, f * k, d)
    ctx.capture = torch.tensor(True)
    ctx.use_cache = torch.tensor(False)
    apply_latent_anchor(x1, ctx, block_idx=5)
    snap = x1.reshape(1, f, h, w, d)[:, 0].reshape(1, k, d)
    torch.testing.assert_close(ctx.anchor_buf[0], snap[0])  # buffer holds x1's anchor tokens
    # Later step (use_cache True, capture False): pull toward the SNAPSHOT,
    # not x2's own anchor frame.
    x2 = torch.randn(1, f * k, d)
    ctx.capture = torch.tensor(False)
    ctx.use_cache = torch.tensor(True)
    out2 = apply_latent_anchor(x2, ctx, block_idx=5)
    grid2 = x2.reshape(1, f, h, w, d)
    expected = _ref_anchor_pull(grid2, snap, snap.mean(dim=1, keepdim=True), strength=0.2, sim_thr=0.5,
                                decay=0.15, anchor_idx=0).reshape(1, -1, d)
    torch.testing.assert_close(out2, expected, rtol=1e-5, atol=1e-6)
    # Buffer untouched by the use-step (capture False).
    torch.testing.assert_close(ctx.anchor_buf[0], snap[0])


def test_latent_anchor_prefix_and_mismatch():
    from fastvideo.models.dits.ltx2_anchor import LatentAnchorContext, apply_latent_anchor
    torch.manual_seed(SEED + 12)
    f, h, w, d = 2, 2, 3, 4
    n_ref = h * w
    ctx = LatentAnchorContext(strength=0.3, blocks=[0], frames=f, height_tokens=h, width_tokens=w,
                              energy_threshold=0.0, token_offset=n_ref)
    x = torch.randn(1, n_ref + f * h * w, d)
    out = apply_latent_anchor(x, ctx, block_idx=0)
    torch.testing.assert_close(out[:, :n_ref], x[:, :n_ref])  # prefix untouched
    assert not torch.allclose(out[:, n_ref:], x[:, n_ref:])
    # Grid mismatch -> silent passthrough.
    bad = torch.randn(1, 7, d)
    ctx2 = LatentAnchorContext(strength=0.3, blocks=[0], frames=f, height_tokens=h, width_tokens=w)
    torch.testing.assert_close(apply_latent_anchor(bad, ctx2, block_idx=0), bad)


def test_tiny_dit_accepts_amp_and_anchor(tiny_ltx2_model):
    from fastvideo.models.dits.ltx2_anchor import LatentAnchorContext
    base = _forward_tiny(tiny_ltx2_model)
    amp = torch.full((1, 8, 1), 1.5, dtype=torch.float32)
    out_amp = _forward_tiny(tiny_ltx2_model, text_amp_weight=amp, text_amp_blocks=[0, 1])
    assert out_amp.shape == base.shape
    assert not torch.allclose(out_amp, base)

    ctx = LatentAnchorContext(strength=0.5, blocks=[0, 1], frames=2, height_tokens=2, width_tokens=2,
                              energy_threshold=0.0)
    out_anchor = _forward_tiny(tiny_ltx2_model, latent_anchor=ctx)
    assert out_anchor.shape == base.shape
    assert not torch.allclose(out_anchor, base)


def test_stage1_cfg_values_validation():
    args = _make_args(ltx2_stage1_cfg_values=[2.0, 1.5, 1.0])
    assert args.ltx2_stage1_cfg_values == [2.0, 1.5, 1.0]
    with pytest.raises(ValueError):
        _make_args(ltx2_stage1_cfg_values=[0.5])
    with pytest.raises(ValueError):
        _make_args(ltx2_stage1_cfg_values=[])
    with pytest.raises(ValueError):
        _make_args(ltx2_text_amp_stage="stage3")
    with pytest.raises(ValueError):
        _make_args(ltx2_anchor_strength=-0.1)


def test_ancestral_repin_preserves_conditioned_frame():
    """Regression: ancestral fresh-noise injection obliterates a partially
    pinned conditioning frame within a few near-sigma-1 steps unless the
    region is re-pinned after every update (ComfyUI KSamplerX0Inpaint
    semantics)."""
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import (
        repin_conditioned_latents, )
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_image_conditioning import (
        apply_ltx2_gaussian_noiser,
        post_process_ltx2_denoised,
    )

    torch.manual_seed(SEED + 20)
    sigmas = [1.0, 0.99987238, 0.99820748, 0.99001548, 0.96332988, 0.89394948, 0.744596, 0.47298248, 0.20186216,
              0.04708576, 0.0]
    clean = torch.randn(1, 4, 3, 4, 4)  # [B, C, F, H, W]
    mask = torch.ones(1, 1, 3, 4, 4)
    mask[:, :, 0] = 0.2  # frame 0 pinned at strength 0.8
    cond_noise = torch.randn_like(clean)

    def run(repin: bool) -> torch.Tensor:
        x = apply_ltx2_gaussian_noiser(noise=torch.randn_like(clean), clean_latent=clean, denoise_mask=mask)
        for i in range(len(sigmas) - 1):
            x0 = post_process_ltx2_denoised(denoised=x * 0.9, denoise_mask=mask, clean_latent=clean)
            x = euler_ancestral_rf_step(x, x0, sigmas[i], sigmas[i + 1], eta=1.0, s_noise=1.0,
                                        noise=torch.randn_like(x))
            if repin:
                x = repin_conditioned_latents(x, clean_latent=clean, denoise_mask=mask, cond_noise=cond_noise,
                                              sigma_next=sigmas[i + 1])
        return x

    fixed = run(repin=True)
    broken = run(repin=False)
    frame0 = clean[:, :, 0]
    err_fixed = (fixed[:, :, 0] - frame0).norm() / frame0.norm()
    err_broken = (broken[:, :, 0] - frame0).norm() / frame0.norm()
    # With the repin the pinned frame ends close to clean; without it the
    # ancestral noise dominates.
    assert err_fixed < 0.35, f"repin failed to preserve the conditioned frame (err={err_fixed:.3f})"
    assert err_broken > 2 * err_fixed, (f"expected collapse without repin (fixed={err_fixed:.3f}, "
                                        f"broken={err_broken:.3f})")
    # sigma_next=0 pins exactly (up to the 0.2 free component of the mask).
    final = repin_conditioned_latents(torch.zeros_like(clean), clean_latent=clean, denoise_mask=mask,
                                      cond_noise=cond_noise, sigma_next=0.0)
    torch.testing.assert_close(final[:, :, 0], clean[:, :, 0] * 0.8, rtol=1e-5, atol=1e-5)


def test_ltx2_images_stage2_override_resolution():
    """Stage-2 can run a reduced keyframe list via ltx2_images_stage2."""
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_image_conditioning import (
        resolve_ltx2_images, )
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    batch = ForwardBatch(data_type="dummy")
    batch.ltx2_images = [("first.png", 0, 1.0), ("last.png", 30, 0.8)]
    batch.ltx2_images_stage2 = [("first.png", 0, 1.0)]

    # Default (stage 1): full list from the batch.
    assert resolve_ltx2_images(batch) == [("first.png", 0, 1.0), ("last.png", 30, 0.8)]
    # Stage-2 override: reduced list.
    assert resolve_ltx2_images(batch, batch.ltx2_images_stage2) == [("first.png", 0, 1.0)]
    # None override falls back to the batch list (backward compatible).
    assert resolve_ltx2_images(batch, None) == resolve_ltx2_images(batch)
    # Empty override disables image conditioning for that stage.
    assert resolve_ltx2_images(batch, []) == []


def test_ltx2_reference_image_path_batch_override():
    """Per-request reference image (server use case) beats the engine arg."""
    from types import SimpleNamespace

    from fastvideo.pipelines.basic.ltx2.stages.ltx2_image_conditioning import (
        resolve_ltx2_reference_image_path, )
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    args = SimpleNamespace(ltx2_reference_image_path="engine.png")
    batch = ForwardBatch(data_type="dummy")

    # No batch override: engine-level arg applies.
    assert resolve_ltx2_reference_image_path(batch, args) == "engine.png"
    # Per-request override wins.
    batch.ltx2_reference_image_path = "request.png"
    assert resolve_ltx2_reference_image_path(batch, args) == "request.png"
    # Neither set: disabled ("" so truthiness gating works at call sites).
    batch.ltx2_reference_image_path = None
    args.ltx2_reference_image_path = ""
    assert resolve_ltx2_reference_image_path(batch, args) == ""
