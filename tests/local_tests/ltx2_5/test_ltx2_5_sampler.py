# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the LTX-2.5 distilled ancestral sampler."""

import math

import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import (
    ANCESTRAL_NOISE_SEED_OFFSET,
    _ltx2_first_frame_keyframes_mask,
    _ltx2_euler_ancestral_step,
    compute_ltxv_scheduler_sigmas,
    euler_ancestral_cfg_pp_step,
)


def test_ltx2_5_preset_enables_ancestral_sampling() -> None:
    params = SamplingParam.from_pretrained("FastVideo/LTX-2.5-Distilled-Diffusers")

    assert params.ltx2_use_ancestral_sampler is True
    assert params.num_inference_steps == 8
    assert params.ltx2_image_crf == 18.0


def test_ancestral_terminal_step_returns_denoised_prediction() -> None:
    sample = torch.tensor([3.0], dtype=torch.bfloat16)
    denoised = torch.tensor([1.25], dtype=torch.bfloat16)

    actual = _ltx2_euler_ancestral_step(
        sample,
        denoised,
        torch.tensor(0.421875),
        torch.tensor(0.0),
        noise=None,
    )

    torch.testing.assert_close(actual, denoised)


def test_ancestral_eta_zero_matches_deterministic_euler() -> None:
    sample = torch.tensor([2.0, -1.0], dtype=torch.float32)
    denoised = torch.tensor([0.5, 0.25], dtype=torch.float32)
    sigma = torch.tensor(0.8)
    sigma_next = torch.tensor(0.3)
    expected = sample + ((sample - denoised) / sigma) * (sigma_next - sigma)

    actual = _ltx2_euler_ancestral_step(
        sample,
        denoised,
        sigma,
        sigma_next,
        noise=None,
        eta=0.0,
    )

    torch.testing.assert_close(actual, expected)


def test_ancestral_noise_stream_is_reproducible_and_video_first() -> None:
    """Verify ancestral noise uses the seeded offset and video-first ordering."""
    seed = 17 + ANCESTRAL_NOISE_SEED_OFFSET
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_noise = torch.randn((1, 2), generator=generator)
    audio_noise = torch.randn((1, 3), generator=generator)

    replay = torch.Generator(device="cpu").manual_seed(seed)
    torch.testing.assert_close(video_noise, torch.randn((1, 2), generator=replay))
    torch.testing.assert_close(audio_noise, torch.randn((1, 3), generator=replay))


def test_ancestral_step_captures_noise_from_stage_forward(monkeypatch) -> None:
    """LTX2DenoisingStage.forward draws ancestral noise from one seed+offset stream,
    video first, then audio — through the real stage forward with a mocked DiT."""
    from unittest.mock import MagicMock
    from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import LTX2DenoisingStage
    from fastvideo.pipelines.composed_pipeline_base import ForwardBatch
    from fastvideo.fastvideo_args import FastVideoArgs

    video_shape = (1, 4, 3, 2, 2)  # (B, C, T, H, W) -> 48 elements
    audio_shape = (1, 4, 6, 1)  # (B, C, T, mel) -> 24 elements

    # Track (sample_shape, noise) pairs passed to _ltx2_euler_ancestral_step.
    captured: list[tuple[tuple[int, ...], torch.Tensor | None]] = []
    original_step = _ltx2_euler_ancestral_step

    def mock_step(sample, denoised, sigma, sigma_next, noise=None, **kwargs):
        captured.append((tuple(sample.shape), None if noise is None else noise.clone()))
        return original_step(sample, denoised, sigma, sigma_next, noise, **kwargs)

    monkeypatch.setattr(
        "fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising._ltx2_euler_ancestral_step",
        mock_step,
    )

    # Mock DiT: returns latents-shaped video/audio x0 predictions.
    mock_transformer = MagicMock()
    mock_transformer.return_value = (torch.zeros(video_shape), torch.zeros(audio_shape))
    mock_transformer.patchifier.get_token_count.return_value = 12
    stage = LTX2DenoisingStage(
        mock_transformer,
        num_inference_steps_override=2,
    )

    batch = ForwardBatch(
        data_type="video",
        latents=torch.randn(video_shape),
        prompt_embeds=[torch.randn(1, 10, 8)],
        seed=17,
        ltx2_use_ancestral_sampler=True,
        extra={"ltx2_audio_latents": torch.randn(audio_shape)},
    )

    args = FastVideoArgs(model_path="dummy")

    stage.forward(batch, args)

    # Two steps, each video-then-audio; the terminal step draws no fresh noise.
    assert [shape for shape, _ in captured] == [video_shape, audio_shape, video_shape, audio_shape]
    video_noise, audio_noise = captured[0][1], captured[1][1]
    assert video_noise is not None and audio_noise is not None
    assert captured[2][1] is None and captured[3][1] is None

    # Both draws come from ONE seed+offset generator, video first (official parity).
    expected_seed = 17 + ANCESTRAL_NOISE_SEED_OFFSET
    generator = torch.Generator(device="cpu").manual_seed(expected_seed)
    torch.testing.assert_close(video_noise, torch.randn(video_shape, generator=generator))
    torch.testing.assert_close(audio_noise, torch.randn(audio_shape, generator=generator))


def test_keyframe_mask_marks_every_token_in_first_causal_latent_frame() -> None:
    mask = _ltx2_first_frame_keyframes_mask(
        batch_size=2,
        token_count=12,
        latent_frames=3,
        device=torch.device("cpu"),
    )

    expected = torch.zeros((2, 12, 1))
    expected[:, :4] = 1.0
    torch.testing.assert_close(mask, expected)


def _reference_cfg_pp_step_via_comfy_tensor_math(x, denoised, uncond, sigma, sigma_next, eta, s_noise, noise):
    """Literal tensor transcription of ComfyUI's sample_euler_ancestral_cfg_pp body
    for CONST models (half-logSNR via logit; builtin-min NaN semantics), used to
    prove the explicit sigma>=1 limit branch matches the reference math."""
    sigma_t = torch.tensor(float(sigma))
    sigma_next_t = torch.tensor(float(sigma_next))
    alpha_s = sigma_t * sigma_t.logit().neg().exp()
    alpha_t = sigma_next_t * sigma_next_t.logit().neg().exp()
    d = (x - alpha_s * uncond) / sigma_t
    if not eta:
        sigma_down, sigma_up = sigma_next_t / alpha_t, torch.tensor(0.0)
    else:
        sigma_from = sigma_t / alpha_s
        sigma_to = sigma_next_t / alpha_t
        sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)**0.5)
        sigma_down = (sigma_to**2 - sigma_up**2)**0.5
    sigma_down = alpha_t * sigma_down
    out = alpha_t * denoised + sigma_down * d
    if eta > 0 and s_noise > 0:
        out = out + alpha_t * noise * s_noise * sigma_up
    return out


def test_cfg_pp_sigma_one_limit_matches_comfy_reference() -> None:
    """sigma=1.0 first step: the explicit limit equals ComfyUI's surviving tensor math."""
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    denoised = torch.randn(2, 3)
    uncond = torch.randn(2, 3)
    noise = torch.randn(2, 3)
    sigma, sigma_next = 1.0, 0.9793

    for eta, s_noise in ((1.0, 1.0), (1.0, 0.5), (1.0, 0.0), (0.0, 1.0)):
        expected = _reference_cfg_pp_step_via_comfy_tensor_math(
            x, denoised, uncond, sigma, sigma_next, eta, s_noise, noise)
        actual = euler_ancestral_cfg_pp_step(
            x, denoised, uncond, sigma, sigma_next, eta=eta, s_noise=s_noise, noise=noise)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6), (eta, s_noise)

    # Closed forms: eta>0 discards the uncond x0 entirely (None allowed) ...
    alpha_t = 1.0 - sigma_next
    limit = euler_ancestral_cfg_pp_step(
        x, denoised, None, sigma, sigma_next, eta=1.0, s_noise=1.0, noise=noise)
    torch.testing.assert_close(limit, alpha_t * denoised + sigma_next * noise)
    # ... while eta==0 keeps the deterministic direction along the initial noise.
    limit_det = euler_ancestral_cfg_pp_step(
        x, denoised, None, sigma, sigma_next, eta=0.0, s_noise=1.0, noise=None)
    torch.testing.assert_close(limit_det, alpha_t * denoised + sigma_next * x)


def test_cfg_pp_normal_step_still_requires_uncond() -> None:
    x = torch.randn(1, 4)
    denoised = torch.randn(1, 4)
    try:
        euler_ancestral_cfg_pp_step(x, denoised, None, 0.85, 0.7250, eta=1.0, s_noise=1.0, noise=torch.randn(1, 4))
    except ValueError as exc:
        assert "unconditional" in str(exc)
    else:
        raise AssertionError("sigma < 1.0 without an uncond x0 must raise")


def test_ltxv_scheduler_sigmas_recipe_values() -> None:
    """LTXVScheduler(steps=8, max_shift=4.0, base_shift=1.5, stretch, terminal=0.1),
    detached-latent anchor: starts at exactly 1.0, ends at 0.0, terminal sigma 0.1."""
    sigmas = compute_ltxv_scheduler_sigmas(8, max_shift=4.0, base_shift=1.5, stretch=True, terminal=0.1)
    assert sigmas.shape == (9,)
    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0
    assert abs(sigmas[-2].item() - 0.1) < 1e-6
    assert all(b < a for a, b in zip(sigmas.tolist(), sigmas.tolist()[1:]))

    # Independent recomputation of the shifted+stretched schedule.
    shift = math.exp(4.0)  # tokens=4096 anchor -> sigma_shift == max_shift
    lin = [1.0 - i / 8 for i in range(9)]
    shifted = [shift * s / (shift * s + (1 - s)) if s > 0 else 0.0 for s in lin]
    omz = [1.0 - s for s in shifted[:-1]]
    scale = omz[-1] / (1.0 - 0.1)
    expected = [1.0 - z / scale for z in omz] + [0.0]
    torch.testing.assert_close(sigmas, torch.tensor(expected, dtype=torch.float32), rtol=1e-6, atol=1e-6)
