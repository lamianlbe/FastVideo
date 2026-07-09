#!/usr/bin/env python3
"""Eager (no torch.compile) FastVideo test run replicating the ComfyUI
workflow `10Eros_10SNodes_I2V_Basic_DMD_V5` setting-for-setting.

Fill in IMAGE_PATH and PROMPT below (or set the LTX23_* env vars), then:

    env -u LD_LIBRARY_PATH python fastvideo_eager_i2v_test.py

Requires the `ltx23-ancestral-reference` branch of
github.com:lamianlbe/FastVideo (ancestral samplers + per-stage sigmas +
reference token conditioning).

Workflow settings replicated
----------------------------
resolution   1024x1376 portrait (stage 1 auto-runs at half res, the x2
             latent upsampler brings stage 2 back to full res — same
             two-pass layout as the ComfyUI graph)
frames/fps   361 frames @ 24 fps (slider 360 + 1; 8k+1 rule holds)
stage 1      euler_ancestral, cfg=1,
             sigmas 1.000,0.955,0.893,0.812,0.715,0.603,0.482,0.241,0.121,0
stage 2      euler_ancestral_cfg_pp, cfg=1 (uncond pass still runs — CFG++
             semantics), sigmas 0.92,0.725,0.421875,0
i2v          input image anchored at frame 0, strength 1.0
             (LTXVImgToVideoInplaceKJ equivalent)
reference    same image injected as attention token prefix, strength 1.0,
             position_mode="reference", timesteps inherited
             (LTXReferenceEnable/Conditioning equivalent)
img CRF      35 (LTXVPreprocess slider). NOTE: ComfyUI uses 35 for stage 1
             and 30 for stage 2; FastVideo applies one value to both —
             negligible difference, called out for completeness.
audio        stage 1 generates, stage 2 refines it (TwoWaySwitch=2
             equivalent — FastVideo's default refine behaviour)

Upsampler
---------
The converted 10Eros repo has no `spatial_upscaler/` subdir. Fetch the
x2 spatial upsampler from FastVideo's official repo once:

    hf download FastVideo/LTX-2.3-Distilled-Diffusers \
        --include "spatial_upscaler/*" --local-dir <MODEL_PATH>

or point LTX23_UPSAMPLER_PATH at any diffusers-format LTX2LatentUpsampler.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

# ---------------------------------------------------------------------------
# Fill these in (env vars override).
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("LTX23_MODEL_PATH", "/workspace/10Eros-LTX-2.3-Distilled-Diffusers")
IMAGE_PATH = os.getenv("LTX23_I2V_IMAGE", "/workspace/photo_2026-07-09_16-48-21.jpg")
PROMPT_BODY = os.getenv("LTX23_I2V_PROMPT",
                        "情色电影，温暖亲密光影。画面右侧的男生用双手持续揉捏女生的乳房，拇指反复刺激乳头，画面左侧的女生保持柔和微张嘴表情"
                        "并发出轻微喘息，两人目光锁定，身体轻微摇摆。")
OUTPUT_DIR = Path(os.getenv("LTX23_OUTPUT_DIR", "outputs_video/eager_i2v_test"))
SEED = int(os.getenv("LTX23_SEED", "635141064074927"))  # workflow node 524

# Workflow geometry (sliders 791/792/796, node 798: frames = 360 + 1).
# These are the FINAL output dims; the refine pipeline runs stage 1 at half
# res automatically. Both must be divisible by 64 (half res must satisfy the
# VAE's /32 rule). NOTE: the ComfyUI sliders say 1024x1376, but node 893
# (Resize v2, divisible_by=32) silently crops the half-res image 688 -> 672,
# so the workflow's true output is 1024x1344 — replicated here explicitly.
WIDTH = 1344
HEIGHT = 768
NUM_FRAMES = int(os.getenv("LTX23_NUM_FRAMES", "241"))  # shrink (e.g. 121) for smoke tests
FPS = 24

# Workflow sampling (nodes 914 / 582 / 910 / 911).
STAGE1_SIGMAS = [1.000, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
STAGE2_SIGMAS = [0.92, 0.725, 0.421875, 0.0]
IMAGE_CRF = 35.0  # LTXVPreprocess slider (node 915)

# Node 537 negative prompt, verbatim. The user prompt is passed through
# as-is (no preamble concatenation).
NEGATIVE_PROMPT = ("3D, phasing, captions, VR, still image, bad quality, subtitles, text, "
                   "watermark, overlay effects, pc game, yelling, console game, video game, "
                   "cartoon, childish, ugly, text, blur, logo, wordmark, static, low quality, "
                   "noise, white noise, bleep, censoring, censor, bleeping, beep, beeping, "
                   "newscast, interview, podcast, non-english, foreign language, russian, "
                   "chinese, japanese, mutant, horror, 70's, film grain, cinematic, comedy, "
                   "stand-up ")

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

from fastvideo import VideoGenerator  # noqa: E402  (after env setup)
from fastvideo.utils import maybe_download_model  # noqa: E402


def resolve_upsampler(model_root: str) -> str:
    override = os.getenv("LTX23_UPSAMPLER_PATH", "")
    candidates = ([override] if override else []) + [
        str(Path(model_root) / "spatial_upscaler"),
        str(Path(model_root) / "spatial_upsampler"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "config.json").is_file():
            return cand
    raise SystemExit(
        "No spatial upsampler found. Download the x2 upsampler into the model repo:\n"
        f"  hf download FastVideo/LTX-2.3-Distilled-Diffusers --include 'spatial_upscaler/*' "
        f"--local-dir {model_root}\n"
        "or set LTX23_UPSAMPLER_PATH.")


def main() -> None:
    if not Path(IMAGE_PATH).is_file():
        raise SystemExit(f"IMAGE_PATH not found: {IMAGE_PATH} (set LTX23_I2V_IMAGE)")
    if "REPLACE ME" in PROMPT_BODY:
        raise SystemExit("Set PROMPT_BODY in the script or the LTX23_I2V_PROMPT env var.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(MODEL_PATH)
    upsampler_path = resolve_upsampler(model_root)
    print(f"model:     {model_root}")
    print(f"upsampler: {upsampler_path}")
    print(f"image:     {IMAGE_PATH}")
    print(f"frames:    {NUM_FRAMES} @ {FPS} fps, {WIDTH}x{HEIGHT}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = NVFP4Config()

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        pipeline_config=pipeline_config,
        # --- eager: no compile flags at all ---
        # --- two-stage refine (ComfyUI first pass + upscale pass) ---
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=upsampler_path,
        ltx2_refine_lora_path="",
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,          # SamplerCustom add_noise=true
        # --- samplers + schedules (exact workflow values) ---
        ltx2_sampler="euler_ancestral",                  # node 910
        ltx2_refine_sampler="euler_ancestral_cfg_pp",    # node 911
        ltx2_stage1_sigmas=STAGE1_SIGMAS,                # node 914
        ltx2_stage2_sigmas=STAGE2_SIGMAS,                # node 582
        # --- reference token conditioning (LTXReferenceEnable/Conditioning) ---
        ltx2_reference_image_path=IMAGE_PATH,
        ltx2_reference_strength=1.0,                     # node 860/870 strength
        ltx2_reference_position_mode="reference",        # node 860/870 position_mode
        ltx2_reference_zero_timesteps=False,             # node 868 default
        # --- keep everything resident on the B200 ---
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    try:
        t0 = time.perf_counter()
        result = generator.generate_video(
            prompt=PROMPT_BODY,
            negative_prompt=NEGATIVE_PROMPT,             # node 537; feeds cfg_pp's uncond pass
            guidance_scale=1.0,                          # cfg=1 in both SamplerCustom nodes
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            fps=FPS,
            num_inference_steps=len(STAGE1_SIGMAS) - 1,
            seed=SEED,
            # i2v: anchor input image at frame 0, full strength (node 772,
            # Conditioning-Fidelity slider = 1.0).
            ltx2_images=[(IMAGE_PATH, 0, 1.0)],
            ltx2_image_crf=IMAGE_CRF,
            # No STG / modality guidance in this workflow revision.
            ltx2_stg_scale_video=0.0,
            ltx2_stg_scale_audio=0.0,
            ltx2_cfg_scale_video=1.0,
            ltx2_cfg_scale_audio=1.0,
            ltx2_modality_scale_video=1.0,
            ltx2_modality_scale_audio=1.0,
            save_video=True,
            output_path=str(OUTPUT_DIR / "eager_i2v_test.mp4"),
        )
        wall = time.perf_counter() - t0
        print(f"\ndone in {wall:.1f}s (eager) -> {OUTPUT_DIR / 'eager_i2v_test.mp4'}")
        if isinstance(result, dict) and result.get("logging_info") is not None:
            stages = getattr(result["logging_info"], "stages", None) or {}
            for name, metrics in stages.items():
                print(f"  {name}: {float(metrics.get('execution_time', 0.0)):.2f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
