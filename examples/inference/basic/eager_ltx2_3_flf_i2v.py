#!/usr/bin/env python3
"""First+last-frame (FLF) variant of the LTX-2.3 DMD workflow port.

    env -u LD_LIBRARY_PATH python eager_ltx2_3_flf_i2v.py

Identical sampling recipe to eager_ltx2_3_ancestral_reference_i2v.py (the
validated DMD stack), plus a second keyframe anchored at the LAST latent
frame. Kept as a separate script while the FLF behavior is being validated;
fold into the main example afterwards.

FLF specifics:
- ``ltx2_images`` frame indices are LATENT-frame indices (unlike ComfyUI's
  AddGuide, which takes pixel frames and divides by 8 internally). The last
  latent index is ``(num_frames - 1) // 8`` — computed below.
- The last anchor pins the final latent frame, which decodes to the final
  ~8 pixel frames (~1/3 s at 24 fps): expect a short settle-in toward the
  target image. LTX23_LAST_STRENGTH trades adherence (higher) against a
  softer transition (lower); start at 0.8.
- End-frame adherence is inherently a bit softer than frame 0 (the causal
  VAE encodes a still image with first-frame statistics) — same behavior as
  ComfyUI's guide mechanism.
- Reference-token conditioning stays on the FIRST image (identity source);
  the last image only enters through the latent anchor, so switching between
  i2v and FLF never changes tensor shapes (no recompile under torch.compile).

Toggles: LTX23_COMPILE=1 / LTX23_QUANT=nvfp4|none, same as the DMD example,
plus LTX23_LAST_IN_UPSCALE=1 to also feed the tail anchor into the stage-2
refine pass (default off — this misbehaved in ComfyUI testing).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Fill these in (env vars override).
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("LTX23_MODEL_PATH", "/workspace/10Eros_v1.4_Stack_Diffusers")
IMAGE_PATH = os.getenv("LTX23_I2V_IMAGE", "/workspace/first_frame.png")
LAST_IMAGE_PATH = os.getenv("LTX23_I2V_LAST_IMAGE", "/workspace/last_frame.png")
LAST_STRENGTH = float(os.getenv("LTX23_LAST_STRENGTH", "0.8"))
# Whether the last-frame anchor also enters the stage-2 refine pass.
# Default OFF: in ComfyUI testing, feeding the tail keyframe into the
# upscale pass caused artifacts/errors — stage 2 then re-anchors only the
# first frame while the last frame's content survives via the upsampled
# stage-1 latent.
LAST_IN_UPSCALE = os.getenv("LTX23_LAST_IN_UPSCALE", "0") == "1"
PROMPT_BODY = os.getenv("LTX23_I2V_PROMPT", "REPLACE ME: describe the motion between the two keyframes.")
OUTPUT_DIR = Path(os.getenv("LTX23_OUTPUT_DIR", "outputs_video/flf_i2v_test"))
SEED = int(os.getenv("LTX23_SEED", "635141064074927"))

WIDTH = 1344
HEIGHT = 768
NUM_FRAMES = int(os.getenv("LTX23_NUM_FRAMES", "241"))  # shrink (e.g. 121) for smoke tests
FPS = 24
# ltx2_images takes LATENT frame indices (8x temporal compression).
LAST_LATENT_IDX = (NUM_FRAMES - 1) // 8

# Validated DMD sampling recipe (same as the main DMD example).
STAGE1_SIGMAS = [1.000, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
STAGE2_SIGMAS = [0.92, 0.725, 0.421875, 0.0]
# LTXVPreprocess port: H.264 CRF re-encode of the conditioning images.
# Higher = more motion (frames look like video, not stills) but a blurrier
# anchored first frame. The stage-2 override lets the refine pass re-anchor
# with a cleaner encode — LTX23_IMAGE_CRF_STAGE2=0 sharpens the final first
# frame while stage 1 keeps the motion-strength CRF (empty = same as stage 1).
IMAGE_CRF = float(os.getenv("LTX23_IMAGE_CRF", "35.0"))
_crf2 = os.getenv("LTX23_IMAGE_CRF_STAGE2", "")
IMAGE_CRF_STAGE2 = float(_crf2) if _crf2 else None

NEGATIVE_PROMPT = ("3D, phasing, captions, VR, still image, bad quality, subtitles, text, "
                   "watermark, overlay effects, pc game, yelling, console game, video game, "
                   "cartoon, childish, ugly, text, blur, logo, wordmark, static, low quality, "
                   "noise, white noise, bleep, censoring, censor, bleeping, beep, beeping, "
                   "newscast, interview, podcast, non-english, foreign language, russian, "
                   "chinese, japanese, mutant, horror, 70's, film grain, cinematic, comedy, "
                   "stand-up ")

COMPILE = os.getenv("LTX23_COMPILE", "0") == "1"
QUANT = os.getenv("LTX23_QUANT", "nvfp4").lower()  # nvfp4 | none
WARMUP_RUNS = int(os.getenv("LTX23_WARMUP_RUNS", "2" if COMPILE else "0"))
MEASURED_RUNS = int(os.getenv("LTX23_MEASURED_RUNS", "1"))

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

if COMPILE:
    import torch._inductor.config as _inductor

    _inductor.shape_padding = False  # mandatory on Blackwell (pad_mm crash)
    _inductor.conv_1x1_as_mm = True
    _inductor.coordinate_descent_tuning = True
    _inductor.coordinate_descent_check_all_directions = True
    _inductor.epilogue_fusion = False

from fastvideo import VideoGenerator  # noqa: E402  (after env setup)
from fastvideo.configs.pipelines.base import PipelineConfig  # noqa: E402
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config  # noqa: E402
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
    raise SystemExit("No spatial upsampler found; set LTX23_UPSAMPLER_PATH or add "
                     "spatial_upscaler/ to the model repo.")


def main() -> None:
    for name, path in (("LTX23_I2V_IMAGE", IMAGE_PATH), ("LTX23_I2V_LAST_IMAGE", LAST_IMAGE_PATH)):
        if not Path(path).is_file():
            raise SystemExit(f"{name} not found: {path}")
    if "REPLACE ME" in PROMPT_BODY:
        raise SystemExit("Set PROMPT_BODY or the LTX23_I2V_PROMPT env var.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(MODEL_PATH)
    upsampler_path = resolve_upsampler(model_root)
    print(f"model:      {model_root}")
    print(f"upsampler:  {upsampler_path}")
    print(f"first:      {IMAGE_PATH}")
    print(f"last:       {LAST_IMAGE_PATH} @ latent idx {LAST_LATENT_IDX}, strength {LAST_STRENGTH}, "
          f"in_upscale={LAST_IN_UPSCALE}")
    print(f"frames:     {NUM_FRAMES} @ {FPS} fps, {WIDTH}x{HEIGHT}")
    print(f"mode:       compile={COMPILE} quant={QUANT}")
    print(f"image_crf:  stage1={IMAGE_CRF} stage2={IMAGE_CRF_STAGE2 if IMAGE_CRF_STAGE2 is not None else '(same)'}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = (NVFP4Config() if QUANT == "nvfp4" else None)

    compile_kwargs: dict = {}
    if COMPILE:
        torch_compile_kwargs = {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "default",
            "dynamic": False,
        }
        compile_kwargs = dict(
            enable_torch_compile=True,
            enable_torch_compile_text_encoder=True,
            enable_torch_compile_vae=True,
            torch_compile_kwargs=torch_compile_kwargs,
            torch_compile_kwargs_vae=torch_compile_kwargs,
        )

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        pipeline_config=pipeline_config,
        **compile_kwargs,
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=upsampler_path,
        ltx2_refine_lora_path="",
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        ltx2_sampler="euler_ancestral",
        ltx2_refine_sampler="euler_ancestral_cfg_pp",
        ltx2_stage1_sigmas=STAGE1_SIGMAS,
        ltx2_stage2_sigmas=STAGE2_SIGMAS,
        # Reference tokens: identity from the FIRST image only.
        ltx2_reference_image_path=IMAGE_PATH,
        ltx2_reference_strength=1.0,
        ltx2_reference_position_mode="reference",
        ltx2_reference_zero_timesteps=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    common_kwargs = dict(
        prompt=PROMPT_BODY,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=1.0,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=FPS,
        num_inference_steps=len(STAGE1_SIGMAS) - 1,
        # First frame hard-anchored; last frame anchored at LAST_STRENGTH.
        ltx2_images=[
            (IMAGE_PATH, 0, 1.0),
            (LAST_IMAGE_PATH, LAST_LATENT_IDX, LAST_STRENGTH),
        ],
        # Stage-2 override: keep the tail anchor out of the refine pass
        # unless LTX23_LAST_IN_UPSCALE=1 (None = same list as stage 1).
        ltx2_images_stage2=(None if LAST_IN_UPSCALE else [(IMAGE_PATH, 0, 1.0)]),
        ltx2_image_crf=IMAGE_CRF,
        ltx2_image_crf_stage2=IMAGE_CRF_STAGE2,
        ltx2_stg_scale_video=0.0,
        ltx2_stg_scale_audio=0.0,
        ltx2_cfg_scale_video=1.0,
        ltx2_cfg_scale_audio=1.0,
        ltx2_modality_scale_video=1.0,
        ltx2_modality_scale_audio=1.0,
        save_video=True,
    )

    try:
        for w in range(WARMUP_RUNS):
            t0 = time.perf_counter()
            print(f"\n[warmup {w + 1}/{WARMUP_RUNS}]…")
            generator.generate_video(output_path=str(OUTPUT_DIR / f"_warmup_{w + 1}.mp4"), seed=SEED, **common_kwargs)
            print(f"[warmup {w + 1}/{WARMUP_RUNS}] wall={time.perf_counter() - t0:.1f}s")
        for w in range(WARMUP_RUNS):
            (OUTPUT_DIR / f"_warmup_{w + 1}.mp4").unlink(missing_ok=True)

        measured: list[float] = []
        stage_times: dict[str, list[float]] = {}
        stage_order: OrderedDict = OrderedDict()
        for m in range(MEASURED_RUNS):
            out_path = OUTPUT_DIR / f"flf_run_{m + 1}.mp4"
            t0 = time.perf_counter()
            result = generator.generate_video(output_path=str(out_path), seed=SEED + m, **common_kwargs)
            wall = time.perf_counter() - t0
            e2e = (result.get("e2e_latency") if isinstance(result, dict) else None) or wall
            measured.append(e2e)
            logging_info = result.get("logging_info") if isinstance(result, dict) else None
            stages = getattr(logging_info, "stages", None) if logging_info else None
            if stages:
                for name, metrics in stages.items():
                    stage_order.setdefault(name, None)
                    stage_times.setdefault(name, []).append(float(metrics.get("execution_time", 0.0)))
            print(f"[run {m + 1}/{MEASURED_RUNS}] e2e={e2e:.2f}s wall={wall:.2f}s -> {out_path}")

        print("\n=== summary (FLF DMD port) ===")
        if measured:
            print(f"e2e (n={len(measured)}): {[round(x, 2) for x in measured]} "
                  f"-> avg {sum(measured) / len(measured):.2f}s")
        for name in stage_order:
            vals = stage_times.get(name) or []
            print(f"  {name}: {sum(vals) / len(vals):.3f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
