#!/usr/bin/env python3
"""FastVideo port of the ComfyUI "all-in-one v2" LTX-2.3 I2V workflow.

    env -u LD_LIBRARY_PATH python eager_ltx2_3_allinone_i2v.py

Differences from the DMD example (eager_ltx2_3_ancestral_reference_i2v.py):
this workflow uses a 10-step eased sigma schedule with a per-step CFG head
([2.0, 1.5, 1, ...] + std-rescale), the LatentAnchorAware identity
stabilizer (blocks 10-30, snapshot locked at step 2), the x1.5 spatial
upscaler, guide-strength 0.8 frame-0 anchoring, and a stage-2 text
cross-attention amplifier (x1.3, blocks 36-47, center-weighted).

Eager-only: the latent anchor's per-block snapshot cache is incompatible
with torch.compile (the stage rejects the combination).

Model: the all-in-one merged repo (base + 3 style LoRAs + gated condsafe
distill LoRA), e.g. built with merge_ltx2_lora_stack.py + convert_ltx23_weights.py.

x1.5 upsampler (one-time):
    hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors
    python scripts/checkpoint_conversion/convert_ltx2_upsampler.py \
        --source ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors \
        --output <MODEL_PATH>/spatial_upscaler_x1_5
    export LTX23_UPSAMPLER_PATH=<MODEL_PATH>/spatial_upscaler_x1_5

Known approximations vs the ComfyUI graph (accepted by design):
- stage-1 keyframe: frame-0 inplace at strength 0.8 approximates the
  appended guide token (same 0.2-sigma timestep semantics; the 0.8
  guide-attention attenuation bias is dropped for FlashAttention
  compatibility);
- stage-2 re-anchors at the same 0.8 (workflow uses 1.0 there).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("LTX23_MODEL_PATH", "/workspace/10Eros-AllInOne-LTX-2.3-Distilled-Diffusers")
IMAGE_PATH = os.getenv("LTX23_I2V_IMAGE", "/workspace/keyframe.png")
PROMPT_BODY = os.getenv("LTX23_I2V_PROMPT", "REPLACE ME: describe the motion and scene here.")
OUTPUT_DIR = Path(os.getenv("LTX23_OUTPUT_DIR", "outputs_video/allinone_i2v_test"))
SEED = int(os.getenv("LTX23_SEED", "635141064074927"))

# Final output dims. x1.5 refine requires both divisible by 96
# (stage 1 = target / 1.5 must hit even /32 latent dims): 1344x768 -> 896x512.
WIDTH = 1344
HEIGHT = 768
NUM_FRAMES = int(os.getenv("LTX23_NUM_FRAMES", "241"))
FPS = 24

# Stage-1 schedule: the workflow's 14-value ManualSigmas after the
# RES4LYF Sigmas Easing transform (cubic in-out, strength 0.7), with the
# three near-1.0 originals (0.99375/0.9875/0.98125) dropped — validated
# as quality-neutral. 10 denoising steps.
STAGE1_SIGMAS = [
    1.0, 0.99987238, 0.99820748, 0.99001548, 0.96332988, 0.89394948, 0.74459600, 0.47298248, 0.20186216, 0.04708576,
    0.0
]
# STGGuiderAdvanced cfg_values head (STG itself is disabled in the workflow
# via out-of-range layer indices; only the CFG schedule remains).
STAGE1_CFG_VALUES = [2.0, 1.5]  # padded with 1.0 for the remaining steps
STAGE2_SIGMAS = [0.85, 0.725, 0.4219, 0.0]
IMAGE_STRENGTH = 0.8  # LTXPlusBatchAddGuide strength (frame-0 anchor)

NEGATIVE_PROMPT = ("still image, bad quality, subtitles, text, watermark, overlay effects, pc game, "
                   "yelling, console game, video game, cartoon, childish, ugly, text, blur, logo, "
                   "wordmark, static, low quality, noise, white noise, bleep, censoring, censor, "
                   "bleeping, beep, beeping, newscast, interview, podcast, non-english, foreign "
                   "language, russian, chinese, japanese, mutant, horror, 70's, film grain, "
                   "cinematic, comedy, stand-up ")

QUANT = os.getenv("LTX23_QUANT", "nvfp4").lower()  # nvfp4 | none
MEASURED_RUNS = int(os.getenv("LTX23_MEASURED_RUNS", "1"))

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

from fastvideo import VideoGenerator  # noqa: E402  (after env setup)
from fastvideo.configs.pipelines.base import PipelineConfig  # noqa: E402
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config  # noqa: E402
from fastvideo.utils import maybe_download_model  # noqa: E402


def resolve_upsampler(model_root: str) -> str:
    override = os.getenv("LTX23_UPSAMPLER_PATH", "")
    candidates = ([override] if override else []) + [
        str(Path(model_root) / "spatial_upscaler_x1_5"),
        str(Path(model_root) / "spatial_upscaler"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "config.json").is_file():
            return cand
    raise SystemExit("No spatial upsampler found; see the docstring for the x1.5 conversion "
                     "commands, or set LTX23_UPSAMPLER_PATH.")


def main() -> None:
    if not Path(IMAGE_PATH).is_file():
        raise SystemExit(f"IMAGE_PATH not found: {IMAGE_PATH} (set LTX23_I2V_IMAGE)")
    if "REPLACE ME" in PROMPT_BODY:
        raise SystemExit("Set PROMPT_BODY or the LTX23_I2V_PROMPT env var.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(MODEL_PATH)
    upsampler_path = resolve_upsampler(model_root)
    print(f"model:     {model_root}")
    print(f"upsampler: {upsampler_path} (x1.5 expected)")
    print(f"image:     {IMAGE_PATH}")
    print(f"frames:    {NUM_FRAMES} @ {FPS} fps, {WIDTH}x{HEIGHT} (stage1 {WIDTH * 2 // 3}x{HEIGHT * 2 // 3})")
    print(f"quant:     {QUANT}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = (NVFP4Config() if QUANT == "nvfp4" else None)

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=1,
        pipeline_config=pipeline_config,
        # --- two-stage refine with the x1.5 upscaler ---
        ltx2_refine_enabled=True,
        ltx2_refine_upsampler_path=upsampler_path,
        ltx2_refine_lora_path="",
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        # --- samplers + schedules (workflow nodes 952/972/943+945/973) ---
        ltx2_sampler="euler_ancestral",
        ltx2_refine_sampler="euler_ancestral_cfg_pp",
        ltx2_stage1_sigmas=STAGE1_SIGMAS,
        ltx2_stage2_sigmas=STAGE2_SIGMAS,
        # --- per-step CFG head (STGGuiderAdvanced node 944, STG disabled) ---
        ltx2_stage1_cfg_values=STAGE1_CFG_VALUES,
        # --- LatentAnchorAware (node 942) ---
        ltx2_anchor_strength=0.11,
        ltx2_anchor_blocks="10-30",
        ltx2_anchor_cache_at_step=2,  # comfy call-counting locks at step 2
        ltx2_anchor_similarity_threshold=0.5,
        ltx2_anchor_decay_with_distance=0.15,
        ltx2_anchor_energy_threshold=0.3,
        ltx2_anchor_frame=0,
        # --- Text attention amplifier (node 969, stage 2) ---
        ltx2_text_amp_scale=1.3,
        ltx2_text_amp_blocks="36-48",  # clamps to 36-47
        ltx2_text_amp_spatial_focus=0.15,
        ltx2_text_amp_stage="refine",
        # --- keep everything resident ---
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    common_kwargs = dict(
        prompt=PROMPT_BODY,
        negative_prompt=NEGATIVE_PROMPT,  # feeds the CFG head + cfg_pp uncond
        guidance_scale=1.0,
        ltx2_rescale_scale=1.0,           # STGGuiderAdvanced std-rescale on CFG steps
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=FPS,
        num_inference_steps=len(STAGE1_SIGMAS) - 1,
        # Keyframe anchor at frame 0, strength 0.8 (guide-token approximation;
        # the workflow's AddGuide path applies no CRF preprocessing).
        ltx2_images=[(IMAGE_PATH, 0, IMAGE_STRENGTH)],
        ltx2_image_crf=0.0,
        ltx2_stg_scale_video=0.0,
        ltx2_stg_scale_audio=0.0,
        ltx2_cfg_scale_video=1.0,
        ltx2_cfg_scale_audio=1.0,
        ltx2_modality_scale_video=1.0,
        ltx2_modality_scale_audio=1.0,
        save_video=True,
    )

    try:
        measured: list[float] = []
        stage_times: dict[str, list[float]] = {}
        stage_order: OrderedDict = OrderedDict()
        for m in range(MEASURED_RUNS):
            out_path = OUTPUT_DIR / f"allinone_run_{m + 1}.mp4"
            t0 = time.perf_counter()
            result = generator.generate_video(
                output_path=str(out_path),
                seed=SEED + m,
                **common_kwargs,
            )
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

        print("\n=== summary (all-in-one workflow port) ===")
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
