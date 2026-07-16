#!/usr/bin/env python3
"""FastVideo port of the ComfyUI "all-in-one v2" LTX-2.3 I2V workflow.

    env -u LD_LIBRARY_PATH python eager_ltx2_3_allinone_i2v.py \
        --first-frame first.png [--last-frame last.png] --prompt "..." \
        --width 1344 --height 768 --fps 24 --num-frames 241 \
        --output out.mp4 --image-crf 0 --image-crf-stage2 0

Per-run params are CLI flags (same surface as the DMD example; see --help).
Passing --last-frame switches i2v -> first+last-frame. This workflow's
frame-0 guide anchor stays at strength 0.8 and defaults to CRF 0 (AddGuide
does no preprocessing); identity comes from LatentAnchorAware, not
reference tokens.

Differences from the DMD example (eager_ltx2_3_ancestral_reference_i2v.py):
this workflow uses a 10-step eased sigma schedule with a per-step CFG head
([2.0, 1.5, 1, ...] + std-rescale), the LatentAnchorAware identity
stabilizer (blocks 10-30, snapshot locked at step 2), the x1.5 spatial
upscaler, guide-strength 0.8 frame-0 anchoring, and a stage-2 text
cross-attention amplifier (x1.3, blocks 36-47, center-weighted).

The latent anchor's snapshot cache is torch.compile-compatible (tensor
buffers + torch.where capture flags); compile support pending B200
validation.

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

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults for the CLI args (env still seeds them for backward compat).
# x1.5 refine requires both output dims divisible by 96 (stage 1 =
# target / 1.5 must hit even /32 latent dims): 1344x768 -> 896x512.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.getenv("LTX23_MODEL_PATH", "/workspace/10Eros_v1_Diffusers")
DEFAULT_PROMPT = os.getenv("LTX23_I2V_PROMPT",
                           "情色电影，温暖亲密光影。画面右侧的男生用双手持续揉捏女生的乳房，拇指反复刺激乳头，画面左侧的女生保持柔和微张嘴表情"
                           "并发出轻微喘息，两人目光锁定，身体轻微摇摆。")

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
# LTXPlusBatchAddGuide frame-0 anchor strength (guide-token approximation);
# a workflow constant, not a CLI arg (mirrors the DMD example's fixed 1.0).
IMAGE_STRENGTH = 0.8

NEGATIVE_PROMPT = ("still image, bad quality, subtitles, text, watermark, overlay effects, pc game, "
                   "yelling, console game, video game, cartoon, childish, ugly, text, blur, logo, "
                   "wordmark, static, low quality, noise, white noise, bleep, censoring, censor, "
                   "bleeping, beep, beeping, newscast, interview, podcast, non-english, foreign "
                   "language, russian, chinese, japanese, mutant, horror, 70's, film grain, "
                   "cinematic, comedy, stand-up ")

QUANT = os.getenv("LTX23_QUANT", "nvfp4").lower()  # nvfp4 | fp8 | fp8_channel | none
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LTX-2.3 all-in-one v2 i2v / first+last-frame "
                                "generation (ComfyUI workflow port; eager-only).")
    p.add_argument("--first-frame", default=os.getenv("LTX23_I2V_IMAGE", ""),
                   help="First-frame image: frame-0 guide anchor + LatentAnchorAware source. Required.")
    p.add_argument("--last-frame", default=None,
                   help="Optional last-frame image; passing it switches i2v -> FLF.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt (passed verbatim).")
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT,
                   help="Negative prompt; feeds the CFG head + cfg_pp uncond.")
    p.add_argument("--width", type=int, default=1344,
                   help="Output width; divisible by 96 for the x1.5 upscaler.")
    p.add_argument("--height", type=int, default=768, help="Output height; same divisibility as --width.")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-frames", type=int, default=int(os.getenv("LTX23_NUM_FRAMES", "241")),
                   help="Frame count (8*k+1).")
    p.add_argument("--output", default=os.getenv("LTX23_OUTPUT", "outputs_video/allinone_i2v.mp4"),
                   help="Output mp4 path (or a directory).")
    p.add_argument("--image-crf", type=float, default=0.0,
                   help="Stage-1 conditioning CRF; this workflow's AddGuide path defaults to 0.")
    p.add_argument("--image-crf-stage2", type=float, default=None,
                   help="Stage-2 (refine) CRF; default = same as --image-crf.")
    p.add_argument("--last-strength", type=float, default=0.8,
                   help="FLF tail-anchor strength in [0, 1] (only with --last-frame).")
    p.add_argument("--last-in-upscale", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether the tail anchor also enters the stage-2 refine pass (FLF only).")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model repo (Diffusers layout).")
    p.add_argument("--seed", type=int, default=int(os.getenv("LTX23_SEED", "635141064074927")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.first_frame or not Path(args.first_frame).is_file():
        raise SystemExit(f"--first-frame not found: {args.first_frame!r}")
    if args.last_frame and not Path(args.last_frame).is_file():
        raise SystemExit(f"--last-frame not found: {args.last_frame}")
    if QUANT not in ("nvfp4", "fp8", "fp8_channel", "none"):
        raise SystemExit(f"LTX23_QUANT must be nvfp4 | fp8 | fp8_channel | none, got {QUANT}")
    if not 0.0 <= args.last_strength <= 1.0:
        raise SystemExit(f"--last-strength must be in [0, 1], got {args.last_strength}")

    output = Path(args.output)
    if output.suffix:
        out_dir, out_file = output.parent, output
    else:
        out_dir, out_file = output, output / "allinone_i2v.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_root = maybe_download_model(args.model)
    upsampler_path = resolve_upsampler(model_root)

    # frame-0 guide anchor at IMAGE_STRENGTH; last frame (FLF) at --last-strength.
    # LATENT frame indices (8x temporal compression).
    images = [(args.first_frame, 0, IMAGE_STRENGTH)]
    images_stage2 = None
    if args.last_frame:
        last_latent_idx = (args.num_frames - 1) // 8
        images.append((args.last_frame, last_latent_idx, args.last_strength))
        if not args.last_in_upscale:
            images_stage2 = [(args.first_frame, 0, IMAGE_STRENGTH)]

    print(f"model:      {model_root}")
    print(f"upsampler:  {upsampler_path} (x1.5 expected)")
    print(f"first:      {args.first_frame} @ strength {IMAGE_STRENGTH}")
    if args.last_frame:
        print(f"last:       {args.last_frame} @ latent idx {(args.num_frames - 1) // 8}, "
              f"strength {args.last_strength}, in_upscale={args.last_in_upscale}")
    print(f"frames:     {args.num_frames} @ {args.fps} fps, {args.width}x{args.height} "
          f"(stage1 {args.width * 2 // 3}x{args.height * 2 // 3})")
    print(f"image_crf:  stage1={args.image_crf} "
          f"stage2={args.image_crf_stage2 if args.image_crf_stage2 is not None else '(same)'}")
    print(f"output:     {out_file}")
    print(f"quant:      {QUANT}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    # Linear quant ladder: nvfp4 (fastest, most lossy) | fp8 | fp8_channel
    # (most conservative quantized tier) | none (bf16).
    if QUANT == "nvfp4":
        pipeline_config.dit_config.quant_config = NVFP4Config()
    elif QUANT in ("fp8", "fp8_channel"):
        from fastvideo.layers.quantization.fp8_config import FP8Config
        pipeline_config.dit_config.quant_config = FP8Config(
            granularity="channel" if QUANT == "fp8_channel" else "tensor")
    else:
        pipeline_config.dit_config.quant_config = None

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
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,  # feeds the CFG head + cfg_pp uncond
        guidance_scale=1.0,
        ltx2_rescale_scale=1.0,           # STGGuiderAdvanced std-rescale on CFG steps
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=len(STAGE1_SIGMAS) - 1,
        # frame-0 guide anchor (+ last frame for FLF); AddGuide path defaults
        # to CRF 0 but --image-crf / --image-crf-stage2 override per stage.
        ltx2_images=images,
        ltx2_images_stage2=images_stage2,
        ltx2_image_crf=args.image_crf,
        ltx2_image_crf_stage2=args.image_crf_stage2,
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
            # Single run -> exactly --output; multiple runs -> _runN siblings.
            out_path = (out_file if MEASURED_RUNS == 1 else out_file.with_name(
                f"{out_file.stem}_run{m + 1}{out_file.suffix or '.mp4'}"))
            t0 = time.perf_counter()
            result = generator.generate_video(
                output_path=str(out_path),
                seed=args.seed + m,
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
