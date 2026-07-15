#!/usr/bin/env python3
"""FastVideo test run replicating the ComfyUI two-pass I2V DMD workflow
setting-for-setting, with eager/compile and bf16/NVFP4 toggles for A/B runs.

    env -u LD_LIBRARY_PATH python eager_ltx2_3_ancestral_reference_i2v.py \
        --first-frame first.png [--last-frame last.png] --prompt "..." \
        --width 1344 --height 768 --fps 24 --num-frames 241 \
        --output out.mp4 --image-crf 35 --image-crf-stage2 0

Per-run params are CLI flags (see --help). Passing --last-frame switches
from i2v to first+last-frame (FLF): the tail image anchors the final
latent frame, while reference-token identity conditioning always stays on
the first frame. Stage-1 and stage-2 conditioning CRF are independent
(--image-crf / --image-crf-stage2). These env vars remain A/B toggles:
    LTX23_COMPILE=1        torch.compile DiT + text encoder + VAE
                           (fullgraph, inductor default mode). First-time
                           compile is ~30-40 min on GB200/B200 and is cached
                           in $TORCHINDUCTOR_CACHE_DIR — point that at a
                           persistent volume on RunPod.
    LTX23_QUANT=nvfp4|none NVFP4 linear quantization (default nvfp4).
    LTX23_WARMUP_RUNS      untimed warmups before measuring (default 2 when
                           compiling, else 0).
    LTX23_MEASURED_RUNS    timed runs (default 2).

Workflow settings replicated: euler_ancestral stage 1 (9-step custom
sigmas) / euler_ancestral_cfg_pp stage 2 ([0.92, 0.725, 0.421875, 0]),
cfg=1, frame-0 image anchor at strength 1.0, reference token conditioning
from the same image, CRF 35 preprocestsing, x2 (or x1.5 via
LTX23_UPSAMPLER_PATH) two-stage refine.

Upsampler: fetch once if the model repo lacks `spatial_upscaler/`:
    hf download FastVideo/LTX-2.3-Distilled-Diffusers \
        --include "spatial_upscaler/*" --local-dir <MODEL_PATH>
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults for the CLI args (env still seeds them for backward compat).
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.getenv("LTX23_MODEL_PATH", "/workspace/10Eros_v1.4_Diffusers")
DEFAULT_PROMPT = os.getenv("LTX23_I2V_PROMPT",
                           "情色电影，温暖亲密光影。画面右侧的男生用双手持续揉捏女生的乳房，拇指反复刺激乳头，画面左侧的女生保持柔和微张嘴表情"
                           "并发出轻微喘息，两人目光锁定，身体轻微摇摆。")

# Workflow sampling (nodes 914 / 582 / 910 / 911).
STAGE1_SIGMAS = [1.000, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
STAGE2_SIGMAS = [0.92, 0.725, 0.421875, 0.0]

# Node 537 negative prompt, verbatim. The user prompt is passed through
# as-is (no preamble concatenation).
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
MEASURED_RUNS = int(os.getenv("LTX23_MEASURED_RUNS", "2"))

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")

if COMPILE:
    import torch._inductor.config as _inductor

    # shape_padding=False is mandatory on Blackwell: pad_mm otherwise hits a
    # cuBLAS INVALID_VALUE crash in the refine path. The rest mirror the
    # official basic_ltx2_3_distilled_i2v example.
    _inductor.shape_padding = False
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
    raise SystemExit(
        "No spatial upsampler found. Download the x2 upsampler into the model repo:\n"
        f"  hf download FastVideo/LTX-2.3-Distilled-Diffusers --include 'spatial_upscaler/*' "
        f"--local-dir {model_root}\n"
        "or set LTX23_UPSAMPLER_PATH.")


def _collect_stage_times(result, stage_times: dict[str, list[float]], stage_order: OrderedDict) -> None:
    logging_info = result.get("logging_info") if isinstance(result, dict) else None
    stages = getattr(logging_info, "stages", None) if logging_info else None
    if not stages:
        return
    for name, metrics in stages.items():
        stage_order.setdefault(name, None)
        stage_times.setdefault(name, []).append(float(metrics.get("execution_time", 0.0)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LTX-2.3 i2v / first+last-frame generation "
                                "(ComfyUI two-pass DMD workflow port).")
    p.add_argument("--first-frame", default=os.getenv("LTX23_I2V_IMAGE", ""),
                   help="First-frame image: anchors frame 0 and the reference tokens. Required.")
    p.add_argument("--last-frame", default=None,
                   help="Optional last-frame image; passing it switches i2v -> FLF.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt (passed verbatim).")
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT,
                   help="Negative prompt; feeds the cfg_pp uncond pass.")
    p.add_argument("--width", type=int, default=1344,
                   help="Output width; divisible by 64 (x2 refine) / 96 (x1.5 upscaler).")
    p.add_argument("--height", type=int, default=768, help="Output height; same divisibility as --width.")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-frames", type=int, default=int(os.getenv("LTX23_NUM_FRAMES", "241")),
                   help="Frame count (8*k+1).")
    p.add_argument("--output", default=os.getenv("LTX23_OUTPUT", "outputs_video/eager_i2v.mp4"),
                   help="Output mp4 path (or a directory).")
    p.add_argument("--image-crf", type=float, default=35.0,
                   help="Stage-1 conditioning CRF (LTXVPreprocess motion strength).")
    p.add_argument("--image-crf-stage2", type=float, default=None,
                   help="Stage-2 (refine) CRF; default = same as --image-crf. 0 = sharpest re-anchor.")
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
    if QUANT not in ("nvfp4", "none"):
        raise SystemExit(f"LTX23_QUANT must be nvfp4 or none, got {QUANT}")
    if not 0.0 <= args.last_strength <= 1.0:
        raise SystemExit(f"--last-strength must be in [0, 1], got {args.last_strength}")

    output = Path(args.output)
    if output.suffix:
        out_dir, out_file = output.parent, output
    else:
        out_dir, out_file = output, output / "eager_i2v.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_root = maybe_download_model(args.model)
    upsampler_path = resolve_upsampler(model_root)

    # FLF vs i2v conditioning (LATENT frame indices; 8x temporal compression).
    images = [(args.first_frame, 0, 1.0)]
    images_stage2 = None
    if args.last_frame:
        last_latent_idx = (args.num_frames - 1) // 8
        images.append((args.last_frame, last_latent_idx, args.last_strength))
        if not args.last_in_upscale:
            images_stage2 = [(args.first_frame, 0, 1.0)]

    print(f"model:      {model_root}")
    print(f"upsampler:  {upsampler_path}")
    print(f"first:      {args.first_frame}")
    if args.last_frame:
        print(f"last:       {args.last_frame} @ latent idx {(args.num_frames - 1) // 8}, "
              f"strength {args.last_strength}, in_upscale={args.last_in_upscale}")
    print(f"frames:     {args.num_frames} @ {args.fps} fps, {args.width}x{args.height}")
    print(f"image_crf:  stage1={args.image_crf} "
          f"stage2={args.image_crf_stage2 if args.image_crf_stage2 is not None else '(same)'}")
    print(f"output:     {out_file}")
    print(f"mode:       compile={COMPILE} quant={QUANT} warmup={WARMUP_RUNS} measured={MEASURED_RUNS}")
    if COMPILE:
        print(f"inductor cache: {os.getenv('TORCHINDUCTOR_CACHE_DIR', '(default, not persistent!)')}")

    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.dit_config.quant_config = (NVFP4Config() if QUANT == "nvfp4" else None)

    compile_kwargs: dict = {}
    if COMPILE:
        torch_compile_kwargs = {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "default",  # matches max-autotune on this pipeline, saves ~7 min cold compile
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
        ltx2_reference_image_path=args.first_frame,
        ltx2_reference_strength=1.0,                     # node 860/870 strength
        ltx2_reference_position_mode="reference",        # node 860/870 position_mode
        ltx2_reference_zero_timesteps=False,             # node 868 default
        # --- keep everything resident on the B200 ---
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        ltx2_vae_tiling=False,
    )

    common_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,        # node 537; feeds cfg_pp's uncond pass
        guidance_scale=1.0,                          # cfg=1 in both SamplerCustom nodes
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=len(STAGE1_SIGMAS) - 1,
        # frame-0 anchor at full strength; last frame (FLF) at --last-strength.
        ltx2_images=images,
        # Stage-2 keyframe override: None = same list; a first-only list keeps
        # the tail anchor out of the refine pass (--no-last-in-upscale).
        ltx2_images_stage2=images_stage2,
        # Independent stage-1 / stage-2 conditioning CRF.
        ltx2_image_crf=args.image_crf,
        ltx2_image_crf_stage2=args.image_crf_stage2,
        # No STG / modality guidance in this workflow revision.
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
            print(f"\n[warmup {w + 1}/{WARMUP_RUNS}] (compile + shape guards settle here)…")
            generator.generate_video(
                output_path=str(out_dir / f"_warmup_{w + 1}.mp4"),
                seed=args.seed,
                **common_kwargs,
            )
            print(f"[warmup {w + 1}/{WARMUP_RUNS}] wall={time.perf_counter() - t0:.1f}s")
        for w in range(WARMUP_RUNS):
            (out_dir / f"_warmup_{w + 1}.mp4").unlink(missing_ok=True)

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
            _collect_stage_times(result, stage_times, stage_order)
            print(f"[measured {m + 1}/{MEASURED_RUNS}] e2e={e2e:.2f}s wall={wall:.2f}s -> {out_path}")

        print(f"\n=== summary (compile={COMPILE} quant={QUANT}) ===")
        if measured:
            print(f"measured e2e (n={len(measured)}): "
                  f"{[round(x, 2) for x in measured]} -> avg {sum(measured) / len(measured):.2f}s")
        if stage_times:
            total = 0.0
            for name in stage_order:
                vals = stage_times.get(name) or []
                avg = sum(vals) / len(vals)
                total += avg
                print(f"  {name}: {avg:.3f}s")
            print(f"  stage_sum_avg: {total:.3f}s")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
