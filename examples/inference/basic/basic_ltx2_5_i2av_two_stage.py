# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 two-stage distilled i2v / flf2v recipe (dev transformer + distilled LoRA).

Replicates the production ComfyUI two-stage workflow on the LTX-2.5 model family
(preset ``ltx2_5_distilled_two_stage_i2v``):

Stage 1 (half resolution, joint AV):
  - conditioning image H.264-re-encoded at CRF 38 (LTXVPreprocess img_compression=38),
    resized/center-cropped so the stage-1 long side lands at ~1024,
  - first frame pinned inplace at strength 0.8 (``--last-frame`` additionally pins the
    last latent frame at the same strength — flf2v, wired the LTX-2.3 way),
  - fresh audio latent generated jointly with the video (empty audio conditioning),
  - sampler euler_ancestral_cfg_pp at cfg=1 (the uncond pass runs every step except the
    degenerate sigma=1.0 first step, whose uncond output ComfyUI itself discards),
  - sigmas from LTXVScheduler(steps=8, max_shift=4.0, base_shift=1.5, stretch=True,
    terminal=0.1) with the scheduler's no-latent token anchor (4096),
  - distilled LoRA merged at strength 0.7 (runtime-LoRA mode; see the deployment
    modes below — pre-merged directories run with no runtime LoRA at all).

Stage 2 (x2 latent upsample + refine):
  - stage-1 latents through LTX-2.5's OWN x2 spatial latent upsampler
    (latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2 — the 2.3 upscaler does
    NOT match the 2.5 latent distribution),
  - the full-resolution image re-pinned inplace at strength 1.0 (the last-frame anchor
    stays out of the refine pass by default, matching the validated 2.3 behavior),
  - manual sigmas [0.85, 0.7250, 0.4219, 0.0], euler_ancestral_cfg_pp at cfg=1,
  - distilled LoRA re-merged at strength 0.5,
  - audio latents passed through from stage 1 (re-noised to sigma 0.85 alongside video).

Decode: video VAE (conv or HQ diffusion decoder, whichever the converted ``vae/``
directory carries) + audio VAE/vocoder, muxed jointly.

Model directory — two deployment modes:

Experimental (runtime per-stage LoRA, the reference workflow's 0.7 / 0.5): convert
with every 2.5 source including the distilled LoRA, e.g.

    python scripts/checkpoint_conversion/convert_ltx2_weights.py \
      --variant dev \
      --transformer-source .../ltx-2.5-22b-dev-transformer-bf16.safetensors \
      --text-encoder-source .../gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
      --vae-source .../ltx-2.5-video-vae-conv-bf16.safetensors \
      --audio-vae-source .../ltx-2.5-audio-vae-bf16.safetensors \
      --spatial-upscaler-source .../ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
      --distilled-lora-source .../ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
      --output /models/LTX-2.5-Dev-Diffusers

The upsampler and distilled LoRA auto-resolve from the converted model_index.json.

Production (pre-merged, zero runtime LoRA cost): merge the distilled LoRA into the
transformer at conversion time instead of bundling it —

    python scripts/checkpoint_conversion/convert_ltx2_weights.py \
      --variant dev \
      --transformer-source .../ltx-2.5-22b-dev-transformer-bf16.safetensors \
      --transformer-lora .../ltx-2.5-22b-distilled-lora-450-bf16.safetensors:0.7 \
      ... remaining --*-source flags ... \
      --output /models/LTX-2.5-Dev-Merged-Diffusers

ONE merged transformer then serves BOTH stages (a deliberate deviation from the
reference workflow's per-stage 0.7/0.5 strengths, accepted for deployment
simplicity; the single merge strength is a tuning choice — A/B 0.6 vs 0.7). This
script detects the pre-merged directory from model_index.json
(fastvideo_transformer_merged_loras present / no fastvideo_refine_lora_path) and
runs with NO runtime LoRA so nothing is double-applied; --pre-merged forces that
behavior explicitly.

Notes / assumptions pending GPU verification:
  - The recipe's numerical parity against the ComfyUI reference has not yet been
    validated on real weights (the mechanics are the validated 2.3 port; the 2.5
    model family, per-stage LoRA strengths, and the sigma=1.0 CFG++ first step are
    wired per spec and CPU-tested only).
  - ComfyUI's LTXVScheduler shifts by latent token count when a latent is attached
    to the node; this script uses the node's detached default (tokens=4096). Pass
    ``--sigmas-follow-latent`` to shift by the actual stage-1 latent size instead.
  - LTX-2.5 additionally supports appended-keyframe conditioning
    (keyframes_mask + use_keyframes_abs_pos_embedding); this script pins the last
    frame inplace (2.3-style) for recipe parity — see docs/inference/ltx2_5.md.
  - Per-stage LoRA strengths re-merge the adapter twice per run (exact
    unmerge-to-pristine + re-merge; no drift, but it costs one weight sweep per
    stage switch). Any LoRA wanted at a fixed strength — the distilled adapter in
    production, extra user LoRAs always — should be offline-merged at conversion
    time with convert_ltx2_weights.py --transformer-lora PATH[:STRENGTH] so the
    runtime slot stays free (or reserved for per-stage experiments).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.pipelines.basic.ltx2.stages import compute_ltxv_scheduler_sigmas
from fastvideo.utils import maybe_download_model

# Stage-1 schedule: ComfyUI LTXVScheduler(steps=8, max_shift=4.0, base_shift=1.5,
# stretch=True, terminal=0.1). Stage-2 schedule: the workflow's manual sigma list.
STAGE1_SCHEDULER_KWARGS = dict(max_shift=4.0, base_shift=1.5, stretch=True, terminal=0.1)
STAGE1_STEPS = 8
STAGE2_SIGMAS = [0.85, 0.7250, 0.4219, 0.0]

FIRST_FRAME_STRENGTH_STAGE1 = 0.8
LAST_FRAME_STRENGTH_STAGE1 = 0.8
FIRST_FRAME_STRENGTH_STAGE2 = 1.0
IMAGE_CRF = 38.0
STAGE1_LORA_STRENGTH = 0.7
REFINE_LORA_STRENGTH = 0.5

# Final (stage-2) long side; stage 1 runs at half of this (~1024, the recipe's
# "input image resized to long side ~1024"). Both stage dims must divide by 32,
# so final dims snap to multiples of 64.
FINAL_LONG_SIDE = 2048


def _snap(value: float, multiple: int = 64) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _derive_final_dims(image_path: Path) -> tuple[int, int]:
    """Final (H, W) from the conditioning image's aspect ratio, long side 2048."""
    with Image.open(image_path) as img:
        src_w, src_h = img.size
    if src_w >= src_h:
        width = FINAL_LONG_SIDE
        height = _snap(FINAL_LONG_SIDE * src_h / src_w)
    else:
        height = FINAL_LONG_SIDE
        width = _snap(FINAL_LONG_SIDE * src_w / src_h)
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="FastVideo/LTX-2.5-Dev-Diffusers",
                        help="Converted LTX-2.5 dev directory (with spatial_upsampler/ and distilled_lora/).")
    parser.add_argument("--first-frame", required=True, help="Conditioning image pinned at frame 0.")
    parser.add_argument("--last-frame", default=None,
                        help="Optional flf2v: image pinned at the LAST latent frame (stage 1 only).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="",
                        help="CFG++ runs the uncond pass even at cfg=1; empty string is fine.")
    parser.add_argument("--output", default="outputs/ltx2_5_i2av_two_stage.mp4")
    parser.add_argument("--height", type=int, default=None, help="Final height (default: from image aspect).")
    parser.add_argument("--width", type=int, default=None, help="Final width (default: from image aspect).")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--last-strength", type=float, default=LAST_FRAME_STRENGTH_STAGE1,
                        help="Stage-1 inplace strength of the last-frame anchor (flf2v).")
    parser.add_argument("--last-in-refine", action="store_true",
                        help="Also feed the last-frame anchor into the stage-2 refine pass "
                             "(default off — misbehaved in ComfyUI testing on 2.3).")
    parser.add_argument("--stage1-lora-strength", type=float, default=STAGE1_LORA_STRENGTH)
    parser.add_argument("--refine-lora-strength", type=float, default=REFINE_LORA_STRENGTH)
    parser.add_argument("--upsampler-path", default=None,
                        help="Override the x2 spatial upsampler directory (default: model_index auto-resolve).")
    parser.add_argument("--distilled-lora", default=None,
                        help="Override the distilled LoRA path (default: model_index auto-resolve).")
    parser.add_argument("--pre-merged", action="store_true",
                        help="The transformer already has the distilled LoRA merged offline "
                             "(--transformer-lora at conversion): run BOTH stages on it with no runtime "
                             "LoRA. Auto-detected from model_index.json; pass this to force it.")
    parser.add_argument("--sigmas-follow-latent", action="store_true",
                        help="Shift the stage-1 schedule by the actual stage-1 latent token count "
                             "(ComfyUI LTXVScheduler with its latent input attached) instead of the "
                             "node's detached tokens=4096 anchor.")
    parser.add_argument("--torch-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first_frame = Path(args.first_frame)
    if not first_frame.is_file():
        raise SystemExit(f"--first-frame not found: {first_frame}")
    last_frame = Path(args.last_frame) if args.last_frame else None
    if last_frame is not None and not last_frame.is_file():
        raise SystemExit(f"--last-frame not found: {last_frame}")

    if args.height is not None and args.width is not None:
        height, width = args.height, args.width
    elif args.height is None and args.width is None:
        height, width = _derive_final_dims(first_frame)
    else:
        raise SystemExit("Pass both --height and --width, or neither.")
    if height % 64 or width % 64:
        raise SystemExit(f"Final dims must be multiples of 64 (stage 1 runs at half): got {width}x{height}")

    stage1_tokens = None
    if args.sigmas_follow_latent:
        # Stage-1 latent grid: T=(frames-1)//8+1, H/32, W/32 of the HALF resolution.
        stage1_tokens = ((args.num_frames - 1) // 8 + 1) * (height // 2 // 32) * (width // 2 // 32)
    stage1_sigmas = compute_ltxv_scheduler_sigmas(
        STAGE1_STEPS, tokens=stage1_tokens, **STAGE1_SCHEDULER_KWARGS).tolist()

    # ltx2_images frame indices are LATENT-frame indices (8x temporal compression);
    # the last anchor pins the final latent frame == final ~8 pixel frames.
    last_latent_idx = (args.num_frames - 1) // 8
    stage1_images = [(str(first_frame), 0, FIRST_FRAME_STRENGTH_STAGE1)]
    if last_frame is not None:
        stage1_images.append((str(last_frame), last_latent_idx, float(args.last_strength)))
    stage2_images = [(str(first_frame), 0, FIRST_FRAME_STRENGTH_STAGE2)]
    if last_frame is not None and args.last_in_refine:
        stage2_images.append((str(last_frame), last_latent_idx, float(args.last_strength)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_root = maybe_download_model(args.model_path)
    pipeline_config = PipelineConfig.from_pretrained(model_root)

    # Runtime-LoRA vs pre-merged detection. A directory converted with
    # --transformer-lora records fastvideo_transformer_merged_loras and omits
    # fastvideo_refine_lora_path, so both stages run the merged transformer
    # with no runtime adapter (nothing gets double-applied).
    if args.pre_merged and args.distilled_lora:
        raise SystemExit("--pre-merged and --distilled-lora contradict each other: a pre-merged "
                         "transformer must not get a runtime LoRA stacked on top.")
    model_index_path = Path(model_root) / "model_index.json"
    model_index = json.loads(model_index_path.read_text()) if model_index_path.is_file() else {}
    merged_loras = model_index.get("fastvideo_transformer_merged_loras")
    runtime_lora_available = bool(args.distilled_lora or model_index.get("fastvideo_refine_lora_path"))
    use_runtime_lora = runtime_lora_available and not args.pre_merged
    if not use_runtime_lora:
        detail = f"offline-merged LoRAs: {merged_loras}" if merged_loras else "no distilled LoRA is wired"
        print(f"Running BOTH stages on the transformer as-is ({detail}); "
              "per-stage runtime LoRA strengths are disabled.")

    engine_kwargs: dict = {}
    if args.upsampler_path:
        engine_kwargs["ltx2_refine_upsampler_path"] = args.upsampler_path
    if args.distilled_lora:
        engine_kwargs["ltx2_refine_lora_path"] = args.distilled_lora
    elif args.pre_merged:
        # Explicit empty path: blocks the model_index fastvideo_refine_lora_path
        # auto-resolve when --pre-merged is forced on a directory that still
        # bundles a runtime distilled LoRA.
        engine_kwargs["ltx2_refine_lora_path"] = ""
    if use_runtime_lora:
        # Per-stage distilled-LoRA strengths (0.7 base denoise, 0.5 refine).
        # Left unset in pre-merged mode: with no refine-LoRA path resolved the
        # pipeline builds no LoRA stages at all.
        engine_kwargs["ltx2_stage1_lora_strength"] = float(args.stage1_lora_strength)
        engine_kwargs["ltx2_refine_lora_strength"] = float(args.refine_lora_strength)
    if args.torch_compile:
        engine_kwargs.update(
            enable_torch_compile=True,
            enable_torch_compile_text_encoder=True,
            enable_torch_compile_vae=True,
        )

    generator = VideoGenerator.from_pretrained(
        model_root,
        num_gpus=args.num_gpus,
        pipeline_config=pipeline_config,
        # Two-stage refine wiring. The upsampler / distilled LoRA paths
        # auto-resolve from the converted model_index.json unless overridden.
        ltx2_refine_enabled=True,
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        # ComfyUI-style CFG++ ancestral sampler in BOTH stages.
        ltx2_sampler="euler_ancestral_cfg_pp",
        ltx2_refine_sampler="euler_ancestral_cfg_pp",
        ltx2_stage1_sigmas=stage1_sigmas,
        ltx2_stage2_sigmas=STAGE2_SIGMAS,
        **engine_kwargs,
    )
    try:
        generator.generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_path=str(output_path),
            save_video=True,
            seed=args.seed,
            height=height,
            width=width,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=STAGE1_STEPS,
            guidance_scale=1.0,
            # Inplace conditioning: stage 1 anchors at 0.8, stage 2 re-pins the
            # full-res first frame at 1.0 with the same CRF-38 preprocessed image.
            ltx2_images=stage1_images,
            ltx2_images_stage2=stage2_images,
            ltx2_image_crf=IMAGE_CRF,
            # Plain CFG++ guider: no STG / modality isolation / rescale, and the
            # official-2.5 ancestral path stays off (the cfg_pp sampler drives).
            ltx2_use_ancestral_sampler=False,
            ltx2_cfg_scale_video=1.0,
            ltx2_cfg_scale_audio=1.0,
            ltx2_modality_scale_video=1.0,
            ltx2_modality_scale_audio=1.0,
            ltx2_rescale_scale=0.0,
            ltx2_stg_scale_video=0.0,
            ltx2_stg_scale_audio=0.0,
        )
        print(f"Two-stage synchronized video+audio written to: {output_path}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
