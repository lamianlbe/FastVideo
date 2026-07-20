#!/usr/bin/env python3
"""Minimal Wan2.2 I2V smoke generation for a converted Wan22-Custom variant.

Sanity-checks that the merged SFW/NSFW directory loads AND produces a
coherent video in FastVideo — validating the dequant + LoRA-merge semantics
end to end. NOT the production recipe: this uses a plain i2v run (no
end_image, no custom 7-step/boundary tuning yet) just to eyeball whether
the merge went sideways.

    env -u LD_LIBRARY_PATH python try_wan22_i2v.py \
        --model Wan22-Custom/sfw --image first.png \
        --prompt "..." --out wan22_sfw_test.mp4

The lightx2v distill in the SFW merge is a 4-step cfg=1 model, so we use
few steps + guidance 1.0 + shift 5 to match how it was trained; a wrong
merge shows up as noise/garbage or ignored conditioning.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Converted variant dir (e.g. Wan22-Custom/sfw)")
    ap.add_argument("--image", required=True, help="First-frame image")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--out", default="wan22_test.mp4")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8, help="Total denoising steps (lightx2v distill: few)")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--boundary", type=float, default=0.875, help="High/low-noise expert switch ratio")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not Path(args.image).is_file():
        raise SystemExit(f"--image not found: {args.image}")

    from fastvideo import VideoGenerator

    print(f"loading {args.model} …")
    t0 = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
    )
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    result = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=args.image,
        output_path=args.out,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        boundary_ratio=args.boundary,
        save_video=True,
    )
    dt = time.perf_counter() - t0
    e2e = (result.get("e2e_latency") if isinstance(result, dict) else None) or dt
    print(f"\ngenerated in {e2e:.1f}s -> {args.out}")
    print("Eyeball: subject follows the image, motion coherent, no static/noise. "
          "If it looks right, the merge is good; refine the recipe next.")
    generator.shutdown()


if __name__ == "__main__":
    main()
