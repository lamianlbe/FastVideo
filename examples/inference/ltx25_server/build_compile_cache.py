#!/usr/bin/env python3
"""Populate the persistent torch.compile cache for every configured mode.

    env -u LD_LIBRARY_PATH python build_compile_cache.py --config config.yaml

Runs one full generation per distinct mode (fps included: the audio latent
length is derived from num_frames / fps, so each fps is its own compile
shape) so all inductor artifacts land in the config's
``inductor_cache_dir``. Ship that directory to every identical machine
(same GPU model / driver / torch / fastvideo stack) and server startup
warmup drops from a cold compile (tens of minutes per shape) to a dynamo
re-trace (about a minute per shape).

Each LTX-2.5 mode compiles TWO transformer shapes — stage 1 at half the
mode's resolution and stage 2 at the full resolution — plus the VAE decode,
so budget accordingly.

The cache is keyed on the full stack: rebuild it after changing torch,
fastvideo, the quantization mode, the attention backend, the VAE flavor
(conv vs HQ diffusion decoder), or the GPU model.

Multi-GPU: the cache is keyed on GPU *model*, not instance, so identical
GPUs share it — each mode only needs compiling ONCE. To halve wall time on
an N-GPU box, run N processes, each pinned to a GPU and building its slice
of the mode list, all writing the shared inductor_cache_dir:

    python build_compile_cache.py --config config.yaml --gpu 0 --shard 0/2 &
    python build_compile_cache.py --config config.yaml --gpu 1 --shard 1/2 &

Disjoint modes -> disjoint top-level cache entries; any shared sub-kernels
are protected by inductor's per-file locking (the cache is designed for
concurrent multi-process use). Assumes all GPUs are the same model.
"""

from __future__ import annotations

import argparse
import time

from ltx25_engine import cache_dir_size, create_generator, load_config, run_warmup, setup_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the server YAML config")
    parser.add_argument("--runs-per-shape", type=int, default=1,
                        help="Generations per compile shape (1 is enough to fill the cache)")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES for this build process, e.g. '0'")
    parser.add_argument("--shard", default=None,
                        help="'i/n': build only modes[i::n] (one process per GPU, shared cache)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.compile:
        raise SystemExit("config has compile: false — nothing to cache. Enable compile first.")
    if not cfg.inductor_cache_dir:
        raise SystemExit("config has no inductor_cache_dir — the compile artifacts would land in "
                         "a non-persistent default location. Set inductor_cache_dir first.")
    if args.gpu is not None:
        cfg.cuda_visible_devices = args.gpu
    shard_label = "all"
    if args.shard is not None:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError as err:
            raise SystemExit(f"--shard must be 'i/n' with integers, got {args.shard!r}") from err
        if not 0 <= i < n:
            raise SystemExit(f"--shard i/n requires 0 <= i < n, got {args.shard}")
        cfg.modes = cfg.modes[i::n]
        shard_label = f"{i}/{n}"
        if not cfg.modes:
            print(f"[compile-cache] shard {shard_label}: no modes in this slice; nothing to do")
            return
    setup_environment(cfg)

    print(f"[compile-cache] cache dir: {cfg.inductor_cache_dir} "
          f"(current size: {cache_dir_size(cfg.inductor_cache_dir)})")
    print(f"[compile-cache] shard {shard_label}, modes: {len(cfg.modes)}, quant={cfg.quant}, "
          f"gpu={cfg.cuda_visible_devices or '(default)'}")
    for mode in cfg.modes:
        stage1_width, stage1_height = mode.stage1_size()
        print(f"[compile-cache]   {mode.width}x{mode.height} f{mode.num_frames} @{mode.fps} "
              f"(stage 1 at {stage1_width}x{stage1_height})")

    t0 = time.perf_counter()
    generator = create_generator(cfg)
    try:
        # encode_check=False: the CPU H.264 encoder has no compile cache.
        run_warmup(generator, cfg, runs_per_shape=args.runs_per_shape, encode_check=False)
    finally:
        generator.shutdown()

    print(f"[compile-cache] done in {(time.perf_counter() - t0) / 60:.1f} min; "
          f"cache size now {cache_dir_size(cfg.inductor_cache_dir)}")
    print("[compile-cache] sync this directory to identical machines and point their "
          "TORCHINDUCTOR_CACHE_DIR / config inductor_cache_dir at it.")


if __name__ == "__main__":
    main()
