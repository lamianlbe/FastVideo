#!/usr/bin/env python3
"""LTX-2.3 production HTTP server (compiled, first-frame / first+last-frame).

    env -u LD_LIBRARY_PATH python server.py --config config.yaml

Startup: loads the config, builds the compiled generator, then runs one
warmup generation per distinct compile shape (dynamo trace + inductor cache
hit) before binding the port. Point TORCHINDUCTOR_CACHE_DIR / the config's
``inductor_cache_dir`` at the cache produced by build_compile_cache.py so
warmup is a re-trace (minutes), not a cold compile (tens of minutes).

Endpoints:
    POST /v1/generate   multipart form; returns the mp4 synchronously
    GET  /v1/modes      supported {width, height, num_frames, fps} combos
    GET  /healthz       liveness + warmup state

Requests whose (width, height, num_frames, fps) don't exactly match a
configured mode are served with the closest-resolution mode; conditioning
images are cover-fit (aspect-preserving resize + center crop, no
letterboxing) to the served resolution inside the pipeline. The actually
served combo is reported in X-LTX23-* response headers.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
import threading
from pathlib import Path

# fastapi doesn't import torch, so these are safe before
# setup_environment() stages the process env. They must be module-level for
# FastAPI to resolve the endpoint's postponed annotations.
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# ltx23_engine keeps all torch/fastvideo imports function-local.
from ltx23_engine import (
    GenerationRequest,
    Ltx23ServerConfig,
    create_generator,
    generate_for_mode,
    load_config,
    match_mode,
    run_warmup,
    setup_environment,
)

_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _save_upload(upload, dest_dir: Path, stem: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in _ALLOWED_IMAGE_SUFFIXES:
        suffix = ".png"
    dest = dest_dir / f"{stem}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    if dest.stat().st_size == 0:
        raise ValueError(f"uploaded {stem} file is empty")
    return str(dest)


def build_app(generator, cfg: Ltx23ServerConfig) -> FastAPI:
    app = FastAPI(title="FastVideo LTX-2.3 server", version="1.0")
    # One GPU pipeline: requests queue on this lock and run strictly
    # serially (sync endpoints run in Starlette's threadpool).
    gpu_lock = threading.Lock()
    scratch_root = cfg.output_dir or None
    if scratch_root:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "busy": gpu_lock.locked()}

    @app.get("/v1/modes")
    def modes() -> dict:
        return {
            "modes": [{
                "width": m.width,
                "height": m.height,
                "num_frames": m.num_frames,
                "fps": m.fps,
            } for m in cfg.modes]
        }

    @app.post("/v1/generate")
    def generate(
        prompt: str = Form(...),
        width: int = Form(...),
        height: int = Form(...),
        num_frames: int = Form(...),
        fps: int = Form(...),
        first_frame: UploadFile = File(...),
        last_frame: UploadFile | None = File(None),
        negative_prompt: str | None = Form(None),
        seed: int | None = Form(None),
        last_frame_strength: float = Form(cfg.last_frame_strength),
        # Product defaults: tail anchor DOES enter the stage-2 refine pass,
        # and stage 2 re-anchors with a clean CRF-0 encode.
        last_in_upscale: bool = Form(True),
        image_crf: float = Form(cfg.image_crf),
        image_crf_stage2: float = Form(cfg.image_crf_stage2),
    ):
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        if width <= 0 or height <= 0 or num_frames <= 0 or fps <= 0:
            raise HTTPException(status_code=400, detail="width/height/num_frames/fps must be positive")
        if not 0.0 <= last_frame_strength <= 1.0:
            raise HTTPException(status_code=400, detail="last_frame_strength must be in [0, 1]")

        mode, exact = match_mode(cfg.modes, width, height, num_frames, fps)
        req_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

        workdir = Path(tempfile.mkdtemp(prefix="ltx23_req_", dir=scratch_root))
        try:
            first_path = _save_upload(first_frame, workdir, "first")
            last_path = (_save_upload(last_frame, workdir, "last")
                         if last_frame is not None and last_frame.filename else None)
        except ValueError as err:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(err)) from err

        request = GenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            first_frame_path=first_path,
            last_frame_path=last_path,
            seed=req_seed,
            last_frame_strength=last_frame_strength,
            last_in_upscale=last_in_upscale,
            image_crf=image_crf,
            image_crf_stage2=image_crf_stage2,
        )

        try:
            with gpu_lock:
                result = generate_for_mode(generator, cfg, mode, request, workdir / "output.mp4")
        except Exception as err:  # noqa: BLE001
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"generation failed: {err}") from err

        video_path = result["video_path"]
        if not Path(video_path).is_file():
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="generation produced no video file")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="output.mp4",
            headers={
                "X-LTX23-Width": str(mode.width),
                "X-LTX23-Height": str(mode.height),
                "X-LTX23-Num-Frames": str(mode.num_frames),
                "X-LTX23-Fps": str(mode.fps),
                "X-LTX23-Exact-Match": "1" if exact else "0",
                "X-LTX23-Seed": str(req_seed),
                "X-LTX23-E2E-Seconds": f"{result['e2e_latency']:.2f}",
            },
            background=BackgroundTask(shutil.rmtree, str(workdir), ignore_errors=True),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the server YAML config")
    parser.add_argument("--host", default=None, help="Override config host")
    parser.add_argument("--port", type=int, default=None, help="Override config port")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip startup warmup (first requests then pay the dynamo trace)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)  # before create_generator imports torch/fastvideo

    print(f"[server] building generator (compile={cfg.compile}, quant={cfg.quant})…")
    generator = create_generator(cfg)

    if cfg.warmup_on_start and not args.skip_warmup:
        print(f"[server] warming up {len(cfg.modes)} mode(s)…")
        run_warmup(generator, cfg)
        print("[server] warmup complete")
    else:
        print("[server] warmup skipped — first request per shape pays the dynamo trace")

    app = build_app(generator, cfg)
    import uvicorn
    try:
        uvicorn.run(app, host=args.host or cfg.host, port=args.port or cfg.port, workers=1)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
