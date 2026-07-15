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

Reliability: every request gets an id (X-LTX23-Request-Id) and a JSON
line in <log_dir>/requests.jsonl; failed generations keep their inputs
under <log_dir>/failed/<id>/ for repro. After max_consecutive_failures
generation errors the process exits(1) so the supervisor (docker
--restart / deploy/run_server.sh) replaces a wedged GPU worker.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import random
import secrets
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# fastapi doesn't import torch, so these are safe before
# setup_environment() stages the process env. They must be module-level for
# FastAPI to resolve the endpoint's postponed annotations.
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

# ltx23_engine keeps all torch/fastvideo imports function-local.
from ltx23_engine import (
    GenerationRequest,
    Ltx23ServerConfig,
    create_generator,
    create_s3_client,
    encode_video_h264,
    generate_for_mode,
    load_config,
    make_lq_frames,
    match_mode,
    run_warmup,
    setup_environment,
    upload_file_to_s3,
)

_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Indirection so tests can intercept the supervisor-restart exit.
_terminate = lambda: os._exit(1)  # noqa: E731


def setup_request_logging(cfg: Ltx23ServerConfig) -> logging.Logger:
    """JSON-lines request log: always mirrored to stdout; also rotated
    into <log_dir>/requests.jsonl when log_dir is set."""
    logger = logging.getLogger("ltx23.requests")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()  # idempotent across rebuilds (tests)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("[request] %(message)s"))
    logger.addHandler(stream)
    if cfg.log_dir:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "requests.jsonl",
            maxBytes=64 * 2**20,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    return logger


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


def build_app(generator, cfg: Ltx23ServerConfig, s3_client=None) -> FastAPI:
    app = FastAPI(title="FastVideo LTX-2.3 server", version="1.0")
    request_logger = setup_request_logging(cfg)
    if s3_client is None and cfg.s3 is not None:
        s3_client = create_s3_client(cfg.s3)
    # One GPU pipeline: requests queue on this lock and run strictly
    # serially (sync endpoints run in Starlette's threadpool). CPU H.264
    # encoding happens OUTSIDE the lock in the request's own thread, so
    # request N+1's generation overlaps request N's encode; the semaphore
    # caps simultaneous encodes.
    gpu_lock = threading.Lock()
    encode_semaphore = threading.Semaphore(max(1, cfg.max_concurrent_encodes))
    scratch_root = cfg.output_dir or None
    if scratch_root:
        Path(scratch_root).mkdir(parents=True, exist_ok=True)

    allowed_keys = [k.strip() for k in cfg.api_keys if k.strip()]

    def require_api_key(
        http_request: Request,
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> None:
        """401 unless the request presents a configured key via X-API-Key
        or Authorization: Bearer. No-op when api_keys is empty. /healthz is
        deliberately outside this gate (docker HEALTHCHECK)."""
        if not allowed_keys:
            return
        presented = x_api_key
        if presented is None and authorization and authorization.startswith("Bearer "):
            presented = authorization[len("Bearer "):].strip()
        if presented and any(secrets.compare_digest(presented, key) for key in allowed_keys):
            return
        request_logger.info(
            json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "event": "auth_rejected",
                "client": http_request.client.host if http_request.client else None,
                "path": http_request.url.path,
                "key_presented": presented is not None,
            }))
        raise HTTPException(status_code=401, detail="invalid or missing API key")

    fail_lock = threading.Lock()
    fail_state = {"consecutive": 0}

    def _register_failure() -> int:
        with fail_lock:
            fail_state["consecutive"] += 1
            count = fail_state["consecutive"]
        if cfg.max_consecutive_failures and count >= cfg.max_consecutive_failures:
            request_logger.critical(
                json.dumps({
                    "event": "too_many_consecutive_failures",
                    "count": count,
                    "action": "exiting in 2s so the supervisor restarts the server",
                }))
            # Delay so the in-flight 500 response flushes first.
            threading.Timer(2.0, _terminate).start()
        return count

    def _register_success() -> None:
        with fail_lock:
            fail_state["consecutive"] = 0

    def _preserve_failed_inputs(workdir: Path, record: dict, tb: str) -> str | None:
        """Keep a failed request's inputs + params for repro (log_dir set)."""
        if not cfg.log_dir:
            shutil.rmtree(workdir, ignore_errors=True)
            return None
        failed_dir = Path(cfg.log_dir) / "failed" / record["request_id"]
        try:
            failed_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(workdir), str(failed_dir))
            (failed_dir / "request.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n" + tb)
            return str(failed_dir)
        except OSError:
            shutil.rmtree(workdir, ignore_errors=True)
            return None

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "busy": gpu_lock.locked(),
            "consecutive_failures": fail_state["consecutive"],
        }

    @app.get("/v1/modes", dependencies=[Depends(require_api_key)])
    def modes() -> dict:
        return {
            "modes": [{
                "width": m.width,
                "height": m.height,
                "num_frames": m.num_frames,
                "fps": m.fps,
            } for m in cfg.modes]
        }

    def _run_generation(
        http_request: Request,
        *,
        endpoint: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        first_frame: UploadFile,
        last_frame: UploadFile | None,
        negative_prompt: str | None,
        seed: int | None,
        last_frame_strength: float,
        last_in_upscale: bool,
        image_crf: float,
        image_crf_stage2: float,
        video_bitrate_kbps: int,
    ) -> dict:
        """Shared validate -> mode-match -> save-uploads -> GENERATE (GPU
        lock) flow for both endpoints. Returns a context dict; raises an
        already-logged HTTPException on any failure."""
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        record: dict = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": request_id,
            "endpoint": endpoint,
            "client": http_request.client.host if http_request.client else None,
            "params": {
                "prompt": prompt,
                "negative_prompt_set": negative_prompt is not None,
                "requested": [width, height, num_frames, fps],
                "has_last_frame": bool(last_frame is not None and last_frame.filename),
                "last_frame_strength": last_frame_strength,
                "last_in_upscale": last_in_upscale,
                "image_crf": image_crf,
                "image_crf_stage2": image_crf_stage2,
                "video_bitrate_kbps": video_bitrate_kbps,
            },
        }
        id_header = {"X-LTX23-Request-Id": request_id}

        def finish(status: int, **extra) -> None:
            record.update(status=status, wall_seconds=round(time.perf_counter() - t0, 2), **extra)
            request_logger.info(json.dumps(record, ensure_ascii=False))

        def bail(status: int, detail: str) -> HTTPException:
            finish(status, error=detail)
            return HTTPException(status_code=status, detail=detail, headers=id_header)

        if not prompt.strip():
            raise bail(400, "prompt must not be empty")
        if width <= 0 or height <= 0 or num_frames <= 0 or fps <= 0:
            raise bail(400, "width/height/num_frames/fps must be positive")
        if not 0.0 <= last_frame_strength <= 1.0:
            raise bail(400, "last_frame_strength must be in [0, 1]")
        if not 100 <= video_bitrate_kbps <= 50000:
            raise bail(400, "video_bitrate_kbps must be in [100, 50000]")

        mode, exact = match_mode(cfg.modes, width, height, num_frames, fps)
        req_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        record["params"]["seed"] = req_seed
        record["params"]["served"] = [mode.width, mode.height, mode.num_frames, mode.fps]
        record["params"]["exact_match"] = exact

        workdir = Path(tempfile.mkdtemp(prefix="ltx23_req_", dir=scratch_root))
        try:
            first_path = _save_upload(first_frame, workdir, "first")
            last_path = (_save_upload(last_frame, workdir, "last")
                         if last_frame is not None and last_frame.filename else None)
        except ValueError as err:
            shutil.rmtree(workdir, ignore_errors=True)
            raise bail(400, str(err)) from err

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

        # Generation holds the GPU lock; encoding runs after it is released
        # so the next queued request starts generating while this thread
        # CPU-encodes. Only GENERATION failures feed the wedged-GPU
        # self-exit counter — encode/upload failures are host-side.
        try:
            with gpu_lock:
                result = generate_for_mode(generator, cfg, mode, request, workdir / "output.mp4")
        except Exception as err:  # noqa: BLE001
            tb = traceback.format_exc()
            failed_dir = _preserve_failed_inputs(workdir, record, tb)
            count = _register_failure()
            finish(500, error=str(err), traceback=tb, failed_inputs=failed_dir,
                   consecutive_failures=count)
            raise HTTPException(
                status_code=500,
                detail=f"generation failed: {err} (request_id={request_id})",
                headers=id_header,
            ) from err
        _register_success()

        return {
            "request_id": request_id,
            "record": record,
            "finish": finish,
            "id_header": id_header,
            "mode": mode,
            "exact": exact,
            "workdir": workdir,
            "result": result,
            "req_seed": req_seed,
            "video_bitrate_kbps": video_bitrate_kbps,
        }

    @app.post("/v1/generate", dependencies=[Depends(require_api_key)])
    def generate(
        http_request: Request,
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
        # Average bitrate of the CPU H.264 (main profile, VBR) encode.
        video_bitrate_kbps: int = Form(cfg.video_bitrate_kbps),
    ):
        ctx = _run_generation(
            http_request,
            endpoint="generate",
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            first_frame=first_frame,
            last_frame=last_frame,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame_strength=last_frame_strength,
            last_in_upscale=last_in_upscale,
            image_crf=image_crf,
            image_crf_stage2=image_crf_stage2,
            video_bitrate_kbps=video_bitrate_kbps,
        )
        mode, exact, workdir, result = ctx["mode"], ctx["exact"], ctx["workdir"], ctx["result"]
        record, finish, id_header = ctx["record"], ctx["finish"], ctx["id_header"]
        request_id, req_seed = ctx["request_id"], ctx["req_seed"]

        video_path = workdir / "output.mp4"
        try:
            with encode_semaphore:
                encode_seconds = encode_video_h264(
                    result["frames"],
                    mode.fps,
                    video_path,
                    bitrate_kbps=video_bitrate_kbps,
                    preset=cfg.x264_preset,
                    audio=result.get("audio"),
                    audio_sample_rate=result.get("audio_sample_rate"),
                    threads=cfg.encode_threads,
                )
            if not video_path.is_file():
                raise RuntimeError("encode produced no video file")
        except Exception as err:  # noqa: BLE001
            tb = traceback.format_exc()
            failed_dir = _preserve_failed_inputs(workdir, record, tb)
            finish(500, error=f"encode failed: {err}", traceback=tb, failed_inputs=failed_dir,
                   gen_seconds=round(result["gen_seconds"], 2))
            raise HTTPException(
                status_code=500,
                detail=f"video encoding failed: {err} (request_id={request_id})",
                headers=id_header,
            ) from err

        finish(200, gen_seconds=round(result["gen_seconds"], 2),
               encode_seconds=round(encode_seconds, 2))
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
                "X-LTX23-Generate-Seconds": f"{result['gen_seconds']:.2f}",
                "X-LTX23-Encode-Seconds": f"{encode_seconds:.2f}",
                **id_header,
            },
            background=BackgroundTask(shutil.rmtree, str(workdir), ignore_errors=True),
        )

    @app.post("/v1/generate_s3", dependencies=[Depends(require_api_key)])
    def generate_s3(
        http_request: Request,
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
        last_in_upscale: bool = Form(True),
        image_crf: float = Form(cfg.image_crf),
        image_crf_stage2: float = Form(cfg.image_crf_stage2),
        video_bitrate_kbps: int = Form(cfg.video_bitrate_kbps),
        # false = HQ only: no LQ encode/upload, and no "lq" field in the
        # response JSON.
        generate_lq: bool = Form(True),
    ):
        """Same generation as /v1/generate, but uploads to S3 and returns
        JSON. By default produces TWO variants — the HQ mp4 plus a
        half-resolution gaussian-blurred LQ mp4 (constrained baseline @
        lq_bitrate_kbps, preset fast, AAC-LC mono 64 kbps) — encoded in
        parallel; generate_lq=false skips the LQ variant."""
        if s3_client is None or cfg.s3 is None:
            raise HTTPException(status_code=503,
                                detail="S3 is not configured — set the 's3' section in the server config")
        ctx = _run_generation(
            http_request,
            endpoint="generate_s3",
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            first_frame=first_frame,
            last_frame=last_frame,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame_strength=last_frame_strength,
            last_in_upscale=last_in_upscale,
            image_crf=image_crf,
            image_crf_stage2=image_crf_stage2,
            video_bitrate_kbps=video_bitrate_kbps,
        )
        mode, exact, workdir, result = ctx["mode"], ctx["exact"], ctx["workdir"], ctx["result"]
        record, finish, id_header = ctx["record"], ctx["finish"], ctx["id_header"]
        request_id, req_seed = ctx["request_id"], ctx["req_seed"]
        record["params"]["generate_lq"] = generate_lq

        audio = result.get("audio")
        audio_sr = result.get("audio_sample_rate")
        key_base = "/".join(
            part for part in (cfg.s3.prefix, datetime.now(timezone.utc).strftime("%Y%m%d"), request_id) if part)

        def _encode_upload_hq() -> dict:
            path = workdir / "hq.mp4"
            enc = encode_video_h264(
                result["frames"],
                mode.fps,
                path,
                bitrate_kbps=ctx["video_bitrate_kbps"],
                preset=cfg.x264_preset,
                audio=audio,
                audio_sample_rate=audio_sr,
                threads=cfg.encode_threads,
            )
            key = f"{key_base}/hq.mp4"
            t_up = time.perf_counter()
            url = upload_file_to_s3(s3_client, cfg.s3, path, key)
            return {
                "url": url,
                "s3_key": key,
                "width": mode.width,
                "height": mode.height,
                "video_bitrate_kbps": ctx["video_bitrate_kbps"],
                "encode_seconds": round(enc, 2),
                "upload_seconds": round(time.perf_counter() - t_up, 2),
            }

        def _encode_upload_lq(lq_frames: list) -> dict:
            path = workdir / "lq.mp4"
            enc = encode_video_h264(
                lq_frames,
                mode.fps,
                path,
                bitrate_kbps=cfg.lq_bitrate_kbps,
                preset=cfg.lq_x264_preset,
                profile="baseline",  # x264 baseline == constrained baseline
                audio=audio,
                audio_sample_rate=audio_sr,
                audio_bitrate_kbps=64,
                audio_mono=True,
                threads=cfg.encode_threads,
            )
            key = f"{key_base}/lq.mp4"
            t_up = time.perf_counter()
            url = upload_file_to_s3(s3_client, cfg.s3, path, key)
            return {
                "url": url,
                "s3_key": key,
                "width": mode.width // 2,
                "height": mode.height // 2,
                "video_bitrate_kbps": cfg.lq_bitrate_kbps,
                "blur_radius": cfg.lq_blur_radius,
                "encode_seconds": round(enc, 2),
                "upload_seconds": round(time.perf_counter() - t_up, 2),
            }

        try:
            lq_info = None
            if generate_lq:
                # GPU half-res + gaussian blur (outside the GPU lock —
                # trivial next to a DiT step), then both encodes + uploads
                # in parallel.
                lq_frames = make_lq_frames(result["frames"], cfg.lq_blur_radius)
                with encode_semaphore:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        hq_future = pool.submit(_encode_upload_hq)
                        lq_future = pool.submit(_encode_upload_lq, lq_frames)
                        hq_info = hq_future.result()
                        lq_info = lq_future.result()
            else:
                with encode_semaphore:
                    hq_info = _encode_upload_hq()
        except Exception as err:  # noqa: BLE001
            tb = traceback.format_exc()
            failed_dir = _preserve_failed_inputs(workdir, record, tb)
            finish(500, error=f"encode/upload failed: {err}", traceback=tb, failed_inputs=failed_dir,
                   gen_seconds=round(result["gen_seconds"], 2))
            raise HTTPException(
                status_code=500,
                detail=f"encode/upload failed: {err} (request_id={request_id})",
                headers=id_header,
            ) from err

        shutil.rmtree(workdir, ignore_errors=True)
        finish(200, gen_seconds=round(result["gen_seconds"], 2), hq=hq_info,
               **({"lq": lq_info} if lq_info is not None else {}))
        payload = {
            "request_id": request_id,
            "seed": req_seed,
            "mode": {
                "width": mode.width,
                "height": mode.height,
                "num_frames": mode.num_frames,
                "fps": mode.fps,
            },
            "exact_match": exact,
            "gen_seconds": round(result["gen_seconds"], 2),
            "hq": hq_info,
        }
        if lq_info is not None:
            payload["lq"] = lq_info
        return JSONResponse(payload, headers=id_header)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the server YAML config")
    parser.add_argument("--host", default=None, help="Override config host")
    parser.add_argument("--port", type=int, default=None, help="Override config port")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip startup warmup (first requests then pay the dynamo trace)")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES for this instance, e.g. '1' (overrides config); "
                        "run one instance per GPU on a multi-GPU box")
    parser.add_argument("--log-dir", default=None,
                        help="Override config log_dir (give each per-GPU instance its own)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # CLI overrides let one config file drive several per-GPU instances.
    if args.gpu is not None:
        cfg.cuda_visible_devices = args.gpu
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    setup_environment(cfg)  # before create_generator imports torch/fastvideo
    if cfg.cuda_visible_devices:
        print(f"[server] CUDA_VISIBLE_DEVICES={cfg.cuda_visible_devices}")

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
