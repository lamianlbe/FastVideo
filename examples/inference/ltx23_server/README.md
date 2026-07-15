# LTX-2.3 Production HTTP Server

Compiled-only serving of the validated LTX-2.3 DMD workflow port
(euler_ancestral stage 1 → latent upsample → euler_ancestral_cfg_pp stage 2,
reference-token identity conditioning, NVFP4 linear quantization). Supports
first-frame (i2v) and first+last-frame (FLF) conditioning in one endpoint —
the two share compiled graphs, so switching between them never recompiles.

## Files

| File | Purpose |
|------|---------|
| `config.example.yaml` | Server config: model paths + the supported `(width, height, num_frames, fps)` modes |
| `build_compile_cache.py` | Offline: compile every mode into a persistent inductor cache |
| `server.py` | Online: FastAPI server, warms up all modes at startup, serves mp4 synchronously |
| `ltx23_engine.py` | Shared engine (recipe, generator construction, mode matching, warmup) |
| `deploy/install.sh` | One-shot dependency install (kernel, FA4, flashinfer, server extras) |
| `deploy/Dockerfile` | Fleet image: deps + server baked in, model/cache on a volume |
| `deploy/run_server.sh` | Bare-metal supervisor: restart-on-crash loop with per-run logs |

## Installing dependencies

`deploy/install.sh` is the single source of truth for the dependency chain
(order matters — the in-tree kernel must land before `pip install .` so pip
never falls back to the broken PyPI sdist):

1. `git submodule update --init --recursive` — cutlass/tk for the kernel
2. `pip install -v ./fastvideo-kernel` — in-tree source (or a `WHEELHOUSE` wheel)
3. `pip install .` — fastvideo itself (pulls flashinfer, fastapi, uvicorn)
4. FA4 `flash_attn.cute`, pinned to the cutlass-4.5-compatible revision
   (same pin as `[tool.uv.sources].flash-attn-4` in pyproject.toml)
5. `python-multipart` + an import self-check

Two ways to consume it:

**Bare machine / RunPod pod** — run it inside the target env:

```bash
bash examples/inference/ltx23_server/deploy/install.sh
# fleets: build the slow kernel wheel once, then reuse it everywhere
pip wheel ./fastvideo-kernel -w /workspace/wheels     # once
WHEELHOUSE=/workspace/wheels bash .../deploy/install.sh   # every other pod
```

**Docker image (recommended for fleets)** — SELF-CONTAINED: torch, the
CUDA toolchain, and all python deps are baked in, so the image runs on any
provider. The host only needs an NVIDIA driver (new enough for CUDA 13)
and nvidia-container-toolkit — the driver always comes from the host and
can never be baked into an image. Model weights, config, and the inductor
cache live on the volume:

```bash
# from a fresh clone with submodules initialized
docker build -f examples/inference/ltx23_server/deploy/Dockerfile \
    -t ltx23-server:$(git rev-parse --short HEAD) .
docker run --gpus all -p 8000:8000 -v /workspace:/workspace \
    ltx23-server:<tag>    # serves /workspace/ltx23/config.yaml
```

Defaults you may need to override with `--build-arg`:

- `BASE_IMAGE=pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel` — the public
  devel image matching the validated stack (torch 2.12.0 + cu130, amd64).
  *devel* (nvcc) is required even at runtime: flashinfer JIT-compiles the
  NVFP4 kernels and torch.compile needs a host toolchain. The pyproject
  pins `torch==2.12.0`, which the image already satisfies, so
  `pip install .` leaves the CUDA torch untouched (the install self-check
  fails loudly if that ever regresses).
- `TORCH_CUDA_ARCH_LIST=10.0` — GPU archs for the kernel build (no GPU is
  visible during `docker build`): `10.0` = B200/GB200 (sm100).

The inductor compile cache is keyed on the GPU model + torch/CUDA stack:
switching providers with the same GPU + this same image keeps the cache
valid; changing BASE_IMAGE or the GPU means re-running
`build_compile_cache.py` once on the new fleet.

### Config and cache are volume-side, not image-side

The image never contains a config: the container reads
`/workspace/ltx23/config.yaml` from the volume at startup, so the config
is written/edited AFTER the image is built and applied by restarting the
container — no rebuild. A different path can be passed per container
(`docker run … ltx23-server:<tag> --config /workspace/other.yaml`
replaces the default CMD args).

The image also contains the full source tree, so the compile cache is
built by the image itself — this is the preferred way, since the cache is
keyed on the exact runtime stack and the image IS that stack:

```bash
docker run --gpus all -v /workspace:/workspace --entrypoint python \
    ltx23-server:<tag> build_compile_cache.py --config /workspace/ltx23/config.yaml
```

The cache lands in the config's `inductor_cache_dir` on the volume and is
picked up by every subsequent server container. Full first-deploy
sequence on a new fleet:

1. `docker build …` (no config involved)
2. put `config.yaml` + model weights on the volume
3. run the cache-build container above once, on one machine
4. start server containers everywhere (same image, same volume contents)

Later config edits: change the file on the volume, restart the container.
Only additions to `modes` need step 3 again first.

`FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN` + `FASTVIDEO_FA4=1` are set by the
image, and the server also derives them from the config's
`attention_backend`/`fa4` fields — no manual exports needed either way.

## Deployment flow

```bash
cp config.example.yaml config.yaml   # edit model_path / modes / cache dir

# 1. One-time (per stack): populate the compile cache. Budget tens of
#    minutes per distinct resolution x num_frames shape on a cold cache.
env -u LD_LIBRARY_PATH python build_compile_cache.py --config config.yaml

# 2. Serve. Startup re-traces each shape against the warm cache
#    (~1 generation per shape), then binds the port.
env -u LD_LIBRARY_PATH python server.py --config config.yaml
```

`env -u LD_LIBRARY_PATH` avoids the system-cuBLAS mismatch on RunPod images.

### Multi-GPU: one instance per GPU

Each server drives a single GPU pipeline, so a 2-GPU box runs two
independent instances. Pin each to a GPU and give it its own port and
`log_dir` (the request log's `RotatingFileHandler` is not multi-process
safe). The model weights and the inductor compile cache are read-only
after warm-up, so both instances **share** them — build the cache once.

Two ways: two config files, or one config + CLI overrides:

```bash
# shared: build the compile cache once (on either GPU)
CUDA_VISIBLE_DEVICES=0 env -u LD_LIBRARY_PATH \
    python build_compile_cache.py --config config.yaml

# instance A -> GPU 0, port 8080, its own logs
env -u LD_LIBRARY_PATH python server.py --config config.yaml \
    --gpu 0 --port 8080 --log-dir /workspace/ltx23/logs/gpu0 &
# instance B -> GPU 1, port 8081, its own logs
env -u LD_LIBRARY_PATH python server.py --config config.yaml \
    --gpu 1 --port 8081 --log-dir /workspace/ltx23/logs/gpu1 &
```

Put a load balancer (nginx/HAProxy) in front to spread requests across
8080/8081. `--gpu`/`--log-dir`/`--port` override the config's
`cuda_visible_devices`/`log_dir`/`port`; or bake those into two separate
config files. Keep `num_gpus: 1` — this is data-parallel across GPUs, not
one model split over both.

The inductor cache is shareable across machines with an **identical** stack
(GPU model, driver, torch, fastvideo, quant mode, attention backend). Sync
`inductor_cache_dir` to each machine; only the per-process dynamo trace
(startup warmup) is paid locally. Rebuild the cache after upgrading any
stack component.

## API

**Auth**: when the config's `api_keys` list is non-empty, every `/v1/*`
request must carry a configured key — `X-API-Key: <key>` or
`Authorization: Bearer <key>` — or it gets a 401 (attempts are logged as
`auth_rejected` events). `/healthz` stays open so the docker HEALTHCHECK
works. An empty `api_keys` list disables auth (dev only).

### `POST /v1/generate` (multipart/form-data)

| Field | Required | Default | Notes |
|-------|----------|---------|-------|
| `prompt` | yes | | passed verbatim (no preamble) |
| `first_frame` | yes | | image file; anchors frame 0 and feeds the identity reference tokens |
| `last_frame` | no | | image file; FLF mode, anchors the final latent frame |
| `width`, `height` | yes | | requested resolution (see matching below) |
| `num_frames` | yes | | requested frame count |
| `fps` | yes | | requested frame rate |
| `negative_prompt` | no | workflow default | feeds the cfg_pp uncond pass |
| `seed` | no | random | |
| `last_frame_strength` | no | 0.8 | tail-anchor strength in [0, 1] |
| `last_in_upscale` | no | **true** | whether the tail anchor also enters the stage-2 refine pass |
| `image_crf` | no | 35 | stage-1 conditioning CRF ("motion strength") |
| `image_crf_stage2` | no | **0** | stage-2 re-anchor CRF; 0 keeps the final first frame sharp |
| `video_bitrate_kbps` | no | 3000 | average bitrate of the H.264 main-profile VBR encode |

Response: the mp4 bytes (`video/mp4`), synchronously. Headers report what
was actually served: `X-LTX23-Width/Height/Num-Frames/Fps`,
`X-LTX23-Exact-Match` (`0` when the request was mapped to the closest
mode), `X-LTX23-Seed`, `X-LTX23-E2E-Seconds`.

```bash
curl -sS -X POST http://localhost:8000/v1/generate \
  -H 'X-API-Key: CHANGE-ME-master-key' \
  -F prompt='describe the motion…' \
  -F first_frame=@first.png \
  -F last_frame=@last.png \
  -F width=1344 -F height=768 -F num_frames=241 -F fps=24 \
  -o out.mp4 -D headers.txt
```

### `POST /v1/generate_s3` (multipart/form-data)

Same fields and generation as `/v1/generate`, but produces **two variants**
and uploads both to S3 (requires the config's `s3` section), returning
JSON instead of the mp4. Extra field: `generate_lq` (default **true**) —
`false` skips the LQ variant entirely and the response JSON then has no
`lq` field.

- **hq**: the same H.264 main-profile mp4 `/v1/generate` returns.
- **lq**: half width/height, GPU gaussian blur (`lq_blur_radius` = sigma
  in pixels at the LQ resolution), H.264 **constrained baseline** at
  `lq_bitrate_kbps` (default 1000) preset *fast*, AAC-LC **mono 64 kbps**.

The two encodes + uploads run in parallel (the pair counts as one
`max_concurrent_encodes` unit), overlapping the next request's generation.

```json
{
  "request_id": "…", "seed": 123,
  "mode": {"width": 1344, "height": 768, "num_frames": 241, "fps": 24},
  "exact_match": true, "gen_seconds": 31.2,
  "hq": {"url": "s3://bucket/ltx23/20260714/<id>/hq.mp4",
         "s3_key": "ltx23/20260714/<id>/hq.mp4",
         "width": 1344, "height": 768, "video_bitrate_kbps": 3000,
         "encode_seconds": 18.4, "upload_seconds": 2.1},
  "lq": {"url": "s3://bucket/ltx23/20260714/<id>/lq.mp4",
         "s3_key": "ltx23/20260714/<id>/lq.mp4",
         "width": 672, "height": 384, "video_bitrate_kbps": 1000,
         "blur_radius": 2.0, "encode_seconds": 4.9, "upload_seconds": 0.6}
}
```

URLs are `s3://bucket/key` URIs — downstream is expected to sign/serve
them itself.

### `GET /v1/modes` — the configured combos. `GET /healthz` — liveness.

## Mode matching & image fitting

- Exact `(width, height, num_frames, fps)` match → served as-is.
- Otherwise the **closest-resolution** mode is used (aspect-aware log
  distance; frames/fps only break ties) — required because only configured
  shapes have compiled kernels.
- Conditioning images of any size are accepted: the pipeline cover-fits
  them (aspect-preserving resize + center crop, never letterboxed) to the
  served resolution.

## Concurrency & recompilation

One GPU pipeline; **generation** is strictly serial (a queue forms under
load), but **H.264 encoding runs outside the GPU lock**: as soon as
request N's frames leave the GPU, request N+1 starts generating while
request N's thread encodes. B200 has no NVENC, so encoding is libx264 on
the CPU — but the RGB→YUV420 color conversion (libav's single-threaded
swscale, the real bottleneck) is done on the GPU and yuv420p is piped
straight to one ffmpeg subprocess (main profile, VBR at
`video_bitrate_kbps`, `+faststart`). Each response returns when its
encode finishes; `X-LTX23-Generate-Seconds` / `X-LTX23-Encode-Seconds`
report the split, and `max_concurrent_encodes` caps simultaneous encodes.
Needs an `ffmpeg` binary (system, or the `imageio-ffmpeg` pip bundle that
`deploy/install.sh` installs).

Per-request parameters — prompt, images, seed, CRF values, bitrate,
`last_in_upscale`, FLF vs i2v — are all value-level and never trigger
recompilation. Only the mode list defines compiled shapes, and **fps is
part of the shape**: the audio latent length is derived from the clip
duration (`num_frames / fps`), so the same frame count at a different
frame rate is a different DiT sequence length. To add a mode, add it to
the config, re-run `build_compile_cache.py`, and restart.

## Reliability: auto-restart + request logs

**Request logging** (built in, both environments). Every request gets an
id, returned as `X-LTX23-Request-Id` on every response including errors,
and one JSON line in `<log_dir>/requests.jsonl` (rotating, 64 MB × 10)
recording the client, all parameters, the served mode, seed, latency, and
on failure the error + traceback. Failed generations additionally keep
their uploaded images and a `request.json` under
`<log_dir>/failed/<request_id>/` so any failure can be reproduced offline.
Point `log_dir` at the network volume; `""` logs to stdout only.

**Crash handling.** Two distinct failure shapes:

- *Process dies* (CUDA abort, OOM kill, segfault) → the supervisor
  restarts it.
- *Process alive but GPU wedged* (every generation errors) → no
  supervisor can see that, so the server converts it into the first
  shape: after `max_consecutive_failures` (default 3) consecutive
  generation errors it logs a critical event and exits(1). Successes
  reset the counter; 4xx validation errors don't count.

**Docker (production):** restart policy + the built-in `HEALTHCHECK`
(hits `/healthz`; `start-period` must cover startup warmup):

```bash
docker run -d --name ltx23 --gpus all --restart unless-stopped \
    -p 8000:8000 -v /workspace:/workspace ltx23-server:<tag>
docker logs -f ltx23                       # startup + [request] lines
tail -f /workspace/ltx23/logs/requests.jsonl
```

Note plain docker only *reports* unhealthy — restarts happen because the
process exits (crash or self-exit) under `--restart unless-stopped`.

**Bare metal (debugging):** `deploy/run_server.sh` wraps the server in a
restart loop with backoff, teeing each run to `logs/server-<ts>.log`:

```bash
cd examples/inference/ltx23_server
CONFIG=config.yaml bash deploy/run_server.sh
```

A clean exit (code 0, e.g. Ctrl-C on uvicorn) stops the loop; crashes and
the self-exit (code 1) restart after `BACKOFF` (default 5 s). On hosts
with systemd, an equivalent unit is `Restart=on-failure` +
`ExecStart=... python server.py --config ...` — the self-exit semantics
are the same.
