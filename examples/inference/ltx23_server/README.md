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
| `deploy/ltx23@.service` + `ltx23.env.example` | systemd template unit — one service per GPU (VMs) |
| `deploy/supervisord.conf` + `runpod_start.sh` | Container process manager for systemd-less hosts (RunPod): auto-restart via supervisord |
| `deploy/run_server.sh` | Zero-dependency restart-loop supervisor (quick/debug on systemd-less hosts) |

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
# shared cache, built in parallel across both GPUs (each does half the
# modes; the cache is keyed on GPU *model*, so both instances then reuse
# every entry). Serial single-GPU build also works — just drop --gpu/--shard.
env -u LD_LIBRARY_PATH python build_compile_cache.py --config config.yaml --gpu 0 --shard 0/2 &
env -u LD_LIBRARY_PATH python build_compile_cache.py --config config.yaml --gpu 1 --shard 1/2 &
wait

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
  "hq": {"url": "s3://bucket/sg/9f2c…-hq-guid.mp4",
         "s3_key": "sg/9f2c…-hq-guid.mp4",
         "width": 1344, "height": 768, "video_bitrate_kbps": 3000,
         "encode_seconds": 18.4, "upload_seconds": 2.1},
  "lq": {"url": "s3://bucket/sg/3a71…-lq-guid.mp4",
         "s3_key": "sg/3a71…-lq-guid.mp4",
         "width": 672, "height": 384, "video_bitrate_kbps": 1000,
         "blur_radius": 2.0, "encode_seconds": 4.9, "upload_seconds": 0.6}
}
```

Each output is keyed `<s3.prefix>/<uuid4>.mp4` (prefix `""` = bucket root;
prefix `"sg"` → `sg/<uuid4>.mp4`). HQ and LQ get independent GUIDs — the
response ties them together. URLs are `s3://bucket/key` URIs; downstream
signs/serves them.

### `GET /v1/modes` — the configured combos.

### `GET /healthz` / `GET /readyz`

The port binds immediately, but warmup (per-shape dynamo trace/compile)
runs in the background. During it, `/v1/generate*` return **503** with
`Retry-After`, so a load balancer sees a live endpoint rather than
connection-refused.

- `GET /healthz` — liveness; 200 even while warming, with a `ready` flag.
- `GET /readyz` — readiness; 200 once warm, 503 while warming. Point the
  load balancer and the docker HEALTHCHECK here so traffic only routes
  when the server can actually serve. A warmup failure exits the process
  (supervisor restarts) rather than lingering at 503.

## Mode matching & image fitting

- Exact `(width, height, num_frames, fps)` match → served as-is.
- Otherwise the **closest-resolution** mode is used (aspect-aware log
  distance; frames/fps only break ties) — required because only configured
  shapes have compiled kernels.
- Conditioning images of any size are accepted: the pipeline cover-fits
  them (aspect-preserving resize + center crop, never letterboxed) to the
  served resolution — or, with `guide_resize: comfy_lanczos_stretch`, the
  ComfyUI geometry instead (see below).

## ComfyUI-parity knobs

The reference ComfyUI workflow (`optimized.json`, the trimmed successor of
`ltx2.3_all_in_one_v2`) differs from this server in three places beyond the
sigma schedules. Each difference is a config knob whose **default is the
server's existing behaviour**, so they can be A/B'd one at a time;
`config.example.yaml` ships them as one commented block that can be
uncommented wholesale. None of them change the compiled shapes. (The old
workflow's LatentAnchorAware / TextAttentionAmplifier ports were removed
after A/B showed no visible effect; the optimized workflow drops both
nodes.)

| Knob | Default | ComfyUI-parity value |
|---|---|---|
| `stage1_conditioning` / `stage1_guide_strength` | `inplace_and_reference` | `guide_only` / `0.8` |
| `guide_attention_bias` | `false` | `true` (log-strength self-attn bias) |
| `stage1_cfg_sigma_list` + `stage1_cfg_values_by_sigma` | empty (flat cfg) | the guider node's two lists |
| `guide_resize` / `guide_longer_size` | `cover_crop` | `comfy_lanczos_stretch` / `1536` |

**Stage-1 conditioning.** The workflow's `LTXPlusBatchAddGuide` calls comfy
core's `LTXVAddGuide.append_keyframe`, which *appends* the encoded guide as
extra tokens at the target's frame-0 RoPE positions with
`noise_mask = 1 - strength`; it never writes latent frame 0. This server
does both: it hard-pins frame 0 at strength 1.0 *and* prepends a clean
reference prefix. `guide_only` drops the in-place pin and switches the
prefix to guide semantics (unscaled latent, per-step noise level
`(1 - strength) * sigma`, matching per-token timestep). Stage 2 keeps its
in-place keyframe at 1.0 either way — that already matches
`LTXVImgToVideoInplace`. FLF (`last_frame`) requests keep the tail anchor
in stage 1 unchanged. Comfy additionally applies a `log(strength)` additive
self-attention bias between guide and non-guide tokens; `guide_attention_bias:
true` ports it, at the cost of routing stage-1 attn1 through masked SDPA
instead of FA4 (O(seq²) bias tensor).

**Per-step CFG.** `STGGuiderAdvanced` selects cfg by *sigma lookup* — the
smallest sigma in its own list that is still ≥ the sampler's current sigma
— and its list is the raw `ManualSigmas`, not the eased schedule the
sampler runs. Paste the node's two strings into
`stage1_cfg_sigma_list` / `stage1_cfg_values_by_sigma`; the engine derives
the per-step list for whatever `stage1_sigmas` is configured and prints it
at startup. STG itself stays off — the workflow's `stg_layers_indices` are
all `[9999]`, so its perturbed pass is a numerical no-op. The node's
`cfg_star_rescale: true` is not ported yet; it only affects the cfg>1
steps.

**Guide geometry.** The workflow lanczos-resizes the upload so its *longer*
edge is 1536 (aspect preserved, nothing cropped) and then lets
`LTXVAddGuide.encode` stretch it to the model resolution with a plain
bilinear resize (`common_upscale(..., crop="disabled")`) — a mismatched
aspect is squashed, not cropped. `comfy_lanczos_stretch` reproduces both
steps by pre-resizing to exactly the mode resolution, which makes the
pipeline's own resize + center crop a no-op.

## Weights: two-stage transformers & fp8

The ComfyUI reference runs **different DiTs in the two passes** — the
distilled LoRA is merged at different strengths per stage (stage 1: 0.88
video/other + 0.90 audio/cross; stage 2: 0.58 + 1.00). Serving both passes
from one merged transformer visibly degrades the refine output, so the
repo can carry a second DiT:

```
<model>/
  transformer/          stage-1 merged DiT
  transformer_stage2/   stage-2 merged DiT   <- auto-detected (or set stage2_transformer_path)
  ...
```

Build both from ComfyUI `ModelSave` exports of the two merged models:

```bash
python scripts/checkpoint_conversion/convert_ltx23_transformer.py \
    --dit stage1_00001_.safetensors --repo <model> --component transformer
python scripts/checkpoint_conversion/convert_ltx23_transformer.py \
    --dit stage2_00001_.safetensors --repo <model> --component transformer_stage2 \
    --set-refine-path --check-connectors-against stage1_00001_.safetensors
```

The converter keeps ComfyUI's **fp8-scaled** storage verbatim (fp8 e4m3
payloads + per-tensor scales for the learned mixed-precision subset, bf16
for the rest): ~21 GB per DiT instead of ~38 GB, and the loader runs those
layers quantized with ComfyUI's exact runtime recipe (input scale 1.0 +
saturating cast, bias fused into the fp8 GEMM) — the same numerics that
produced the reference videos. `--dequant` emits plain bf16 instead. With
fp8 repos set `quant: none`; any other `quant` value is ignored per
component with a warning, because a pre-quantized checkpoint dictates its
own format. Audit the result with
`scripts/checkpoint_conversion/verify_ltx23_conversion.py`.

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
recompilation.

The encoder is configurable: `video_codec` (default `libx264`; e.g.
`libx265` if your ffmpeg has it — the H.264 main/baseline profile is
skipped for non-H.264 codecs) and `extra_video_args` (shlex-split, appended
to the video-encoder options for both HQ and LQ), e.g.
`extra_video_args: "-x265-params asm=avx512 -tag:v hvc1"`. Only the mode list defines compiled shapes, and **fps is
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

**VM / bare metal (recommended): systemd.** `deploy/ltx23@.service` is a
template unit — one instance per GPU, the instance name being the GPU id.
`Restart=on-failure` picks up both crashes and the server's own exit(1) on
a wedged GPU; `StartLimitBurst` stops an unrecoverable loop.

```bash
cd examples/inference/ltx23_server/deploy
# edit the venv/repo/config paths in ltx23@.service, then:
sudo cp ltx23@.service /etc/systemd/system/
sudo mkdir -p /etc/ltx23
sudo cp ltx23.env.example /etc/ltx23/0.env   # PORT=8080, LOG_DIR=.../gpu0
sudo cp ltx23.env.example /etc/ltx23/1.env   # edit: PORT=8081, LOG_DIR=.../gpu1
sudo systemctl daemon-reload
sudo systemctl enable --now ltx23@0 ltx23@1  # start both, and on boot
```

Manage: `systemctl status ltx23@0`, `journalctl -u ltx23@0 -f` (warmup +
`[request]` lines), `systemctl restart ltx23@1`. The JSON request log
still lands in each instance's `LOG_DIR/requests.jsonl`.

**RunPod / containers without systemd (recommended: supervisord).** RunPod
runs your container exactly once and will **not** restart a crashed PID 1 —
there is no systemd (`Restart=on-failure`) and you don't control the
`docker run` line (`--restart unless-stopped`). So the *only* way to get
auto-restart is a supervisor **inside** the container. `deploy/supervisord.conf`
is the 1:1 port of `ltx23@.service`: same `on-failure` semantics (exit 0 =
clean, no restart; anything else = crash or the wedged-GPU self-exit → fresh
process), same SIGINT drain, same `startretries=5` give-up. Launch it via
`deploy/runpod_start.sh`, which prepares the log dirs and hands PID 1 to
`supervisord -n`.

Set the pod's **Container Start Command** (overrides the image ENTRYPOINT, so
the docker path above is untouched):

```bash
bash /opt/FastVideo/examples/inference/ltx23_server/deploy/runpod_start.sh
```

Tunable via pod env vars (all default): `CONFIG` `GPU` `PORT` `LOG_DIR`.
Control it like systemctl:

```bash
supervisorctl status              # like systemctl status
supervisorctl restart ltx23
supervisorctl tail -f ltx23       # live stdout ([request] JSON lines)
```

RunPod gotchas that bite this server specifically:

- **Mount a Network Volume at `/workspace`.** The container root fs is wiped
  on pod restart/redeploy; the config, merged checkpoints, and especially the
  `inductor_cache` must live on the volume or every restart re-pays the full
  warmup (the `HEALTHCHECK` `start-period` is 60m for a reason).
- **Expose a TCP port, not the HTTP proxy.** The `*.proxy.runpod.net` proxy
  has a ~100s timeout and buffers responses (built for web UIs); use direct
  TCP port mapping for the synchronous mp4 API.
- **supervisor restarts the process, not the host.** A dead pod host does not
  migrate (only serverless does) — for HA run two pods behind your own LB.

Prefer a persistent **Pod** over Serverless: the 100+GB weights and per-shape
compile make serverless cold starts prohibitive.

**Quick/debug fallback (no supervisor install):** `deploy/run_server.sh` wraps
the server in a bash restart loop with backoff, teeing each run to
`logs/server-<ts>.log` (`CONFIG=config.yaml bash deploy/run_server.sh`). Fine
for a first bring-up; use supervisord above for anything long-running.
