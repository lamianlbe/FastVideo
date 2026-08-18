# LTX-2.5 Production HTTP Server

Compiled-only serving of the validated LTX-2.5 **two-stage distilled**
recipe: CFG++ ancestral sampling at half resolution → x2 latent upsample →
3-step refine at the final resolution, with joint video+audio. Supports
first-frame (i2v) and first+last-frame (flf2v) conditioning in one endpoint —
the two share compiled graphs, so switching between them never recompiles.

The recipe is a 1:1 port of
[`examples/inference/basic/basic_ltx2_5_i2av_two_stage.py`](../basic/basic_ltx2_5_i2av_two_stage.py)
and the `ltx2_5_distilled_two_stage_i2v` preset; see
[`docs/inference/ltx2_5.md`](../../../docs/inference/ltx2_5.md) for the model
and conversion background. `tests/local_tests/ltx2_5/test_ltx2_5_server.py`
asserts the server's recipe constants never drift from that example.

> **Fork, not a shared library.** This directory is a deliberate,
> self-contained fork of [`../ltx23_server/`](../ltx23_server/). That server
> runs live production traffic and must stay untouched, so the operational
> machinery (auth, readiness gate, request logging, GPU lock, encoding, S3)
> is duplicated here rather than imported. The single exception is
> `deploy/install.sh`, which is shared infrastructure and is **not**
> duplicated — use [`../ltx23_server/deploy/install.sh`](../ltx23_server/deploy/install.sh).

## Files

| File | Purpose |
|------|---------|
| `config.example.yaml` | Server config: model paths + the supported `(width, height, num_frames, fps)` modes |
| `build_compile_cache.py` | Offline: compile every mode into a persistent inductor cache |
| `server.py` | Online: FastAPI server, warms up all modes at startup, serves mp4 synchronously |
| `ltx25_engine.py` | Shared engine (recipe, generator construction, mode matching, encoding, warmup) |
| `deploy/Dockerfile` | Fleet image: deps + server baked in, model/cache on a volume |
| `deploy/ltx25@.service` + `ltx25.env.example` | systemd template unit — one service per GPU (VMs) |
| `deploy/supervisord.conf` + `runpod_start.sh` | Container process manager for systemd-less hosts (RunPod) |
| `deploy/run_server.sh` | Zero-dependency restart-loop supervisor (quick/debug on systemd-less hosts) |
| *(shared)* `../ltx23_server/deploy/install.sh` | One-shot dependency install — kernel, FA4, flashinfer, **natten**, server extras |

## What differs from the 2.3 server

Operationally: nothing. Every production behaviour — API-key auth, the
readiness gate, request ids + rotating JSONL logs + failed-input capture,
the GPU lock with encoding outside it, bounded concurrent encodes, the
`max_consecutive_failures` self-exit — is identical. The recipe differs:

| | LTX-2.3 server | LTX-2.5 server |
|---|---|---|
| Stage-1 sampler | `euler_ancestral` | `euler_ancestral_cfg_pp` (cfg=1) |
| Stage-2 sampler | `euler_ancestral_cfg_pp` | `euler_ancestral_cfg_pp` (cfg=1) |
| Stage-1 sigmas | one hardcoded 9-step list | **computed per mode** — LTXVScheduler(steps=8, max_shift=4.0, base_shift=1.5, stretch, terminal=0.1) shifted by that mode's stage-1 latent token count |
| Stage-2 sigmas | `[0.92, 0.725, 0.421875, 0.0]` | `[0.85, 0.7250, 0.4219, 0.0]` |
| Mode resolution | the resolution rendered | the **FINAL** size; stage 1 renders at **half** |
| Upsampler | 2.3 x1.5/x2 upscaler | LTX-2.5's **own** x2 latent upsampler (the 2.3 one does not match the 2.5 latent distribution) |
| Conditioning | reference identity tokens + first-frame pin at 1.0 | inplace pin **0.8** (stage 1) / **1.0** (stage 2), no reference tokens |
| Conditioning CRF | 35 stage 1, 0 stage 2 (clean re-anchor) | **38 in both stages** (one `image_crf` field) |
| `last_in_upscale` default | `true` | **`false`** — the validated 2.5 example keeps the tail anchor out of the refine pass |
| Negative prompt | long workflow default | `""` (CFG++ only needs uncond embeddings to exist) |
| Quantization | `nvfp4` | `none` (bf16) — quantized 2.5 deployment is follow-up work |
| Distilled LoRA | n/a | pre-merged (production) or runtime per-stage 0.7/0.5 (experiments) |

Compile recipe is unchanged: inductor, `fullgraph=True`, `mode=default`,
`dynamic=False` — one static graph per shape.

## Model directory: conversion

Convert the official LTX-2.5 sources into one FastVideo directory with
`scripts/checkpoint_conversion/convert_ltx2_weights.py`. **Production wants a
pre-merged transformer** so neither stage pays a runtime LoRA merge:

```bash
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --variant dev \
  --transformer-source /weights/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --transformer-lora /weights/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors:0.7 \
  --text-encoder-source /weights/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --vae-source /weights/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae-source /weights/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler-source /weights/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --output /workspace/LTX-2.5-Dev-Merged-Diffusers
```

The server auto-detects that layout (`_fastvideo_transformer_merged_loras`
present, no `fastvideo_refine_lora_path` — the legacy unprefixed spelling is
accepted too) and runs **both stages with no runtime adapter**, so nothing is
double-applied. The startup log states which mode it picked.

The experimental alternative — `--distilled-lora-source` at conversion, or
`distilled_lora_path` in the config — re-merges the adapter at
`stage1_lora_strength` (0.7) and `refine_lora_strength` (0.5). Each switch is
an exact unmerge + re-merge weight sweep **per stage per request**: a
strength-sweep tool, not a serving configuration. `pre_merged: true` forces
merged behaviour on a directory that still bundles `distilled_lora/`;
combining it with `distilled_lora_path` is rejected at config load.

### Video decoder: conv vs HQ, and the natten requirement

There is **no config field** for the decoder — whichever VAE the converted
`vae/` directory carries decides it, and the server prints which one loaded:

```
[engine] video decoder: convolutional decoder (CausalVideoAutoencoder)
[engine] video decoder: HQ diffusion decoder (CausalDiffusionVAE; needs natten, ~2-3x conv decode cost)
```

- **conv** (`ltx-2.5-video-vae-conv-bf16.safetensors`) — the default, and the
  right choice for throughput-bound serving.
- **HQ diffusion decoder** (`ltx-2.5-video-vae-bf16.safetensors`) — a
  neighborhood-attention transformer that denoises pixels conditioned on the
  latent. Noticeably sharper, at roughly **2-3x the decode time**. Convert a
  second model directory (only `--vae-source` differs) and point a separate
  server at it rather than swapping in place.
- **natten is REQUIRED for the HQ decoder on CUDA** (`natten>=0.21.7`,
  installed by the shared `install.sh`). A missing natten raises an
  ImportError instead of silently falling back. Prefer the prebuilt wheel for
  your torch/CUDA build over the PyPI sdist, e.g. for torch 2.12.0 + cu130:
  `pip install natten==0.21.7+torch2120cu130 -f https://whl.natten.org`.
  `na_backend: triton|eager` in the config forces the slow debug backends.
- Set `vae_tiling: true` with the HQ decoder at 720p and above.

## Installing dependencies

Use the **shared** `../ltx23_server/deploy/install.sh` — it is the single
source of truth for the dependency chain (in-tree kernel before
`pip install .`, FA4 pinned to the cutlass-4.5-compatible revision,
flashinfer, natten, `python-multipart`, and an import self-check):

```bash
bash examples/inference/ltx23_server/deploy/install.sh
# fleets: build the slow kernel wheel once, then reuse it everywhere
pip wheel ./fastvideo-kernel -w /workspace/wheels                        # once
WHEELHOUSE=/workspace/wheels bash examples/inference/ltx23_server/deploy/install.sh
```

**Docker image (recommended for fleets)** — `deploy/Dockerfile` is
self-contained (torch, CUDA toolchain, every python dep baked in; the host
only supplies the NVIDIA driver + nvidia-container-toolkit). It calls the
shared install.sh during the build:

```bash
docker build -f examples/inference/ltx25_server/deploy/Dockerfile \
    -t ltx25-server:$(git rev-parse --short HEAD) .
docker run --gpus all -p 8000:8000 -v /workspace:/workspace \
    ltx25-server:<tag>    # serves /workspace/ltx25/config.yaml
```

`--build-arg BASE_IMAGE=…` / `TORCH_CUDA_ARCH_LIST=…` work exactly as in the
[2.3 README](../ltx23_server/README.md#installing-dependencies), which also
covers the image-vs-volume split (config, weights, and the inductor cache
live on the volume; edits apply by restarting the container, no rebuild).

## Deployment flow

```bash
cp config.example.yaml config.yaml   # edit model_path / modes / cache dir

# 1. One-time (per stack): populate the compile cache. Each mode compiles
#    TWO transformer shapes (half-res stage 1 + full-res stage 2), so budget
#    roughly double the 2.3 per-mode time on a cold cache.
env -u LD_LIBRARY_PATH python build_compile_cache.py --config config.yaml

# 2. Serve. Startup re-traces each shape against the warm cache
#    (~1 generation per shape) in the background; /readyz flips to 200 after.
env -u LD_LIBRARY_PATH python server.py --config config.yaml
```

`env -u LD_LIBRARY_PATH` avoids the system-cuBLAS mismatch on RunPod images.

Multi-GPU is one instance per GPU (`--gpu N --port … --log-dir …`, shared
read-only weights + inductor cache, `--shard i/n` to build the cache in
parallel) — identical to the
[2.3 multi-GPU section](../ltx23_server/README.md#multi-gpu-one-instance-per-gpu),
with `ltx23` → `ltx25` in the paths.

## Configuration

Full annotated reference: `config.example.yaml`. The fields that are specific
to this server:

| Key | Default | Meaning |
|-----|---------|---------|
| `model_path` | — | Converted LTX-2.5 directory (pre-merged transformer for production) |
| `upsampler_path` | `""` | `""` auto-detects `<model>/spatial_upsampler` (2.5's own x2 upsampler) |
| `distilled_lora_path` | `""` | Runtime-LoRA experiments only; contradicts `pre_merged` |
| `pre_merged` | `false` | Force "no runtime LoRA"; auto-detected from `model_index.json` |
| `quant` | `none` | bf16. `nvfp4`/`fp8`/`fp8_channel` exist but are unvalidated on 2.5 |
| `na_backend` | `""` | HQ decoder only: `""` = natten (required), `triton`/`eager` = slow debug |
| `vae_tiling` | `false` | Recommended `true` with the HQ decoder at 720p+ |
| `modes` | — | **FINAL** `(width, height, num_frames, fps)`; both dims must divide by 64, `num_frames = 8k+1` |
| `stage1_steps` / `stage1_max_shift` / `stage1_base_shift` / `stage1_stretch` / `stage1_terminal` | 8 / 4.0 / 1.5 / true / 0.1 | LTXVScheduler parameters for the per-mode stage-1 schedule |
| `sigmas_token_anchor` | `false` | `true` uses the scheduler node's detached `tokens=4096` anchor instead of the mode's real token count |
| `stage2_sigmas` | `[0.85, 0.7250, 0.4219, 0.0]` | Manual refine schedule |
| `image_crf` | `38.0` | LTXVPreprocess `img_compression`; **both** stages |
| `first_frame_strength_stage1` / `_stage2` | 0.8 / 1.0 | Inplace conditioning strengths |
| `last_frame_strength` | `0.8` | flf2v tail anchor (stage 1; stage 2 only if `last_in_upscale`) |
| `stage1_lora_strength` / `refine_lora_strength` | 0.7 / 0.5 | Runtime-LoRA mode only |

Everything else (`api_keys`, `log_dir`, `max_consecutive_failures`,
`inductor_cache_dir`, `output_dir`, encoder settings, `lq_*`, `s3`,
`cuda_visible_devices`, `attention_backend`, `fa4*`) is identical in name and
meaning to the 2.3 server.

**A mode's `width`/`height` is the FINAL size.** Stage 1 denoises at half of
it, so `2048x1152` runs stage 1 at `1024x576`. Both dims must be multiples of
64 (the halves must still divide by 32) — `load_config` rejects anything else.

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
| `first_frame` | yes | | image file; pinned inplace at latent frame 0 |
| `last_frame` | no | | image file; flf2v, pins the final latent frame |
| `width`, `height` | yes | | requested **final** resolution (see matching below) |
| `num_frames` | yes | | requested frame count |
| `fps` | yes | | requested frame rate |
| `negative_prompt` | no | `""` | feeds the CFG++ uncond pass; empty is valid |
| `seed` | no | random | |
| `last_frame_strength` | no | 0.8 | tail-anchor strength in [0, 1] |
| `last_in_upscale` | no | **false** | whether the tail anchor also enters the stage-2 refine pass |
| `image_crf` | no | 38 | conditioning CRF, [0, 51]; used by **both** stages |
| `video_bitrate_kbps` | no | 3000 | average bitrate of the H.264 main-profile VBR encode |

There is no `image_crf_stage2` field (the 2.3 server has one): the validated
2.5 recipe reuses the stage-1 CRF-38 image when it re-pins in stage 2.

Response: the mp4 bytes (`video/mp4`), synchronously. Headers report what was
actually served: `X-LTX25-Width/Height/Num-Frames/Fps`,
`X-LTX25-Exact-Match` (`0` when the request was mapped to the closest mode),
`X-LTX25-Seed`, `X-LTX25-Generate-Seconds`, `X-LTX25-Encode-Seconds`, and
`X-LTX25-Request-Id` (present on errors too).

```bash
curl -sS -X POST http://localhost:8000/v1/generate \
  -H 'X-API-Key: CHANGE-ME-master-key' \
  -F prompt='the camera pushes in as the scene comes alive with sound' \
  -F first_frame=@first.png \
  -F last_frame=@last.png \
  -F width=2048 -F height=1152 -F num_frames=121 -F fps=24 \
  -o out.mp4 -D headers.txt
```

### `POST /v1/generate_s3` (multipart/form-data)

Same fields and generation as `/v1/generate`, but produces **two variants**
and uploads both to S3 (requires the config's `s3` section), returning JSON
instead of the mp4. Extra field: `generate_lq` (default **true**) — `false`
skips the LQ variant entirely and the response JSON then has no `lq` field.

- **hq**: the same H.264 main-profile mp4 `/v1/generate` returns.
- **lq**: half width/height, GPU gaussian blur (`lq_blur_radius` = sigma in
  pixels at the LQ resolution), H.264 **constrained baseline** at
  `lq_bitrate_kbps` (default 1000), AAC-LC **mono 64 kbps**.

The two encodes + uploads run in parallel (the pair counts as one
`max_concurrent_encodes` unit), overlapping the next request's generation.

```json
{
  "request_id": "…", "seed": 123,
  "mode": {"width": 2048, "height": 1152, "num_frames": 121, "fps": 24},
  "exact_match": true, "gen_seconds": 41.7,
  "hq": {"url": "s3://bucket/sg/9f2c….mp4", "s3_key": "sg/9f2c….mp4",
         "width": 2048, "height": 1152, "video_bitrate_kbps": 3000,
         "encode_seconds": 18.4, "upload_seconds": 2.1},
  "lq": {"url": "s3://bucket/sg/3a71….mp4", "s3_key": "sg/3a71….mp4",
         "width": 1024, "height": 576, "video_bitrate_kbps": 1000,
         "blur_radius": 2.0, "encode_seconds": 4.9, "upload_seconds": 0.6}
}
```

Each output is keyed `<s3.prefix>/<uuid4>.mp4` (prefix `""` = bucket root).
HQ and LQ get independent GUIDs; the response ties them together. URLs are
`s3://bucket/key` URIs; downstream signs/serves them.

### `GET /v1/modes`

The configured combos. Each entry also reports `stage1_width`/`stage1_height`
(half the final size) so clients can see what stage 1 actually denoises.

### `GET /healthz` / `GET /readyz`

The port binds immediately, but warmup (per-shape dynamo trace/compile) runs
in the background. During it, `/v1/generate*` return **503** with
`Retry-After`, so a load balancer sees a live endpoint rather than
connection-refused.

- `GET /healthz` — liveness; 200 even while warming, with `ready`, `busy`,
  and `consecutive_failures` flags.
- `GET /readyz` — readiness; 200 once warm, 503 while warming. Point the load
  balancer and the docker HEALTHCHECK here. A warmup failure exits the
  process (supervisor restarts) rather than lingering at 503.

## Mode matching & image fitting

- Exact `(width, height, num_frames, fps)` match → served as-is.
- Otherwise the **closest-resolution** mode is used (aspect-aware log
  distance; frames/fps only break ties) — required because only configured
  shapes have compiled kernels.
- Conditioning images of any size are accepted: the pipeline cover-fits them
  (aspect-preserving resize + center crop, never letterboxed) to the stage
  resolution.

## Concurrency & recompilation

One GPU pipeline; **generation** is strictly serial (a queue forms under
load), but **H.264 encoding runs outside the GPU lock**: as soon as request
N's frames leave the GPU, request N+1 starts generating while request N's
thread encodes. B200 has no NVENC, so encoding is libx264 on the CPU — but
the RGB→YUV420 color conversion (libav's single-threaded swscale, the real
bottleneck) is done on the GPU and yuv420p is piped straight to one ffmpeg
subprocess (main profile, VBR, `+faststart`). `X-LTX25-Generate-Seconds` /
`X-LTX25-Encode-Seconds` report the split, and `max_concurrent_encodes` caps
simultaneous encodes. Needs an `ffmpeg` binary (system, or the
`imageio-ffmpeg` pip bundle that install.sh installs).

Per-request parameters — prompt, images, seed, CRF, bitrate,
`last_in_upscale`, flf2v vs i2v — are all value-level and never trigger
recompilation. The per-mode stage-1 **sigma schedule** is value-level too: it
is re-applied to the resident engine under the GPU lock before each
generation, so multiple modes coexist without recompiling.

Only the mode list defines compiled shapes, and **fps is part of the shape**:
the audio latent length is derived from the clip duration (`num_frames /
fps`), so the same frame count at a different frame rate is a different DiT
sequence length. To add a mode: add it to the config, re-run
`build_compile_cache.py`, restart.

## Reliability: auto-restart + request logs

Identical to the 2.3 server; the full rationale is in the
[2.3 reliability section](../ltx23_server/README.md#reliability-auto-restart--request-logs).
In short:

- Every request gets an id (`X-LTX25-Request-Id`, on errors too) and one JSON
  line in `<log_dir>/requests.jsonl` (rotating, 64 MB × 10) with the client,
  parameters, served mode, stage-1 size, seed, latency, and on failure the
  error + traceback. Failed generations keep their uploaded images and a
  `request.json` under `<log_dir>/failed/<request_id>/`.
- *Process dies* → the supervisor restarts it. *Process alive but GPU wedged*
  → after `max_consecutive_failures` (default 3) consecutive generation
  errors the server logs a critical event and exits(1). Successes reset the
  counter; 4xx validation errors don't count.

Supervisors, all renamed for 2.5 and otherwise identical to the 2.3 set:

```bash
# Docker
docker run -d --name ltx25 --gpus all --restart unless-stopped \
    -p 8000:8000 -v /workspace:/workspace ltx25-server:<tag>

# systemd (VMs) — one instance per GPU, instance name IS the GPU id
sudo cp deploy/ltx25@.service /etc/systemd/system/
sudo mkdir -p /etc/ltx25
sudo cp deploy/ltx25.env.example /etc/ltx25/0.env   # PORT=8080, LOG_DIR=…/gpu0
sudo systemctl daemon-reload && sudo systemctl enable --now ltx25@0

# RunPod / systemd-less containers — set the pod's Container Start Command to:
bash /opt/FastVideo/examples/inference/ltx25_server/deploy/runpod_start.sh
# then: supervisorctl status | restart ltx25 | tail -f ltx25

# Quick/debug restart loop, no supervisor install
CONFIG=config.yaml bash deploy/run_server.sh
```

RunPod gotchas (network volume at `/workspace`, expose a TCP port instead of
the buffering HTTP proxy, supervisor restarts the process not the host) apply
unchanged — see the
[2.3 RunPod notes](../ltx23_server/README.md#reliability-auto-restart--request-logs).
