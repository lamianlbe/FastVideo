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

The inductor cache is shareable across machines with an **identical** stack
(GPU model, driver, torch, fastvideo, quant mode, attention backend). Sync
`inductor_cache_dir` to each machine; only the per-process dynamo trace
(startup warmup) is paid locally. Rebuild the cache after upgrading any
stack component.

## API

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

Response: the mp4 bytes (`video/mp4`), synchronously. Headers report what
was actually served: `X-LTX23-Width/Height/Num-Frames/Fps`,
`X-LTX23-Exact-Match` (`0` when the request was mapped to the closest
mode), `X-LTX23-Seed`, `X-LTX23-E2E-Seconds`.

```bash
curl -sS -X POST http://localhost:8000/v1/generate \
  -F prompt='describe the motion…' \
  -F first_frame=@first.png \
  -F last_frame=@last.png \
  -F width=1344 -F height=768 -F num_frames=241 -F fps=24 \
  -o out.mp4 -D headers.txt
```

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

One GPU pipeline; requests are served strictly serially (a queue forms
under load). Per-request parameters — prompt, images, seed, CRF values,
`last_in_upscale`, FLF vs i2v — are all value-level and never trigger
recompilation. Only the mode list defines compiled shapes; to add a mode,
add it to the config, re-run `build_compile_cache.py`, and restart.
