# LTX-2.5 inference

FastVideo supports BF16 inference for the LTX-2.5 dev and distilled
transformers. Both text-to-video and image-to-video generate synchronized
video and audio. The native path supports sequence parallelism, component
offload, and `torch.compile`.

## Convert the official checkpoint

The gated [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
repository publishes separate transformer, packed Gemma 4, convolutional video
VAE, audio VAE/vocoder, and spatial upscaler files. Accept its license and
download those files, then convert them into one FastVideo model directory:

```bash
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --variant distilled \
  --transformer-source /weights/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-source /weights/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --vae-source /weights/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae-source /weights/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler-source /weights/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --output /models/LTX-2.5-Distilled-Diffusers
```

Use `--variant dev` and the dev transformer to convert the development model.
The converter emits a standard component directory and preserves LTX-2.5's
architecture metadata, packed tokenizer, joint audio components, and refine
configuration. The convolutional video VAE is the default decode path.

Add `--distilled-lora-source loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`
to bundle the distilled LoRA (used by the dev-transformer two-stage distilled
recipe below) as `distilled_lora/` inside the output directory — or merge it
into the transformer offline with `--transformer-lora` (see the deployment
modes under the two-stage recipe below).

### Swapping individual components

Every `--*-source` flag converts independently into `--output`, so a single
component can be replaced without re-converting the rest:

```bash
# Swap the conv video VAE for the diffusion/HQ decoder VAE in place:
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --vae-source /weights/ltx-2.5-video-vae-bf16.safetensors \
  --output /models/LTX-2.5-Distilled-Diffusers

# Swap the dev transformer in for the distilled one:
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --transformer-source /weights/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --variant dev \
  --output /models/LTX-2.5-Distilled-Diffusers
```

Cross-component wiring is handled automatically: a transformer swap also
refreshes the video/audio embeddings connectors that live inside
`text_encoder/model.safetensors` (they ship in the transformer file), and
`model_index.json` is rewritten after every run from the directory's current
contents (e.g. the `vae` class after a conv <-> HQ swap) while preserving
recorded fields such as the variant when they are not re-specified. A
text-encoder-only conversion needs a previously converted
`transformer/config.json` in the output directory (or `--transformer-source`
in the same run) to derive its config.

Only the official `-bf16` files are supported. The quantized variants
(`-comfy-int8-convrot`, `-nvfp4`) are rejected with a "quantized source not
supported" error — quantized deployment happens after conversion, not through
it.

## High-quality diffusion video decoder (DiffVAE)

LTX-2.5 also ships a diffusion-based video decoder
(`vae/ltx-2.5-video-vae-bf16.safetensors`): a neighborhood-attention
transformer that denoises pixels conditioned on the latent, trading decode
time for noticeably sharper detail than the convolutional decoder. Both
decoders share the same encoder and latent space, so they are interchangeable
per run.

To use it, pass the diffusion VAE file as `--vae-source` during conversion.
The converter detects the decoder flavor from the checkpoint's safetensors
metadata (`config.vae._class_name`) and writes a `vae/` directory whose
`config.json` declares `CausalDiffusionVAE`; at load time FastVideo
instantiates the matching decoder automatically. Convert twice (once per VAE
file) to keep a conv and an HQ model directory side by side — whichever `vae/`
the model directory carries decides the decode path.

Notes:

- [`natten`](https://natten.org) is a required dependency on Linux
  (`natten>=0.21.7` in `pyproject.toml`) and the only supported HQ decode
  backend on CUDA — a missing natten raises an ImportError instead of
  silently falling back. NATTEN auto-selects its fastest kernel per GPU,
  including the CUTLASS Blackwell sm100 FNA backend on B200/B300 (CUDA >=
  12.8 builds). Prefer the prebuilt libnatten wheel matching your torch/CUDA
  build over the PyPI sdist, e.g. for torch 2.12.0 + cu130:
  `pip install natten==0.21.7+torch2120cu130 -f https://whl.natten.org`.
  The Triton port and the pure-PyTorch tiled-SDPA path remain as explicit
  opt-ins (`FASTVIDEO_LTX2_NA_BACKEND=triton|eager`) and as the automatic
  CPU path — correct but slow.
- The HQ decode costs roughly 2-3x the convolutional decode.
- Decode noise is seeded from the request seed, so results are reproducible
  per seed.
- Enable VAE tiling (`vae_tiling`) at 720p and above: stages 1-4 of the
  decoder run once, and the memory-dominant final stage and diffusion blocks
  run per overlapping tile with blended seams.

## Generate video and audio

Run distilled text-to-audio-video across four GPUs with sequence parallelism:

```bash
python examples/inference/basic/basic_ltx2_5_t2av.py \
  --model-path /models/LTX-2.5-Distilled-Diffusers \
  --prompt "A jazz trio performs in a candlelit club, synchronized live sound" \
  --num-gpus 4 \
  --torch-compile
```

Condition on a first frame with the I2AV example:

```bash
python examples/inference/basic/basic_ltx2_5_i2av.py \
  --model-path /models/LTX-2.5-Distilled-Diffusers \
  --image /images/first-frame.png \
  --prompt "The train pulls away from the platform as its horn sounds" \
  --num-gpus 4 \
  --torch-compile
```

Pass `--variant dev` with a converted dev directory to use the 30-step dev
guidance preset. Distilled inference uses the official eight-step ancestral
schedule and a three-step spatial refinement pass by default.

## Two-stage distilled i2v / flf2v recipe

The `ltx2_5_distilled_two_stage_i2v` preset plus
`examples/inference/basic/basic_ltx2_5_i2av_two_stage.py` replicate the
production ComfyUI two-stage workflow on the dev transformer + distilled LoRA:

- Stage 1 runs at half resolution (long side ~1024) with the
  `euler_ancestral_cfg_pp` sampler at cfg=1 and LTXVScheduler sigmas
  (steps=8, max_shift=4.0, base_shift=1.5, stretch, terminal=0.1). The
  conditioning image is H.264 re-encoded at CRF 38 (`ltx2_image_crf`) and
  pinned inplace at strength 0.8; the audio latent starts empty and denoises
  jointly. The distilled LoRA is merged at strength 0.7.
- Stage 2 upsamples the latents through LTX-2.5's own x2 spatial latent
  upsampler (do not reuse the 2.3 upscaler — the latent distributions
  differ), re-pins the full-resolution image at strength 1.0, denoises with
  manual sigmas `[0.85, 0.7250, 0.4219, 0.0]` under the same cfg_pp sampler,
  re-merges the distilled LoRA at strength 0.5
  (`ltx2_stage1_lora_strength` / `ltx2_refine_lora_strength`), and carries
  the stage-1 audio latents through.

```bash
python examples/inference/basic/basic_ltx2_5_i2av_two_stage.py \
  --model-path /models/LTX-2.5-Dev-Diffusers \
  --first-frame /images/first.png \
  --prompt "The camera pushes in as the scene comes alive with sound"
```

### Deployment modes for the distilled LoRA

The distilled LoRA can be applied two ways; both use the same converter and
example script.

**Production — offline merge (`--transformer-lora`).** Merge the LoRA into the
transformer once at conversion time; ONE merged transformer then serves BOTH
stages with zero runtime LoRA cost (no per-run unmerge/re-merge weight sweeps,
no adapter bookkeeping):

```bash
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --variant dev \
  --transformer-source /weights/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --transformer-lora /weights/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors:0.7 \
  --text-encoder-source /weights/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --vae-source /weights/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae-source /weights/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler-source /weights/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --output /models/LTX-2.5-Dev-Merged-Diffusers
```

`--transformer-lora PATH[:STRENGTH]` is repeatable (chain order, strength
defaults to 1.0) and merges with the ComfyUI/official semantics
`W += strength * (alpha/rank) * (B @ A)` — fp32 accumulation, bf16 output;
`--device cuda` accelerates the merge GEMMs. The converted `model_index.json`
records the merge as `fastvideo_transformer_merged_loras`, implies
`fastvideo_refine_enabled`, and never emits `fastvideo_refine_lora_path`, so
the runtime cannot double-apply an adapter on top of pre-merged weights. The
example script auto-detects such directories and runs both stages with no
runtime LoRA (force with `--pre-merged`).

Note this deviates from the reference workflow's per-stage strengths (0.7 for
stage 1, 0.5 for stage 2): a single shared strength is a deliberate
simplification for deployment. The merge strength is a quality-tuning choice —
A/B test around it (e.g. 0.6 vs 0.7) rather than treating 0.7 as canonical.

**Experimental — runtime per-stage strengths.** Keep the LoRA separate
(`--distilled-lora-source`, or `--distilled-lora` at run time) and let the
pipeline re-merge it per stage: `ltx2_stage1_lora_strength` (~0.7) and
`ltx2_refine_lora_strength` (~0.5), exposed by the example as
`--stage1-lora-strength` / `--refine-lora-strength`. Each strength switch is an
exact unmerge-to-pristine + re-merge (no drift), costing one weight sweep per
stage per run — the right tool for strength sweeps and recipe experiments, not
for serving. Both strength parameters default to no-ops when unset: with no
refine LoRA wired, the pipeline builds no LoRA stages at all.

Pass `--last-frame /images/last.png` for first+last-frame conditioning
(flf2v): the last image is pinned inplace at the final latent frame
(strength 0.8, stage 1 only), exactly like the validated LTX-2.3 path.
LTX-2.5's transformer additionally supports appended-keyframe conditioning
(`keyframes_mask` with `use_keyframes_abs_pos_embedding`, where keyframes are
extra tokens with learned absolute-position markers); FastVideo currently
marks only the first causal latent frame in that mask, so end-frame anchoring
via keyframe tokens is a possible future alternative to the inplace pin.

CFG++ notes: `euler_ancestral_cfg_pp` needs negative prompt embeddings for
its unconditional pass even at cfg=1 (an empty `negative_prompt` works). At
the schedule's sigma=1.0 first step the uncond output is mathematically
discarded (ComfyUI's own degenerate-limit behavior), so FastVideo skips that
one forward pass.

## Current scope

The initial inference path includes the native transformer, packed Gemma 4
text stack, convolutional and diffusion (DiffVAE/NATTEN) video decoders, audio
VAE/vocoder, dev guidance, and the distilled ancestral sampler. The DiffVAE
port covers the combined pathway with eager/Triton fallbacks; torch.compile
for the decoder, the chunked/Blackwell-DSL DiffVAE modes, and the
memory-budget auto-tiling recommendation are follow-up work, as are generated
keyframes, temporal upsampling, automatic duration selection, HDR, training,
fine-tuning, and quantized deployment.
