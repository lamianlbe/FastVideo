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

- Install [`natten`](https://natten.org) for production HQ decode. NATTEN
  auto-selects its fastest kernel per GPU, including the CUTLASS Blackwell
  sm100 FNA backend on B200/B300. Without natten, FastVideo falls back to a
  Triton port (CUDA) or a pure-PyTorch tiled-SDPA path — correct but slow, and
  it logs a warning. `FASTVIDEO_LTX2_NA_BACKEND=natten|triton|eager` forces a
  backend.
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

## Current scope

The initial inference path includes the native transformer, packed Gemma 4
text stack, convolutional and diffusion (DiffVAE/NATTEN) video decoders, audio
VAE/vocoder, dev guidance, and the distilled ancestral sampler. The DiffVAE
port covers the combined pathway with eager/Triton fallbacks; torch.compile
for the decoder, the chunked/Blackwell-DSL DiffVAE modes, and the
memory-budget auto-tiling recommendation are follow-up work, as are generated
keyframes, temporal upsampling, automatic duration selection, HDR, training,
fine-tuning, and quantized deployment.
