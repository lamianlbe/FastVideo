# LTX-2.3 server image for fleet deployment.
#
# Build from the repo root of a FRESH clone with submodules initialized
# (the context is COPY'd wholesale — a dev checkout drags outputs/models in):
#
#   git clone <fork> FastVideo && cd FastVideo
#   git checkout ltx23-ancestral-reference
#   git submodule update --init --recursive
#   docker build -f examples/inference/ltx23_server/deploy/Dockerfile \
#       --build-arg BASE_IMAGE=<image of your validated pod> \
#       -t ltx23-server:$(git rev-parse --short HEAD) .
#
# BASE_IMAGE is deliberately required: the torch.compile cache is keyed on
# the exact GPU/driver/torch/CUDA stack, so the image MUST be built on the
# same base your compile cache was produced with (e.g. the RunPod torch
# image your B200 pods run).
#
# Big mutable state stays on the network volume, not in the image:
#   /workspace/ltx23/config.yaml     server config (modes, paths)
#   /workspace/<model dirs>          merged checkpoints
#   /workspace/inductor_cache        compile cache from build_compile_cache.py

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/FastVideo
COPY . .

RUN bash examples/inference/ltx23_server/deploy/install.sh

# FA4 under the FLASH_ATTN backend + NVFP4 (flashinfer) is the validated
# serving stack. LD_LIBRARY_PATH is cleared to keep pip-installed CUDA libs
# ahead of any system cuBLAS baked into the base image (the docker
# equivalent of `env -u LD_LIBRARY_PATH`).
ENV FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
    FASTVIDEO_FA4=1 \
    LD_LIBRARY_PATH=

WORKDIR /opt/FastVideo/examples/inference/ltx23_server
EXPOSE 8000

# Liveness probe. start-period must cover startup warmup (one dynamo trace
# per mode against a warm inductor cache) — size it to your mode count.
# Note plain docker only REPORTS unhealthy; restarts come from the process
# exiting (crash, or the server's max_consecutive_failures self-exit)
# combined with `docker run --restart unless-stopped`. Adjust the port if
# your config overrides it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5)"

ENTRYPOINT ["python", "server.py"]
CMD ["--config", "/workspace/ltx23/config.yaml"]
