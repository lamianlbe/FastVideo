# LTX-2.3 server image for fleet deployment — SELF-CONTAINED: torch, the
# full CUDA toolchain, and every python dep are baked in, so the image runs
# on any provider's GPU hosts. The only host requirements are an NVIDIA
# driver new enough for CUDA 13 and nvidia-container-toolkit (the driver
# always comes from the host; it cannot be baked into an image).
#
# Build from the repo root of a FRESH clone with submodules initialized
# (the context is COPY'd wholesale — a dev checkout drags outputs/models in):
#
#   git clone <fork> FastVideo && cd FastVideo
#   git checkout ltx23-ancestral-reference
#   git submodule update --init --recursive
#   docker build -f examples/inference/ltx23_server/deploy/Dockerfile \
#       -t ltx23-server:$(git rev-parse --short HEAD) .
#
# The default base is the public pytorch devel image matching the validated
# stack (torch 2.12.0 + cu130, amd64). "devel" (nvcc present) is REQUIRED
# even at runtime: flashinfer JIT-compiles NVFP4 kernels and torch.compile
# needs a host toolchain. The compile cache is keyed on this stack — after
# changing BASE_IMAGE (or the GPU model), re-run build_compile_cache.py.
ARG BASE_IMAGE=pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel
# GPU archs to compile fastvideo-kernel for (no GPU is visible during
# docker build, so this cannot be probed): 10.0 = B200/GB200 (sm100).
# Add e.g. "10.0;12.0" to also cover sm120 consumer Blackwell.
ARG TORCH_CUDA_ARCH_LIST=10.0

FROM ${BASE_IMAGE}
ARG TORCH_CUDA_ARCH_LIST

# git: pip installs FA4 from a pinned git revision.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/FastVideo
COPY . .

RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    bash examples/inference/ltx23_server/deploy/install.sh

# Big mutable state stays on the volume, not in the image:
#   /workspace/ltx23/config.yaml     server config (modes, paths, api keys)
#   /workspace/<model dirs>          merged checkpoints
#   /workspace/inductor_cache        compile cache from build_compile_cache.py
#   /workspace/ltx23/logs            request logs + failed-request inputs

# FA4 under the FLASH_ATTN backend + NVFP4 (flashinfer) is the validated
# serving stack. LD_LIBRARY_PATH is cleared to keep pip-installed CUDA libs
# ahead of any system cuBLAS baked into the base image (the docker
# equivalent of `env -u LD_LIBRARY_PATH`).
ENV FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN \
    FASTVIDEO_FA4=1 \
    LD_LIBRARY_PATH=

WORKDIR /opt/FastVideo/examples/inference/ltx23_server
EXPOSE 8000

# Readiness probe: /readyz is 503 while warming, 200 once warm — so the
# container reads "healthy" only when it can actually serve. start-period
# must cover startup warmup (one dynamo trace per mode against a warm
# inductor cache) — size it to your mode count. Note plain docker only
# REPORTS unhealthy; restarts come from the process exiting (crash, or the
# server's max_consecutive_failures self-exit) combined with
# `docker run --restart unless-stopped`. Adjust the port if your config
# overrides it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=5)"

ENTRYPOINT ["python", "server.py"]
CMD ["--config", "/workspace/ltx23/config.yaml"]
