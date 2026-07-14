#!/usr/bin/env bash
# One-shot dependency install for the LTX-2.3 server on a B200/GB200 machine.
#
#   bash examples/inference/ltx23_server/deploy/install.sh
#
# Run inside the target Python environment (conda/venv) of a machine that
# already has the matching torch + CUDA stack. Also used by the Dockerfile
# in this directory — keep it the single source of truth for install steps.
#
# Env knobs:
#   PYTHON=python           interpreter to install into
#   PIP_EXTRA_ARGS=""       e.g. "--trusted-host pypi.org --trusted-host files.pythonhosted.org"
#   WHEELHOUSE=""           dir with prebuilt wheels (fastvideo_kernel-*.whl);
#                           skips the slow in-tree kernel source build.
#                           Populate once per stack with:
#                             pip wheel ./fastvideo-kernel -w $WHEELHOUSE
#   TORCH_CUDA_ARCH_LIST    REQUIRED on GPU-less hosts (docker build): GPU
#                           archs for the kernel build, e.g. "10.0" for
#                           B200/GB200. With a visible GPU it is probed.
set -euo pipefail

PYTHON="${PYTHON:-python}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"
WHEELHOUSE="${WHEELHOUSE:-}"
# FA4 revision must stay in sync with [tool.uv.sources].flash-attn-4 in
# pyproject.toml (cutlass-4.5-compatible pin).
FA4_REV="82d6441eec5d4dfec120153db2c0145ae855a083"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

pip_install() {
    # shellcheck disable=SC2086
    "$PYTHON" -m pip install $PIP_EXTRA_ARGS "$@"
}

echo "== [0/5] sanity: torch =="
"$PYTHON" - <<'EOF'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"cuda_available={torch.cuda.is_available()}")
EOF

echo "== [1/5] git submodules (cutlass/tk for the kernel source build) =="
if [ -d .git ]; then
    git submodule update --init --recursive fastvideo-kernel
fi
if [ ! -e fastvideo-kernel/include/cutlass/include ]; then
    echo "ERROR: fastvideo-kernel/include/cutlass is empty. Run" >&2
    echo "  git submodule update --init --recursive" >&2
    echo "in a git checkout (docker builds must COPY an initialized tree)." >&2
    exit 1
fi

echo "== [2/5] fastvideo-kernel =="
# Install BEFORE fastvideo: the in-tree version satisfies pyproject's
# fastvideo-kernel==0.3.2 pin, so pip won't fall back to the PyPI sdist
# (which ships without the cutlass submodule and fails to build).
if [ -n "$WHEELHOUSE" ] && compgen -G "$WHEELHOUSE/fastvideo_kernel-*.whl" > /dev/null; then
    echo "   using prebuilt wheel from $WHEELHOUSE"
    pip_install "$WHEELHOUSE"/fastvideo_kernel-*.whl
else
    echo "   building from in-tree source (slow; consider WHEELHOUSE for fleets)"
    pip_install -v ./fastvideo-kernel
fi

echo "== [3/5] fastvideo (pulls flashinfer-python, fastapi, uvicorn, ...) =="
pip_install .

echo "== [4/5] FA4 (flash_attn.cute) pinned + server extras =="
pip_install "git+https://github.com/Dao-AILab/flash-attention.git@${FA4_REV}#subdirectory=flash_attn/cute"
pip_install python-multipart  # FastAPI multipart Form/File parsing

echo "== [5/5] verify imports =="
"$PYTHON" - <<'EOF'
import importlib

import torch

# A CPU-only torch here means pip replaced the base image's CUDA build
# during `pip install .` — the pyproject pin must match the image's torch.
assert torch.version.cuda, "torch lost its CUDA build during install!"

for mod in ("fastvideo", "flashinfer", "fastapi", "yaml", "uvicorn"):
    importlib.import_module(mod)
    print(f"  {mod}: ok")
# python-multipart's import name changed across versions.
assert any(importlib.util.find_spec(m) for m in ("python_multipart", "multipart")), \
    "python-multipart missing"
print("  python-multipart: ok")
if torch.cuda.is_available():
    from flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401
    print("  flash_attn.cute (FA4): ok")
else:
    # flash_attn.cute needs a visible CUDA device to import; on a GPU-less
    # build host just confirm the package landed.
    assert importlib.util.find_spec("flash_attn.cute") is not None
    print("  flash_attn.cute (FA4): installed (import deferred, no CUDA device)")
print("ALL DEPENDENCIES OK")
EOF

cat <<'EOF'

Install complete. Serve with (or bake into the image / config):
  export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
  export FASTVIDEO_FA4=1
  env -u LD_LIBRARY_PATH python server.py --config config.yaml
(the server config's attention_backend/fa4 fields set both vars for you)
EOF
