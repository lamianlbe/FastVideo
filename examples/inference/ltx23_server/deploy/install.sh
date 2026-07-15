#!/usr/bin/env bash
# One-shot dependency install for the LTX-2.3 server on a B200/GB200 machine.
#
#   bash examples/inference/ltx23_server/deploy/install.sh
#
# Run inside the target Python environment (conda / venv / uv venv) of a
# machine that already has the matching torch + CUDA stack. Also used by the
# Dockerfile in this directory — keep it the single source of truth.
#
# uv venvs: activate it (or export PYTHON=$VIRTUAL_ENV/bin/python) and make
# sure pip is present (`uv pip install pip` once). The kernel is built with
# --no-build-isolation so CMake finds this env's Python headers + torch.
# If you hit "Could NOT find Python (missing: Development.Module)" the venv's
# Python has no headers — use a uv-managed Python, or `apt install python3-dev`
# for a system-Python venv.
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
#   DEB_PIP_CONFLICTS       deb python packages to purge before pip installs
#                           (default "python3-blinker"). Ubuntu ships these
#                           as distutils installs that pip cannot uninstall
#                           ("Cannot uninstall blinker"). Purged via apt
#                           when possible, else shadowed with a scoped
#                           pip --ignore-installed.
#   TORCH_BACKEND=cu130     PyTorch wheel variant (download.pytorch.org index).
#                           cu130 is the validated B200 stack; PyPI's default
#                           torch wheel is cu126, which skews against a
#                           CUDA-13 system toolchain (FA4 / flashinfer JIT).
#   TORCH_VERSION=2.12.0    must match pyproject's torch pin.
set -euo pipefail

PYTHON="${PYTHON:-python}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"
WHEELHOUSE="${WHEELHOUSE:-}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"
# cu130 -> "13.0", cu126 -> "12.6"
_cuda_digits="${TORCH_BACKEND#cu}"
EXPECTED_CUDA="${_cuda_digits%?}.${_cuda_digits: -1}"
# FA4 revision must stay in sync with [tool.uv.sources].flash-attn-4 in
# pyproject.toml (cutlass-4.5-compatible pin).
FA4_REV="82d6441eec5d4dfec120153db2c0145ae855a083"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

pip_install() {
    # shellcheck disable=SC2086
    "$PYTHON" -m pip install $PIP_EXTRA_ARGS "$@"
}

echo "== [0/6] torch ${TORCH_VERSION} (${TORCH_BACKEND}) =="
# Install torch from the matching CUDA index BEFORE anything compiles
# against it (the kernel build links against the installed torch). Skips
# when the right build is already present (e.g. the docker base image).
if EXPECTED_CUDA="$EXPECTED_CUDA" TORCH_VERSION="$TORCH_VERSION" "$PYTHON" - <<'EOF'
import os
import sys

try:
    import torch
except Exception:
    sys.exit(1)
version_ok = torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"]
cuda_ok = (torch.version.cuda or "") == os.environ["EXPECTED_CUDA"]
sys.exit(0 if (version_ok and cuda_ok) else 1)
EOF
then
    echo "   torch ${TORCH_VERSION}+${TORCH_BACKEND} already installed; keeping it"
else
    echo "   installing torch ${TORCH_VERSION} (${TORCH_BACKEND}) + torchvision/torchaudio"
    pip_install "torch==${TORCH_VERSION}" torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}"
fi
"$PYTHON" - <<'EOF'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"cuda_available={torch.cuda.is_available()}")
EOF

echo "== [1/6] git submodules (cutlass/tk for the kernel source build) =="
if [ -d .git ]; then
    git submodule update --init --recursive fastvideo-kernel
fi
if [ ! -e fastvideo-kernel/include/cutlass/include ]; then
    echo "ERROR: fastvideo-kernel/include/cutlass is empty. Run" >&2
    echo "  git submodule update --init --recursive" >&2
    echo "in a git checkout (docker builds must COPY an initialized tree)." >&2
    exit 1
fi

echo "== [2/6] fastvideo-kernel =="
# Install BEFORE fastvideo: the in-tree version satisfies pyproject's
# fastvideo-kernel==0.3.2 pin, so pip won't fall back to the PyPI sdist
# (which ships without the cutlass submodule and fails to build).
if [ -n "$WHEELHOUSE" ] && compgen -G "$WHEELHOUSE/fastvideo_kernel-*.whl" > /dev/null; then
    echo "   using prebuilt wheel from $WHEELHOUSE"
    pip_install "$WHEELHOUSE"/fastvideo_kernel-*.whl
else
    echo "   building from in-tree source (slow; consider WHEELHOUSE for fleets)"
    # Preflight: the kernel is a C++/CUDA extension and needs Python dev
    # headers (Python.h). A uv/venv built on a bare system Python without
    # python3-dev has none, and CMake fails cryptically with
    # "missing: Development.Module". Fail fast with the actual fix instead.
    if ! "$PYTHON" -c 'import sysconfig, os, sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path("include"), "Python.h")) else 1)'; then
        _inc="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
        echo "ERROR: Python headers (Python.h) not found under ${_inc}." >&2
        echo "  Debian/Ubuntu system-Python venv:  apt-get install -y python3-dev" >&2
        echo "  or recreate the venv from a uv-managed Python (headers bundled)." >&2
        exit 1
    fi
    # Build deps into THIS env + --no-build-isolation, mirroring the kernel's
    # own build.sh. With pip's default build isolation, scikit-build-core's
    # CMake find_package(Python COMPONENTS Development.Module) fails against
    # the isolated env (and uv-managed standalone Pythons):
    #   "Could NOT find Python (missing: Development.Module)".
    # Without isolation, CMake targets this venv's Python (headers present)
    # and the torch installed in step 0.
    pip_install scikit-build-core cmake ninja setuptools wheel
    # Point CMake's FindPython straight at this interpreter. uv-managed
    # standalone Pythons otherwise trip "Could NOT find Python (missing:
    # Development.Module)" even with --no-build-isolation. Harmless when
    # scikit-build-core already resolves it. If it STILL fails, the venv's
    # Python has no headers (see the header notes at the top of this file).
    _py_exec="$("$PYTHON" -c 'import sys; print(sys.executable)')"
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DPython_EXECUTABLE=${_py_exec}"
    pip_install -v --no-build-isolation ./fastvideo-kernel
fi

echo "== [3/6] purge deb python packages pip cannot upgrade =="
DEB_PIP_CONFLICTS="${DEB_PIP_CONFLICTS:-python3-blinker}"
for deb_pkg in $DEB_PIP_CONFLICTS; do
    if command -v dpkg >/dev/null 2>&1 && dpkg -s "$deb_pkg" >/dev/null 2>&1; then
        echo "   purging $deb_pkg (deb distutils install conflicts with pip)"
        if [ "$(id -u)" -eq 0 ]; then
            apt-get purge -y "$deb_pkg" || true
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get purge -y "$deb_pkg" || true
        fi
        if dpkg -s "$deb_pkg" >/dev/null 2>&1; then
            # No root: shadow the deb files instead (pip installs land
            # earlier on sys.path than /usr/lib/python3/dist-packages).
            pip_name="${deb_pkg#python3-}"
            echo "   purge unavailable; shadowing via pip --ignore-installed $pip_name"
            pip_install --ignore-installed "$pip_name"
        fi
    fi
done

echo "== [4/6] fastvideo (pulls flashinfer-python, fastapi, uvicorn, ...) =="
pip_install .

echo "== [5/6] FA4 (flash_attn.cute) pinned + server extras =="
pip_install "git+https://github.com/Dao-AILab/flash-attention.git@${FA4_REV}#subdirectory=flash_attn/cute"
pip_install python-multipart boto3 imageio-ffmpeg  # multipart + S3 + ffmpeg binary

echo "== [6/6] verify imports =="
EXPECTED_CUDA="$EXPECTED_CUDA" "$PYTHON" - <<'EOF'
import importlib
import os

import torch

# A different CUDA flavor here means a later pip step replaced the torch
# installed in step 0 (e.g. a PyPI cu126 wheel sneaking back in).
expected = os.environ["EXPECTED_CUDA"]
assert torch.version.cuda == expected, \
    f"torch CUDA {torch.version.cuda!r} != expected {expected!r} — replaced during install!"

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
