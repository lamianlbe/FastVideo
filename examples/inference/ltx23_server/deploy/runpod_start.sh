#!/usr/bin/env bash
# RunPod (or any systemd-less container) entrypoint for the LTX-2.3 server.
#
# WHY THIS EXISTS: RunPod runs your container exactly once and will NOT
# restart a crashed PID 1 — there is no systemd and you don't control the
# `docker run` line, so neither Restart=on-failure nor --restart is
# available. This script hands PID 1 to supervisord (see supervisord.conf),
# which restarts the server in-container on crash. It is the RunPod
# equivalent of enabling ltx23@.service on a VM.
#
# Set this as the pod's "Container Start Command" (Docker Command), which
# overrides the image's default ENTRYPOINT so the VM path
# (`python server.py`, restarted by docker) stays untouched:
#
#   bash /opt/FastVideo/examples/inference/ltx23_server/deploy/runpod_start.sh
#
# Tunables (set as pod env vars; all have defaults):
#   CONFIG   server config on the persistent volume  [/workspace/ltx23/config.yaml]
#   GPU      GPU index inside the pod                 [0]
#   PORT     listen port                              [8000]
#   LOG_DIR  supervisor + server logs (persist it!)   [/workspace/ltx23/logs/gpu$GPU]
set -euo pipefail

export CONFIG="${CONFIG:-/workspace/ltx23/config.yaml}"
export GPU="${GPU:-0}"
export PORT="${PORT:-8000}"
export LOG_DIR="${LOG_DIR:-/workspace/ltx23/logs/gpu${GPU}}"

CONF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logs and the config's parent must exist and live on the persistent volume,
# or a pod restart on a new host loses them (and the expensive inductor
# cache). supervisord creates log FILES but not their parent dirs.
mkdir -p "$LOG_DIR"

if [ ! -f "$CONFIG" ]; then
    echo "[runpod_start] ERROR: config not found: $CONFIG" >&2
    echo "[runpod_start] put it on the persistent volume, or set CONFIG=..." >&2
    exit 1
fi

# supervisor may not be baked into the image yet — self-bootstrap so this
# works on a plain RunPod PyTorch base pod too. Bake `pip install supervisor`
# into the Dockerfile to skip this at cold start.
if ! command -v supervisord >/dev/null 2>&1; then
    echo "[runpod_start] supervisor not found; installing"
    python -m pip install --no-cache-dir supervisor
fi

echo "[runpod_start] CONFIG=$CONFIG GPU=$GPU PORT=$PORT LOG_DIR=$LOG_DIR"
echo "[runpod_start] handing PID 1 to supervisord (auto-restart on crash)"
# exec: supervisord becomes PID 1, so it reaps zombies and receives the
# pod's SIGTERM directly (forwarded to the server as SIGINT to drain).
exec supervisord -n -c "$CONF_DIR/supervisord.conf"
