#!/usr/bin/env bash
# Bare-metal / pod supervisor for the LTX-2.5 server: restart-on-crash loop
# with backoff and timestamped log files. For debugging hosts without
# systemd (e.g. RunPod pods); production should use the Dockerfile with
# `docker run --restart unless-stopped`.
#
#   cd examples/inference/ltx25_server
#   CONFIG=config.yaml bash deploy/run_server.sh
#
# Env knobs:
#   CONFIG=config.yaml    server config
#   LOG_DIR=logs          supervisor log files (server-<ts>.log); the JSON
#                         request log location is the config's log_dir
#   BACKOFF=5             seconds between restarts
#   PYTHON=python         interpreter
set -uo pipefail  # deliberately no -e: the loop handles failures

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-config.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
BACKOFF="${BACKOFF:-5}"
PYTHON="${PYTHON:-python}"

cd "$SERVER_DIR"
mkdir -p "$LOG_DIR"

echo "[supervisor] serving $CONFIG; Ctrl-C stops the loop"
while true; do
    ts="$(date +%Y%m%d-%H%M%S)"
    log_file="$LOG_DIR/server-$ts.log"
    echo "[supervisor] starting server (log: $log_file)"
    # env -u LD_LIBRARY_PATH: keep pip-installed CUDA libs ahead of any
    # system cuBLAS (RunPod images).
    env -u LD_LIBRARY_PATH "$PYTHON" server.py --config "$CONFIG" 2>&1 | tee "$log_file"
    code=${PIPESTATUS[0]}
    if [ "$code" -eq 0 ]; then
        echo "[supervisor] server exited cleanly (code 0); not restarting"
        break
    fi
    echo "[supervisor] server crashed (code $code); restarting in ${BACKOFF}s"
    sleep "$BACKOFF"
done
