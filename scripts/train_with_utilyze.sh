#!/usr/bin/env bash
# Run training with utilyze sidecar to capture true MFU + memory bandwidth utilization.
#
# Prereqs (run once on the box):
#   curl -sSfL https://systalyze.com/utilyze/install.sh | sh
#   # then ensure passwordless sudo for utlz, e.g.:
#   #   echo "$USER ALL=(root) NOPASSWD: $(command -v utlz)" | sudo tee /etc/sudoers.d/utlz
#
# Usage:
#   RUN_NAME=my_run scripts/train_with_utilyze.sh
#   RUN_NAME=my_run UTLZ_DEVICES=0,1 scripts/train_with_utilyze.sh
#
# Output:
#   ./profiles/<RUN_NAME>_utlz.log -- per-second SOL metrics from utlz
#   stdout                         -- training output

set -euo pipefail

RUN_NAME="${RUN_NAME:?RUN_NAME required}"
PROFILE_DIR="${PROFILE_DIR:-./profiles}"
mkdir -p "$PROFILE_DIR"
UTLZ_LOG="$PROFILE_DIR/${RUN_NAME}_utlz.log"

if ! command -v utlz >/dev/null 2>&1; then
    echo "utlz not found on PATH. Install with:"
    echo "  curl -sSfL https://systalyze.com/utilyze/install.sh | sh"
    exit 1
fi

UTLZ_ARGS=(--log "$UTLZ_LOG")
if [[ -n "${UTLZ_DEVICES:-}" ]]; then
    UTLZ_ARGS+=(--devices "$UTLZ_DEVICES")
fi

echo "Starting utlz sidecar -> $UTLZ_LOG"
sudo -b utlz "${UTLZ_ARGS[@]}" >/dev/null 2>&1 || true
sleep 1
UTLZ_PID=$(pgrep -nx utlz || true)
if [[ -z "$UTLZ_PID" ]]; then
    echo "utlz did not start (check sudoers / install)"
    exit 1
fi
echo "utlz pid: $UTLZ_PID"

cleanup() {
    if [[ -n "${UTLZ_PID:-}" ]] && kill -0 "$UTLZ_PID" 2>/dev/null; then
        echo "Stopping utlz ($UTLZ_PID)"
        sudo kill "$UTLZ_PID" || true
        wait "$UTLZ_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

RUN_NAME="$RUN_NAME" .venv/bin/python -m ueaj.train.train "$@"

echo
echo "=== utlz log tail ($UTLZ_LOG) ==="
tail -n 20 "$UTLZ_LOG" || true
