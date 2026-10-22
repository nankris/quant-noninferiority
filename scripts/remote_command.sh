#!/usr/bin/env bash
# Send a short shell command to only this experiment's runtime.
set -eu
if [[ -n "${COLAB_SETUP:-}" ]]; then source "$COLAB_SETUP"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
{ sleep 5; printf '%s\n' "$@"; sleep 8; } | timeout 55 colab console -s quant-noninferiority > results/remote_latest.console.log 2>&1
"${PYTHON:-python}" scripts/clean_console.py results/remote_latest.console.log
