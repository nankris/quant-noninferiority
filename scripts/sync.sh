#!/usr/bin/env bash
set -eu
if [[ -n "${COLAB_SETUP:-}" ]]; then source "$COLAB_SETUP"; fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Jupyter channels fail on this runtime; its terminal websocket works.
{ sleep 5; printf '%s\n' 'python /content/quant-noninferiority/poll.py'; sleep 10; } | timeout 55 colab console -s quant-noninferiority > results/poll_latest.console.log 2>&1
"${PYTHON:-python}" scripts/clean_console.py results/poll_latest.console.log
timeout 55 colab download -s quant-noninferiority /content/quant-noninferiority/checkpoint.tar.gz results/checkpoint.download.tar.gz
gzip -t results/checkpoint.download.tar.gz
mv results/checkpoint.download.tar.gz results/checkpoint.tar.gz
tar -xzf results/checkpoint.tar.gz
"${PYTHON:-python}" -c 'import json,time; d=json.load(open("results/collection.json")); age=time.time()-d["collected_unix"]; assert -60<age<180, f"Stale checkpoint: {age:.1f} seconds"; print(f"Fresh checkpoint: {age:.1f} seconds old")'
"${PYTHON:-python}" scripts/compare_completed.py
