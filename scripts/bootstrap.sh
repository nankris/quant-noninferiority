#!/usr/bin/env bash
set -eu
mkdir -p /content/quant-noninferiority
cd /content/quant-noninferiority
tar -xzf /content/bundle-quant.tar.gz
cp scripts/*.py scripts/install.sh .
mkdir -p results
nvidia-smi > results/nvidia_smi.txt
date -u +%FT%TZ > results/bootstrap_start.txt
bash install.sh
exec python -u driver.py
