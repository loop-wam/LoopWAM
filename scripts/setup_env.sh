#!/usr/bin/env bash
# Create the training environment and install this package.
# Matches the Fast-WAM setup: Python 3.10, torch 2.7.1+cu128, then `pip install -e .`.
#
#   bash scripts/setup_env.sh
#
# If conda is available and the current env is not already `loopwam`, this
# creates and activates that env. Otherwise it installs into the active Python.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "loopwam" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx loopwam; then
    conda create -n loopwam python=3.10 -y
  fi
  conda activate loopwam
fi

python -m pip install -U pip
python -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128

echo "[setup_env] done. Active Python: $(command -v python)"
