#!/usr/bin/env bash
# Download Wan2.2-TI2V-5B (video DiT, VAE, umT5, tokenizer) and build the
# ActionDiT backbone used by every LoopWAM config:
#   checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
#
# Wan files land under ${DIFFSYNTH_MODEL_BASE_PATH} (default: ./checkpoints).
# Download source defaults to ModelScope, same as the loader. Use Hugging Face with:
#   DIFFSYNTH_DOWNLOAD_SOURCE=huggingface bash scripts/download_pretrained.sh
#
#   bash scripts/download_pretrained.sh
#   DEVICE=cpu bash scripts/download_pretrained.sh
#   FORCE=1 bash scripts/download_pretrained.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=false
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}"

OUTPUT="${DIFFSYNTH_MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [[ -f "${OUTPUT}" && "${FORCE:-0}" != "1" ]]; then
  echo "[download_pretrained] already exists: ${OUTPUT}"
  echo "[download_pretrained] set FORCE=1 to rebuild it."
  exit 0
fi

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output "${OUTPUT}" \
  --device "${DEVICE:-cuda}" \
  --dtype bfloat16

echo "[download_pretrained] wrote ${OUTPUT}"
echo "[download_pretrained] Wan weights are under ${DIFFSYNTH_MODEL_BASE_PATH}"
