#!/usr/bin/env bash
# Download the RoboCasa365 checkpoint from loop-wam/loopwam:
#   robocasa365/step_112500.pt
#   robocasa365/dataset_stats.json
# into:
#   checkpoints/robocasa365/
#
#   bash scripts/download_robocasa365.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_robocasa365.sh
#   FORCE=1 bash scripts/download_robocasa365.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

REPO_ID="${REPO_ID:-loop-wam/loopwam}"
REMOTE_DIR="${REMOTE_DIR:-robocasa365}"
OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints/robocasa365}"
FORCE="${FORCE:-0}"

WEIGHT_NAME="step_112500.pt"
STATS_NAME="dataset_stats.json"
WEIGHT_OUT="${OUT_DIR}/${WEIGHT_NAME}"
STATS_OUT="${OUT_DIR}/${STATS_NAME}"

mkdir -p "${OUT_DIR}"

if [[ -f "${WEIGHT_OUT}" && -f "${STATS_OUT}" && "${FORCE}" != "1" ]]; then
  echo "[download_robocasa365] already exists:"
  echo "  ${WEIGHT_OUT}"
  echo "  ${STATS_OUT}"
  echo "[download_robocasa365] set FORCE=1 to re-download."
  exit 0
fi

echo "[download_robocasa365] HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
echo "[download_robocasa365] repo=${REPO_ID}  remote=${REMOTE_DIR}/  -> ${OUT_DIR}"

ensure_hf() {
  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
  else
    python -m pip install -q "huggingface_hub[cli]"
    if command -v hf >/dev/null 2>&1; then
      HF_DOWNLOAD=(hf download)
    else
      HF_DOWNLOAD=(huggingface-cli download)
    fi
  fi
}

ensure_hf
"${HF_DOWNLOAD[@]}" "${REPO_ID}" \
  "${REMOTE_DIR}/${WEIGHT_NAME}" \
  "${REMOTE_DIR}/${STATS_NAME}" \
  --repo-type dataset \
  --local-dir "${OUT_DIR}/_hf_tmp"

# Flatten robocasa365/ nesting if the CLI preserved remote paths
if [[ -f "${OUT_DIR}/_hf_tmp/${REMOTE_DIR}/${WEIGHT_NAME}" ]]; then
  cp -f "${OUT_DIR}/_hf_tmp/${REMOTE_DIR}/${WEIGHT_NAME}" "${WEIGHT_OUT}"
  cp -f "${OUT_DIR}/_hf_tmp/${REMOTE_DIR}/${STATS_NAME}" "${STATS_OUT}"
elif [[ -f "${OUT_DIR}/_hf_tmp/${WEIGHT_NAME}" ]]; then
  cp -f "${OUT_DIR}/_hf_tmp/${WEIGHT_NAME}" "${WEIGHT_OUT}"
  cp -f "${OUT_DIR}/_hf_tmp/${STATS_NAME}" "${STATS_OUT}"
else
  echo "[download_robocasa365] downloaded files not found under ${OUT_DIR}/_hf_tmp" >&2
  find "${OUT_DIR}/_hf_tmp" -maxdepth 3 -type f 2>/dev/null | head -40 >&2 || true
  exit 1
fi
rm -rf "${OUT_DIR}/_hf_tmp"

echo "[download_robocasa365] ready:"
echo "  ${WEIGHT_OUT}"
echo "  ${STATS_OUT}"
echo "[download_robocasa365] serve with: bash eval_robocasa.sh serve"
