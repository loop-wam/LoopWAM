#!/usr/bin/env bash
# Prepare a LoopWAM checkout for local RoboTwin evaluation:
# soft-link assets / cuRobo / policy / shared pretrained weights, then download
# the LoopWAM RoboTwin checkpoint if missing.
#
# Assumes Hugging Face is reachable in the current shell (proxy / mirror as needed).
#
#   bash scripts/prepare_eval.sh
#
# Override the prebuilt RoboTwin tree (must contain assets/ and envs/curobo/):
#   ROBOTWIN_PREBUILT=/path/to/RoboTwin bash scripts/prepare_eval.sh
#
# Override shared pretrained checkpoints (ActionDiT / DiffSynth / Wan-AI):
#   CHECKPOINTS_SRC=/path/to/checkpoints bash scripts/prepare_eval.sh
#
# Or override pieces separately:
#   ASSETS_SRC=/path/to/assets CUROBO_SRC=/path/to/curobo bash scripts/prepare_eval.sh
#
# If CHECKPOINTS_SRC is unset or incomplete, falls back to:
#   bash scripts/download_pretrained.sh
#
# Links only (no hf download of robotwin ckpt):
#   SKIP_DOWNLOAD=1 bash scripts/prepare_eval.sh
#
# Force re-download of robotwin ckpt even if present:
#   FORCE_CKPT_DOWNLOAD=1 bash scripts/prepare_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROBOTWIN_PREBUILT="${ROBOTWIN_PREBUILT:-}"
ASSETS_SRC="${ASSETS_SRC:-${ROBOTWIN_PREBUILT:+${ROBOTWIN_PREBUILT}/assets}}"
CUROBO_SRC="${CUROBO_SRC:-${ROBOTWIN_PREBUILT:+${ROBOTWIN_PREBUILT}/envs/curobo}}"
CHECKPOINTS_SRC="${CHECKPOINTS_SRC:-}"

POLICY_SRC="${ROOT}/experiments/robotwin/fastwam_policy"
POLICY_DST="${ROOT}/third_party/RoboTwin/policy/fastwam_policy"
ASSETS_DST="${ROOT}/third_party/RoboTwin/assets"
CUROBO_DST="${ROOT}/third_party/RoboTwin/envs/curobo"
CKPT_DST="${ROOT}/checkpoints"

# Shared pretrained pieces (eval needs text encoder + ActionDiT).
# Layout matches DIFFSYNTH_MODEL_BASE_PATH=./checkpoints.
SHARED_CKPT_NAMES=(
  ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  DiffSynth-Studio
  Wan-AI
)

link_path() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "${src}" ]]; then
    echo "[prepare_eval] missing path: ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  ln -sfn "$(realpath "${src}")" "${dst}"
  echo "[prepare_eval] ${dst} -> $(realpath "${src}")"
}

echo "[prepare_eval] root: ${ROOT}"

link_path "${POLICY_SRC}" "${POLICY_DST}"

if [[ -z "${ASSETS_SRC}" || -z "${CUROBO_SRC}" ]]; then
  echo "[prepare_eval] set ROBOTWIN_PREBUILT=/path/to/RoboTwin (or ASSETS_SRC + CUROBO_SRC) to soft-link assets/curobo" >&2
  exit 1
fi
link_path "${ASSETS_SRC}" "${ASSETS_DST}"
link_path "${CUROBO_SRC}" "${CUROBO_DST}"

mkdir -p "${CKPT_DST}"
link_shared_ckpts=1
if [[ -z "${CHECKPOINTS_SRC}" ]]; then
  link_shared_ckpts=0
else
  for name in "${SHARED_CKPT_NAMES[@]}"; do
    if [[ ! -e "${CHECKPOINTS_SRC}/${name}" ]]; then
      echo "[prepare_eval] CHECKPOINTS_SRC missing ${name}, will download pretrained instead"
      link_shared_ckpts=0
      break
    fi
  done
fi

if [[ "${link_shared_ckpts}" == "1" ]]; then
  echo "[prepare_eval] linking shared pretrained weights from ${CHECKPOINTS_SRC}"
  for name in "${SHARED_CKPT_NAMES[@]}"; do
    link_path "${CHECKPOINTS_SRC}/${name}" "${CKPT_DST}/${name}"
  done
fi

echo "[prepare_eval] verifying RoboTwin links ..."
ls "${ASSETS_DST}/objects" "${CUROBO_DST}" "${POLICY_DST}" >/dev/null
echo "[prepare_eval] links ok"

if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  echo "[prepare_eval] SKIP_DOWNLOAD=1, skipping downloads"
  exit 0
fi

if [[ "${link_shared_ckpts}" != "1" ]]; then
  echo "[prepare_eval] downloading text encoder / ActionDiT backbone ..."
  bash "${ROOT}/scripts/download_pretrained.sh"
else
  ls "${CKPT_DST}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" >/dev/null
  ls "${CKPT_DST}/DiffSynth-Studio/Wan-Series-Converted-Safetensors" >/dev/null
  ls "${CKPT_DST}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" >/dev/null
fi

ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
    return
  fi
  echo "[prepare_eval] installing huggingface_hub (provides hf / huggingface-cli) ..."
  python -m pip install -U 'huggingface_hub[cli]'
  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
  else
    echo "[prepare_eval] huggingface_hub installed but CLI not on PATH" >&2
    exit 1
  fi
}

CKPT_PT="${CKPT_DST}/robotwin/step_037645.pt"
CKPT_STATS="${CKPT_DST}/robotwin/dataset_stats.json"
if [[ -f "${CKPT_PT}" && -f "${CKPT_STATS}" && "${FORCE_CKPT_DOWNLOAD:-0}" != "1" ]]; then
  echo "[prepare_eval] robotwin checkpoint already present, skip download"
  echo "[prepare_eval] set FORCE_CKPT_DOWNLOAD=1 to re-download"
else
  ensure_hf_cli
  echo "[prepare_eval] downloading LoopWAM RoboTwin checkpoint (~12GB) ..."
  "${HF_DOWNLOAD[@]}" loop-wam/loopwam \
    robotwin/step_037645.pt \
    robotwin/dataset_stats.json \
    --repo-type dataset \
    --local-dir "${CKPT_DST}"
fi

for f in "${CKPT_PT}" "${CKPT_STATS}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[prepare_eval] expected file missing: ${f}" >&2
    exit 1
  fi
done

echo "[prepare_eval] done"
echo "[prepare_eval] robotwin ckpt under checkpoints/robotwin/"
