#!/usr/bin/env bash
# RoboCasa365 evaluation entrypoint.
#
# Policy server lives in this repo. The simulator client lives under
# third_party/RoboCasa365 (robocasa + robosuite). Existing scripts under
# experiments/robocasa/ are left unchanged and remain usable.
#
#   bash scripts/download_robocasa365.sh
#   ASSET_SRC_ROOT=... ROBOSUITE_ASSET_SRC=... bash eval_robocasa.sh prepare
#   bash eval_robocasa.sh serve
#   bash eval_robocasa.sh client --server_url http://HOST:7891 --task_groups "..." ...
#
# Optional env for serve:
#   CKPT  STATS  GPU_ID  HOST  PORT  QUALITY

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-${ROOT}/third_party/RoboCasa365}"
MODE="${1:-}"
if [[ -n "${MODE}" ]]; then
  shift
fi

usage() {
  cat <<'EOF'
Usage:
  bash eval_robocasa.sh prepare   # link RoboCasa / robosuite assets
  bash eval_robocasa.sh serve     # start LoopWAM policy HTTP server
  bash eval_robocasa.sh client --server_url http://HOST:7891 [client args...]

prepare requires:
  ASSET_SRC_ROOT=/path/to/robocasa_assets
  ROBOSUITE_ASSET_SRC=/path/to/robosuite/robosuite/models/assets
EOF
}

cmd_prepare() {
  if [[ ! -x "${ROBOCASA365_ROOT}/setup_symlinks.sh" ]]; then
    echo "[eval_robocasa] missing ${ROBOCASA365_ROOT}/setup_symlinks.sh" >&2
    exit 1
  fi
  bash "${ROBOCASA365_ROOT}/setup_symlinks.sh"
}

cmd_serve() {
  export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"
  export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

  CKPT="${CKPT:-${ROOT}/checkpoints/robocasa365/step_112500.pt}"
  STATS="${STATS:-${ROOT}/checkpoints/robocasa365/dataset_stats.json}"
  GPU_ID="${GPU_ID:-0}"
  HOST="${HOST:-0.0.0.0}"
  PORT="${PORT:-7891}"
  QUALITY="${QUALITY:-5}"

  if [[ ! -f "${CKPT}" ]]; then
    echo "[eval_robocasa] missing checkpoint: ${CKPT}" >&2
    echo "  Run: bash scripts/download_robocasa365.sh" >&2
    exit 1
  fi
  if [[ ! -f "${STATS}" ]]; then
    echo "[eval_robocasa] missing dataset stats: ${STATS}" >&2
    echo "  Run: bash scripts/download_robocasa365.sh" >&2
    exit 1
  fi

  echo "[eval_robocasa] serve ckpt=${CKPT}"
  echo "[eval_robocasa] stats=${STATS}"
  echo "[eval_robocasa] gpu=${GPU_ID} ${HOST}:${PORT} quality=${QUALITY}"

  exec python experiments/robocasa/serve_robocasa_policy.py \
    "ckpt=${CKPT}" \
    "EVALUATION.dataset_stats_path=${STATS}" \
    "+EVALUATION.quality_score=${QUALITY}" \
    "gpu_id=${GPU_ID}" \
    "server.host=${HOST}" \
    "server.port=${PORT}" \
    "$@"
}

cmd_client() {
  local launcher="${ROBOCASA365_ROOT}/robocasa365_inference_client_pro.sh"
  if [[ ! -x "${launcher}" && ! -f "${launcher}" ]]; then
    echo "[eval_robocasa] missing client launcher: ${launcher}" >&2
    exit 1
  fi
  if [[ ! -e "${ROBOCASA365_ROOT}/robosuite/robosuite/models/assets" ]]; then
    echo "[eval_robocasa] robosuite assets not linked; run: bash eval_robocasa.sh prepare" >&2
    exit 1
  fi
  for name in fixtures objects textures generative_textures; do
    if [[ ! -e "${ROBOCASA365_ROOT}/robocasa/robocasa/models/assets/${name}" ]]; then
      echo "[eval_robocasa] missing asset link: ${name}" >&2
      echo "  Run: bash eval_robocasa.sh prepare" >&2
      exit 1
    fi
  done

  echo "[eval_robocasa] client root=${ROBOCASA365_ROOT}"
  exec bash "${launcher}" "$@"
}

case "${MODE}" in
  prepare) cmd_prepare "$@" ;;
  serve) cmd_serve "$@" ;;
  client) cmd_client "$@" ;;
  -h|--help|help|"") usage; exit 1 ;;
  *)
    echo "[eval_robocasa] unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
