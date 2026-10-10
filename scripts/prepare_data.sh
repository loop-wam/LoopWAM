#!/usr/bin/env bash
# Prepare datasets.
#
# RoboTwin expert demos are generated with the official RoboTwin 2.0 data
# generator (https://github.com/RoboTwin-Platform/RoboTwin), then converted
# into the stitched LeRobot tree. LoopWAM rollout archives live under
# data/robotwin2_0-ours/ on https://huggingface.co/datasets/loop-wam/loopwam:
#   bash scripts/prepare_data.sh robotwin
#
# If you already have the processed tree (stitched expert + rollouts + text caches):
#   DATA_SRC=/path/to/data bash scripts/prepare_data.sh robotwin
#
# Real-robot LeRobot trees used by the four train_*_loopwam.sh scripts:
#   CONVERTED_DATA=/path/to/converted bash scripts/prepare_data.sh real
#
# Both:
#   DATA_SRC=/path/to/data CONVERTED_DATA=/path/to/converted bash scripts/prepare_data.sh all

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TARGET="${1:-}"

link_dir() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "${src}" ]]; then
    echo "[prepare_data] missing path: ${src}" >&2
    exit 1
  fi
  ln -sfn "$(realpath "${src}")" "${dst}"
  echo "[prepare_data] ${dst} -> $(realpath "${src}")"
}

extract_archive() {
  local archive="$1"
  local dest="$2"
  mkdir -p "${dest}"
  case "${archive}" in
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${dest}"
      ;;
    *.tar.bz2|*.tbz2)
      tar -xjf "${archive}" -C "${dest}"
      ;;
    *.tar)
      tar -xf "${archive}" -C "${dest}"
      ;;
    *.zip)
      unzip -qo "${archive}" -d "${dest}"
      ;;
    *)
      echo "[prepare_data] skip unknown archive type: ${archive}" >&2
      return 1
      ;;
  esac
  echo "[prepare_data] extracted $(basename "${archive}") -> ${dest}"
}

download_robotwin_rollouts() {
  local hf_dir="${ROOT}/data/_hf_loopwam"
  local src="${hf_dir}/data/robotwin2_0-ours"
  mkdir -p "${hf_dir}" "${ROOT}/data"

  if [[ "${SKIP_ROLLOUT_DOWNLOAD:-0}" == "1" ]]; then
    echo "[prepare_data] SKIP_ROLLOUT_DOWNLOAD=1, skipping LoopWAM rollout download"
    return 0
  fi

  echo "[prepare_data] downloading LoopWAM RoboTwin rollouts from loop-wam/loopwam (data/robotwin2_0-ours/)"
  if command -v hf >/dev/null 2>&1; then
    hf download loop-wam/loopwam \
      --repo-type dataset \
      --include "data/robotwin2_0-ours/*" \
      --local-dir "${hf_dir}"
  else
    huggingface-cli download loop-wam/loopwam \
      --repo-type dataset \
      --include "data/robotwin2_0-ours/*" \
      --local-dir "${hf_dir}"
  fi

  if [[ ! -d "${src}" ]]; then
    echo "[prepare_data] expected directory missing after download: ${src}" >&2
    echo "[prepare_data] upload may still be in progress; re-run once data/robotwin2_0-ours/ is on the Hub." >&2
    return 1
  fi

  shopt -s nullglob
  local archives=("${src}"/*.tar.gz "${src}"/*.tgz "${src}"/*.tar "${src}"/*.zip "${src}"/*.tar.bz2)
  shopt -u nullglob

  if (( ${#archives[@]} == 0 )); then
    # Already-extracted tree uploaded as files/dirs
    if [[ -d "${src}/lerobot_format_new" ]]; then
      mkdir -p "${ROOT}/data/robotwin2_0-ours"
      cp -a "${src}/." "${ROOT}/data/robotwin2_0-ours/"
      echo "[prepare_data] copied extracted rollout tree into data/robotwin2_0-ours/"
      return 0
    fi
    echo "[prepare_data] no archives found under ${src}" >&2
    echo "[prepare_data] upload may still be in progress; re-run after the three packages finish uploading." >&2
    return 1
  fi

  local archive
  for archive in "${archives[@]}"; do
    extract_archive "${archive}" "${ROOT}/data"
  done

  echo "[prepare_data] rollout packages extracted under ./data/"
  echo "[prepare_data] expected:"
  echo "  data/robotwin2_0-ours/lerobot_format_new/fwam_processed_stitched/"
  echo "  data/robotwin2_0-ours/lerobot_format_new/pi05_processed_stitched/"
  echo "  data/robotwin2_0-ours/lerobot_format_new/xvla_processed_stitched/"
}

prepare_robotwin() {
  if [[ -n "${DATA_SRC:-}" ]]; then
    link_dir "${DATA_SRC}" "${ROOT}/data"
    return
  fi

  mkdir -p "${ROOT}/data"
  download_robotwin_rollouts

  echo "[prepare_data] expert demos are generated with official RoboTwin 2.0, then converted to:"
  echo "  data/robotwin2_0_stitched/"
  echo "[prepare_data] mixed LoopWAM training also expects:"
  echo "  data/robotwin2_0-ours/text_embeds_cache/"
  echo "[prepare_data] build or link those (and dataset_stats.json) before train_robotwin_loopwam.sh."
}

prepare_real() {
  if [[ -z "${CONVERTED_DATA:-}" ]]; then
    if [[ -e "${ROOT}/converted_data" ]]; then
      echo "[prepare_data] converted_data already present: ${ROOT}/converted_data"
      return
    fi
    echo "[prepare_data] set CONVERTED_DATA to the converted real-robot LeRobot root." >&2
    exit 1
  fi
  link_dir "${CONVERTED_DATA}" "${ROOT}/converted_data"
}

case "${TARGET}" in
  robotwin) prepare_robotwin ;;
  real) prepare_real ;;
  all)
    prepare_robotwin
    prepare_real
    ;;
  *)
    echo "Usage: bash scripts/prepare_data.sh [robotwin|real|all]" >&2
    exit 1
    ;;
esac
