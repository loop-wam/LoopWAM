#!/usr/bin/env bash
# Prepare datasets.
#
# RoboTwin expert data, the public Fast-WAM release
# (https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam):
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

prepare_robotwin() {
  if [[ -n "${DATA_SRC:-}" ]]; then
    link_dir "${DATA_SRC}" "${ROOT}/data"
    return
  fi

  local dest="${ROOT}/data/robotwin2.0"
  mkdir -p "${dest}"
  echo "[prepare_data] downloading yuanty/robotwin2.0-fastwam into ${dest}"
  huggingface-cli download yuanty/robotwin2.0-fastwam \
    --repo-type dataset \
    --local-dir "${dest}"

  shopt -s nullglob
  local parts=("${dest}"/robotwin2.0.tar.gz.part-*)
  shopt -u nullglob
  if (( ${#parts[@]} > 0 )); then
    echo "[prepare_data] extracting ${#parts[@]} archive parts"
    cat "${dest}"/robotwin2.0.tar.gz.part-* | tar -xzf - -C "${dest}"
  fi

  echo "[prepare_data] public expert set is under ${dest}"
  echo "[prepare_data] mixed LoopWAM training also expects these directories under ./data:"
  echo "  robotwin2_0_stitched/"
  echo "  robotwin2_0-ours/lerobot_format_new/fwam_processed_stitched/"
  echo "  robotwin2_0-ours/lerobot_format_new/pi05_processed_stitched/"
  echo "  robotwin2_0-ours/lerobot_format_new/xvla_processed_stitched/"
  echo "  robotwin2_0-ours/text_embeds_cache/"
  echo "[prepare_data] those rollout trees are not in the public archive."
  echo "[prepare_data] point DATA_SRC at a tree that already contains them, then re-run."
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
