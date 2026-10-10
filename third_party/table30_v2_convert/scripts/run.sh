#!/usr/bin/env bash
# Table30-V2 -> LeRobot. Usage:
#   bash scripts/run.sh w1|aloha|arx5|ur5
#   RAW_ROOT=... OUT_DIR=... WORKERS=8 bash scripts/run.sh arx5
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROBOT="${1:-}"
if [[ -z "${ROBOT}" ]]; then
  echo "Usage: bash scripts/run.sh w1|aloha|arx5|ur5" >&2
  exit 1
fi
shift || true

RAW_ROOT="${RAW_ROOT:-../data/table30}"
WORKERS="${WORKERS:-5}"

case "${ROBOT}" in
  w1)
    TASK_LIST=./task_lists/rc_w1_4.txt
    OUT_DIR="${OUT_DIR:-../data/converted_data/table30-rc_w1_4-lerobot}"
    EXTRA=( )
    ;;
  aloha)
    TASK_LIST=./task_lists/rc_aloha_10.txt
    OUT_DIR="${OUT_DIR:-../data/converted_data/table30-rc_aloha_10-lerobot}"
    EXTRA=( )
    ;;
  arx5)
    TASK_LIST=./task_lists/rc_arx5_5.txt
    OUT_DIR="${OUT_DIR:-../data/converted_data/table30-rc_arx5_5-lerobot}"
    EXTRA=( --fps 30 --video-codec h264 )
    WORKERS="${WORKERS:-4}"
    ;;
  ur5)
    TASK_LIST=./task_lists/rc_ur5_4.txt
    OUT_DIR="${OUT_DIR:-../data/converted_data/table30-rc_ur5_4-lerobot}"
    EXTRA=( )
    WORKERS="${WORKERS:-8}"
    ;;
  *)
    echo "Unknown robot: ${ROBOT} (expected w1|aloha|arx5|ur5)" >&2
    exit 1
    ;;
esac

echo "[run] robot=${ROBOT} raw=${RAW_ROOT} out=${OUT_DIR} workers=${WORKERS}"
python convert_mp.py \
  --raw-root "${RAW_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --task-list "${TASK_LIST}" \
  --num-workers "${WORKERS}" \
  --overwrite \
  "${EXTRA[@]}" \
  "$@"
