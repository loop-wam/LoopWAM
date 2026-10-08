#!/usr/bin/env bash
# Table30-V2 real-robot training. One embodiment per run.
#
#   bash train_table30_v2.sh ur5
#   bash train_table30_v2.sh aloha 8
#   bash train_table30_v2.sh arx5 1
#   AUTO_RESUME=0 bash train_table30_v2.sh w1 8

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROBOT="${1:-}"
if [[ -z "${ROBOT}" ]]; then
  echo "Usage: bash train_table30_v2.sh {ur5|aloha|arx5|w1} [num_gpus]" >&2
  exit 1
fi
shift

case "${ROBOT}" in
  ur5)
    TASK="ur5_rollout_subtask_delta"
    ;;
  aloha)
    TASK="aloha_rollout_delta_nopackpen"
    export FASTWAM_EXCLUDE_RAW_PREFIXES="${FASTWAM_EXCLUDE_RAW_PREFIXES:-pack_the_items/,pack_the_toothbrush_holder/,put_the_pencil_case_into_the_schoolbag/}"
    ;;
  arx5)
    TASK="arx5_newrollout_subtask_delta"
    ;;
  w1)
    TASK="w1_rollout_nofoldlace_delta"
    export FASTWAM_EXCLUDE_RAW_PREFIXES="${FASTWAM_EXCLUDE_RAW_PREFIXES:-fold_the_clothes/,untie_the_shoelaces/}"
    ;;
  *)
    echo "Unknown robot '${ROBOT}'. Use ur5, aloha, arx5, or w1." >&2
    exit 1
    ;;
esac

export RUN_ID="${RUN_ID:-${TASK}}"
export FASTWAM_TASK="${FASTWAM_TASK:-${TASK}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${FASTWAM_TASK}/${RUN_ID}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${SCRIPT_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

exec bash "${SCRIPT_DIR}/scripts/train.sh" "$@"
