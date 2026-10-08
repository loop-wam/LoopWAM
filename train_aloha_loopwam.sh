#!/usr/bin/env bash
# Train the Aloha LoopWAM run released as
#   checkpoints/aloha_rollout_delta_nopackpen/step_061320.pt
#
#   bash train_aloha_loopwam.sh
#   bash train_aloha_loopwam.sh 8
#   AUTO_RESUME=0 bash train_aloha_loopwam.sh 8

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export RUN_ID="${RUN_ID:-aloha_rollout_delta_nopackpen}"
export FASTWAM_TASK="${FASTWAM_TASK:-aloha_rollout_delta_nopackpen}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${FASTWAM_TASK}/${RUN_ID}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${SCRIPT_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
# Stats file is dataset_stats_relative_joint_no_pack_pencil.json.
export FASTWAM_EXCLUDE_RAW_PREFIXES="${FASTWAM_EXCLUDE_RAW_PREFIXES:-pack_the_items/,pack_the_toothbrush_holder/,put_the_pencil_case_into_the_schoolbag/}"

exec bash "${SCRIPT_DIR}/scripts/train.sh" "$@"
