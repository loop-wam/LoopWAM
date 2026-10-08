#!/usr/bin/env bash
# Train the W1 LoopWAM run released as
#   checkpoints/w1_rollout_nofoldlace_delta/step_072970.pt
#
#   bash train_w1_loopwam.sh
#   bash train_w1_loopwam.sh 8
#   AUTO_RESUME=0 bash train_w1_loopwam.sh 8

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export RUN_ID="${RUN_ID:-w1_rollout_nofoldlace_delta}"
export FASTWAM_TASK="${FASTWAM_TASK:-w1_rollout_nofoldlace_delta}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${FASTWAM_TASK}/${RUN_ID}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${SCRIPT_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export FASTWAM_EXCLUDE_RAW_PREFIXES="${FASTWAM_EXCLUDE_RAW_PREFIXES:-fold_the_clothes/,untie_the_shoelaces/}"

exec bash "${SCRIPT_DIR}/scripts/train.sh" "$@"
