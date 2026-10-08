#!/usr/bin/env bash
# Train the ARX5 LoopWAM run released as
#   checkpoints/arx5_newrollout_subtask_delta/step_057355.pt
#
#   bash train_arx5_loopwam.sh
#   bash train_arx5_loopwam.sh 8
#   AUTO_RESUME=0 bash train_arx5_loopwam.sh 8

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export RUN_ID="${RUN_ID:-arx5_newrollout_subtask_delta}"
export FASTWAM_TASK="${FASTWAM_TASK:-arx5_newrollout_subtask_delta}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${FASTWAM_TASK}/${RUN_ID}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${SCRIPT_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

exec bash "${SCRIPT_DIR}/scripts/train.sh" "$@"
