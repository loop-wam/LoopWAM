#!/usr/bin/env bash
# Collect XVLA RoboTwin hdf5 rollouts.
#
#   ROBOTWIN_ROOT=/path/to/RoboTwin \
#   XVLA_REPO_ROOT=/path/to/X-VLA \
#   MODEL_PATH=/path/to/X-VLA-checkpoints \
#   bash collect_xvla.sh [TASK_LIST] [SAVE_ROOT]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_env ROBOTWIN_ROOT dir
require_env XVLA_REPO_ROOT dir
require_env MODEL_PATH dir

TASK_LIST=${1:-task_list.txt}
SAVE_ROOT=${2:-./rollout_data/xvla}
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

exec python "$SCRIPT_DIR/collect.py" \
  --policy xvla \
  --task-list "$TASK_LIST" \
  --save-root "$SAVE_ROOT" \
  --task-config "${TASK_CONFIG:-demo_clean}" \
  --num-episodes "${EVAL_NUM_EPISODES:-20}" \
  --seed "${SEED:-74}" \
  --gpu "${GPU:-0}" \
  --device "${DEVICE:-cuda}" \
  --inference-steps "${INFERENCE_STEPS:-10}" \
  --success-delay-steps "${SUCCESS_DELAY_STEPS:-30}" \
  --torch-dtype "${TORCH_DTYPE:-float32}" \
  "$@"
