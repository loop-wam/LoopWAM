#!/usr/bin/env bash
# Collect pi0.5 RoboTwin hdf5 rollouts.
#
#   ROBOTWIN_ROOT=/path/to/RoboTwin \
#   bash collect_pi05.sh [TASK_LIST] [SAVE_ROOT]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_env ROBOTWIN_ROOT dir

TASK_LIST=${1:-task_list.txt}
SAVE_ROOT=${2:-./rollout_data/pi05}
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

exec python "$SCRIPT_DIR/collect.py" \
  --policy pi05 \
  --task-list "$TASK_LIST" \
  --save-root "$SAVE_ROOT" \
  --task-config "${TASK_CONFIG:-demo_clean}" \
  --num-episodes "${EVAL_NUM_EPISODES:-20}" \
  --seed "${SEED:-76}" \
  --gpu "${GPU:-0}" \
  --train-config-name "${TRAIN_CONFIG_NAME:-pi05_robotwin2}" \
  --model-name "${MODEL_NAME:-${TASK_CONFIG:-demo_clean}}" \
  --checkpoint-id "${CHECKPOINT_ID:-2000}" \
  --pi0-step "${PI0_STEP:-50}" \
  --success-delay-steps "${SUCCESS_DELAY_STEPS:-30}" \
  --instruction-type "${INSTRUCTION_TYPE:-unseen}" \
  "$@"
