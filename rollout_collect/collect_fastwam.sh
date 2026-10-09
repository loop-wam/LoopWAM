#!/usr/bin/env bash
# Collect FastWAM RoboTwin hdf5 rollouts.
#
#   ROBOTWIN_ROOT=/path/to/RoboTwin \
#   FASTWAM_ROOT=/path/to/FastWAM \
#   CKPT=/path/to/ckpt.pt \
#   bash collect_fastwam.sh [TASK_LIST] [SAVE_ROOT]
#
# dataset_stats.json is taken from DATASET_STATS_PATH, or from the checkpoint
# parent directories when that variable is unset.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_env ROBOTWIN_ROOT dir
require_env FASTWAM_ROOT dir
require_env CKPT nonempty

TASK_LIST=${1:-task_list.txt}
SAVE_ROOT=${2:-./rollout_data/fastwam}
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

infer_args=()
if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
  infer_args+=(--num-inference-steps "$NUM_INFERENCE_STEPS")
fi

exec python "$SCRIPT_DIR/collect.py" \
  --policy fastwam \
  --task-list "$TASK_LIST" \
  --save-root "$SAVE_ROOT" \
  --task-config "${TASK_CONFIG:-demo_clean}" \
  --num-episodes "${EVAL_NUM_EPISODES:-20}" \
  --seed "${SEED:-76}" \
  --gpu "${GPU:-0}" \
  --sim-task "${SIM_TASK:-robotwin_uncond_3cam_384_1e-4}" \
  --replan-steps "${REPLAN_STEPS:-24}" \
  --mixed-precision "${MIXED_PRECISION:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --text-cfg-scale "${TEXT_CFG_SCALE:-1.0}" \
  --instruction-type "${INSTRUCTION_TYPE:-unseen}" \
  "${infer_args[@]}" \
  "$@"
