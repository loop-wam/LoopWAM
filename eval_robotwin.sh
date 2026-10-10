#!/usr/bin/env bash
# RoboTwin 2.0 evaluation for the released LoopWAM checkpoint.
# Matches the multi-GPU layout used by the internal final eval
# (one process per GPU, up to two tasks per GPU; not video/action pairing).
#
# Prerequisites: soft-links + weights (see scripts/prepare_eval.sh).
#
# Default: 8× A100 (NUM_GPUS=8), full 50-task suite.
#
#   bash eval_robotwin.sh              # 8× A100
#   bash eval_robotwin.sh 4            # fewer GPUs
#   TASK_NAME=place_shoe bash eval_robotwin.sh
#   bash eval_robotwin.sh 8 EVALUATION.eval_num_episodes=10
#
# Extra Hydra overrides go after the optional GPU count.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CKPT="${CKPT:-${ROOT}/checkpoints/robotwin/step_037645.pt}"
STATS="${STATS:-${ROOT}/checkpoints/robotwin/dataset_stats.json}"
TASK="${FASTWAM_TASK:-robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_START="${GPU_START:-0}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$1"
  shift
fi

if (( NUM_GPUS <= 0 )); then
  echo "[eval_robotwin] NUM_GPUS must be > 0 (got ${NUM_GPUS})" >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[eval_robotwin] missing checkpoint: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${STATS}" ]]; then
  echo "[eval_robotwin] missing dataset stats: ${STATS}" >&2
  exit 1
fi
if [[ ! -e "${ROOT}/third_party/RoboTwin/policy/fastwam_policy" ]]; then
  echo "[eval_robotwin] missing policy link; run: ROBOTWIN_PREBUILT=... bash scripts/prepare_eval.sh" >&2
  exit 1
fi

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

EXTRA_ARGS=()
if [[ -n "${TASK_NAME:-}" ]]; then
  EXTRA_ARGS+=("EVALUATION.task_name=${TASK_NAME}")
fi

echo "[eval_robotwin] ckpt=${CKPT}"
echo "[eval_robotwin] gpus=${NUM_GPUS} (start=${GPU_START}, max_tasks_per_gpu=${MAX_TASKS_PER_GPU}) task=${TASK}"

exec python experiments/robotwin/run_robotwin_manager.py \
  "task=${TASK}" \
  "ckpt=${CKPT}" \
  "EVALUATION.dataset_stats_path=${STATS}" \
  EVALUATION.prompt_quality_suffix=null \
  EVALUATION.prompt_quality_score=5 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.binarize_gripper=true \
  EVALUATION.smooth_action_chunk=false \
  EVALUATION.action_horizon=64 \
  EVALUATION.replan_steps=32 \
  model.video_cross_attend_proprio=false \
  data.train.num_frames=65 \
  data.val.num_frames=65 \
  data.train.action_video_freq_ratio=8 \
  data.val.action_video_freq_ratio=8 \
  EVALUATION.save_rollout=false \
  EVALUATION.multi_gpu=false \
  MULTIRUN.multi_gpu=false \
  "MULTIRUN.gpu_start=${GPU_START}" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
