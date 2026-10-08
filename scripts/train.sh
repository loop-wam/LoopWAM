#!/usr/bin/env bash
# Cluster-friendly training with preemption resume.
#
# Set a stable RUN_ID once per experiment so requeued jobs continue the same run:
#   export RUN_ID="robotwin_4task_v1"
#   bash scripts/train.sh
#
# Env:
#   RUN_ID / FASTWAM_RUN_ID   Stable run name (required for resume across preemption)
#   FASTWAM_TASK              Hydra task (default: robotwin_uncond_3cam_384_1e-4)
#   FASTWAM_NPROC_PER_NODE    GPUs per node (default: 8)
#   AUTO_RESUME               1 = auto pick latest checkpoint (default), 0 = always fresh
#   RESUME                    Explicit state dir (overrides AUTO_RESUME), e.g. ./runs/.../checkpoints/state/step_002500
#   OUTPUT_DIR                Override output root (default: ./runs/<task>/<RUN_ID>)
#
# Examples:
#   export RUN_ID=exp1 && bash scripts/train.sh
#   AUTO_RESUME=0 RUN_ID=exp1 bash scripts/train.sh          # restart weights from scratch, same output dir
#   RESUME=./runs/.../checkpoints/state/step_010000 bash scripts/train.sh 8

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

FASTWAM_TASK="${FASTWAM_TASK:-robotwin_joint_mixed_3cam_384_1e-4}"
NPROC_PER_NODE="${FASTWAM_NPROC_PER_NODE:-8}"
AUTO_RESUME="${AUTO_RESUME:-1}"

TASK_CFG="${FASTWAM_TASK#task=}"
TASK_CFG="${TASK_CFG%.yaml}"
TASK_BASENAME="${TASK_CFG##*/}"

NPROC_PER_NODE="${1:-${NPROC_PER_NODE}}"
if [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  shift || true
else
  NPROC_PER_NODE="${FASTWAM_NPROC_PER_NODE:-8}"
fi
HYDRA_EXTRA=("$@")

has_hydra_override() {
  local key="$1"
  local arg
  for arg in "${HYDRA_EXTRA[@]}"; do
    if [[ "${arg}" == "${key}" || "${arg}" == "${key}="* ]]; then
      return 0
    fi
  done
  return 1
}

find_latest_state_dir() {
  local output_dir="$1"
  local state_root="${output_dir}/checkpoints/state"
  if [[ ! -d "${state_root}" ]]; then
    return 0
  fi

  local best_dir="" best_step=-1
  local d step
  for d in "${state_root}"/step_*; do
    [[ -d "${d}" ]] || continue
    step="${d##*/step_}"
    step="$((10#${step}))"
    if (( step > best_step )); then
      best_step="${step}"
      best_dir="${d}"
    fi
  done
  if [[ -n "${best_dir}" ]]; then
    printf '%s' "${best_dir}"
  fi
}

# Stable RUN_ID across preemptions; timestamp only when unset (not resumable across requeue).
if [[ -n "${RUN_ID:-}" ]]; then
  :
elif [[ -n "${FASTWAM_RUN_ID:-}" ]]; then
  RUN_ID="${FASTWAM_RUN_ID}"
else
  RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  echo "[train] RUN_ID not set; using ${RUN_ID}. Set export RUN_ID=<name> before submit to enable preemption resume." >&2
fi
export RUN_ID

DEFAULT_OUTPUT_DIR="./runs/${TASK_BASENAME}/${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

HYDRA_ARGS=( "task=${FASTWAM_TASK}" )

if ! has_hydra_override "output_dir"; then
  HYDRA_ARGS+=( "output_dir=${OUTPUT_DIR}" )
fi

RESUME_PATH=""
if [[ -n "${RESUME:-}" ]]; then
  RESUME_PATH="${RESUME}"
elif [[ "${AUTO_RESUME}" == "1" ]] && ! has_hydra_override "resume"; then
  RESUME_PATH="$(find_latest_state_dir "${OUTPUT_DIR}")"
fi

if [[ -n "${RESUME_PATH}" ]]; then
  if [[ ! -d "${RESUME_PATH}" ]]; then
    echo "[train] ERROR: resume path is not a directory: ${RESUME_PATH}" >&2
    exit 1
  fi
  HYDRA_ARGS+=( "resume=${RESUME_PATH}" )
  echo "[train] resume=${RESUME_PATH}"
else
  echo "[train] no checkpoint found; starting new run in ${OUTPUT_DIR}"
fi

echo "[train] task=${FASTWAM_TASK} nproc=${NPROC_PER_NODE} output_dir=${OUTPUT_DIR} run_id=${RUN_ID}"

HYDRA_ARGS+=( "${HYDRA_EXTRA[@]}" )

bash "${REPO_ROOT}/scripts/train_zero1.sh" "${NPROC_PER_NODE}" "${HYDRA_ARGS[@]}"
