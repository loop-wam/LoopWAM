#!/usr/bin/env bash
# ============================================================================
# FastWAM multi-node launcher (Tencent Cloud / TI-ONE / manual cluster).
#
# Thin task wrappers (e.g. scripts/train/train_robotwin_joint_bidirectional.sh)
# set FASTWAM_TASK / RUN_ID and call this script.
#
# Usage:
#   bash scripts/train/train_multinode.sh [gpus_per_node] [hydra_overrides...]
#   bash scripts/train/train_multinode.sh 8
#   bash scripts/train/train_multinode.sh 8 batch_size=8
#
# Multi-node env (腾讯云 TI-ONE 等平台会自动注入；手动启动时需自行 export):
#   NODE_NUM / NNODES / NUM_MACHINES     节点总数 (e.g. 2)
#   RANK / INDEX / NODE_RANK / MACHINE_RANK  当前节点 rank (0 = master)
#   CHIEF_IP / MASTER_ADDR / MASTER_IP   master 节点 IP（worker 必须能访问）
#   MAIN_PROCESS_PORT / MASTER_PORT      rendezvous 端口 (default 29500)
#
# Optional:
#   RUN_ID / FASTWAM_RUN_ID    固定实验名，多机 resume 必备
#   FASTWAM_TASK               Hydra task
#   TRAIN_ENV_KIND             conda (default) or skip if already activated
#   TRAIN_CONDA_ENV_NAME       default: loopwam
#   AUTO_RESUME                1 = 自动找最新 checkpoint (default)
#   RESUME                     显式 state 目录
#   NCCL_DEBUG=INFO            排查网络时打开
#   NCCL_SOCKET_IFNAME         指定 NCCL OOB 网卡 (default: auto eth0)
#
# Flow: train_multinode.sh -> scripts/train_zero1.sh -> accelerate launch
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Environment activation (conda, same style as psi-policy)
# ---------------------------------------------------------------------------
TRAIN_ENV_KIND="${TRAIN_ENV_KIND:-conda}"
TRAIN_CONDA_ENV_NAME="${TRAIN_CONDA_ENV_NAME:-loopwam}"

find_conda_sh() {
  local candidate=""
  if command -v conda >/dev/null 2>&1; then
    local conda_base=""
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "$conda_base/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  for candidate in \
    /root/anaconda3/etc/profile.d/conda.sh \
    /root/miniconda3/etc/profile.d/conda.sh \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    /opt/conda/etc/profile.d/conda.sh
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

activate_training_env() {
  case "$TRAIN_ENV_KIND" in
    conda)
      if [[ "${CONDA_DEFAULT_ENV:-}" == "$TRAIN_CONDA_ENV_NAME" ]]; then
        return 0
      fi
      local conda_sh=""
      if ! conda_sh="$(find_conda_sh)"; then
        echo "[train_multinode] conda not found; activate ${TRAIN_CONDA_ENV_NAME} manually or set TRAIN_ENV_KIND=skip" >&2
        exit 1
      fi
      local had_nounset=0
      case "$-" in *u*) had_nounset=1; set +u ;; esac
      # shellcheck disable=SC1090
      source "$conda_sh"
      conda activate "$TRAIN_CONDA_ENV_NAME"
      if [[ "$had_nounset" -eq 1 ]]; then set -u; fi
      ;;
    skip|"")
      ;;
    *)
      echo "[train_multinode] Unsupported TRAIN_ENV_KIND: $TRAIN_ENV_KIND (use conda, skip)" >&2
      exit 1
      ;;
  esac
}

activate_training_env

# ---------------------------------------------------------------------------
# Parse CLI: optional gpus_per_node + hydra overrides
# ---------------------------------------------------------------------------
NPROC_PER_NODE="${FASTWAM_NPROC_PER_NODE:-8}"
EXTRA_ARGS=()

if [[ $# -gt 0 && "${1:-}" =~ ^[0-9]+$ ]]; then
  NPROC_PER_NODE="$1"
  shift
fi
EXTRA_ARGS=("$@")

FASTWAM_TASK="${FASTWAM_TASK:-robotwin_joint_mixed_3cam_384_1e-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"

TASK_CFG="${FASTWAM_TASK#task=}"
TASK_CFG="${TASK_CFG%.yaml}"
TASK_BASENAME="${TASK_CFG##*/}"

has_hydra_override() {
  local key="$1"
  local arg
  for arg in "${EXTRA_ARGS[@]}"; do
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
    # Only consider plain numeric step dirs; skip suffixed ones like
    # step_032500_universal (converted checkpoints) so auto-resume stays robust.
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
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

detect_gpu_count() {
  local cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "$cuda_visible" && "$cuda_visible" != "-1" ]]; then
    local IFS=,
    local devices=()
    read -ra devices <<< "$cuda_visible"
    local count=0 device
    for device in "${devices[@]}"; do
      device="${device//[[:space:]]/}"
      if [[ -n "$device" && "$device" != "-1" ]]; then
        count=$((count + 1))
      fi
    done
    if (( count > 0 )); then
      echo "$count"
      return 0
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')" || count=0
    if [[ "$count" =~ ^[0-9]+$ && "$count" -gt 0 ]]; then
      echo "$count"
      return 0
    fi
  fi

  echo "8"
}

if [[ -n "${GPUS_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${GPUS_PER_NODE}"
elif [[ -n "${GPU_NUM_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${GPU_NUM_PER_NODE}"
elif [[ "${NPROC_PER_NODE}" == "8" && -z "${FASTWAM_NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(detect_gpu_count)"
fi

# ---------------------------------------------------------------------------
# Multi-node coordination (psi-policy compatible aliases)
# ---------------------------------------------------------------------------
NODE_COUNT="${NUM_MACHINES:-${NODE_NUM:-${NNODES:-${WORLD_SIZE:-1}}}}"
MACHINE_RANK="${MACHINE_RANK:-${INDEX:-${NODE_RANK:-${RANK:-0}}}}"
MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-${MASTER_IP:-}}}"
MASTER_PORT="${MASTER_PORT:-${MAIN_PROCESS_PORT:-29500}}"

export NNODES="${NODE_COUNT}"
export NODE_RANK="${MACHINE_RANK}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT}"

if (( NODE_COUNT > 1 )); then
  if [[ -z "${MASTER_ADDR}" || "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "[train_multinode] ERROR: multi-node requires MASTER_ADDR or CHIEF_IP (master node reachable IP)." >&2
    echo "  Example: export CHIEF_IP=10.0.0.1 NNODES=2 NODE_RANK=0" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# NCCL (Tencent / generic Ethernet; lighter than Baige RDMA defaults)
# ---------------------------------------------------------------------------
pick_oob_ifname() {
  if [[ -n "${NCCL_SOCKET_IFNAME_OVERRIDE:-}" ]]; then
    echo "${NCCL_SOCKET_IFNAME_OVERRIDE}"
    return
  fi
  for cand in eth0 bond0 ens3 ens5 eno1 enp0s3; do
    if [[ -d "/sys/class/net/${cand}" ]]; then
      echo "${cand}"
      return
    fi
  done
  echo "eth0"
}

if [[ -z "${NCCL_SOCKET_IFNAME:-}" || "${NCCL_SOCKET_IFNAME}" == rdma* ]]; then
  export NCCL_SOCKET_IFNAME="$(pick_oob_ifname)"
fi

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# ---------------------------------------------------------------------------
# RUN_ID / resume (aligned with scripts/train.sh)
# ---------------------------------------------------------------------------
if [[ -n "${RUN_ID:-}" ]]; then
  :
elif [[ -n "${FASTWAM_RUN_ID:-}" ]]; then
  RUN_ID="${FASTWAM_RUN_ID}"
elif (( NODE_COUNT <= 1 )); then
  RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  echo "[train_multinode] RUN_ID not set; using ${RUN_ID}. Set RUN_ID=<name> for preemption resume." >&2
fi
export RUN_ID

DEFAULT_OUTPUT_DIR="./runs/${TASK_BASENAME}/${RUN_ID:-pending}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

HYDRA_ARGS=( "task=${FASTWAM_TASK}" )

if ! has_hydra_override "output_dir" && [[ -n "${RUN_ID:-}" ]]; then
  HYDRA_ARGS+=( "output_dir=${OUTPUT_DIR}" )
fi

RESUME_PATH=""
if [[ -n "${RESUME:-}" ]]; then
  RESUME_PATH="${RESUME}"
elif [[ "${AUTO_RESUME}" == "1" ]] && ! has_hydra_override "resume" && [[ -n "${RUN_ID:-}" ]]; then
  RESUME_PATH="$(find_latest_state_dir "${OUTPUT_DIR}")"
fi

if [[ -n "${RESUME_PATH}" ]]; then
  if [[ ! -d "${RESUME_PATH}" ]]; then
    echo "[train_multinode] ERROR: resume path is not a directory: ${RESUME_PATH}" >&2
    exit 1
  fi
  HYDRA_ARGS+=( "resume=${RESUME_PATH}" )
  echo "[train_multinode] resume=${RESUME_PATH}"
elif [[ -n "${RUN_ID:-}" ]]; then
  echo "[train_multinode] no checkpoint found; starting new run in ${OUTPUT_DIR}"
fi

HYDRA_ARGS+=( "${EXTRA_ARGS[@]}" )

TOTAL_PROCESSES=$((NPROC_PER_NODE * NODE_COUNT))

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
{
  echo "=================== FastWAM multi-node launcher ==================="
  echo "[host]       hostname=$(hostname) cwd=$(pwd)"
  echo "[rendezvous] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
  echo "[cluster]    node_count=${NODE_COUNT} machine_rank=${MACHINE_RANK}"
  echo "[train]      task=${FASTWAM_TASK} gpus_per_node=${NPROC_PER_NODE} total_gpus=${TOTAL_PROCESSES}"
  echo "[train]      run_id=${RUN_ID:-<sync-on-launch>} output_dir=${OUTPUT_DIR}"
  echo "[env]        conda=${CONDA_DEFAULT_ENV:-<none>} TRAIN_ENV_KIND=${TRAIN_ENV_KIND}"
  echo "[net]        NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "[ifaces]     $(ls /sys/class/net 2>/dev/null | tr '\n' ' ')"
  echo "===================================================================="
} >&2

exec bash "${REPO_ROOT}/scripts/train_zero1.sh" "${NPROC_PER_NODE}" "${HYDRA_ARGS[@]}"
