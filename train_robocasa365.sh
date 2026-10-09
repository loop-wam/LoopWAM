#!/usr/bin/env bash
# RoboCasa365 mixed-quality LoopWAM.
# Launch defaults match robocasa365_mq_sft_3rollout.sh:
# GPU count comes from NPROC_PER_NODE, otherwise from torch.cuda.device_count(),
# and falls back to 16 only when that detection fails.
#
#   bash train_robocasa365.sh
#   NPROC_PER_NODE=16 bash train_robocasa365.sh
#   EXTRA_ARGS="log_every=5 save_every=2500" bash train_robocasa365.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

FASTWAM_TASK="${FASTWAM_TASK:-robocasa365_join_mq_3rollout}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-log_every=5 save_every=2500}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NGPU="${NPROC_PER_NODE}"
else
  NGPU="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  if [[ "${NGPU}" == "0" || -z "${NGPU}" ]]; then
    NGPU=16
    echo "[entrypoint] cannot detect GPU count, fallback NGPU=${NGPU}."
  fi
fi

export NNODES="${NNODES:-${WORLD_SIZE:-1}}"
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export FASTWAM_TASK
export RUN_ID="${RUN_ID:-robocasa365_join_mq_3rollout}"

MY_LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${MY_LOG_DIR}"
MY_LOG_FILE="${MY_LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "[entrypoint] tee-ing all output to ${MY_LOG_FILE}"
  exec > >(tee -a "${MY_LOG_FILE}") 2>&1
fi

export NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=24
export NCCL_SOCKET_TIMEOUT=7200
export OMP_NUM_THREADS=4
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

echo "[entrypoint] ================ start quality mixed training ================"
echo "[entrypoint] task=${FASTWAM_TASK}"
echo "[entrypoint] NGPU(per node)=${NGPU} NNODES=${NNODES} NODE_RANK=${NODE_RANK}"
echo "[entrypoint] MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "[entrypoint] RUN_ID=${RUN_ID}"
echo "[entrypoint] extra_args=${EXTRA_ARGS_STR}"
echo "[entrypoint] =============================================================="

# shellcheck disable=SC2086
exec bash "${REPO_ROOT}/scripts/train_zero1.sh" "${NGPU}" "task=${FASTWAM_TASK}" ${EXTRA_ARGS_STR}
