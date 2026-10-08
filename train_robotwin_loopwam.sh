#!/usr/bin/env bash
# RoboTwin 2.0 mixed-rollout LoopWAM.
# Final recipe from wa_ali/run_multinode.sh:
#   expert robotwin2_0_stitched + FastWAM / pi0.5 / XVLA rollouts,
#   quality-weighted action loss, video branch does not cross-attend proprio,
#   num_frames 65, action_video_freq_ratio 8.
#
#   bash train_robotwin_loopwam.sh
#   bash train_robotwin_loopwam.sh 16
#   AUTO_RESUME=0 bash train_robotwin_loopwam.sh 16

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export RUN_ID="${RUN_ID:-ours_quality_score_3low_96_actweight_nostate_future}"
export FASTWAM_TASK="${FASTWAM_TASK:-robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${FASTWAM_TASK}/${RUN_ID}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${SCRIPT_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export TRAIN_ENV_KIND="${TRAIN_ENV_KIND:-skip}"

NPROC="${NPROC_PER_NODE:-16}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NPROC="$1"
  shift
fi

exec bash "${SCRIPT_DIR}/scripts/train/train_multinode.sh" "${NPROC}" \
  model.video_cross_attend_proprio=false \
  data.train.num_frames=65 \
  data.val.num_frames=65 \
  data.train.action_video_freq_ratio=8 \
  data.val.action_video_freq_ratio=8 \
  "$@"
