#!/usr/bin/env bash
# RoboCasa365 parallel eval client (resume-capable).
# Lives under LoopWAM/third_party/RoboCasa365 with local robocasa/ + robosuite/.
# Link assets first: bash setup_symlinks.sh
# Prefer launching via: bash eval_robocasa.sh client --server_url http://HOST:7891 ...
#
# bash robocasa365_inference_client_pro.sh \
#   --task_groups "RecycleBottlesByType,WaffleReheat;\
# ArrangeBreadBasket,WeighIngredients;\
# BreadSelection,CuttingToolSelection;\
# GarnishPancake,ArrangeTea" \
#   --server_url http://127.0.0.1:7891 \
#   --split pretrain \
#   --replan_steps 48 \
#   --num_trials 50 \
#   --save_video_every 0 \
#   --run_name robocasa365_eval_unseen1
set -euo pipefail

echo "Start Robocasa Parallel Eval Client (with resume support)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODE_PATH="$SCRIPT_DIR"
ROBOCASA_REPO="$SCRIPT_DIR/robocasa"
ROBOCASA_WORKDIR="$SCRIPT_DIR/robocasa/robocasa"
ROBOSUITE_REPO="$SCRIPT_DIR/robosuite"
CLIENT_EXP="eval_robocasa_client_pro.py"
ASSET_SRC_ROOT="${ASSET_SRC_ROOT:-}"

if [[ -x "/usr/local/bin/python" ]]; then
    PYTHON_BIN="/usr/local/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: no python executable found"
    exit 1
fi


TASKS=""
TASK_SET=""
TASK_GROUPS=""
PARALLEL_CLIENTS="1"

SERVER_URL=""
SPLIT="pretrain"
NUM_TRIALS="50"
SAVE_VIDEO_EVERY="10"
REPLAN_STEPS=""

VIDEO_DIR=""
RUN_NAME=""
LOG_FILE=""

EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --code_path)
            CODE_PATH="$2"
            shift
            ;;
        --robocasa_repo)
            ROBOCASA_REPO="$2"
            shift
            ;;
        --robocasa_workdir)
            ROBOCASA_WORKDIR="$2"
            shift
            ;;
        --robosuite_repo)
            ROBOSUITE_REPO="$2"
            shift
            ;;
        --asset_src_root)
            ASSET_SRC_ROOT="$2"
            shift
            ;;
        --client_exp)
            CLIENT_EXP="$2"
            shift
            ;;
        --python_bin)
            PYTHON_BIN="$2"
            shift
            ;;
        --tasks)
            TASKS="$2"
            shift
            ;;
        --task_set)
            TASK_SET="$2"
            shift
            ;;
        --task_groups)
            TASK_GROUPS="$2"
            shift
            ;;
        --parallel_clients)
            PARALLEL_CLIENTS="$2"
            shift
            ;;
        --server_url)
            SERVER_URL="$2"
            shift
            ;;
        --split)
            SPLIT="$2"
            shift
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift
            ;;
        --save_video_every)
            SAVE_VIDEO_EVERY="$2"
            shift
            ;;
        --replan_steps)
            REPLAN_STEPS="$2"
            shift
            ;;
        --video_dir)
            VIDEO_DIR="$2"
            shift
            ;;
        --run_name)
            RUN_NAME="$2"
            shift
            ;;
        --log_file)
            LOG_FILE="$2"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$SERVER_URL" ]]; then
    echo "ERROR: --server_url is required"
    exit 1
fi

if ! [[ "$PARALLEL_CLIENTS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --parallel_clients must be a positive integer"
    exit 1
fi

if [[ "$PARALLEL_CLIENTS" -lt 1 ]]; then
    echo "ERROR: --parallel_clients must be >= 1"
    exit 1
fi

MODE_COUNT=0
[[ -n "$TASKS" ]] && MODE_COUNT=$((MODE_COUNT + 1))
[[ -n "$TASK_SET" ]] && MODE_COUNT=$((MODE_COUNT + 1))
[[ -n "$TASK_GROUPS" ]] && MODE_COUNT=$((MODE_COUNT + 1))

if [[ "$MODE_COUNT" -ne 1 ]]; then
    echo "ERROR: exactly one of --tasks, --task_set, or --task_groups must be specified"
    exit 1
fi

if [[ -n "$TASK_SET" && "$PARALLEL_CLIENTS" -gt 1 ]]; then
    echo "ERROR: --task_set cannot be automatically split."
    echo "If you want parallel eval, expand the task_set into --tasks or use --task_groups."
    exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="robocasa_parallel_eval_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "$VIDEO_DIR" ]]; then
    VIDEO_DIR="$CODE_PATH/eval_videos"
fi

if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$CODE_PATH/logs/eval/${RUN_NAME}.log"
fi

CLIENT_SCRIPT="$CODE_PATH/$CLIENT_EXP"

if [[ ! -d "$CODE_PATH" ]]; then
    echo "ERROR: CODE_PATH does not exist: $CODE_PATH"
    exit 1
fi

if [[ ! -d "$ROBOCASA_REPO" ]]; then
    echo "ERROR: ROBOCASA_REPO does not exist: $ROBOCASA_REPO"
    exit 1
fi

if [[ ! -d "$ROBOCASA_WORKDIR" ]]; then
    echo "ERROR: ROBOCASA_WORKDIR does not exist: $ROBOCASA_WORKDIR"
    exit 1
fi

if [[ ! -d "$ROBOSUITE_REPO" ]]; then
    echo "ERROR: ROBOSUITE_REPO does not exist: $ROBOSUITE_REPO"
    exit 1
fi

if [[ ! -f "$CLIENT_SCRIPT" ]]; then
    echo "ERROR: CLIENT_SCRIPT does not exist: $CLIENT_SCRIPT"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$VIDEO_DIR"

SERVER_HOST="$("$PYTHON_BIN" - <<PY
from urllib.parse import urlparse
u = urlparse("$SERVER_URL")
print(u.hostname or "")
PY
)"

# Do not force a machine-local HTTP proxy. Set HTTP_PROXY / HTTPS_PROXY yourself if needed.
if [[ -n "${SERVER_HOST}" ]]; then
  export NO_PROXY="127.0.0.1,localhost,${SERVER_HOST}${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="127.0.0.1,localhost,${SERVER_HOST}${no_proxy:+,$no_proxy}"
fi

export PYTHONPATH="$CODE_PATH:$ROBOCASA_REPO:$ROBOCASA_WORKDIR:$ROBOSUITE_REPO:${PYTHONPATH:-}"

echo "============================================================"
echo "Robocasa parallel eval client config"
echo "CODE_PATH: $CODE_PATH"
echo "ROBOCASA_REPO: $ROBOCASA_REPO"
echo "ROBOCASA_WORKDIR: $ROBOCASA_WORKDIR"
echo "ROBOSUITE_REPO: $ROBOSUITE_REPO"
echo "ASSET_SRC_ROOT: $ASSET_SRC_ROOT"
echo "CLIENT_SCRIPT: $CLIENT_SCRIPT"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "TASKS: ${TASKS:-<empty>}"
echo "TASK_SET: ${TASK_SET:-<empty>}"
echo "TASK_GROUPS: ${TASK_GROUPS:-<empty>}"
echo "PARALLEL_CLIENTS: $PARALLEL_CLIENTS"
echo "SERVER_URL: $SERVER_URL"
echo "SERVER_HOST: $SERVER_HOST"
echo "SPLIT: $SPLIT"
echo "NUM_TRIALS: $NUM_TRIALS"
echo "SAVE_VIDEO_EVERY: $SAVE_VIDEO_EVERY"
echo "REPLAN_STEPS: ${REPLAN_STEPS:-<default>}"
echo "VIDEO_DIR: $VIDEO_DIR"
echo "RUN_NAME: $RUN_NAME"
echo "LOG_FILE: $LOG_FILE"
echo "PYTHONPATH: $PYTHONPATH"
echo "Hostname: $(hostname)"
echo "IP candidates:"
hostname -I || true
echo "============================================================"

echo "Python executable:"
"$PYTHON_BIN" -c "import sys; print(sys.executable)"

echo "Changing workdir to robocasa:"
cd "$ROBOCASA_WORKDIR"
pwd

if [[ ! -d "models" ]]; then
    echo "ERROR: models directory does not exist under $(pwd)"
    exit 1
fi

if [[ ! -d "models/assets" ]]; then
    echo "models/assets does not exist, creating it..."
    mkdir -p models/assets
fi

echo "Ensuring robocasa asset subdirectory symlinks..."
ASSET_DST_ROOT="models/assets"

for name in fixtures objects textures generative_textures; do
    dst="$ASSET_DST_ROOT/$name"
    if [[ -n "$ASSET_SRC_ROOT" ]]; then
        src="$ASSET_SRC_ROOT/$name"
        if [[ ! -d "$src" ]]; then
            echo "ERROR: asset source does not exist or is not a directory: $src"
            exit 1
        fi
        if [[ -L "$dst" ]]; then
            current_target="$(readlink "$dst")"
            if [[ "$current_target" == "$src" || "$current_target" == "$(realpath "$src")" ]]; then
                echo "OK symlink $dst -> $src"
                continue
            fi
        fi
        rm -rf "$dst"
        ln -sfn "$(realpath "$src")" "$dst"
        echo "Linked $dst -> $(realpath "$src")"
    elif [[ -e "$dst" ]]; then
        echo "OK existing asset path $dst"
    else
        echo "ERROR: missing $dst"
        echo "  Run: ASSET_SRC_ROOT=... ROBOSUITE_ASSET_SRC=... bash setup_symlinks.sh"
        echo "  Or pass --asset_src_root /path/to/robocasa_assets"
        exit 1
    fi
done

# robosuite assets are expected as a symlink under this folder already
if [[ ! -e "$ROBOSUITE_REPO/robosuite/models/assets" ]]; then
    echo "ERROR: robosuite models/assets missing: $ROBOSUITE_REPO/robosuite/models/assets"
    echo "  Run: ASSET_SRC_ROOT=... ROBOSUITE_ASSET_SRC=... bash setup_symlinks.sh"
    exit 1
fi

echo "Robocasa asset symlinks:"
ls -l models/assets/fixtures
ls -l models/assets/objects
ls -l models/assets/textures
ls -l models/assets/generative_textures
echo "Robosuite assets:"
ls -ld "$ROBOSUITE_REPO/robosuite/models/assets"

echo "Environment check after asset setup:"
"$PYTHON_BIN" -c "import robosuite; print('robosuite:', robosuite.__file__)"
"$PYTHON_BIN" -c "import robocasa; print('robocasa:', robocasa.__file__)"
"$PYTHON_BIN" -c "import dexbotic; print('dexbotic:', dexbotic.__file__)" || true

EVAL_GROUPS=()

if [[ -n "$TASK_SET" ]]; then
    EVAL_GROUPS+=("__TASK_SET__:$TASK_SET")
elif [[ -n "$TASK_GROUPS" ]]; then
    IFS=';' read -r -a RAW_GROUPS <<< "$TASK_GROUPS"
    for task_group in "${RAW_GROUPS[@]}"; do
        if [[ -n "$task_group" ]]; then
            EVAL_GROUPS+=("__TASKS__:$task_group")
        fi
    done
else
    IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

    declare -a GROUP_TASKS
    for ((i = 0; i < PARALLEL_CLIENTS; i++)); do
        GROUP_TASKS[$i]=""
    done

    for task_idx in "${!TASK_ARRAY[@]}"; do
        task="${TASK_ARRAY[$task_idx]}"
        group_id=$((task_idx % PARALLEL_CLIENTS))

        if [[ -z "${GROUP_TASKS[$group_id]}" ]]; then
            GROUP_TASKS[$group_id]="$task"
        else
            GROUP_TASKS[$group_id]="${GROUP_TASKS[$group_id]},$task"
        fi
    done

    for ((i = 0; i < PARALLEL_CLIENTS; i++)); do
        if [[ -n "${GROUP_TASKS[$i]}" ]]; then
            EVAL_GROUPS+=("__TASKS__:${GROUP_TASKS[$i]}")
        fi
    done
fi

if [[ "${#EVAL_GROUPS[@]}" -eq 0 ]]; then
    echo "ERROR: no eval groups generated"
    exit 1
fi

echo "============================================================"
echo "Generated eval groups:"
for group_idx in "${!EVAL_GROUPS[@]}"; do
    echo "Group $group_idx: ${EVAL_GROUPS[$group_idx]}"
done
echo "============================================================"

PIDS=()

cleanup() {
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        echo "Cleaning up child eval processes: ${PIDS[*]}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" || true
            fi
        done
    fi
}

trap cleanup INT TERM

for group_idx in "${!EVAL_GROUPS[@]}"; do
    eval_group="${EVAL_GROUPS[$group_idx]}"
    group_log="${LOG_FILE%.log}_group${group_idx}.log"
    group_run_name="${RUN_NAME}_group${group_idx}"

    if [[ "$eval_group" == __TASK_SET__:* ]]; then
        value="${eval_group#__TASK_SET__:}"
        TASK_ARGS=(--task_set "$value")
    else
        value="${eval_group#__TASKS__:}"
        TASK_ARGS=(--tasks "$value")
    fi

    echo "Starting eval group $group_idx"
    echo "Group $group_idx run_name: $group_run_name"
    echo "Group $group_idx log: $group_log"
    echo "Group $group_idx output dir: $VIDEO_DIR/$group_run_name"
    echo "Command: $PYTHON_BIN $CLIENT_SCRIPT ${TASK_ARGS[*]} --server_url $SERVER_URL --split $SPLIT --num_trials $NUM_TRIALS --save_video_every $SAVE_VIDEO_EVERY --video_dir $VIDEO_DIR --run_name $group_run_name ${EXTRA_ARGS[*]:-}"

    (
        set -o pipefail

        (
            echo "============================================================"
            echo "Eval group $group_idx started at $(date)"
            echo "TASK_ARGS: ${TASK_ARGS[*]}"
            echo "SERVER_URL: $SERVER_URL"
            echo "SPLIT: $SPLIT"
            echo "NUM_TRIALS: $NUM_TRIALS"
            echo "SAVE_VIDEO_EVERY: $SAVE_VIDEO_EVERY"
            echo "VIDEO_DIR: $VIDEO_DIR"
            echo "RUN_NAME: $group_run_name"
            echo "PYTHONPATH: $PYTHONPATH"
            echo             "PWD: $(pwd)"
            echo "============================================================"

            if [[ "$group_idx" -gt 0 ]]; then
                sleep 2
            fi

            "$PYTHON_BIN" "$CLIENT_SCRIPT" \
                "${TASK_ARGS[@]}" \
                --server_url "$SERVER_URL" \
                --split "$SPLIT" \
                --num_trials "$NUM_TRIALS" \
                --save_video_every "$SAVE_VIDEO_EVERY" \
                ${REPLAN_STEPS:+--replan_steps "$REPLAN_STEPS"} \
                --video_dir "$VIDEO_DIR" \
                --run_name "$group_run_name" \
                "${EXTRA_ARGS[@]}"
        ) 2>&1 | sed -u "s/^/[group${group_idx}] /" | tee -a "$group_log"
    ) &

    PIDS+=("$!")
done

echo "============================================================"
echo "All eval groups launched."
echo "PIDs: ${PIDS[*]}"
echo "Waiting for all groups..."
echo "============================================================"

FAILED=0

for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: eval process failed, pid=$pid"
        FAILED=1
    fi
done

trap - INT TERM

if [[ "$FAILED" -ne 0 ]]; then
    echo "Some eval groups failed. Please check logs:"
    ls -lh "${LOG_FILE%.log}"_group*.log || true
    exit 1
fi

echo "All eval groups finished successfully."
echo "Group logs:"
ls -lh "${LOG_FILE%.log}"_group*.log || true

echo "============================================================"
echo "Merging group summaries..."
echo "============================================================"

VIDEO_DIR="$VIDEO_DIR" RUN_NAME="$RUN_NAME" "$PYTHON_BIN" - <<'PY'
import glob
import json
import os
import sys

video_dir = os.environ["VIDEO_DIR"]
run_name = os.environ["RUN_NAME"]

summary_paths = sorted(
    glob.glob(os.path.join(video_dir, f"{run_name}_group*", "summary.json"))
)

if not summary_paths:
    raise RuntimeError(
        f"No summary.json found under {video_dir}/{run_name}_group*/summary.json"
    )

all_tasks = []
source_summaries = []

for path in summary_paths:
    with open(path, "r") as f:
        summary = json.load(f)

    source_summaries.append(path)
    all_tasks.extend(summary.get("tasks", []))

if not all_tasks:
    raise RuntimeError("Found group summaries, but no task results inside them.")

task_average_success_rate = (
    sum(float(x.get("success_rate", 0.0)) for x in all_tasks) / len(all_tasks)
)

total_successes = sum(int(x.get("num_successes", 0)) for x in all_tasks)
total_episodes = sum(int(x.get("num_episodes", 0)) for x in all_tasks)
episode_average_success_rate = (
    total_successes / total_episodes if total_episodes > 0 else 0.0
)

subset_buckets = {}
for item in all_tasks:
    subset = item.get("task_subset", "custom")
    subset_buckets.setdefault(subset, []).append(item)

subset_summaries = {}
for subset, items in subset_buckets.items():
    subset_task_average_success_rate = (
        sum(float(x.get("success_rate", 0.0)) for x in items) / len(items)
    )

    subset_successes = sum(int(x.get("num_successes", 0)) for x in items)
    subset_episodes = sum(int(x.get("num_episodes", 0)) for x in items)
    subset_episode_average_success_rate = (
        subset_successes / subset_episodes if subset_episodes > 0 else 0.0
    )

    subset_summaries[subset] = {
        "num_tasks": len(items),
        "task_average_success_rate": subset_task_average_success_rate,
        "episode_average_success_rate": subset_episode_average_success_rate,
        "num_successes": subset_successes,
        "num_episodes": subset_episodes,
        "tasks": [x.get("task") for x in items],
    }

merged = {
    "run_name": run_name,
    "video_dir": video_dir,
    "num_groups": len(summary_paths),
    "num_tasks": len(all_tasks),
    "num_successes": total_successes,
    "num_episodes": total_episodes,
    "overall_success_rate": task_average_success_rate,
    "task_average_success_rate": task_average_success_rate,
    "episode_average_success_rate": episode_average_success_rate,
    "resume": True,
    "subset_summaries": subset_summaries,
    "tasks": all_tasks,
    "source_summaries": source_summaries,
}

out_path = os.path.join(video_dir, f"{run_name}_merged_summary.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print("=" * 80)
print(f"Merged {len(summary_paths)} group summaries")
print(f"Num tasks: {len(all_tasks)}")
print(f"Num episodes: {total_episodes}")
print(f"Num successes: {total_successes}")
print(f"Task-average success rate: {task_average_success_rate * 100:.2f}%")
print(f"Episode-average success rate: {episode_average_success_rate * 100:.2f}%")
print(f"Merged summary saved to: {out_path}")
print("=" * 80)
PY

echo "============================================================"
echo "Parallel eval finished."
echo "Merged summary:"
echo "$VIDEO_DIR/${RUN_NAME}_merged_summary.json"
echo "Group output dirs:"
ls -d "$VIDEO_DIR/${RUN_NAME}"_group* 2>/dev/null || true
echo "Group logs:"
ls -lh "${LOG_FILE%.log}"_group*.log 2>/dev/null || true
echo "============================================================"


