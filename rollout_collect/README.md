# Rollout Collection

This directory is self-contained and may be copied out of the repository. Paths to RoboTwin, FastWAM, and X-VLA are supplied through environment variables; the scripts do not assume a fixed parent checkout.

Collection drives the simulator entry points listed below.

| Policy | Required environment variables | Entry point |
| --- | --- | --- |
| xvla | `ROBOTWIN_ROOT`, `XVLA_REPO_ROOT`, `MODEL_PATH` | `$ROBOTWIN_ROOT/run_xvla.py` |
| pi05 | `ROBOTWIN_ROOT` | `$ROBOTWIN_ROOT/script/eval_policy_gwx.py` (policy under `$ROBOTWIN_ROOT/policy/pi05`) |
| fastwam | `ROBOTWIN_ROOT`, `FASTWAM_ROOT`, `CKPT` | Same eval script; policy under `$FASTWAM_ROOT/experiments/robotwin/fastwam_policy` |

## Environment

- `ROBOTWIN_ROOT`: checkout of [loop-wam/Robotwin-Rollout](https://github.com/loop-wam/Robotwin-Rollout). It must include `envs/`, `task_config/`, `run_xvla.py`, and `script/eval_policy_gwx.py`.
- `FASTWAM_ROOT`: FastWAM (or LoopWAM) root that contains `configs/sim_robotwin.yaml` and `experiments/robotwin/fastwam_policy`.
- `XVLA_REPO_ROOT`: X-VLA repository that provides the `models` package.
- `MODEL_PATH`: X-VLA checkpoint directory.
- `CKPT`: FastWAM checkpoint file (`.pt`).

Optional overrides:

| Variable | Default |
| --- | --- |
| `PROCESSOR_PATH` | Same as `MODEL_PATH` |
| `DATASET_STATS_PATH` | Resolved from directories above `CKPT` (`dataset_stats.json`) |
| `EVAL_NUM_EPISODES` | Per-script / CLI default |
| `TASK_CONFIG` | `demo_clean` or `demo_randomized` |
| `GPU` | Device id for the worker |
| `SEED` | Base random seed |

Use a separate save root for each policy. Collection resumes from episodes already present in that root.

## Output layout

```text
<save_root>/<task>/<task_config>/data/episode*.hdf5
<save_root>/<task>/<task_config>/scene_info.json
<save_root>/<task>/<task_config>/instructions/episode*.json
```

Evaluation logs are written under `<save_root>/_logs/`. When the number of existing episodes meets the target, the task is skipped. Otherwise only the remaining episodes are collected, continuing from `eval_seed_cursor.json`.

## Usage

Run the scripts from any working directory. Paths in `task_list.txt` and the save root are resolved relative to the current directory.

```bash
git clone https://github.com/loop-wam/Robotwin-Rollout.git
export ROBOTWIN_ROOT=$PWD/Robotwin-Rollout
export XVLA_REPO_ROOT=/path/to/X-VLA
export MODEL_PATH=/path/to/X-VLA-checkpoints
export FASTWAM_ROOT=/path/to/FastWAM
export CKPT=/path/to/ckpt.pt

bash /path/to/rollout_collect/collect_xvla.sh task_list.txt rollout_data/xvla
bash /path/to/rollout_collect/collect_pi05.sh task_list.txt rollout_data/pi05
bash /path/to/rollout_collect/collect_fastwam.sh task_list.txt rollout_data/fastwam
```

The Python entrypoint accepts the same environment variables:

```bash
python /path/to/rollout_collect/collect.py --policy xvla --task-list task_list.txt --save-root rollout_data/xvla
python /path/to/rollout_collect/collect.py --policy pi05 --task-list task_list.txt --save-root rollout_data/pi05
python /path/to/rollout_collect/collect.py --policy fastwam --task-list task_list.txt --save-root rollout_data/fastwam
```

`--dry-run` prints the commands that would be executed without launching the simulator. If an individual task fails, collection continues with the remaining tasks and exits with a non-zero status at the end.

FastWAM loads `$FASTWAM_ROOT/experiments/robotwin/fastwam_policy` through `PYTHONPATH` and does not create a symlink under RoboTwin `policy/`. By default every simulation step stores an observation frame. Pass `--skip-get-obs-within-replan` to observe only at replan boundaries.

## Conversion

After HDF5 rollouts are written, convert them to the LeRobot layout used by mixed-quality training:

```bash
python /path/to/rollout_collect/convert_robotwin_to_lerobot.py ...
```

See the script’s `--help` output for the full argument list.
