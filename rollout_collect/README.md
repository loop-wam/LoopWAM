# Rollout 数据采集

这个目录可以单独拷走运行。RoboTwin、FastWAM、X-VLA 的位置都由环境变量指定，不依赖当前目录是谁的子目录。

仿真仍调用这些环境里的入口：

| 策略 | 需要的环境变量 | 调用的入口 |
|------|----------------|------------|
| xvla | `ROBOTWIN_ROOT` `XVLA_REPO_ROOT` `MODEL_PATH` | `$ROBOTWIN_ROOT/run_xvla.py` |
| pi05 | `ROBOTWIN_ROOT` | `$ROBOTWIN_ROOT/script/eval_policy_gwx.py`，策略在 `$ROBOTWIN_ROOT/policy/pi05` |
| fastwam | `ROBOTWIN_ROOT` `FASTWAM_ROOT` `CKPT` | 同上，策略在 `$FASTWAM_ROOT/experiments/robotwin/fastwam_policy` |

`ROBOTWIN_ROOT` 是包含 `envs/`、`task_config/`、`run_xvla.py`、`script/eval_policy_gwx.py` 的 RoboTwin 根目录。`FASTWAM_ROOT` 是包含 `configs/sim_robotwin.yaml` 和 `experiments/robotwin/fastwam_policy` 的 FastWAM 根目录。`XVLA_REPO_ROOT` 是包含 `models` 包的 X-VLA 仓库。`MODEL_PATH` 是 XVLA checkpoint 目录。

可选：`PROCESSOR_PATH`（默认等于 `MODEL_PATH`）、`DATASET_STATS_PATH`（默认沿 `CKPT` 的上级目录找 `dataset_stats.json`）、`EVAL_NUM_EPISODES`、`TASK_CONFIG`、`GPU`、`SEED`。

`task_config` 使用 `demo_clean` 或 `demo_randomized`。三条策略请使用各自的保存目录。续采按目录里已有 episode 计数。

## 目录

```
<save_root>/<task>/<task_config>/data/episode*.hdf5
<save_root>/<task>/<task_config>/scene_info.json
<save_root>/<task>/<task_config>/instructions/episode*.json
```

评测日志写在 `<save_root>/_logs/`。已有 episode 数达到目标则跳过；不足则只补差额，并从 `eval_seed_cursor.json` 接着采。

## 用法

在任意工作目录执行脚本。`task_list.txt` 和保存路径相对当前目录解析。

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
export XVLA_REPO_ROOT=/path/to/X-VLA
export MODEL_PATH=/path/to/X-VLA-checkpoints
export FASTWAM_ROOT=/path/to/FastWAM
export CKPT=/path/to/ckpt.pt

bash /path/to/rollout_collect/collect_xvla.sh task_list.txt rollout_data/xvla
bash /path/to/rollout_collect/collect_pi05.sh task_list.txt rollout_data/pi05
bash /path/to/rollout_collect/collect_fastwam.sh task_list.txt rollout_data/fastwam
```

Python 入口读同一组环境变量：

```bash
python /path/to/rollout_collect/collect.py --policy xvla --task-list task_list.txt --save-root rollout_data/xvla
python /path/to/rollout_collect/collect.py --policy pi05 --task-list task_list.txt --save-root rollout_data/pi05
python /path/to/rollout_collect/collect.py --policy fastwam --task-list task_list.txt --save-root rollout_data/fastwam
```

`--dry-run` 只打印将要执行的命令。某个任务失败时会继续后面的任务，全部结束后以非 0 退出。

FastWAM 通过 `PYTHONPATH` 加载 `$FASTWAM_ROOT/experiments/robotwin/fastwam_policy`，不往 RoboTwin 的 `policy/` 里写链接。默认每个仿真步都存一帧。若要只在 replan 时取观测，加上 `--skip-get-obs-within-replan`。
