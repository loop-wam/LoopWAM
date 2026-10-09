# LoopWAM

**Closed-loop generalist policy improvement from deployment rollouts.**

[Project page](https://loopwam.github.io/) · [arXiv](https://arxiv.org/abs/XXXX.XXXXX)

Expert demonstrations show a robot how to succeed. Deployment shows what happens when it does not. LoopWAM learns from that full mixture: successful, suboptimal, and failed rollouts from itself and from other deployed policies. World-model supervision stays on for every trajectory. Action imitation is weighted by trajectory quality, and failed rollouts contribute no direct action-imitation loss.

| Benchmark | Setting | LoopWAM |
| --- | --- | --- |
| [RoboChallenge Table30-V2](https://loopwam.github.io/) | 30 real tasks, 4 embodiments | **43.33%** success, **56.80** score |
| [RoboTwin 2.0](https://loopwam.github.io/) | 50 bimanual tasks, clean and randomized | **94.5%** average success |
| [RoboCasa365](https://loopwam.github.io/) | 50 tasks, seen and unseen compositions | **49.5%** overall success |

These are the best overall numbers among the methods compared in the submitted paper. No extra embodied-data pretraining is used on RoboTwin 2.0 or RoboCasa365. The video backbone is pretrained Wan2.2.

## Method

LoopWAM jointly denoises future video and actions. Bidirectional attention connects the two streams at every layer, so learned action consequences can inform control. Both streams are conditioned on the instruction and on trajectory quality. At deployment, an expert-quality target steers the policy.

Three deployment-and-training rounds on Table30-V2 take a single multi-task model per embodiment from an expert-trained policy to the final result:

| Stage | Success | Score |
| --- | ---: | ---: |
| Expert-trained | 23.67% | 36.10 |
| Round 1 | 28.00% | 42.63 |
| Round 2 | 37.67% | 51.65 |
| Round 3 | 43.33% | 56.80 |

Shirt folding, after single-task post-training, moves from 13.20% to 55.56% in two further rounds. Numbers are from the [project page](https://loopwam.github.io/) and the paper (Figure 3, Section 4.4).

## Results

### RoboChallenge Table30-V2

30 real-world tasks. One multi-task model per embodiment. Overall score uses task weights 10 / 10 / 7 / 3 for ALOHA, DOS-W1, ARX5, and UR5.

| Method | ALOHA | DOS-W1 | ARX5 | UR5 | Score | SR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LoopWAM** | **26.00** | **42.00** | 55.71 | **76.67** | **56.80** | **43.33** |
| VLA-DM0.5 | 18.00 | 42.00 | **58.57** | 70.00 | 54.42 | 40.67 |
| LoopWAM (expert-trained) | 8.00 | 38.00 | 24.29 | 26.67 | 36.10 | 23.67 |

Full comparison: [project page, Table 1](https://loopwam.github.io/).

### RoboTwin 2.0

50 bimanual tasks. Mixed expert–rollout training. Rollouts come from an expert-trained LoopWAM, X-VLA, and π0.5. Collection and evaluation use different random seeds.

| Method | Embodied pretraining | Clean | Randomized | Average |
| --- | --- | ---: | ---: | ---: |
| **LoopWAM** | No | **94.6** | **94.4** | **94.5** |
| LingBot-VA 2.0 | Yes | 93.8 | 93.4 | 93.6 |
| AHA-WAM | No | 93.4 | 92.2 | 92.8 |
| Fast-WAM | No | 91.9 | 91.8 | 91.8 |
| π0.5 | Yes | 82.7 | 76.8 | 79.8 |

### RoboCasa365

All 16 Composite-Unseen tasks are held out of both expert data and rollout data. Overall success uses weights 18 / 16 / 16.

| Method | Atomic seen | Composite seen | Composite unseen | Overall |
| --- | ---: | ---: | ---: | ---: |
| **LoopWAM** | **79.7** | **50.3** | 14.9 | **49.5** |
| ABot-M0.6 | 79.4 | 48.3 | 7.9 | 46.6 |
| PRTS | 66.3 | 30.3 | **18.8** | 39.6 |

## Repository

| Path | What it is |
| --- | --- |
| `train_table30_v2.sh` | Real-robot training for UR5, ALOHA, ARX5, and DOS-W1 |
| `train_robotwin_loopwam.sh` | RoboTwin 2.0 training |
| `train_robocasa365.sh` | RoboCasa365 mixed-quality training |
| `checkpoints/` | Released weights, real copies, each with `dataset_stats.json` |
| `configs/` | Hydra task and data configs |
| `scripts/setup_env.sh` | Conda env, PyTorch, and `pip install -e .` |
| `scripts/download_pretrained.sh` | Wan2.2-TI2V-5B download and ActionDiT backbone |
| `scripts/prepare_data.sh` | RoboTwin expert data |
| `rollout_collect/` | RoboTwin rollout collection for FastWAM, π0.5, and X-VLA |
| `experiments/robotwin/` | RoboTwin 2.0 evaluation manager and policy |
| `experiments/robocasa/` | RoboCasa365 policy server and simulator client |
| `scripts/` | Training launcher (`train.py`, DeepSpeed / Accelerate configs) |
| `src/` | Model and dataset code |

## Environment

Python 3.10 and PyTorch 2.7.1+cu128.

```bash
bash scripts/setup_env.sh
conda activate loopwam
```

## Pretrained weights

Training and inference both need Wan2.2-TI2V-5B under `checkpoints/`, plus the ActionDiT backbone interpolated from the Wan DiT. The script downloads the Wan files (ModelScope by default) and writes `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`.

```bash
bash scripts/download_pretrained.sh

# Hugging Face instead of ModelScope
DIFFSYNTH_DOWNLOAD_SOURCE=huggingface bash scripts/download_pretrained.sh
```

Released LoopWAM checkpoints already live in `checkpoints/` next to that backbone. Training scripts set `DIFFSYNTH_MODEL_BASE_PATH` to this directory and skip a second download.

## Dataset

### RoboTwin 2.0

The public expert set is the Fast-WAM release, [yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam). The script downloads the split archives and extracts them:

```bash
bash scripts/prepare_data.sh robotwin
```

Extracted layout:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

`dataset_stats.json` in that directory can be used as the normalization file. Mixed LoopWAM training reads a stitched expert set and three rollout sets, which are not in the public archive:

```text
data/robotwin2_0_stitched/
data/robotwin2_0-ours/lerobot_format_new/fwam_processed_stitched/
data/robotwin2_0-ours/lerobot_format_new/pi05_processed_stitched/
data/robotwin2_0-ours/lerobot_format_new/xvla_processed_stitched/
data/robotwin2_0-ours/text_embeds_cache/
```

If that tree already exists, link it and skip the public download:

```bash
DATA_SRC=/path/to/data bash scripts/prepare_data.sh robotwin
```

`rollout_collect/` gathers the three rollout sets in RoboTwin. It writes HDF5 episodes and can resume from what is already saved. The simulator root is [loop-wam/Robotwin-Rollout](https://github.com/loop-wam/Robotwin-Rollout). Clone it and set `ROBOTWIN_ROOT` to that checkout. FastWAM and X-VLA paths come from environment variables. Details are in `rollout_collect/README.md`.

```bash
git clone https://github.com/loop-wam/Robotwin-Rollout.git
export ROBOTWIN_ROOT=$PWD/Robotwin-Rollout

bash rollout_collect/collect_fastwam.sh task_list.txt rollout_data/fastwam
bash rollout_collect/collect_pi05.sh task_list.txt rollout_data/pi05
bash rollout_collect/collect_xvla.sh task_list.txt rollout_data/xvla
```

`rollout_collect/convert_robotwin_to_lerobot.py` turns those HDF5 episodes into the LeRobot layout used by mixed training.

Before the first training run, cache umT5 embeddings for the task prompts:

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted

# or several GPUs
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted
```

### Real robots

Real-robot training uses [RoboChallenge Table30-V2](https://loopwam.github.io/).

## Training

Real-robot scripts default to 8 GPUs. RoboTwin 2.0 defaults to 16 GPUs per node. RoboCasa365 uses `NPROC_PER_NODE` when it is set, otherwise the number of visible GPUs, and falls back to 16 only if that count cannot be detected.

```bash
bash train_table30_v2.sh ur5       # UR5
bash train_table30_v2.sh aloha     # ALOHA
bash train_table30_v2.sh arx5      # ARX5
bash train_table30_v2.sh w1        # DOS-W1
bash train_robotwin_loopwam.sh     # RoboTwin 2.0
bash train_robocasa365.sh          # RoboCasa365
```

Pass a GPU count after the robot name, for example `bash train_table30_v2.sh ur5 1`. RoboTwin takes the GPU count as its first argument. RoboCasa365 keeps the original launch defaults, including `log_every=5` and `save_every=2500`; set `NPROC_PER_NODE` to choose the GPU count.

Real-robot training uses RoboChallenge Table30-V2. Released checkpoints, each with its `dataset_stats.json`:

| Robot | Checkpoint |
| --- | --- |
| UR5 | `checkpoints/ur5_rollout_subtask_delta/step_029925.pt` |
| ALOHA | `checkpoints/aloha_rollout_delta_nopackpen/step_061320.pt` |
| ARX5 | `checkpoints/arx5_newrollout_subtask_delta/step_057355.pt` |
| DOS-W1 | `checkpoints/w1_rollout_nofoldlace_delta/step_072970.pt` |
| RoboTwin 2.0 | `checkpoints/robotwin/step_037645.pt` |

`save_every` is 2500, so a fresh run writes `step_002500.pt`, `step_005000.pt`, and so on. The released step numbers are the checkpoints used for evaluation; they are not multiples of 2500.

RoboTwin 2.0 follows the final mixed-rollout recipe: expert `robotwin2_0_stitched` plus FastWAM, π0.5, and X-VLA rollouts, quality-weighted action loss, `num_frames=65`, `action_video_freq_ratio=8`. Prepare that tree with `scripts/prepare_data.sh` before launching.

RoboCasa365 uses expert demonstrations plus three rollout sources, quality-weighted action loss, and the `robocasa` three-camera stitch (`256x384`). Dataset roots stay on the training cluster under `/mnt/data/dm05/dexmal-aa-wzg-data/robocasa365/`. Normalization stats are computed from the training set on the first run and saved as `dataset_stats.json` in the run directory. Text-embedding caches are the paths already written in `configs/data/robocasa365_mq_3rollout.yaml`.

## RoboTwin 2.0 evaluation

Released weights: `checkpoints/robotwin/step_037645.pt` and `checkpoints/robotwin/dataset_stats.json`. Reported result: clean **94.6%**, randomized **94.4%**, average **94.5%**.

The simulator is not in this repository. Install [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), including cuRobo and the task assets, at `third_party/RoboTwin`, or pass another checkout with `EVALUATION.robotwin_root`. That checkout must contain `script/eval_policy.py`, `policy/`, and `task_config/_eval_step_limit.yml`. The task list in `_eval_step_limit.yml` is the full 50-task suite.

Evaluation also loads the Wan2.2 text encoder. Download it once if `checkpoints/` does not already contain the Wan files:

```bash
bash scripts/download_pretrained.sh
```

`configs/sim_robotwin.yaml` skips loading the ActionDiT backbone from scratch (`skip_dit_load_from_pretrain=true`). The released `.pt` supplies the trained weights.

Run from the repository root:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted \
  ckpt=./checkpoints/robotwin/step_037645.pt \
  EVALUATION.dataset_stats_path=./checkpoints/robotwin/dataset_stats.json \
  EVALUATION.prompt_quality_suffix=null \
  EVALUATION.prompt_quality_score=5 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.action_horizon=64 \
  EVALUATION.replan_steps=32 \
  model.video_cross_attend_proprio=false \
  data.train.num_frames=65 \
  data.val.num_frames=65 \
  data.train.action_video_freq_ratio=8 \
  data.val.action_video_freq_ratio=8 \
  EVALUATION.save_rollout=false \
  EVALUATION.multi_gpu=true \
  EVALUATION.video_device=cuda:1 \
  EVALUATION.action_device=cuda:0 \
  MULTIRUN.multi_gpu=true \
  MULTIRUN.gpu_start=0 \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

What that command does:

1. `run_robotwin_manager.py` reads every task name in `third_party/RoboTwin/task_config/_eval_step_limit.yml`. Set `EVALUATION.task_name=<one task>` or `EVALUATION.task_names=[a,b]` to run a subset.
2. Each task is evaluated twice: `demo_clean`, then `demo_randomized`. Each phase runs `EVALUATION.eval_num_episodes` episodes (100 in `configs/sim_robotwin.yaml`) with unseen instructions.
3. With `MULTIRUN.multi_gpu=true` and `num_gpus=8`, the manager pairs devices `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)` and keeps one task on each pair. `max_tasks_per_gpu=1` means one task per pair. `num_gpus` must be even. Video runs on `cuda:1` of the pair and the action expert on `cuda:0`, leaving the simulator on the action GPU.
4. Each worker is `experiments/robotwin/eval_robotwin_single.py`. It links `experiments/robotwin/fastwam_policy` to `<RoboTwin>/policy/fastwam_policy` when that link is missing, then calls `script/eval_policy.py` inside the RoboTwin checkout.
5. `fastwam_policy/deploy_policy.py` rebuilds the Hydra config, loads `dataset_stats.json` and the checkpoint, and steps the simulator with absolute joint positions. The policy samples 64 actions and executes 32 before replanning. `prompt_quality_score=5` asks for the expert-quality prompt used at deployment. `skip_get_obs_within_replan=true` skips RGB rendering inside one action chunk; set it to `false` when you need every frame rendered.

`configs/sim_robotwin.yaml` defaults to `task=robotwin_uncond_3cam_384_1e-4`. The command above replaces that with the mixed-rollout task, matching the released checkpoint.

Results are written under `evaluate_results/robotwin/`. For `checkpoints/robotwin/step_037645.pt` the run directory is `evaluate_results/robotwin/step_037645/robotwin/step_037645/<timestamp>/`. The manager writes `manager.log`, `summary.csv`, `summary.json`, and `failed_tasks.txt`. Each task directory contains `_result_clean.txt` and `_result_random.txt`; the last number in each file is that phase's success rate. `summary.json` reports the mean of both phases. A worker failure stops the remaining tasks and records the return code in `failed_tasks.txt`.

## RoboCasa365 evaluation

RoboCasa365 is split across two processes. This repository serves the policy. The simulator runs in a separate container and calls the server over HTTP.

The server reuses `task=robocasa365_join_mq_3rollout`, so the camera stitch, 11-dimensional action, and 14-dimensional state match training. `configs/sim_robocasa.yaml` loads the Wan2.2 text encoder and skips a fresh ActionDiT initialization; the trained `.pt` supplies the weights. `dataset_stats.json` is taken from `EVALUATION.dataset_stats_path` or from the checkpoint's parent directories. Download the text encoder first if it is not already under `checkpoints/`:

```bash
bash scripts/download_pretrained.sh
```

Start the server from the repository root. `ckpt` is resolved from this root, so pass an absolute path when the weights are not inside the repo. `+EVALUATION.quality_score=5` appends the expert-quality prompt. The server listens on `0.0.0.0:7891` and answers `POST /process_frame` with an action chunk of length `num_frames - 1` (64 for this task).

```bash
python experiments/robocasa/serve_robocasa_policy.py \
  ckpt=runs/robocasa365_expert_sft_from_3rollout/robocasa365_expert_sft_from_3rollout/checkpoints/weights/step_112500.pt \
  +EVALUATION.quality_score=5 \
  gpu_id=0 \
  server.host=0.0.0.0 \
  server.port=7891
```

On the simulator machine, install RoboCasa at `/workspace/robocasa` (override with `--robocasa_repo` and `--robocasa_workdir`). Then run the client from `experiments/robocasa/`. It sends three PNGs and the 14-dimensional state, converts the returned 11-dimensional action back to the 12-dimensional simulator action, and executes `--replan_steps` actions before asking for the next chunk. The command below runs eight task pairs, 50 trials each, on the `pretrain` split, and does not save videos (`--save_video_every 0`). Change `--server_url` to the machine that is serving the policy.

```bash
cd experiments/robocasa && \
bash robocasa365_inference_client_pro.sh \
  --task_groups "RecycleBottlesByType,WaffleReheat;\
ArrangeBreadBasket,WeighIngredients;\
BreadSelection,CuttingToolSelection;\
GarnishPancake,ArrangeTea" \
  --server_url http://10.2.0.243:7891 \
  --split pretrain \
  --replan_steps 48 \
  --num_trials 50 \
  --save_video_every 0 \
  --video_dir /mlp_vepfs/share/wzg/project/eval_robocasa365/eval3e_mq_3rollout_resft_48/unseen16 \
  --run_name robocasa365_eval_parallel_test_unseen1
```

The client resumes from its log when the same `--run_name` is used again. Logs go to `experiments/robocasa/logs/eval/<run_name>.log`.

