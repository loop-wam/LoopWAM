# Table30-V2 → LeRobot (four embodiments)

Convert raw Table30 exports to LeRobot v2.1 for LoopWAM real-robot training.

| Robot | Command | Task list |
| --- | --- | --- |
| W1 | `bash scripts/run.sh w1` | `task_lists/rc_w1_4.txt` |
| ALOHA | `bash scripts/run.sh aloha` | `task_lists/rc_aloha_10.txt` |
| ARX5 | `bash scripts/run.sh arx5` | `task_lists/rc_arx5_5.txt` |
| UR5 | `bash scripts/run.sh ur5` | `task_lists/rc_ur5_4.txt` |

## Layout

```text
convertv2.py           # Read Table30, write one shard
convert_mp.py          # Multiprocess + merge (all four robots)
utils/merge_lerobot.py
task_lists/            # Task names per embodiment
scripts/run.sh
src/fastwam/...        # Minimal LeRobot writer API only (no models)
```

## Usage

From this directory:

```bash
cd third_party/table30_v2_convert

# Default raw root: ../data/table30
bash scripts/run.sh w1

RAW_ROOT=/path/to/table30 OUT_DIR=/path/to/out WORKERS=8 \
  bash scripts/run.sh arx5
```

If the `--task-list` filename contains `arx5` or `ur5`, the converter uses the single-arm camera map and `states.jsonl`. Otherwise it uses the bimanual Aloha / W1 layout.

`src/fastwam` is a **minimal subset** for writing LeRobot parquet and videos only. It does not include training or model code.

After conversion, link the output trees for training:

```bash
# from the LoopWAM repo root
CONVERTED_DATA=/path/to/converted bash scripts/prepare_data.sh real
bash train_table30_v2.sh w1
```
