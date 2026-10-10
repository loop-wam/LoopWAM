"""Merge LeRobot v2.1 shard directories into one dataset (used by convert_mp)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
_src = REPO_ROOT / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(REPO_ROOT))

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata

TASK_INDEX_COLS = (
    "coarse_task_index",
    "task_index",
    "coarse_quality_index",
    "quality_index",
)


def _ensure_task_index(meta: LeRobotDatasetMetadata, task: str) -> int:
    idx = meta.get_task_index(task)
    if idx is None:
        meta.add_task(task)
        idx = meta.get_task_index(task)
    assert idx is not None
    return idx


def _copy_episode_artifacts(
    src_root: Path,
    src_meta: LeRobotDatasetMetadata,
    dst_root: Path,
    dst_meta: LeRobotDatasetMetadata,
    local_episode_index: int,
    global_episode_index: int,
    frame_offset: int,
) -> None:
    src_parquet = src_root / src_meta.get_data_file_path(local_episode_index)
    dst_parquet = dst_root / dst_meta.get_data_file_path(global_episode_index)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(src_parquet)
    columns = {name: table[name] for name in table.column_names}
    ep_idx_type = table.schema.field("episode_index").type
    columns["episode_index"] = pa.array(
        [global_episode_index] * table.num_rows, type=ep_idx_type
    )
    columns["index"] = pc.add(
        columns["index"],
        pa.scalar(frame_offset, type=table.schema.field("index").type),
    )
    for col in TASK_INDEX_COLS:
        if col in columns:
            col_type = table.schema.field(col).type
            columns[col] = pa.array(
                [
                    _ensure_task_index(dst_meta, src_meta.tasks[int(old)])
                    for old in table[col].to_pylist()
                ],
                type=col_type,
            )
    pq.write_table(pa.table(columns), dst_parquet)

    for vid_key in dst_meta.video_keys:
        src_video = src_root / src_meta.get_video_file_path(local_episode_index, vid_key)
        dst_video = dst_root / dst_meta.get_video_file_path(global_episode_index, vid_key)
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_video, dst_video)


def merge_lerobot_parts(part_dirs: list[Path], output_dir: Path) -> None:
    """Merge LeRobot v2.1 shards into a single dataset."""
    part_dirs = [p.resolve() for p in part_dirs]
    output_dir = output_dir.resolve()

    for part_dir in part_dirs:
        if not (part_dir / "meta/info.json").is_file():
            raise FileNotFoundError(f"Not a LeRobot dataset: {part_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(part_dirs[0], output_dir)

    dst_meta = LeRobotDatasetMetadata(repo_id=str(output_dir), root=output_dir)
    print(
        f"Merge base {part_dirs[0].name}: "
        f"{dst_meta.total_episodes} episodes, {dst_meta.total_frames} frames"
    )

    for part_dir in part_dirs[1:]:
        src_meta = LeRobotDatasetMetadata(repo_id=str(part_dir), root=part_dir)
        frame_offset = dst_meta.total_frames
        for local_ep in range(src_meta.total_episodes):
            global_ep = dst_meta.total_episodes
            _copy_episode_artifacts(
                part_dir,
                src_meta,
                output_dir,
                dst_meta,
                local_ep,
                global_ep,
                frame_offset,
            )
            ep_info = src_meta.episodes[local_ep]
            ep_stats = src_meta.episodes_stats[local_ep]
            dst_meta.save_episode(
                global_ep,
                ep_info["length"],
                ep_info["tasks"],
                ep_stats,
                ep_info.get("raw_file_name"),
            )
            frame_offset += ep_info["length"]
        print(
            f"Merged {part_dir.name}: +{src_meta.total_episodes} episodes "
            f"(total {dst_meta.total_episodes})"
        )

    print(
        f"Merge done -> {output_dir}: "
        f"{dst_meta.total_episodes} episodes, {dst_meta.total_frames} frames"
    )
