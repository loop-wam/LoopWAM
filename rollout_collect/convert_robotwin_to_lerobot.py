#!/usr/bin/env python3
"""Convert RoboTwin-collected HDF5 data to FastWAM / LeRobot v2.1 format.

Source layout (RoboTwin):
  {source_root}/{task_name}/{task_config}/data/episode{N}.hdf5
  {source_root}/{task_name}/{task_config}/scene_info.json
  {source_root}/{task_name}/{task_config}/instructions/episode{N}.json  (optional)

Target layout matches ``data/robotwin2_0-fastwam/robotwin2_0``:
  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/stats.json
  data/chunk-*/episode_*.parquet
  videos/chunk-*/observation.images.*/episode_*.mp4

Example:
  python scripts/convert_robotwin_to_lerobot.py \\
    --source-root ./data/robotwin2_0-ours/5task \\
    --output-dir ./data/robotwin2_0-fastwam/robotwin2_0_ours \\
    --task-config demo_clean
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Repo root on PYTHONPATH when invoked from FastWAM root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import get_safe_default_codec
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "RoboTwin"
DESCRIPTION_ROOT = ROBOTWIN_ROOT / "description"
OBJECTS_DESC_ROOT = DESCRIPTION_ROOT / "objects_description"

# RoboTwin HDF5 keys -> FastWAM / LeRobot camera feature keys
CAMERA_HDF5_TO_LEROBOT = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}

JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def parse_img_array(data: np.ndarray) -> np.ndarray:
    """Decode JPEG bitstreams stored in RoboTwin HDF5 rgb datasets.

    RoboTwin encodes Sapien RGB frames with ``cv2.imencode`` without swapping to
    BGR first (see ``envs/utils/pkl2hdf5.py``). After ``cv2.imdecode``, channel
    order already matches the original RGB layout, so do **not** call BGR2RGB
    (that swap is what makes red objects look blue in the output videos).
    """
    flat = np.asarray(data).ravel()
    imgs = []
    for buf in flat:
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported rgb buffer type: {type(buf)}")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed on an rgb frame buffer")
        imgs.append(img)
    return np.stack(imgs, axis=0)


def build_features(image_shape: tuple[int, int, int] = (480, 640, 3)) -> dict[str, dict]:
    h, w, c = image_shape
    joint_block = {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES),),
        "names": [JOINT_NAMES],
    }
    cam_block = {
        "dtype": "video",
        "shape": [h, w, c],
        "names": ["height", "width", "rgb"],
    }
    features = {
        "observation.state": joint_block,
        "action": joint_block,
    }
    for lerobot_key in CAMERA_HDF5_TO_LEROBOT.values():
        features[lerobot_key] = dict(cam_block)
    return features


def extract_placeholders(instruction: str) -> list[str]:
    return re.findall(r"{([^}]+)}", instruction)


def filter_instructions(instructions: list[str], episode_params: dict[str, str]) -> list[str]:
    filtered: list[str] = []
    shuffled = list(instructions)
    random.shuffle(shuffled)
    stripped = {k.strip("{}"): v for k, v in episode_params.items()}
    arm_params = {k for k in stripped if len(k) == 1 and "a" <= k <= "z"}
    for instruction in shuffled:
        placeholders = extract_placeholders(instruction)
        if set(placeholders) == set(stripped.keys()) or (
            arm_params
            and set(placeholders).union(arm_params) == set(stripped.keys())
            and not arm_params.intersection(set(placeholders))
        ):
            filtered.append(instruction)
    return filtered


def replace_placeholders(instruction: str, episode_params: dict[str, str]) -> str:
    stripped = {k.strip("{}"): v for k, v in episode_params.items()}
    for key, value in stripped.items():
        placeholder = "{" + key + "}"
        json_path = OBJECTS_DESC_ROOT / f"{value}.json"
        if json_path.is_file():
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            description = random.choice(json_data.get("seen", []) or json_data.get("unseen", []) or [value])
            value = f"the {description}"
        elif len(key) == 1 and "a" <= key <= "z":
            value = f"the {value} arm"
        instruction = instruction.replace(placeholder, value)
    return instruction


def load_task_instruction_templates(task_name: str) -> dict[str, Any]:
    path = DESCRIPTION_ROOT / "task_instruction" / f"{task_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing task instruction template: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_episode_instructions(
    instructions_dir: Path,
    episode_index: int,
) -> list[str] | None:
    path = instructions_dir / f"episode{episode_index}.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seen = data.get("seen", [])
    unseen = data.get("unseen", [])
    combined = list(seen) + list(unseen)
    return combined if combined else None


def load_rollout_instruction_from_scene_info(
    scene_info_path: Path,
    episode_index: int,
) -> list[str] | None:
    if not scene_info_path.is_file():
        return None
    with open(scene_info_path, "r", encoding="utf-8") as f:
        scene_info = json.load(f)
    episode_key = f"episode_{episode_index}"
    if episode_key not in scene_info:
        return None
    instruction = scene_info[episode_key].get("instruction")
    if instruction is None or str(instruction).strip() == "":
        return None
    return [str(instruction)]


def generate_episode_instructions(
    task_name: str,
    episode_params: dict[str, str],
    max_descriptions: int,
) -> list[str]:
    task_data = load_task_instruction_templates(task_name)
    seen_instructions = task_data.get("seen", [])
    filtered = filter_instructions(seen_instructions, episode_params)
    if not filtered:
        return [task_data.get("full_description", task_name.replace("_", " "))]

    descriptions: list[str] = []
    flag = True
    while len(descriptions) < max_descriptions and flag and filtered:
        for instruction in filtered:
            if len(descriptions) >= max_descriptions:
                flag = False
                break
            descriptions.append(replace_placeholders(instruction, episode_params))
    return descriptions


def load_scene_episode_params(scene_info_path: Path, episode_index: int) -> dict[str, str]:
    if not scene_info_path.is_file():
        return {}
    with open(scene_info_path, "r", encoding="utf-8") as f:
        scene_info = json.load(f)
    episode_key = f"episode_{episode_index}"
    if episode_key not in scene_info:
        return {}
    return scene_info[episode_key].get("info", {})


def discover_hdf5_episodes(
    source_root: Path,
    task_configs: list[str],
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for task_dir in sorted(source_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for task_config in task_configs:
            config_dir = task_dir / task_config
            data_dir = config_dir / "data"
            if not data_dir.is_dir():
                continue
            for hdf5_path in sorted(data_dir.glob("episode*.hdf5")):
                local_idx = int(hdf5_path.stem.replace("episode", ""))
                episodes.append(
                    {
                        "task_name": task_dir.name,
                        "task_config": task_config,
                        "local_episode_index": local_idx,
                        "hdf5_path": hdf5_path,
                        "scene_info_path": config_dir / "scene_info.json",
                        "instructions_dir": config_dir / "instructions",
                    }
                )
    episodes.sort(
        key=lambda ep: (ep["task_name"], ep["task_config"], ep["local_episode_index"])
    )
    return episodes


def decode_hdf5_episode(hdf5_path: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as f:
        vectors = np.asarray(f["joint_action/vector"], dtype=np.float32)
        images = {
            hdf5_cam: parse_img_array(f[f"observation/{hdf5_cam}/rgb"])
            for hdf5_cam in CAMERA_HDF5_TO_LEROBOT
        }
    if len({len(v) for v in images.values()}) != 1:
        raise ValueError(f"Inconsistent frame counts in {hdf5_path}")
    if images["head_camera"].shape[0] != len(vectors):
        raise ValueError(
            f"Frame count mismatch in {hdf5_path}: "
            f"{images['head_camera'].shape[0]} rgb vs {len(vectors)} joint vectors"
        )
    return {"vectors": vectors, "images": images}


def save_frame_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=95)


def open_dataset_for_recording(
    output_dir: Path,
    fps: int,
    features: dict[str, dict],
    video_codec: str,
    resume: bool,
) -> LeRobotDataset:
    if resume and (output_dir / "meta/info.json").is_file():
        ds = LeRobotDataset.__new__(LeRobotDataset)
        ds.repo_id = str(output_dir)
        ds.root = output_dir
        ds.tolerance_s = 1e-4
        ds.video_codec = video_codec
        ds.is_compute_episode_stats_image = True
        ds.image_writer = None
        ds.image_transforms = None
        ds.delta_timestamps = None
        ds.episodes = None
        ds.delta_indices = None
        ds.during_training = True
        ds.revision = None
        ds.meta = LeRobotDatasetMetadata(repo_id=ds.repo_id, root=ds.root)
        ds.episode_buffer = ds.create_episode_buffer()
        ds.hf_dataset = ds.create_hf_dataset()
        ds.video_backend = get_safe_default_codec()
        return ds

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} exists; pass --resume to continue or choose another --output-dir"
        )
    return LeRobotDataset.create(
        repo_id=str(output_dir),
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="aloha",
        use_videos=True,
        video_codec=video_codec,
        is_compute_episode_stats_image=True,
    )


def convert_episode(
    dataset: LeRobotDataset,
    episode: dict[str, Any],
    language_num: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    data = decode_hdf5_episode(episode["hdf5_path"])
    vectors = data["vectors"]
    images = data["images"]
    num_frames = len(vectors)

    episode_params = load_scene_episode_params(
        episode["scene_info_path"], episode["local_episode_index"]
    )
    instructions = load_episode_instructions(
        episode["instructions_dir"], episode["local_episode_index"]
    )
    if instructions is None:
        instructions = load_rollout_instruction_from_scene_info(
            episode["scene_info_path"], episode["local_episode_index"]
        )
    if instructions is None:
        instructions = generate_episode_instructions(
            episode["task_name"], episode_params, language_num
        )
    if not instructions:
        instructions = [episode["task_name"].replace("_", " ")]

    task_templates = load_task_instruction_templates(episode["task_name"])
    coarse_task = task_templates.get("full_description", episode["task_name"].replace("_", " "))

    global_episode_index = dataset.meta.total_episodes
    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=global_episode_index)

    for frame_idx in range(num_frames):
        frame: dict[str, Any] = {
            "observation.state": vectors[frame_idx].astype(np.float32),
            "action": (
                vectors[frame_idx + 1] if frame_idx + 1 < num_frames else vectors[frame_idx]
            ).astype(np.float32),
        }
        for hdf5_cam, lerobot_cam in CAMERA_HDF5_TO_LEROBOT.items():
            rgb = data["images"][hdf5_cam][frame_idx]
            img_path = dataset._get_image_file_path(
                episode_index=global_episode_index,
                image_key=lerobot_cam,
                frame_index=frame_idx,
            )
            save_frame_jpeg(img_path, rgb)
            frame[lerobot_cam] = rgb

        low_instruction = instructions[frame_idx % len(instructions)]
        task_tuple = [coarse_task, low_instruction, "", ""]
        dataset.add_frame(frame, task=task_tuple)

    raw_name = f"{episode['task_name']}/{episode['task_config']}/episode{episode['local_episode_index']}"
    dataset.save_episode(raw_file_name=raw_name)


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
    columns["index"] = pc.add(columns["index"], pa.scalar(frame_offset, type=table.schema.field("index").type))
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
    """Merge LeRobot v2.1 shards (e.g. one per RoboTwin task) into a single dataset."""
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


def run_convert(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    task_configs = args.task_configs
    if args.task_config is not None:
        task_configs = [args.task_config]

    episodes = discover_hdf5_episodes(source_root, task_configs)
    if args.tasks:
        allowed = set(args.tasks)
        episodes = [ep for ep in episodes if ep["task_name"] in allowed]
    if not episodes:
        raise FileNotFoundError(
            f"No HDF5 episodes under {source_root}/*/{{{','.join(task_configs)}}}/data/"
        )

    sample = decode_hdf5_episode(episodes[0]["hdf5_path"])
    h, w, _ = sample["images"]["head_camera"].shape[1:]
    features = build_features((h, w, 3))

    dataset = open_dataset_for_recording(
        output_dir=output_dir,
        fps=args.fps,
        features=features,
        video_codec=args.video_codec,
        resume=args.resume,
    )

    print(f"Converting {len(episodes)} episodes -> {output_dir}")
    for offset, episode in enumerate(tqdm(episodes)):
        convert_episode(
            dataset,
            episode,
            language_num=args.language_num,
            seed=args.seed + offset,
        )

    print(
        f"Done. Episodes: {dataset.meta.total_episodes}, "
        f"frames: {dataset.meta.total_frames}, tasks: {dataset.meta.total_tasks}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "data/robotwin2_0-ours/5task",
        help="Root with per-task RoboTwin export folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/robotwin2_0-fastwam/robotwin2_0_ours",
        help="LeRobot dataset root (same layout as robotwin2_0)",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="Single RoboTwin task config (deprecated; use --task-configs)",
    )
    parser.add_argument(
        "--task-configs",
        nargs="+",
        default=["demo_clean", "demo_randomized"],
        help="RoboTwin task config subfolders to include (default: clean + randomized)",
    )
    parser.add_argument("--fps", type=int, default=50, help="Dataset fps metadata (match FastWAM config)")
    parser.add_argument(
        "--language-num",
        type=int,
        default=100,
        help="Max language instructions per episode when instructions/ is missing",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libsvtav1",
        choices=["libsvtav1", "h264", "hevc", "h264_nvenc"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue writing into an existing output-dir",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task folder names under source-root",
    )
    parser.add_argument(
        "--merge-parts",
        nargs="+",
        type=Path,
        default=None,
        help="Merge existing LeRobot shard directories into --output-dir and exit",
    )
    args = parser.parse_args()

    if args.merge_parts is not None:
        merge_lerobot_parts(args.merge_parts, args.output_dir.resolve())
        return

    run_convert(args)


if __name__ == "__main__":
    main()
