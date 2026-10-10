#!/usr/bin/env python3
"""Convert Table30 multi-task exports to FastWAM / LeRobot v2.1 format.

Source layout (Table30):
  {raw_root}/{task_name}/meta/task_info.json
  {raw_root}/{task_name}/data/episode_*/states/{left,right}_states.jsonl
  {raw_root}/{task_name}/data/episode_*/videos/*.mp4

Target layout matches ``convert_robotwin_to_lerobot_new.py``:
  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/stats.json
  data/chunk-*/episode_*.parquet
  videos/chunk-*/observation.images.*/episode_*.mp4

Example:
  python convertv2.py \\
    --raw-root ./data/table30-debug \\
    --output-dir ./data/table30-debug-lerobot

Note:
  ``dataset_stats.json`` in reference datasets (e.g. fast-wam-train-converted) is
  generated during FastWAM training, not by this conversion script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_fastwam_src() -> Path:
    """Prefer local ./src, then FASTWAM_ROOT/LOOPWAM_ROOT env, then sibling FastWAM."""
    import os

    candidates: list[Path] = [REPO_ROOT / "src"]
    for env_key in ("FASTWAM_ROOT", "LOOPWAM_ROOT"):
        raw = os.environ.get(env_key)
        if raw:
            candidates.append(Path(raw) / "src")
            candidates.append(Path(raw))
    candidates.append(REPO_ROOT.parent / "FastWAM" / "src")
    candidates.append(
        REPO_ROOT.parent
        / "realrobot_inference-lyp-dm05-w1-drawer"
        / "loopwam_train"
        / "src"
    )
    for path in candidates:
        if (path / "fastwam").is_dir():
            return path
    raise FileNotFoundError(
        "Cannot find fastwam package. Symlink or copy FastWAM/loopwam_train "
        "`src` to ./src, or set FASTWAM_ROOT=/path/to/FastWAM."
    )


sys.path.insert(0, str(_resolve_fastwam_src()))
sys.path.insert(0, str(REPO_ROOT))

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import get_safe_default_codec
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)


CAM_MAP_ALOHA = {
    "observation.images.cam_high": "cam_high_rgb.mp4",
    "observation.images.cam_left_wrist": "cam_left_wrist_rgb.mp4",
    "observation.images.cam_right_wrist": "cam_right_wrist_rgb.mp4",
}

# ARX5 single-arm exports (see utils/rrd_to_video.py ARM_CAMERA_FILES)
CAM_MAP_ARX5 = {
    "observation.images.cam_high": "cam_global_rgb.mp4",
    "observation.images.cam_left_wrist": "cam_arm_rgb.mp4",
    "observation.images.cam_right_wrist": "cam_side_rgb.mp4",
}

# UR5 single-arm exports: only global + arm cameras
CAM_MAP_UR5 = {
    "observation.images.cam_high": "cam_global_rgb.mp4",
    "observation.images.cam_left_wrist": "cam_arm_rgb.mp4",
}

CAM_MAP = CAM_MAP_ALOHA
LEROBOOT_CAM_KEYS = list(CAM_MAP_ALOHA.keys())


@dataclass(frozen=True)
class RobotLayout:
    cam_map: dict[str, str]
    single_arm: bool


def resolve_robot_layout(task_list: Path | None = None) -> RobotLayout:
    """Pick camera filenames and state layout from the task-list group name."""
    if task_list is not None:
        stem = task_list.stem.lower()
        if "arx5" in stem:
            return RobotLayout(cam_map=CAM_MAP_ARX5, single_arm=True)
        if "ur5" in stem:
            return RobotLayout(cam_map=CAM_MAP_UR5, single_arm=True)
    return RobotLayout(cam_map=CAM_MAP_ALOHA, single_arm=False)

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_episode_states(
    states_dir: Path,
    single_arm: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load per-frame joint states for bimanual or single-arm Table30 exports."""
    single_path = states_dir / "states.jsonl"
    if single_arm or single_path.is_file():
        states = load_jsonl(single_path)
        empty = {"joint_positions": [], "gripper_width": 0.0}
        return states, [empty] * len(states)

    return (
        load_jsonl(states_dir / "left_states.jsonl"),
        load_jsonl(states_dir / "right_states.jsonl"),
    )


def pad(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros(n, dtype=np.float32)
    out[: min(n, arr.shape[0])] = arr[:n]
    return out


def state_to_14(left: dict[str, Any], right: dict[str, Any]) -> np.ndarray:
    """6 joint positions + 1 gripper width per arm -> 14-dim vector."""
    out = np.zeros(14, dtype=np.float32)

    left_joints = pad(left.get("joint_positions", []), 6)
    right_joints = pad(right.get("joint_positions", []), 6)

    out[0:6] = left_joints
    out[6] = float(left.get("gripper_width", 0.0))

    out[7:13] = right_joints
    out[13] = float(right.get("gripper_width", 0.0))

    return out


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
    for lerobot_key in LEROBOOT_CAM_KEYS:
        features[lerobot_key] = dict(cam_block)
    return features


def load_task_desc(task_dir: Path) -> dict[str, Any]:
    task_info_path = task_dir / "meta" / "task_info.json"
    if not task_info_path.is_file():
        raise FileNotFoundError(f"Missing task metadata: {task_info_path}")
    task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
    return task_info["task_desc"]


def coarse_task_from_desc(task_name: str, task_desc: dict[str, Any]) -> str:
    return (
        task_desc.get("full_description")
        or task_desc.get("description")
        or task_name.replace("_", " ")
    )


def discover_table30_episodes(
    raw_root: Path,
    tasks: list[str] | None,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for task_dir in sorted(raw_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "meta" / "task_info.json").is_file():
            continue
        if tasks and task_dir.name not in tasks:
            continue

        data_dir = task_dir / "data"
        if not data_dir.is_dir():
            continue

        task_desc = load_task_desc(task_dir)
        task_info = json.loads((task_dir / "meta" / "task_info.json").read_text(encoding="utf-8"))

        for ep_dir in sorted(data_dir.glob("episode_*")):
            local_idx = int(ep_dir.name.replace("episode_", ""))
            episodes.append(
                {
                    "task_name": task_dir.name,
                    "task_dir": task_dir,
                    "episode_dir": ep_dir,
                    "local_episode_index": local_idx,
                    "task_desc": task_desc,
                    "task_info": task_info,
                }
            )

    episodes.sort(key=lambda ep: (ep["task_name"], ep["local_episode_index"]))
    return episodes


def read_frame(
    caps: dict[str, cv2.VideoCapture],
    video_names: dict[str, str],
    required_cam_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    cam_keys = required_cam_keys or list(video_names.keys())
    frames: dict[str, np.ndarray] = {}
    shape: tuple[int, int, int] | None = None

    for key in cam_keys:
        if key in caps:
            ok, frame = caps[key].read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame from {video_names[key]}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[key] = rgb
            shape = rgb.shape
            continue

        if shape is None:
            raise RuntimeError(f"no opened camera available to infer shape for missing {key}")
        frames[key] = np.zeros(shape, dtype=np.uint8)

    return frames


def open_episode_captures(
    videos_dir: Path,
    cam_map: dict[str, str],
) -> dict[str, cv2.VideoCapture]:
    caps: dict[str, cv2.VideoCapture] = {}
    for key, name in cam_map.items():
        video_path = videos_dir / name
        if not video_path.is_file():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        caps[key] = cap

    if not caps:
        raise RuntimeError(f"no videos opened under {videos_dir} for cam_map={cam_map}")

    return caps


def skip_frames(caps: dict[str, cv2.VideoCapture], num_frames: int) -> None:
    for cap in caps.values():
        for _ in range(num_frames):
            cap.grab()


def save_frame_jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=95)


def open_dataset_for_recording(
    output_dir: Path,
    fps: int,
    features: dict[str, dict],
    video_codec: str,
    resume: bool,
    overwrite: bool = False,
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

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} exists; pass --resume to continue or choose another --output-dir"
            )
        else:
            output_dir.rmdir()

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
    frame_interval: int,
    max_frames: int | None,
    layout: RobotLayout | None = None,
) -> None:
    ep_dir: Path = episode["episode_dir"]
    task_desc = episode["task_desc"]
    coarse_task = coarse_task_from_desc(episode["task_name"], task_desc)
    low_instruction = task_desc.get("prompt", episode["task_name"].replace("_", " "))

    states_dir = ep_dir / "states"
    videos_dir = ep_dir / "videos"
    robot_layout = layout or resolve_robot_layout()
    cam_map = robot_layout.cam_map

    left_states, right_states = load_episode_states(states_dir, robot_layout.single_arm)

    caps = open_episode_captures(videos_dir, cam_map)

    try:
        counts = {
            key: int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for key, cap in caps.items()
        }
        n = min(len(left_states), len(right_states), *counts.values())
        if max_frames is not None:
            n = min(n, max_frames)
        if n <= 0:
            print(f"skip empty episode {ep_dir.name}: n={n}")
            return

        global_episode_index = dataset.meta.total_episodes
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=global_episode_index)

        frame_idx = 0
        written = 0
        while frame_idx < n:
            frames = read_frame(caps, cam_map, required_cam_keys=LEROBOOT_CAM_KEYS)
            if frame_interval > 1:
                skip_frames(caps, frame_interval - 1)

            state = state_to_14(left_states[frame_idx], right_states[frame_idx])
            next_idx = frame_idx + frame_interval
            action = state_to_14(
                left_states[next_idx] if next_idx < n else left_states[frame_idx],
                right_states[next_idx] if next_idx < n else right_states[frame_idx],
            )

            frame: dict[str, Any] = {
                "observation.state": state,
                "action": action,
            }
            for cam_key, rgb in frames.items():
                img_path = dataset._get_image_file_path(
                    episode_index=global_episode_index,
                    image_key=cam_key,
                    frame_index=written,
                )
                save_frame_jpeg(img_path, rgb)
                frame[cam_key] = rgb

            task_tuple = [coarse_task, low_instruction, "", ""]
            dataset.add_frame(frame, task=task_tuple)

            written += 1
            frame_idx += frame_interval

        if written == 0:
            print(f"skip empty episode {ep_dir.name}")
            return

        raw_name = f"{episode['task_name']}/episode{episode['local_episode_index']}"
        dataset.save_episode(raw_file_name=raw_name)

    finally:
        for cap in caps.values():
            cap.release()


def probe_image_shape(
    episodes: list[dict[str, Any]],
    layout: RobotLayout | None = None,
) -> tuple[int, int, int]:
    robot_layout = layout or resolve_robot_layout()
    cam_map = robot_layout.cam_map
    first_ep = episodes[0]["episode_dir"]
    high_video = first_ep / "videos" / cam_map["observation.images.cam_high"]
    cap = cv2.VideoCapture(str(high_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open first video: {high_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height, width, 3


def run_convert(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    episodes = discover_table30_episodes(raw_root, args.tasks)
    if not episodes:
        raise FileNotFoundError(f"No Table30 episodes under {raw_root}")

    h, w, c = probe_image_shape(episodes, layout=resolve_robot_layout())
    features = build_features((h, w, c))

    fps = args.fps
    if args.frame_interval > 1:
        print(
            f"warning: frame-interval={args.frame_interval} subsamples frames; "
            f"fps metadata stays {fps}"
        )

    dataset = open_dataset_for_recording(
        output_dir=output_dir,
        fps=fps,
        features=features,
        video_codec=args.video_codec,
        resume=args.resume,
    )

    print(f"Converting {len(episodes)} episodes -> {output_dir}")
    for episode in tqdm(episodes):
        convert_episode(
            dataset,
            episode,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames_per_episode,
        )

    print(
        f"Done. Episodes: {dataset.meta.total_episodes}, "
        f"frames: {dataset.meta.total_frames}, tasks: {dataset.meta.total_tasks}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "data/table30-debug",
        help="Root with per-task Table30 export folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/table30-debug-lerobot",
        help="LeRobot dataset root (all tasks merged into one dataset)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Sample every N source frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Dataset fps metadata (default: 50, same as convert_robotwin_to_lerobot_new.py)",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Optional cap on frames converted per episode",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libsvtav1",
        choices=["libsvtav1", "h264", "hevc", "h264_nvenc"],
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task folder names under --raw-root",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue writing into an existing output-dir",
    )
    args = parser.parse_args()

    run_convert(args)


if __name__ == "__main__":
    main()
