#!/usr/bin/env python3
"""Multiprocess Table30-V2 -> LeRobot conversion (all 4 embodiments).

Workers write temporary shards, then ``merge_lerobot_parts`` merges them.
Robot layout (cameras / single-arm vs bimanual) is inferred from ``--task-list``
filename: ``*arx5*`` / ``*ur5*`` -> single-arm maps; otherwise Aloha/W1 bimanual.

Example:
  python convert_mp.py \\
    --raw-root ../data/table30 \\
    --output-dir ../data/converted_data/table30-rc_w1_4-lerobot \\
    --task-list ./task_lists/rc_w1_4.txt \\
    --num-workers 4 --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import os
import signal
import shutil
import sys
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_fastwam_src() -> Path:
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
        "Cannot find fastwam package. Symlink ./src -> FastWAM/src "
        "or set FASTWAM_ROOT."
    )


sys.path.insert(0, str(_resolve_fastwam_src()))
sys.path.insert(0, str(REPO_ROOT))

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
from utils.merge_lerobot import merge_lerobot_parts
from convertv2 import (
    RobotLayout,
    build_features,
    convert_episode,
    discover_table30_episodes,
    open_dataset_for_recording,
    probe_image_shape,
    resolve_robot_layout,
)


@dataclass(frozen=True)
class ShardJob:
    shard_id: int
    part_dir: str
    episodes: list[dict[str, Any]]
    fps: int
    frame_interval: int
    max_frames_per_episode: int | None
    video_codec: str
    image_shape: tuple[int, int, int]
    cam_map: dict[str, str]
    single_arm: bool


def episode_to_payload(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": episode["task_name"],
        "episode_dir": str(episode["episode_dir"]),
        "local_episode_index": episode["local_episode_index"],
        "task_desc": episode["task_desc"],
        "task_info": episode["task_info"],
    }


def payload_to_episode(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": payload["task_name"],
        "episode_dir": Path(payload["episode_dir"]),
        "local_episode_index": payload["local_episode_index"],
        "task_desc": payload["task_desc"],
        "task_info": payload["task_info"],
    }


def load_task_list(path: Path) -> list[str]:
    tasks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    if not tasks:
        raise ValueError(f"task list file is empty: {path}")
    return tasks


def resolve_tasks(args: argparse.Namespace) -> list[str] | None:
    if args.task_list is not None and args.tasks:
        raise ValueError("pass only one of --task-list or --tasks")

    if args.task_list is not None:
        task_list_path = args.task_list.resolve()
        if not task_list_path.is_file():
            raise FileNotFoundError(f"task list not found: {task_list_path}")
        tasks = load_task_list(task_list_path)
        print(f"loaded {len(tasks)} tasks from {task_list_path}")
        return tasks

    return args.tasks


def split_episode_shards(
    episodes: list[dict[str, Any]],
    num_workers: int,
) -> list[list[dict[str, Any]]]:
    if not episodes:
        return []
    num_workers = max(1, min(num_workers, len(episodes)))
    chunk_size = (len(episodes) + num_workers - 1) // num_workers
    return [episodes[i : i + chunk_size] for i in range(0, len(episodes), chunk_size)]


def episode_label(episode: dict[str, Any]) -> str:
    return f"{episode['task_name']}/episode_{episode['local_episode_index']}"


def cleanup_failed_episode(dataset: LeRobotDataset, episode_index: int) -> None:
    """Remove partial artifacts for a failed episode and reset in-memory state."""
    root = Path(dataset.root)

    for image_key in dataset.meta.video_keys:
        img_dir = dataset._get_image_file_path(
            episode_index=episode_index,
            image_key=image_key,
            frame_index=0,
        ).parent
        if img_dir.is_dir():
            shutil.rmtree(img_dir)

    for vid_key in dataset.meta.video_keys:
        video_path = root / dataset.meta.get_video_file_path(episode_index, vid_key)
        if video_path.is_file():
            video_path.unlink()

    parquet_path = root / dataset.meta.get_data_file_path(episode_index)
    if parquet_path.is_file():
        parquet_path.unlink()

    dataset.episode_buffer = dataset.create_episode_buffer(
        episode_index=dataset.meta.total_episodes
    )
    if dataset.meta.total_episodes > 0:
        dataset.hf_dataset = dataset.load_hf_dataset()
    else:
        dataset.hf_dataset = dataset.create_hf_dataset()


def shard_worker_log_path(parts_root: Path, shard_id: int) -> Path:
    return parts_root / "logs" / f"shard_{shard_id:03d}.log"


def _redirect_stdio_to_file(log_fp) -> tuple[Any, Any, int, int]:
    """Redirect Python and native stdout/stderr to the same log file."""
    log_fd = log_fp.fileno()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1, closefd=False)
    stderr = os.fdopen(2, "w", encoding="utf-8", buffering=1, closefd=False)
    return stdout, stderr, saved_stdout_fd, saved_stderr_fd


def _restore_stdio(
    stdout,
    stderr,
    saved_stdout_fd: int,
    saved_stderr_fd: int,
) -> None:
    try:
        stdout.flush()
        stderr.flush()
    except Exception:
        pass
    os.dup2(saved_stdout_fd, 1)
    os.dup2(saved_stderr_fd, 2)
    os.close(saved_stdout_fd)
    os.close(saved_stderr_fd)
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", encoding="utf-8", buffering=1, closefd=False)


@contextlib.contextmanager
def worker_log_context(shard_id: int, parts_root: Path):
    log_path = shard_worker_log_path(parts_root, shard_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8", buffering=1)
    stdout, stderr, saved_stdout_fd, saved_stderr_fd = _redirect_stdio_to_file(log_fp)
    sys.stdout = stdout
    sys.stderr = stderr
    faulthandler.enable(file=log_fp, all_threads=True)

    try:
        print(f"=== worker start: shard={shard_id} pid={os.getpid()} log={log_path} ===")
        yield log_path
        print(f"=== worker finish: shard={shard_id} ===")
    except BaseException as exc:
        print(f"=== worker failed: shard={shard_id} err={type(exc).__name__}: {exc} ===")
        traceback.print_exc()
        raise
    finally:
        faulthandler.disable()
        _restore_stdio(stdout, stderr, saved_stdout_fd, saved_stderr_fd)
        log_fp.close()


def write_process_crash_log(
    log_path: Path,
    shard_id: int,
    exc: Exception,
    futures: dict[Any, int],
    parts_root: Path | None = None,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pending_shards = sorted(
        futures[f] for f in futures if not f.done()
    )
    worker_log_path = (
        shard_worker_log_path(parts_root, shard_id) if parts_root else None
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"time: {now}\n")
        f.write(f"failed_shard: {shard_id}\n")
        f.write(f"exception_type: {type(exc).__name__}\n")
        f.write(f"exception_message: {exc}\n")
        f.write(f"pending_shards: {pending_shards}\n")
        if worker_log_path is not None:
            f.write(f"worker_log: {worker_log_path}\n")
        f.write("traceback:\n")
        f.write(traceback.format_exc())
        if not traceback.format_exc().endswith("\n"):
            f.write("\n")


def convert_shard(job: ShardJob) -> str:
    part_dir = Path(job.part_dir)
    parts_root = part_dir.parent

    def _on_signal(signum: int, _frame: Any) -> None:
        print(
            f"received signal={signum} ({signal.Signals(signum).name}); exiting",
            flush=True,
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    with worker_log_context(job.shard_id, parts_root):
        layout = RobotLayout(cam_map=job.cam_map, single_arm=job.single_arm)
        features = build_features(job.image_shape)
        dataset = open_dataset_for_recording(
            output_dir=part_dir,
            fps=job.fps,
            features=features,
            video_codec=job.video_codec,
            resume=False,
            overwrite=True,
        )

        skipped: list[str] = []
        for payload in job.episodes:
            episode = payload_to_episode(payload)
            label = episode_label(episode)
            episode_index = dataset.meta.total_episodes
            print(f"start episode idx={episode_index} label={label}")
            try:
                convert_episode(
                    dataset,
                    episode,
                    frame_interval=job.frame_interval,
                    max_frames=job.max_frames_per_episode,
                    layout=layout,
                )
                print(f"done episode idx={episode_index} label={label}")
            except Exception as exc:
                print(
                    f"skip failed episode idx={episode_index} label={label} "
                    f"err={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                cleanup_failed_episode(dataset, episode_index)
                skipped.append(label)

        if skipped:
            print(
                f"skipped {len(skipped)}/{len(job.episodes)} episodes: {skipped}"
            )
        print(
            f"shard summary: shard={job.shard_id} "
            f"converted={len(job.episodes) - len(skipped)} skipped={len(skipped)} "
            f"total={len(job.episodes)}"
        )

    return str(part_dir)


def run_convert_mp(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    parts_root = args.parts_dir.resolve() if args.parts_dir else output_dir.parent / f"{output_dir.name}_parts"

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    tasks = resolve_tasks(args)
    layout = resolve_robot_layout(args.task_list)
    episodes = discover_table30_episodes(raw_root, tasks)
    if not episodes:
        scope = "all tasks" if tasks is None else f"tasks={tasks}"
        raise FileNotFoundError(f"No Table30 episodes under {raw_root} ({scope})")

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            print(f"removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} exists; pass --overwrite or choose another --output-dir"
            )

    if parts_root.exists():
        if args.overwrite:
            print(f"removing existing parts dir: {parts_root}")
            shutil.rmtree(parts_root)
        else:
            raise FileExistsError(
                f"{parts_root} exists; pass --overwrite or set --parts-dir"
            )

    print(
        f"robot layout: single_arm={layout.single_arm} cam_map={layout.cam_map}"
    )

    image_shape = probe_image_shape(episodes, layout=layout)
    fps = args.fps
    if args.frame_interval > 1:
        print(
            f"warning: frame-interval={args.frame_interval} subsamples frames; "
            f"fps metadata stays {fps}"
        )

    shards = split_episode_shards(episodes, args.num_workers)
    parts_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        ShardJob(
            shard_id=i,
            part_dir=str(parts_root / f"part_{i:03d}"),
            episodes=[episode_to_payload(ep) for ep in shard],
            fps=fps,
            frame_interval=args.frame_interval,
            max_frames_per_episode=args.max_frames_per_episode,
            video_codec=args.video_codec,
            image_shape=image_shape,
            cam_map=layout.cam_map,
            single_arm=layout.single_arm,
        )
        for i, shard in enumerate(shards)
    ]

    print(
        f"Converting {len(episodes)} episodes with {len(jobs)} workers "
        f"-> {output_dir}"
    )
    for job in jobs:
        print(f"  shard {job.shard_id}: {len(job.episodes)} episodes -> {job.part_dir}")
    print(f"worker logs -> {parts_root / 'logs'}")

    part_dirs: list[Path] = []
    if len(jobs) == 1:
        part_dirs.append(Path(convert_shard(jobs[0])))
    else:
        process_crash_log = parts_root / "process_crash.log"
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(convert_shard, job): job.shard_id for job in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="shards"):
                shard_id = futures[future]
                try:
                    part_dirs.append(Path(future.result()))
                except Exception as exc:
                    write_process_crash_log(
                        process_crash_log, shard_id, exc, futures, parts_root=parts_root
                    )
                    print(
                        f"[main] process-level shard failure logged to: {process_crash_log}",
                        file=sys.stderr,
                    )
                    worker_log = shard_worker_log_path(parts_root, shard_id)
                    if worker_log.is_file():
                        print(
                            f"[main] see full worker log: {worker_log}",
                            file=sys.stderr,
                        )
                    raise RuntimeError(f"shard {shard_id} failed: {exc}") from exc

    part_dirs = sorted(part_dirs, key=lambda p: p.name)
    print(f"Merging {len(part_dirs)} shards -> {output_dir}")
    merge_lerobot_parts(part_dirs, output_dir)

    if not args.keep_parts:
        for part_dir in parts_root.glob("part_*"):
            if part_dir.is_dir():
                shutil.rmtree(part_dir)
        print(f"removed temporary shard dirs under: {parts_root}")
        print(f"kept worker logs at: {parts_root / 'logs'}")
    else:
        print(f"kept temporary parts dir: {parts_root}")

    print(f"Done -> {output_dir}")


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
        help="Final merged LeRobot dataset root",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Temporary shard output directory (default: <output-dir>_parts)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Number of parallel worker processes",
    )
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libsvtav1",
        choices=["libsvtav1", "h264", "hevc", "h264_nvenc"],
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task folder names")
    parser.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help="Optional txt file with task folder names (one per line); default converts all tasks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output-dir / parts-dir before converting",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep temporary shard directories after merge",
    )
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")

    run_convert_mp(args)


if __name__ == "__main__":
    main()
