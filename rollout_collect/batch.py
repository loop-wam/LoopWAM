"""Shared task-list batching for rollout collection."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def resolve_user_path(path: str) -> Path:
    expanded = Path(os.path.expanduser(os.path.expandvars(path)))
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve()


def require_env_dir(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise SystemExit(f"{name} is required.")
    path = resolve_user_path(raw)
    if not path.is_dir():
        raise SystemExit(f"{name} is not a directory: {path}")
    return path


def load_task_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Task list not found: {path}")
    tasks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tasks.append(line)
    if not tasks:
        raise ValueError(f"Task list is empty: {path}")
    return tasks


def count_hdf5_episodes(save_root: Path, task: str, task_config: str) -> int:
    data_dir = save_root / task / task_config / "data"
    if not data_dir.is_dir():
        return 0
    return len(list(data_dir.glob("episode*.hdf5")))


@dataclass
class Launch:
    cmd: list[str]
    env: dict[str, str]
    cwd: Path


def run_launch(launch: Launch) -> int:
    print("Running:")
    print(" ".join(launch.cmd))
    completed = subprocess.run(
        launch.cmd,
        cwd=str(launch.cwd),
        env=launch.env,
        check=False,
    )
    return int(completed.returncode)
