from __future__ import annotations

from pathlib import Path


def resolve_eval_result_subpath(ckpt_path: Path) -> tuple[str, str, str, str | None]:
    """Return (task_name, run_id, step_name, extra_name) for evaluate_results layout.

    Examples:
      outputs/<task>/<run_id>/step_020000.pt -> (task, run_id, step_020000, None)
      runs/<task>/<run_id>/.../step_020000.pt -> (task, run_id, step_020000, extra_name?)
    """
    ckpt_path = ckpt_path.resolve()
    parts = ckpt_path.parts
    step_name = ckpt_path.stem

    if "outputs" in parts:
        idx = parts.index("outputs")
        if len(parts) >= idx + 4:
            return parts[idx + 1], parts[idx + 2], step_name, None

    if "runs" in parts:
        idx = parts.index("runs")
        if len(parts) >= idx + 3:
            extra_name: str | None = None
            if len(parts) >= idx + 4:
                candidate = parts[idx + 3]
                if candidate not in {"checkpoints", "weights"}:
                    extra_name = candidate
            return parts[idx + 1], parts[idx + 2], step_name, extra_name

    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if len(parts) >= idx + 3:
            return parts[idx + 2].removesuffix(".pt"), parts[idx + 1], step_name, None

    parent = ckpt_path.parent.name
    if parent and parent not in {".", "/"}:
        return parent, "default", step_name, None
    return "misc", "default", step_name, None


def resolve_eval_run_output_dir(
    ckpt_path: Path,
    run_ts: str,
    *,
    project_root: Path,
) -> Path:
    task_name, run_id, step_name, extra_name = resolve_eval_result_subpath(ckpt_path)
    base = (
        project_root
        / "evaluate_results"
        / "robotwin"
        / task_name
        / run_id
    )
    if extra_name:
        base = base / extra_name
    return base / step_name / run_ts
