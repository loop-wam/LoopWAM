"""Build one-task rollout commands for xvla, pi05, and fastwam."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from batch import Launch


def _bool_token(value: bool) -> str:
    return "True" if value else "False"


def _base_env(gpu: str, robotwin_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["ROBOTWIN_ROOT"] = str(robotwin_root)
    return env


def build_xvla(
    *,
    robotwin_root: Path,
    task: str,
    task_config: str,
    num_episodes: int,
    save_root: Path,
    seed: int,
    gpu: str,
    model_path: str,
    processor_path: str,
    xvla_repo_root: str,
    device: str,
    inference_steps: int,
    success_delay_steps: int,
    save_only_success: bool,
    resume: bool,
    torch_dtype: str,
    eval_log_dir: Path,
    output_path: Path,
    instruction_type: str | None,
    extra: list[str],
) -> Launch:
    entry = robotwin_root / "run_xvla.py"
    if not entry.is_file():
        raise FileNotFoundError(f"XVLA rollout entry not found: {entry}")
    cmd = [
        sys.executable,
        str(entry),
        "--model_path",
        model_path,
        "--processor_path",
        processor_path,
        "--xvla_repo_root",
        xvla_repo_root,
        "--device",
        device,
        "--task_name",
        task,
        "--task_config",
        task_config,
        "--num_episodes",
        str(num_episodes),
        "--eval_log_dir",
        str(eval_log_dir),
        "--output_path",
        str(output_path),
        "--rollout_save_root",
        str(save_root),
        "--rollout_save_only_success",
        "true" if save_only_success else "false",
        "--rollout_resume",
        "true" if resume else "false",
        "--inference_steps",
        str(inference_steps),
        "--success_delay_steps",
        str(success_delay_steps),
        "--seed",
        str(seed),
        "--torch_dtype",
        torch_dtype,
        "--save_rollout",
        "true",
    ]
    if instruction_type:
        cmd.extend(["--instruction_type", instruction_type])
    cmd.extend(extra)
    return Launch(cmd=cmd, env=_base_env(gpu, robotwin_root), cwd=robotwin_root)


def build_pi05(
    *,
    robotwin_root: Path,
    task: str,
    task_config: str,
    num_episodes: int,
    save_root: Path,
    seed: int,
    gpu: str,
    train_config_name: str,
    model_name: str,
    checkpoint_id: int,
    pi0_step: int,
    success_delay_steps: int,
    save_only_success: bool,
    resume: bool,
    instruction_type: str,
    eval_output_dir: Path,
    extra: list[str],
) -> Launch:
    entry = robotwin_root / "script" / "eval_policy_gwx.py"
    config = robotwin_root / "policy" / "pi05" / "deploy_policy.yml"
    if not entry.is_file():
        raise FileNotFoundError(f"pi05 rollout entry not found: {entry}")
    if not config.is_file():
        raise FileNotFoundError(f"pi05 deploy config not found: {config}")
    env = _base_env(gpu, robotwin_root)
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
    cmd = [
        sys.executable,
        str(entry),
        "--config",
        str(config),
        "--overrides",
        "--task_name",
        task,
        "--task_config",
        task_config,
        "--train_config_name",
        train_config_name,
        "--model_name",
        model_name,
        "--ckpt_setting",
        model_name,
        "--checkpoint_id",
        str(checkpoint_id),
        "--pi0_step",
        str(pi0_step),
        "--seed",
        str(seed),
        "--policy_name",
        "pi05",
        "--instruction_type",
        instruction_type,
        "--save_rollout",
        _bool_token(True),
        "--rollout_save_root",
        str(save_root),
        "--rollout_save_only_success",
        _bool_token(save_only_success),
        "--rollout_resume",
        _bool_token(resume),
        "--success_delay_steps",
        str(success_delay_steps),
        "--eval_num_episodes",
        str(num_episodes),
        "--eval_output_dir",
        str(eval_output_dir),
        *extra,
    ]
    return Launch(cmd=cmd, env=env, cwd=robotwin_root)


def fastwam_policy_dir(fastwam_root: Path) -> Path:
    return fastwam_root / "experiments" / "robotwin" / "fastwam_policy"


def resolve_fastwam_dataset_stats(ckpt: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(explicit))).resolve())
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "dataset_stats.json was not found next to the checkpoint. "
        "Set DATASET_STATS_PATH=/path/to/dataset_stats.json."
    )


def build_fastwam(
    *,
    robotwin_root: Path,
    fastwam_root: Path,
    task: str,
    task_config: str,
    num_episodes: int,
    save_root: Path,
    seed: int,
    gpu: str,
    ckpt: Path,
    dataset_stats_path: Path,
    sim_task: str,
    replan_steps: int,
    num_inference_steps: int | None,
    action_horizon: int | None,
    mixed_precision: str,
    device: str,
    text_cfg_scale: float,
    skip_get_obs_within_replan: bool,
    save_only_success: bool,
    resume: bool,
    instruction_type: str,
    eval_output_dir: Path,
    extra: list[str],
) -> Launch:
    entry = robotwin_root / "script" / "eval_policy_gwx.py"
    policy_dir = fastwam_policy_dir(fastwam_root)
    config = policy_dir / "deploy_policy.yml"
    sim_cfg = fastwam_root / "configs" / "sim_robotwin.yaml"
    if not entry.is_file():
        raise FileNotFoundError(f"FastWAM rollout entry not found: {entry}")
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"FastWAM policy package not found: {policy_dir}")
    if not config.is_file():
        raise FileNotFoundError(f"FastWAM deploy config not found: {config}")
    if not sim_cfg.is_file():
        raise FileNotFoundError(f"FastWAM sim config not found: {sim_cfg}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    env = _base_env(gpu, robotwin_root)
    env["FASTWAM_ROOT"] = str(fastwam_root)
    policy_parent = str(policy_dir.parent)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = policy_parent if not previous else policy_parent + os.pathsep + previous

    cmd = [
        sys.executable,
        str(entry),
        "--config",
        str(config),
        "--overrides",
        "--task_name",
        task,
        "--task_config",
        task_config,
        "--ckpt_setting",
        str(ckpt),
        "--seed",
        str(seed),
        "--policy_name",
        "fastwam_policy",
        "--instruction_type",
        instruction_type,
        "--eval_num_episodes",
        str(num_episodes),
        "--eval_output_dir",
        str(eval_output_dir),
        "--sim_cfg_path",
        str(sim_cfg),
        "--sim_task",
        sim_task,
        "--mixed_precision",
        mixed_precision,
        "--device",
        device,
        "--dataset_stats_path",
        str(dataset_stats_path),
        "--replan_steps",
        str(replan_steps),
        "--text_cfg_scale",
        str(text_cfg_scale),
        "--skip_get_obs_within_replan",
        _bool_token(skip_get_obs_within_replan),
        "--save_rollout",
        _bool_token(True),
        "--rollout_save_root",
        str(save_root),
        "--rollout_save_only_success",
        _bool_token(save_only_success),
        "--rollout_resume",
        _bool_token(resume),
    ]
    if num_inference_steps is not None:
        cmd.extend(["--num_inference_steps", str(num_inference_steps)])
    if action_horizon is not None:
        cmd.extend(["--action_horizon", str(action_horizon)])
    cmd.extend(extra)
    return Launch(cmd=cmd, env=env, cwd=robotwin_root)
