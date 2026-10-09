#!/usr/bin/env python3
"""Collect RoboTwin hdf5 rollouts with xvla, pi05, or fastwam.

Code locations come from the environment:

  ROBOTWIN_ROOT     RoboTwin checkout (run_xvla.py, script/eval_policy_gwx.py, policy/pi05)
  XVLA_REPO_ROOT    X-VLA repo that contains the models package
  MODEL_PATH        XVLA checkpoint directory
  FASTWAM_ROOT      FastWAM checkout (experiments/robotwin/fastwam_policy, configs/)
  CKPT              FastWAM checkpoint file
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch import count_hdf5_episodes, load_task_list, require_env_dir, resolve_user_path, run_launch
from policies import build_fastwam, build_pi05, build_xvla, resolve_fastwam_dataset_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rollout data for xvla, pi05, or fastwam.")
    parser.add_argument("--policy", required=True, choices=["xvla", "pi05", "fastwam"])
    parser.add_argument("--task-list", default="task_list.txt")
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--task-config", action="append", dest="task_configs")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--instruction-type", default=None)
    parser.add_argument("--save-only-success", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--success-delay-steps", type=int, default=None)
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--eval-log-dir", default=None)
    parser.add_argument("--output-path", default=None)

    parser.add_argument("--train-config-name", default="pi05_robotwin2")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint-id", type=int, default=2000)
    parser.add_argument("--pi0-step", type=int, default=50)

    parser.add_argument("--sim-task", default="robotwin_uncond_3cam_384_1e-4")
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--skip-get-obs-within-replan", action="store_true")

    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    return args


def _required_text(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required.")
    return value


def _defaults(args: argparse.Namespace) -> None:
    args.robotwin_root = require_env_dir("ROBOTWIN_ROOT")
    if not args.task_configs:
        args.task_configs = ["demo_clean"]
    if args.seed is None:
        args.seed = 74 if args.policy == "xvla" else 76
    if args.success_delay_steps is None:
        args.success_delay_steps = 30 if args.policy != "fastwam" else 0
    if args.instruction_type is None and args.policy != "xvla":
        args.instruction_type = "unseen"
    if args.num_episodes <= 0:
        raise ValueError(f"--num-episodes must be > 0, got {args.num_episodes}")

    if args.policy == "xvla":
        args.xvla_repo_root = str(require_env_dir("XVLA_REPO_ROOT"))
        args.model_path = str(require_env_dir("MODEL_PATH"))
        processor = os.environ.get("PROCESSOR_PATH", "").strip()
        args.processor_path = str(require_env_dir("PROCESSOR_PATH")) if processor else args.model_path
    elif args.policy == "fastwam":
        args.fastwam_root = require_env_dir("FASTWAM_ROOT")
        args.ckpt = resolve_user_path(_required_text("CKPT"))
        stats = os.environ.get("DATASET_STATS_PATH", "").strip()
        args.dataset_stats_path = stats or None


def _count(save_root: Path, task: str, task_config: str) -> int:
    return count_hdf5_episodes(save_root, task, task_config)


def _launch_for(args: argparse.Namespace, task: str, task_config: str, need: int, save_root: Path):
    log_dir = save_root / "_logs" / task_config / task
    if args.policy == "xvla":
        eval_log_dir = resolve_user_path(args.eval_log_dir) if args.eval_log_dir else save_root / "_logs" / "xvla"
        output_path = resolve_user_path(args.output_path) if args.output_path else save_root / "_preview.mp4"
        if not args.dry_run:
            eval_log_dir.mkdir(parents=True, exist_ok=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return build_xvla(
            robotwin_root=args.robotwin_root,
            task=task,
            task_config=task_config,
            num_episodes=need,
            save_root=save_root,
            seed=args.seed,
            gpu=args.gpu,
            model_path=args.model_path,
            processor_path=args.processor_path,
            xvla_repo_root=args.xvla_repo_root,
            device=args.device,
            inference_steps=args.inference_steps,
            success_delay_steps=args.success_delay_steps,
            save_only_success=args.save_only_success,
            resume=not args.no_resume,
            torch_dtype=args.torch_dtype,
            eval_log_dir=eval_log_dir,
            output_path=output_path,
            instruction_type=args.instruction_type,
            extra=args.extra,
        )
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    if args.policy == "pi05":
        return build_pi05(
            robotwin_root=args.robotwin_root,
            task=task,
            task_config=task_config,
            num_episodes=need,
            save_root=save_root,
            seed=args.seed,
            gpu=args.gpu,
            train_config_name=args.train_config_name,
            model_name=args.model_name or task_config,
            checkpoint_id=args.checkpoint_id,
            pi0_step=args.pi0_step,
            success_delay_steps=args.success_delay_steps,
            save_only_success=args.save_only_success,
            resume=not args.no_resume,
            instruction_type=args.instruction_type,
            eval_output_dir=log_dir,
            extra=args.extra,
        )
    stats = resolve_fastwam_dataset_stats(args.ckpt, args.dataset_stats_path)
    return build_fastwam(
        robotwin_root=args.robotwin_root,
        fastwam_root=args.fastwam_root,
        task=task,
        task_config=task_config,
        num_episodes=need,
        save_root=save_root,
        seed=args.seed,
        gpu=args.gpu,
        ckpt=args.ckpt,
        dataset_stats_path=stats,
        sim_task=args.sim_task,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        mixed_precision=args.mixed_precision,
        device=args.device,
        text_cfg_scale=args.text_cfg_scale,
        skip_get_obs_within_replan=args.skip_get_obs_within_replan,
        save_only_success=args.save_only_success,
        resume=not args.no_resume,
        instruction_type=args.instruction_type,
        eval_output_dir=log_dir,
        extra=args.extra,
    )


def main() -> int:
    args = _parse_args()
    _defaults(args)
    task_list = resolve_user_path(args.task_list)
    save_root = resolve_user_path(args.save_root)
    tasks = load_task_list(task_list)
    if not args.dry_run:
        save_root.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print(f"Policy: {args.policy}")
    print(f"RoboTwin: {args.robotwin_root}")
    if args.policy == "xvla":
        print(f"XVLA repo: {args.xvla_repo_root}")
        print(f"Model: {args.model_path}")
    if args.policy == "fastwam":
        print(f"FastWAM: {args.fastwam_root}")
        print(f"Checkpoint: {args.ckpt}")
        print(f"Sim task: {args.sim_task}")
    print(f"Task list: {task_list}")
    print(f"Save root: {save_root}")
    print(f"Task configs: {' '.join(args.task_configs)}")
    print(f"Episodes per (task, config): {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print(f"GPU: {args.gpu}")
    print("========================================")

    failed: list[str] = []
    for task_config in args.task_configs:
        for task in tasks:
            existing = _count(save_root, task, task_config)
            label = f"{task} / {task_config}"
            print("========================================")
            print(f"Task: {label}")
            print(f"Existing episodes: {existing}")
            print(f"Target episodes: {args.num_episodes}")
            if existing >= args.num_episodes:
                print(f"Skip {label}: already has {existing} episodes.")
                continue
            need = args.num_episodes - existing
            print(f"Need to generate: {need}")
            launch = _launch_for(args, task, task_config, need, save_root)
            if args.dry_run:
                print(" ".join(launch.cmd))
                continue
            code = run_launch(launch)
            after = _count(save_root, task, task_config)
            print(f"Episodes after run: {after}")
            if code != 0:
                print(f"Task failed ({code}): {label}")
                failed.append(label)

    if failed:
        print("Failed tasks:")
        for label in failed:
            print(f"  {label}")
        return 1
    if args.dry_run:
        print("Dry run finished. No rollout was started.")
        return 0
    print(f"All done. Data saved under: {save_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
