import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import (
    build_robotwin_prompt,
    normalize_prompt_quality_score,
    normalize_prompt_quality_suffix,
)
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)


def _maybe_silence_logging() -> None:
    if os.environ.get("FASTWAM_QUIET", "0").strip().lower() not in {"1", "true", "yes"}:
        return
    logging.getLogger().setLevel(logging.CRITICAL)
    logger.setLevel(logging.CRITICAL)


def _binarize_robotwin_gripper(
    action: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
    mid_delta: float = 0.05,
) -> np.ndarray:
    """Hysteresis-style gripper post-process for RoboTwin qpos action.

    Layout is [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)].
    Values < low_threshold -> 0 (closed); values > high_threshold -> 1 (open);
    values in [low_threshold, high_threshold] are reduced by mid_delta.
    """
    action = np.asarray(action, dtype=np.float32).copy()
    if action.ndim == 2:
        for i in range(action.shape[0]):
            action[i] = _binarize_robotwin_gripper(
                action[i],
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                mid_delta=mid_delta,
            )
        return action
    if action.ndim != 1:
        raise ValueError(f"Expected action shape [D] or [T,D], got {action.shape}")
    dim = int(action.shape[0])
    if dim < 2 or dim % 2 != 0:
        raise ValueError(f"Expected even dual-arm action dim (>=2), got {dim}")
    if float(low_threshold) > float(high_threshold):
        raise ValueError(
            f"low_threshold ({low_threshold}) must be <= high_threshold ({high_threshold})"
        )
    left_gripper_idx = dim // 2 - 1
    right_gripper_idx = dim - 1
    for idx in (left_gripper_idx, right_gripper_idx):
        value = float(action[idx])
        if value < low_threshold:
            action[idx] = 0.0
        elif value > high_threshold:
            action[idx] = 1.0
        else:
            action[idx] = float(np.clip(value - float(mid_delta), 0.0, 1.0))
    return action


def _smooth_action_chunk_dreamzero(
    action_chunk: np.ndarray,
    *,
    upsample_factor: int = 2,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    """DreamZero action-chunk smoothing: 2x cubic upsample → Savitzky–Golay → downsample.

    Applied on the full predicted chunk (e.g. T=64) before taking the first
    ``replan_steps`` actions for execution, so the filter sees more temporal context.
    """
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk [T, D], got {action_chunk.shape}")
    t_len, _ = action_chunk.shape
    if t_len < 2:
        return action_chunk.astype(np.float32)

    upsample_factor = int(max(1, upsample_factor))
    t = np.arange(t_len, dtype=np.float64)
    t_up = np.linspace(0.0, t_len - 1, t_len * upsample_factor, dtype=np.float64)
    upsampled = interp1d(t, action_chunk, axis=0, kind="cubic", assume_sorted=True)(t_up)

    up_len = upsampled.shape[0]
    # window must be odd, <= length, and > polyorder
    max_window = up_len if up_len % 2 == 1 else up_len - 1
    window = min(int(window_length), max_window)
    if window % 2 == 0:
        window -= 1
    if window <= int(polyorder):
        logger.warning(
            "Skip action-chunk smoothing: upsampled length=%d too short for "
            "window=%d polyorder=%d.",
            up_len,
            window_length,
            polyorder,
        )
        return action_chunk.astype(np.float32)

    smoothed_up = savgol_filter(
        upsampled, window_length=window, polyorder=int(polyorder), axis=0, mode="interp"
    )
    smoothed = interp1d(t_up, smoothed_up, axis=0, kind="cubic", assume_sorted=True)(t)
    return smoothed.astype(np.float32)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        multi_gpu: bool = False,
        video_device: str = "cuda:1",
        action_device: str = "cuda:0",
        prompt_quality_suffix: Optional[str] = None,
        prompt_quality_score: Any = None,
        save_denoised_video: bool = False,
        denoised_video_save_root: Optional[Path] = None,
        denoised_video_fps: int = 8,
        binarize_gripper: bool = False,
        gripper_binarize_low: float = 0.3,
        gripper_binarize_high: float = 0.7,
        gripper_binarize_mid_delta: float = 0.05,
        smooth_action_chunk: bool = False,
        smooth_upsample_factor: int = 2,
        smooth_savgol_window: int = 21,
        smooth_savgol_polyorder: int = 3,
    ) -> None:
        _maybe_silence_logging()
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        # Load on CPU first when splitting across GPUs: avoids stacking video+action
        # weights on cuda:0 before enable_multi_gpu_inference (common OOM on 24GB cards).
        load_device = "cpu" if multi_gpu else device
        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=load_device)
        self.model.load_checkpoint(checkpoint_path)
        if multi_gpu:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError(
                    "multi_gpu inference requires at least 2 visible CUDA devices."
                )
            self.model.enable_multi_gpu_inference(
                video_device=video_device,
                action_device=action_device,
            )
        else:
            self.model = self.model.to(device).eval()
        self.model.eval()
        self._multi_gpu = bool(multi_gpu)

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.prompt_quality_suffix = normalize_prompt_quality_suffix(prompt_quality_suffix)
        self.prompt_quality_score = normalize_prompt_quality_score(
            prompt_quality_score,
            field_name="prompt_quality_score",
        )
        if isinstance(self.prompt_quality_score, dict):
            raise ValueError(
                "`prompt_quality_score` success/failure mapping is only supported during training; "
                "pass a fixed integer score for evaluation."
            )
        self.save_denoised_video = bool(save_denoised_video)
        self.denoised_video_save_root = denoised_video_save_root
        self.denoised_video_fps = int(denoised_video_fps)
        self.binarize_gripper = bool(binarize_gripper)
        self.gripper_binarize_low = float(gripper_binarize_low)
        self.gripper_binarize_high = float(gripper_binarize_high)
        self.gripper_binarize_mid_delta = float(gripper_binarize_mid_delta)
        self.smooth_action_chunk = bool(smooth_action_chunk)
        self.smooth_upsample_factor = int(smooth_upsample_factor)
        self.smooth_savgol_window = int(smooth_savgol_window)
        self.smooth_savgol_polyorder = int(smooth_savgol_polyorder)
        self._video_frame_stride = max(
            1,
            self.action_horizon // max(1, self._num_video_frames - 1),
        )

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._denoised_video_chunks: list[list[Image.Image]] = []
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d | "
            "multi_gpu=%s | prompt_quality_suffix=%s | prompt_quality_score=%s | "
            "binarize_gripper=%s (low=%.2f, high=%.2f, mid_delta=%.2f) | smooth_action_chunk=%s "
            "(upsample=%d, window=%d, poly=%d)",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self._multi_gpu,
            self.prompt_quality_suffix,
            self.prompt_quality_score,
            self.binarize_gripper,
            self.gripper_binarize_low,
            self.gripper_binarize_high,
            self.gripper_binarize_mid_delta,
            self.smooth_action_chunk,
            self.smooth_upsample_factor,
            self.smooth_savgol_window,
            self.smooth_savgol_polyorder,
        )

    def _denoised_video_path(self, *, success: Optional[bool] = None) -> Path:
        if self.denoised_video_save_root is None:
            raise ValueError("`denoised_video_save_root` must be set when saving denoised videos.")
        suffix = "" if success is None else f"_success-{str(bool(success)).lower()}"
        return self.denoised_video_save_root / (
            f"episode{self.episode_count:06d}_denoised_exec{suffix}.mp4"
        )

    def _record_denoised_video_prediction(self, video: list[Image.Image]) -> None:
        if len(video) <= 1:
            return
        # Skip the conditioning observation at t0. The remaining frames are future predictions.
        self._denoised_video_chunks.append([frame.copy() for frame in video[1:]])

    def _denoised_video_frames_for_replan_window(self) -> list[Image.Image]:
        frames: list[Image.Image] = []
        frames_per_chunk = max(1, self.replan_steps // self._video_frame_stride)
        for chunk_frames in self._denoised_video_chunks:
            keep_frames = min(len(chunk_frames), frames_per_chunk)
            if keep_frames > 0:
                frames.extend(chunk_frames[:keep_frames])
        return frames

    def finalize_episode(self, *, success: Optional[bool] = None) -> None:
        if not self.save_denoised_video:
            return
        frames = self._denoised_video_frames_for_replan_window()
        if len(frames) == 0:
            logger.warning("No denoised video frames to save for episode %d.", self.episode_count)
            return
        video_path = self._denoised_video_path(success=success)
        save_mp4(frames, str(video_path), fps=self.denoised_video_fps)
        logger.info("Saved replan-window denoised video to %s", video_path)

    @property
    def _input_device(self) -> torch.device:
        if self._multi_gpu:
            return self.model._infer_video_device()
        return self.model.device

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self._input_device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = build_robotwin_prompt(
            instruction,
            self.prompt_quality_suffix,
            quality_score=self.prompt_quality_score,
        )
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            if self.save_denoised_video:
                joint_kwargs = dict(infer_kwargs)
                joint_signature = inspect.signature(self.model.infer_joint).parameters
                if "num_video_frames" in joint_signature:
                    joint_kwargs["num_video_frames"] = int(self._num_video_frames)
                if "test_action_with_infer_action" in joint_signature:
                    joint_kwargs["test_action_with_infer_action"] = False
                pred = self.model.infer_joint(**joint_kwargs)
            else:
                action_kwargs = dict(infer_kwargs)
                if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
                    action_kwargs["num_video_frames"] = int(self._num_video_frames)
                pred = self.model.infer_action(**action_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        # Smooth the full predicted chunk (e.g. 64), then `_fill_action_queue`
        # keeps only the first `replan_steps` (e.g. 32) for execution.
        if self.smooth_action_chunk:
            action_chunk = _smooth_action_chunk_dreamzero(
                action_chunk,
                upsample_factor=self.smooth_upsample_factor,
                window_length=self.smooth_savgol_window,
                polyorder=self.smooth_savgol_polyorder,
            )
        if self.binarize_gripper:
            action_chunk = _binarize_robotwin_gripper(
                action_chunk,
                low_threshold=self.gripper_binarize_low,
                high_threshold=self.gripper_binarize_high,
                mid_delta=self.gripper_binarize_mid_delta,
            )
        if self.save_denoised_video:
            self._record_denoised_video_prediction(pred["video"])
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self._denoised_video_chunks.clear()
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    multi_gpu = _parse_bool(usr_args.get("multi_gpu", cfg.EVALUATION.get("multi_gpu", False)))
    video_device = str(usr_args.get("video_device", cfg.EVALUATION.get("video_device", "cuda:0")))
    action_device = str(usr_args.get("action_device", cfg.EVALUATION.get("action_device", "cuda:1")))
    save_denoised_video = _parse_bool(
        usr_args.get("save_denoised_video", cfg.EVALUATION.get("save_denoised_video", False))
    )
    denoised_video_fps = int(usr_args.get("denoised_video_fps", cfg.EVALUATION.get("denoised_video_fps", 8)))
    denoised_video_save_root = usr_args.get(
        "denoised_video_save_root", cfg.EVALUATION.get("denoised_video_save_root")
    )
    if save_denoised_video:
        if multi_gpu:
            raise ValueError(
                "`save_denoised_video=true` currently requires `multi_gpu=false` because joint video+action "
                "sampling runs both experts on one device."
            )
        if _is_none_like(denoised_video_save_root):
            eval_output_dir = usr_args.get("eval_output_dir")
            if _is_none_like(eval_output_dir):
                raise ValueError(
                    "`save_denoised_video=true` requires `eval_output_dir` or `denoised_video_save_root`."
                )
            denoised_video_save_root = (
                Path(str(eval_output_dir)) / "denoised_videos" / str(usr_args.get("task_config", "unknown"))
            )
        else:
            denoised_video_save_root = (
                Path(str(denoised_video_save_root))
                / str(usr_args.get("task_name", "unknown"))
                / str(usr_args.get("task_config", "unknown"))
            )
        denoised_video_save_root.mkdir(parents=True, exist_ok=True)
    else:
        denoised_video_save_root = None

    prompt_quality_suffix = usr_args.get(
        "prompt_quality_suffix", cfg.EVALUATION.get("prompt_quality_suffix")
    )
    if _is_none_like(prompt_quality_suffix):
        prompt_quality_suffix = None

    prompt_quality_score = usr_args.get(
        "prompt_quality_score", cfg.EVALUATION.get("prompt_quality_score")
    )
    if _is_none_like(prompt_quality_score):
        prompt_quality_score = None

    binarize_gripper = _parse_bool(
        usr_args.get("binarize_gripper", cfg.EVALUATION.get("binarize_gripper", False))
    )
    gripper_binarize_low = float(
        usr_args.get(
            "gripper_binarize_low",
            cfg.EVALUATION.get("gripper_binarize_low", 0.3),
        )
    )
    gripper_binarize_high = float(
        usr_args.get(
            "gripper_binarize_high",
            cfg.EVALUATION.get("gripper_binarize_high", 0.7),
        )
    )
    gripper_binarize_mid_delta = float(
        usr_args.get(
            "gripper_binarize_mid_delta",
            cfg.EVALUATION.get("gripper_binarize_mid_delta", 0.05),
        )
    )
    smooth_action_chunk = _parse_bool(
        usr_args.get("smooth_action_chunk", cfg.EVALUATION.get("smooth_action_chunk", False))
    )
    smooth_upsample_factor = int(
        usr_args.get(
            "smooth_upsample_factor",
            cfg.EVALUATION.get("smooth_upsample_factor", 2),
        )
    )
    smooth_savgol_window = int(
        usr_args.get(
            "smooth_savgol_window",
            cfg.EVALUATION.get("smooth_savgol_window", 21),
        )
    )
    smooth_savgol_polyorder = int(
        usr_args.get(
            "smooth_savgol_polyorder",
            cfg.EVALUATION.get("smooth_savgol_polyorder", 3),
        )
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        multi_gpu=multi_gpu,
        video_device=video_device,
        action_device=action_device,
        prompt_quality_suffix=prompt_quality_suffix,
        prompt_quality_score=prompt_quality_score,
        save_denoised_video=save_denoised_video,
        denoised_video_save_root=denoised_video_save_root,
        denoised_video_fps=denoised_video_fps,
        binarize_gripper=binarize_gripper,
        gripper_binarize_low=gripper_binarize_low,
        gripper_binarize_high=gripper_binarize_high,
        gripper_binarize_mid_delta=gripper_binarize_mid_delta,
        smooth_action_chunk=smooth_action_chunk,
        smooth_upsample_factor=smooth_upsample_factor,
        smooth_savgol_window=smooth_savgol_window,
        smooth_savgol_polyorder=smooth_savgol_polyorder,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()


def finalize_model_episode(model, episode_idx: Optional[int] = None, success: Optional[bool] = None):
    del episode_idx
    model.finalize_episode(success=success)
