import hashlib
import os
import re
from typing import Any, Optional, Sequence, Union
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, ListConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
QUALITY_SUFFIX_VALUES = frozenset({"high", "low"})
QUALITY_SCORE_MIN = 1
QUALITY_SCORE_MAX = 5


def normalize_prompt_quality_suffix(value: Any) -> Optional[str]:
    """Parse a single quality suffix config value. None means no suffix."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none", "null"):
            return None
        if text not in QUALITY_SUFFIX_VALUES:
            raise ValueError(
                f"Invalid prompt quality suffix {value!r}. "
                f"Expected one of: {sorted(QUALITY_SUFFIX_VALUES)}, or null to disable."
            )
        return text
    raise ValueError(
        f"Invalid prompt quality suffix type {type(value).__name__}: {value!r}. "
        "Expected a string or null."
    )


def _normalize_quality_score_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{field_name}` score must be an integer from 1 to 5, got bool.")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("quality:"):
            text = text[len("quality:") :].strip()
        if text.startswith("score:"):
            text = text[len("score:") :].strip()
        text = text.rstrip(".")
        if not text.isdigit():
            raise ValueError(f"`{field_name}` score must be an integer from 1 to 5, got {value!r}.")
        score = int(text)
    else:
        raise ValueError(
            f"`{field_name}` score must be an integer from 1 to 5, got "
            f"{type(value).__name__}: {value!r}."
        )
    if score < QUALITY_SCORE_MIN or score > QUALITY_SCORE_MAX:
        raise ValueError(f"`{field_name}` score must be in [1, 5], got {score}.")
    return score


def normalize_prompt_quality_score(value: Any, field_name: str = "dataset_prompt_quality_score"):
    """Parse numeric quality-score config.

    Accepted forms:
      - null: disabled
      - 5 / "5" / "Quality: 5": fixed score
      - {success: 4, failure: 1}: score selected from episode success metadata
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        if "success" not in value or "failure" not in value:
            raise ValueError(
                f"`{field_name}` mapping must contain `success` and `failure` scores, got {value!r}."
            )
        return {
            "success": _normalize_quality_score_value(value["success"], f"{field_name}.success"),
            "failure": _normalize_quality_score_value(value["failure"], f"{field_name}.failure"),
        }
    return _normalize_quality_score_value(value, field_name)


def _coerce_config_sequence(values: Any, field_name: str) -> list[Any]:
    """Convert Hydra/OmegaConf list configs to a plain Python list."""
    if values is None:
        raise ValueError(f"`{field_name}` is None.")
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple, ListConfig)):
        return list(values)
    if isinstance(values, DictConfig):
        container = OmegaConf.to_container(values, resolve=True)
        if not isinstance(container, list):
            raise ValueError(
                f"`{field_name}` must be a list matching `dataset_dirs`, "
                f"got {type(container).__name__}."
            )
        return container
    raise ValueError(
        f"`{field_name}` must be null, a string, or a list matching `dataset_dirs`, "
        f"got {type(values).__name__}."
    )


def normalize_prompt_quality_suffix_list(
    values: Any,
    num_dataset_dirs: int,
    field_name: str = "dataset_prompt_quality_suffix",
) -> list[Optional[str]]:
    """Align per-dataset quality suffix settings with ``dataset_dirs``."""
    if values is None:
        return [None] * num_dataset_dirs
    values = _coerce_config_sequence(values, field_name)
    if len(values) != num_dataset_dirs:
        raise ValueError(
            f"`{field_name}` length must match `dataset_dirs`: "
            f"got {len(values)} suffix entries and {num_dataset_dirs} dirs."
        )
    return [normalize_prompt_quality_suffix(v) for v in values]


def normalize_prompt_quality_score_list(
    values: Any,
    num_dataset_dirs: int,
    field_name: str = "dataset_prompt_quality_score",
) -> list[Any]:
    """Align per-dataset numeric quality-score settings with ``dataset_dirs``."""
    if values is None:
        return [None] * num_dataset_dirs
    values = _coerce_config_sequence(values, field_name)
    if len(values) != num_dataset_dirs:
        raise ValueError(
            f"`{field_name}` length must match `dataset_dirs`: "
            f"got {len(values)} score entries and {num_dataset_dirs} dirs."
        )
    return [
        normalize_prompt_quality_score(v, field_name=f"{field_name}[{idx}]")
        for idx, v in enumerate(values)
    ]


def normalize_text_embedding_cache_dirs(
    values: Any,
    num_dataset_dirs: int,
    field_name: str = "text_embedding_cache_dir",
) -> Optional[list[str]]:
    """Align text embedding cache dirs with ``dataset_dirs``.

    A single path is treated as the legacy shared cache for every dataset root.
    """
    if values is None:
        return None
    cache_dirs = _coerce_config_sequence(values, field_name)
    if len(cache_dirs) == 1:
        cache_dirs = cache_dirs * num_dataset_dirs
    if len(cache_dirs) != num_dataset_dirs:
        raise ValueError(
            f"`{field_name}` length must be 1 or match `dataset_dirs`: "
            f"got {len(cache_dirs)} cache dirs and {num_dataset_dirs} dirs."
        )
    normalized_dirs = []
    for cache_dir in cache_dirs:
        if cache_dir is None:
            raise ValueError(f"`{field_name}` entries must be non-empty paths.")
        cache_dir = str(cache_dir).strip()
        if not cache_dir:
            raise ValueError(f"`{field_name}` entries must be non-empty paths.")
        normalized_dirs.append(cache_dir)
    return normalized_dirs


def build_robotwin_prompt(
    task: str,
    quality_suffix: Optional[str] = None,
    quality_score: Any = None,
) -> str:
    """Build the full text prompt used for T5 encoding and cache lookup."""
    base = DEFAULT_PROMPT.format(task=task)
    if quality_score is not None:
        score = normalize_prompt_quality_score(quality_score, field_name="quality_score")
        if isinstance(score, dict):
            raise ValueError("`quality_score` must be resolved to a fixed integer before building a prompt.")
        return f"{base} Quality: {score}."
    if quality_suffix is None:
        return base
    suffix = normalize_prompt_quality_suffix(quality_suffix)
    if suffix is None:
        return base
    return f"{base} Quality: {suffix}."


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        dataset_supervise_action: Optional[Sequence[bool]] = None,
        dataset_prompt_quality_suffix: Optional[Union[str, Sequence[Optional[str]]]] = None,
        dataset_prompt_quality_score: Optional[Any] = None,
        decode_video_sampled_frames_only: bool = False,
    ):
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))
        self.decode_video_sampled_frames_only = bool(decode_video_sampled_frames_only)

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_sample_indices=self.video_sample_indices if self.decode_video_sampled_frames_only else None,
        )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_dirs = normalize_text_embedding_cache_dirs(
            text_embedding_cache_dir,
            len(dataset_dirs),
        )
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        if dataset_supervise_action is not None:
            flags = [
                bool(x)
                for x in _coerce_config_sequence(
                    dataset_supervise_action, "dataset_supervise_action"
                )
            ]
            if len(flags) != len(dataset_dirs):
                raise ValueError(
                    "`dataset_supervise_action` length must match `dataset_dirs`: "
                    f"got {len(flags)} flags and {len(dataset_dirs)} dirs."
                )
            self.dataset_supervise_action = flags
        else:
            self.dataset_supervise_action = None

        self.dataset_prompt_quality_suffix = normalize_prompt_quality_suffix_list(
            dataset_prompt_quality_suffix,
            len(dataset_dirs),
            field_name="dataset_prompt_quality_suffix",
        )
        self.dataset_prompt_quality_score = normalize_prompt_quality_score_list(
            dataset_prompt_quality_score,
            len(dataset_dirs),
            field_name="dataset_prompt_quality_score",
        )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.decode_video_sampled_frames_only:
                processor.image_obs_steps = len(self.video_sample_indices)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.decode_video_sampled_frames_only:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.decode_video_sampled_frames_only:
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if not self.decode_video_sampled_frames_only:
            image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction

        quality_suffix = None
        quality_score = None
        if any(s is not None for s in self.dataset_prompt_quality_score):
            score_spec = self._get_dataset_aligned_value(
                sample,
                self.dataset_prompt_quality_score,
                "dataset_prompt_quality_score",
            )
            quality_score = self._resolve_prompt_quality_score(score_spec, sample)
        elif any(s is not None for s in self.dataset_prompt_quality_suffix):
            quality_suffix = self._get_dataset_aligned_value(
                sample,
                self.dataset_prompt_quality_suffix,
                "dataset_prompt_quality_suffix",
            )

        instruction = build_robotwin_prompt(task, quality_suffix, quality_score=quality_score)

        cache_dir = self._get_text_embedding_cache_dir(sample)
        context, context_mask = self._get_cached_text_context(instruction, cache_dir)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.dataset_supervise_action is not None:
            di = sample.get("dataset_index", None)
            if di is None:
                raise ValueError(
                    "`dataset_supervise_action` requires a multi-root LeRobot setup that provides "
                    "`dataset_index` on each frame (use multiple entries in `dataset_dirs`)."
                )
            di_i = int(di.item()) if torch.is_tensor(di) else int(di)
            supervise = bool(self.dataset_supervise_action[di_i])
            data["supervise_action"] = torch.tensor(1.0 if supervise else 0.0, dtype=torch.float32)

        return data

    def _get_dataset_aligned_value(self, sample, values: Sequence[Any], field_name: str):
        if len(values) == 1:
            return values[0]
        di = sample.get("dataset_index", None)
        if di is None:
            raise ValueError(
                f"`{field_name}` with multiple `dataset_dirs` requires "
                "`dataset_index` on each frame."
            )
        di_i = int(di.item()) if torch.is_tensor(di) else int(di)
        if di_i < 0 or di_i >= len(values):
            raise IndexError(f"`dataset_index` {di_i} is out of range for `{field_name}`.")
        return values[di_i]

    def _get_sample_dataset_index(self, sample) -> int:
        di = sample.get("dataset_index", 0)
        return int(di.item()) if torch.is_tensor(di) else int(di)

    def _get_sample_episode_record(self, sample) -> dict[str, Any]:
        episode_index = sample.get("episode_index", None)
        if episode_index is None:
            raise ValueError(
                "`dataset_prompt_quality_score` success/failure mapping requires "
                "`episode_index` on each frame."
            )
        ep_i = int(episode_index.item()) if torch.is_tensor(episode_index) else int(episode_index)
        di_i = self._get_sample_dataset_index(sample)
        dataset = self.lerobot_dataset.multi_dataset._datasets[di_i]
        return dataset.meta.episodes.get(ep_i) or dataset.meta.episodes.get(str(ep_i)) or {}

    @staticmethod
    def _parse_success_value(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "success", "succeeded"}:
                return True
            if text in {"false", "0", "no", "n", "failure", "failed"}:
                return False
        return None

    def _get_sample_episode_success(self, sample) -> bool:
        record = self._get_sample_episode_record(sample)
        for key in ("success", "eval_success", "episode_success"):
            parsed = self._parse_success_value(record.get(key))
            if parsed is not None:
                return parsed

        raw_file_name = str(record.get("raw_file_name", ""))
        match = re.search(r"success[-_=](true|false|1|0)", raw_file_name, flags=re.IGNORECASE)
        if match is not None:
            parsed = self._parse_success_value(match.group(1))
            if parsed is not None:
                return parsed

        raise ValueError(
            "Unable to infer episode success for `dataset_prompt_quality_score`. "
            "Expected `success` in meta/episodes.jsonl or `success-true/false` in `raw_file_name`."
        )

    def _resolve_prompt_quality_score(self, score_spec: Any, sample) -> Optional[int]:
        if score_spec is None:
            return None
        if isinstance(score_spec, dict):
            success = self._get_sample_episode_success(sample)
            return int(score_spec["success"] if success else score_spec["failure"])
        return int(score_spec)

    def _get_text_embedding_cache_dir(self, sample):
        if self.text_embedding_cache_dirs is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        if all(cache_dir == self.text_embedding_cache_dirs[0] for cache_dir in self.text_embedding_cache_dirs):
            return self.text_embedding_cache_dirs[0]

        di = sample.get("dataset_index", None)
        if di is None:
            raise ValueError(
                "`text_embedding_cache_dir` with multiple entries requires "
                "`dataset_index` on each frame."
            )
        di_i = int(di.item()) if torch.is_tensor(di) else int(di)
        if di_i < 0 or di_i >= len(self.text_embedding_cache_dirs):
            raise IndexError(
                f"`dataset_index` {di_i} is out of range for "
                f"{len(self.text_embedding_cache_dirs)} text embedding cache dirs."
            )
        return self.text_embedding_cache_dirs[di_i]

    def _get_cached_text_context(self, prompt: str, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
