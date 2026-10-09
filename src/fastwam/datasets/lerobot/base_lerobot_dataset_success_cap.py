"""Success-aware episode capping for LeRobot datasets.

Drop-in subclass of BaseLerobotDataset that extends max_episodes_per_dataset so
each per-dir cap may be, in addition to the existing forms:

    * None      -> no cap (keep every episode for that dir)
    * int N     -> keep at most N episodes (success-agnostic, same as base)

a NEW mapping form:

    * {"success": S, "failure": F}
          -> keep at most S successful and F failed episodes for that dir.
             None for either side means "no cap for that class".

Motivation: rollout sources are dominated by FAILURE episodes (often 85%+ of
frames) whose action loss is masked anyway. When trimming a rollout source to
rebalance the expert-vs-rollout frame ratio, we want to KEEP ALL the (scarce)
successes and only cap the failures -- which a plain integer cap cannot do,
because the base cap simply takes the first N of a shuffled index list.

Design goals:
    * ZERO changes to base_lerobot_dataset.py -- everything lives here.
    * Preserve the base train/val split behaviour EXACTLY (same seed, same
      shuffle, same split index) so switching a dir between an int cap and a
      dict cap does not disturb the other dirs.
    * Read per-episode success from meta.episodes (same source used elsewhere,
      e.g. quality-score resolution).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_lerobot_dataset import BaseLerobotDataset
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _parse_success_value(value: Any) -> Optional[bool]:
    """Best-effort parse of an episode success field into a bool.

    Returns None when success cannot be determined, so unknown episodes form
    their own class and are never silently mislabeled.
    """
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


def _episode_success(meta: LeRobotDatasetMetadata, ep_idx: int) -> Optional[bool]:
    """Resolve the success flag of a single episode from dataset metadata.

    Mirrors the resolution order used elsewhere (RobotVideoDataset / precompute):
    explicit success-like fields first, then a success-<bool> token embedded in
    raw_file_name.
    """
    record = meta.episodes.get(ep_idx)
    if record is None:
        record = meta.episodes.get(str(ep_idx))
    if not record:
        return None

    for key in ("success", "eval_success", "episode_success"):
        parsed = _parse_success_value(record.get(key))
        if parsed is not None:
            return parsed

    raw_file_name = str(record.get("raw_file_name", ""))
    match = re.search(r"success[-_=](true|false|1|0)", raw_file_name, flags=re.IGNORECASE)
    if match is not None:
        parsed = _parse_success_value(match.group(1))
        if parsed is not None:
            return parsed
    return None


def _normalize_success_cap(value: Any) -> Any:
    """Normalize a single per-dir cap entry.

    Accepted forms:
        None                          -> None
        int / numeric str             -> int
        {"success": S, "failure": F}  -> {"success": S|None, "failure": F|None}
    """
    if value is None:
        return None
    # dict-like (plain dict or OmegaConf DictConfig both expose keys()/get()).
    if hasattr(value, "keys"):
        d = dict(value)
        unknown = set(d) - {"success", "failure"}
        if unknown:
            raise ValueError(
                f"`max_episodes_per_dataset` mapping has unknown keys {sorted(unknown)}; "
                "only 'success' and 'failure' are allowed."
            )

        def _coerce(x: Any) -> Optional[int]:
            if x is None:
                return None
            xi = int(x)
            if xi < 0:
                raise ValueError(f"cap must be >= 0, got {xi}")
            return xi

        return {"success": _coerce(d.get("success")), "failure": _coerce(d.get("failure"))}
    return int(value)


def _apply_success_cap(
    ep_indices: List[int],
    cap: Any,
    meta: LeRobotDatasetMetadata,
) -> List[int]:
    """Return a capped copy of ep_indices honouring cap.

    ep_indices is assumed to already be the (shuffled) train/val split for this
    dir, so order is preserved for int caps (identical to base behaviour).

    For dict caps we split the indices into success / failure / unknown buckets
    (order preserved), then keep the first S successes and first F failures.
    Unknown-success episodes are conservatively DROPPED under a dict cap (we
    cannot honour a success/failure budget for them); use an int cap to keep
    them.
    """
    if cap is None:
        return list(ep_indices)

    # Legacy int cap: identical to base (take first N of the split order).
    if not isinstance(cap, dict):
        n = int(cap)
        if n > 0:
            return list(ep_indices[:n])
        return list(ep_indices)

    s_cap = cap.get("success")
    f_cap = cap.get("failure")

    success_eps: List[int] = []
    failure_eps: List[int] = []
    unknown_eps: List[int] = []
    for ep in ep_indices:
        flag = _episode_success(meta, ep)
        if flag is True:
            success_eps.append(ep)
        elif flag is False:
            failure_eps.append(ep)
        else:
            unknown_eps.append(ep)

    if unknown_eps:
        logger.warning(
            "[success-cap] %s: %d episodes have UNKNOWN success and are dropped "
            "under a success/failure cap (use an int cap to keep them).",
            meta.repo_id,
            len(unknown_eps),
        )

    kept_success = success_eps if s_cap is None else success_eps[:s_cap]
    kept_failure = failure_eps if f_cap is None else failure_eps[:f_cap]

    logger.info(
        "[success-cap] %s: kept success %d/%d, failure %d/%d.",
        meta.repo_id,
        len(kept_success), len(success_eps),
        len(kept_failure), len(failure_eps),
    )

    kept = kept_success + kept_failure
    # Re-sort by original split order for deterministic downstream indexing,
    # independent of the success/failure grouping above.
    order = {ep: i for i, ep in enumerate(ep_indices)}
    kept.sort(key=lambda e: order[e])
    return kept


class BaseLerobotDatasetSuccessCap(BaseLerobotDataset):
    """BaseLerobotDataset with success-aware max_episodes_per_dataset.

    Behaviour is identical to the base class EXCEPT the construction of the
    per-dir episodes list, where a dict cap {success, failure} is honoured by
    keeping successes and failures separately.
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,
        obs_size: int = 1,
        past_obs_size: int = 0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        global_sample_stride: int = 1,
        image_sample_indices=None,
        max_episodes_per_dataset: Optional[Any] = None,
    ):
        # ---- guards (mirror base) ----
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None

        metas: List[LeRobotDatasetMetadata] = []
        for ds_dir in dataset_dirs:
            meta = LeRobotDatasetMetadata(repo_id=ds_dir, root=Path(ds_dir))
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]

        self.global_sample_stride = global_sample_stride
        self.image_sample_indices = list(image_sample_indices) if image_sample_indices is not None else None
        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        # ---- CAP NORMALIZATION (extended: allow dict entries) ----
        if max_episodes_per_dataset is None:
            max_eps_list: List[Any] = [None] * len(dataset_dirs)
        elif isinstance(max_episodes_per_dataset, int):
            max_eps_list = [int(max_episodes_per_dataset)] * len(dataset_dirs)
        else:
            raw_list = list(max_episodes_per_dataset)
            if len(raw_list) != len(dataset_dirs):
                raise ValueError(
                    "`max_episodes_per_dataset` length must match `dataset_dirs`: "
                    f"got {len(raw_list)} caps and {len(dataset_dirs)} dirs."
                )
            max_eps_list = [_normalize_success_cap(v) for v in raw_list]
        self.max_episodes_per_dataset = max_eps_list

        # ---- shape meta + delta_timestamps (identical to base) ----
        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps: Dict[str, Any] = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
            image_time_indices = (
                self.image_sample_indices
                if self.image_sample_indices is not None
                else range(-past_obs_size, -past_obs_size + obs_size)
            )
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in image_time_indices
            ]
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps
                for t in range(-past_action_size, -past_action_size + action_size)
            ]

        # ---- EPISODE SELECTION (split identical to base; cap extended) ----
        episodes: Dict[str, List[int]] = {}
        if val_set_proportion < 1e-6:
            for di, meta in enumerate(metas):
                ep = list(range(meta.total_episodes))
                ep = _apply_success_cap(ep, self.max_episodes_per_dataset[di], meta)
                episodes[meta.repo_id] = ep
        else:
            for di, meta in enumerate(metas):
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                episode_indices = list(range(meta.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)  # SAME seed/shuffle as base
                if self.is_training_set:
                    ep = [episode_indices[i] for i in range(split_idx)]
                else:
                    ep = [episode_indices[i] for i in range(split_idx, meta.total_episodes)]
                ep = _apply_success_cap(ep, self.max_episodes_per_dataset[di], meta)
                episodes[meta.repo_id] = ep

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )

        # ---- episode_data_index bookkeeping (identical to base) ----
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([d["from"] for d in episode_data_index]),
            "to": torch.cat([d["to"] for d in episode_data_index]),
        }