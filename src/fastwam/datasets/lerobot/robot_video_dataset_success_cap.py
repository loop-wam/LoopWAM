"""RobotVideoDataset variant that supports success-aware episode capping.

This is a thin subclass of RobotVideoDataset. It behaves EXACTLY like the base
dataset except that the underlying LeRobot dataset is built with
BaseLerobotDatasetSuccessCap, which lets each entry of
``max_episodes_per_dataset`` optionally be a mapping::

    {success: S, failure: F}

to cap successful and failed episodes independently (keep ALL successes, trim
failures). Plain ``None`` / ``int`` caps keep working unchanged, so this class
is a safe drop-in replacement.

Use it by pointing a data config's ``train._target_`` (and ``val._target_``) at::

    fastwam.datasets.lerobot.robot_video_dataset_success_cap.RobotVideoDatasetSuccessCap

Everything else in the config stays identical.
"""
from __future__ import annotations

from .robot_video_dataset import RobotVideoDataset
from .base_lerobot_dataset_success_cap import BaseLerobotDatasetSuccessCap


class RobotVideoDatasetSuccessCap(RobotVideoDataset):
    """RobotVideoDataset backed by success-aware episode capping."""

    def _build_lerobot_dataset(self, **kwargs):
        return BaseLerobotDatasetSuccessCap(**kwargs)
