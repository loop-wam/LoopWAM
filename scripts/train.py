import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def _enable_rollout_recipe() -> None:
    """Match the released real-robot rollout recipes that current wa_ali dataset code rejects.

    Text caches use Quality scores in [0, 12], including halves such as 7.5, and
    the literal config value ``episode`` (read ``score`` from meta/episodes.jsonl).
    Expert subtask prompts are switched from offline annotations:

    - UR5 shred_paper: after the gripper-release frame, ``Retract gripper.``
    - ARX5 press_the_button: four E3-valley segments
    - ARX5 water_the_flowers: split at qx_peak
    - Aloha stamp_positioning: split at right_peak_1 and left_peak

    ``FASTWAM_EXCLUDE_RAW_PREFIXES`` drops episodes whose ``raw_file_name``
    starts with one of the comma-separated prefixes.

    RoboTwin 2.0 and RoboCasa365 use quality scores 1-5 and stay on the stock dataset code.
    """
    import os
    import sys

    task_name = ""
    for arg in sys.argv[1:]:
        if arg.startswith("task="):
            task_name = arg.split("=", 1)[1]
            break
    if not task_name:
        task_name = os.environ.get("FASTWAM_TASK", "")
    task_name = task_name.strip()
    if task_name.endswith(".yaml"):
        task_name = task_name[:-5]
    if task_name.startswith("robotwin") or task_name.startswith("robocasa"):
        return

    import json
    from pathlib import Path

    import fastwam.datasets.lerobot.robot_video_dataset as ds
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import MultiLeRobotDataset
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor

    ds.QUALITY_SCORE_MIN = 0
    ds.QUALITY_SCORE_MAX = 12

    def _as_float(value, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"`{field_name}` score must be a number in [0, 12], got bool.")
        if isinstance(value, (int, float)):
            score = float(value)
        elif isinstance(value, str):
            text = value.strip().lower().rstrip(".")
            for prefix in ("quality:", "score:"):
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
            score = float(text)
        else:
            raise ValueError(
                f"`{field_name}` score must be a number in [0, 12], got {type(value).__name__}."
            )
        if score < ds.QUALITY_SCORE_MIN or score > ds.QUALITY_SCORE_MAX:
            raise ValueError(f"`{field_name}` score must be in [0, 12], got {score}.")
        return score

    def _format_score(score: float) -> str:
        if abs(score - round(score)) < 1e-6:
            return str(int(round(score)))
        return f"{score:.1f}".rstrip("0").rstrip(".")

    orig_normalize = ds.normalize_prompt_quality_score

    def normalize_prompt_quality_score(value, field_name="dataset_prompt_quality_score"):
        if isinstance(value, str) and value.strip().lower() == "episode":
            return "episode"
        if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none", "null")):
            return None
        if isinstance(value, dict) or value.__class__.__name__ == "DictConfig":
            return orig_normalize(value, field_name=field_name)
        return _as_float(value, field_name)

    ds.normalize_prompt_quality_score = normalize_prompt_quality_score
    ds._normalize_quality_score_value = _as_float

    orig_build = ds.build_robotwin_prompt

    def build_robotwin_prompt(task, quality_suffix=None, quality_score=None):
        if quality_score is None or isinstance(quality_score, dict):
            return orig_build(task, quality_suffix=quality_suffix, quality_score=quality_score)
        if quality_score == "episode":
            raise ValueError("`quality_score` must be resolved to a number before building a prompt.")
        shown = _format_score(float(quality_score))
        base = ds.DEFAULT_PROMPT.format(task=task)
        return f"{base} Quality: {shown}."

    ds.build_robotwin_prompt = build_robotwin_prompt

    orig_resolve = ds.RobotVideoDataset._resolve_prompt_quality_score

    def _resolve_prompt_quality_score(self, score_spec, sample):
        if score_spec == "episode":
            record = self._get_sample_episode_record(sample)
            if "score" not in record:
                raise ValueError(
                    "dataset_prompt_quality_score=episode requires `score` in meta/episodes.jsonl."
                )
            return _as_float(record["score"], "episodes.jsonl.score")
        return orig_resolve(self, score_spec, sample)

    ds.RobotVideoDataset._resolve_prompt_quality_score = _resolve_prompt_quality_score

    prefixes = [
        item.strip()
        for item in os.environ.get("FASTWAM_EXCLUDE_RAW_PREFIXES", "").split(",")
        if item.strip()
    ]
    if prefixes:
        orig_multi_init = MultiLeRobotDataset.__init__

        def _multi_init(self, dataset_dirs, episodes=None, *args, **kwargs):
            if episodes:
                filtered = {}
                for repo_id, indices in episodes.items():
                    drop = set()
                    ep_path = Path(repo_id) / "meta" / "episodes.jsonl"
                    if ep_path.is_file():
                        for line in ep_path.read_text(encoding="utf-8").splitlines():
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            raw = record.get("raw_file_name") or ""
                            if any(raw.startswith(prefix) for prefix in prefixes):
                                drop.add(int(record["episode_index"]))
                    filtered[repo_id] = [idx for idx in indices if int(idx) not in drop]
                episodes = filtered
            return orig_multi_init(self, dataset_dirs, episodes, *args, **kwargs)

        MultiLeRobotDataset.__init__ = _multi_init

    root = Path(__file__).resolve().parent.parent / "converted_data"

    def _load_episodes(rel_path):
        path = root / rel_path
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            episode
            for episode in (payload.get("episodes") or [])
            if episode.get("status") == "ok"
        ]

    shred_release = {}
    for episode in _load_episodes(
        "table30-rc_ur5_4-lerobot_stitched/meta/shred_paper_gripper_release.json"
    ):
        release = (episode.get("release") or {}).get("frame_index")
        if release is not None:
            shred_release[int(episode["episode_index"])] = int(release)

    press_bounds = {}
    for episode in _load_episodes(
        "table30-rc_arx5_5-lerobot_stitched/meta/press_the_button_e3_valleys.json"
    ):
        valleys = sorted(episode.get("valleys") or [], key=lambda item: int(item.get("button_index", 0)))
        frames = [int(item["frame_index"]) for item in valleys if item.get("frame_index") is not None]
        if frames:
            press_bounds[int(episode["episode_index"])] = frames

    water_bounds = {}
    for episode in _load_episodes(
        "table30-rc_arx5_5-lerobot_stitched/meta/water_the_flowers_keypoints.json"
    ):
        qx = None
        for item in episode.get("keypoints") or []:
            if item.get("keypoint") == "qx_peak" and item.get("frame_index") is not None:
                qx = int(item["frame_index"])
        if qx is not None:
            water_bounds[int(episode["episode_index"])] = [qx]

    stamp_bounds = {}
    for episode in _load_episodes(
        "table30-rc_aloha_10-lerobot_stitched/meta/stamp_positioning_gripper_peaks.json"
    ):
        keypoints = episode.get("keypoints") or {}
        right = (keypoints.get("right_peak_1") or {}).get("frame_index")
        left = (keypoints.get("left_peak") or {}).get("frame_index")
        if right is not None and left is not None:
            stamp_bounds[int(episode["episode_index"])] = [int(right), int(left)]

    press_prompts = (
        "Press the pink button.",
        "Press the blue button.",
        "Press the green button.",
        "Press the yellow button, then retract the gripper.",
    )
    water_prompts = (
        "Pick up the watering can and water the potted plant.",
        "The watering is finished, put the watering can down.",
    )
    stamp_prompts = (
        "Pick up the stamp.",
        "Dip into the ink.",
        "Stamp the signature area and return.",
    )

    def _scalar(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "reshape"):
            flat = value.reshape(-1)
            if int(flat.shape[0]) == 0:
                return None
            return int(flat[0].item())
        return int(value)

    def _segment_prompt(frame, bounds, prompts, role):
        idx = 0
        for bound in bounds:
            crossed = frame >= bound if role == "segment_start" else frame > bound
            if crossed:
                idx += 1
            else:
                break
        return prompts[min(idx, len(prompts) - 1)]

    orig_augment = FastWAMProcessor.augment_instruction

    def augment_instruction(self, data):
        dataset_index = _scalar(data.get("dataset_index", 0))
        frame_index = _scalar(data.get("frame_index"))
        episode_index = _scalar(data.get("episode_index"))
        task = data.get("task") or ""
        task_l = task.lower() if isinstance(task, str) else ""
        replacement = None
        if dataset_index == 0 and frame_index is not None and episode_index is not None:
            if "shred" in task_l and episode_index in shred_release:
                if frame_index > shred_release[episode_index]:
                    replacement = "Retract gripper."
            elif "button" in task_l and episode_index in press_bounds:
                replacement = _segment_prompt(
                    frame_index, press_bounds[episode_index], press_prompts, "segment_end"
                )
            elif ("water" in task_l or "potted" in task_l) and episode_index in water_bounds:
                replacement = _segment_prompt(
                    frame_index, water_bounds[episode_index], water_prompts, "segment_start"
                )
            elif "stamp" in task_l and episode_index in stamp_bounds:
                replacement = _segment_prompt(
                    frame_index, stamp_bounds[episode_index], stamp_prompts, "segment_end"
                )
        if replacement is not None:
            data = dict(data)
            data["task"] = replacement
        return orig_augment(self, data)

    FastWAMProcessor.augment_instruction = augment_instruction


_enable_rollout_recipe()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
