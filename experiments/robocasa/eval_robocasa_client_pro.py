'''

python playground/dm05/eval/eval_robocasa_client.py \
  --tasks OpenDrawer,CloseFridge,NavigateKitchen,CloseBlenderLid,TurnSinkSpout,AddLemonToFish,OrganizeVegetables,PortionYogurt,BowlAndCup,PackIdenticalLunches \
  --server_url http://127.0.0.1:7891 \
  --split pretrain \
  --num_trials 50 \
  --save_video_every 10

nohup torchrun --nproc_per_node=1 playground/dm05/sft/dm05_sft_robocasa.py \
    --task inference \
    --model_name_or_path user_checkpoints/dm05_sft/dm05_sft_0630/checkpoint-14000 \
    --port 7891 \
    > server.log 2>&1 &



nohup python playground/dm05/eval/eval_robocasa_client.py \
  --tasks OpenDrawer,CloseFridge,NavigateKitchen,CloseBlenderLid,TurnSinkSpout,AddLemonToFish,OrganizeVegetables,PortionYogurt,BowlAndCup,PackIdenticalLunches \
  --server_url http://127.0.0.1:7891 \
  --split pretrain \
  --num_trials 50 \
  --save_video_every 10 \
  > eval.log 2>&1 &



python playground/dm05/eval/eval_robocasa_client.py \
  --task_set target_atomic18 \
  --server_url http://127.0.0.1:7891 \
  --split pretrain \
  --num_trials 50 \
  --save_video_every 10



python playground/dm05/eval/eval_robocasa_client.py \
  --task_set target_all50 \
  --server_url http://127.0.0.1:7891 \
  --split pretrain \
  --num_trials 50 \
  --save_video_every 20



'''



import argparse
import collections
import io
import json
import logging
import os
import time
from datetime import datetime

import imageio
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_ATOMIC18_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

TARGET_COMPOSITE_SEEN16_TASKS = [
    "StackBowlsCabinet",
    "PreSoakPan",
    "ScrubCuttingBoard",
    "WashLettuce",
    "RinseSinkBasin",
    "KettleBoiling",
    "LoadDishwasher",
    "StoreLeftoversInBowl",
    "SetUpCuttingStation",
    "StirVegetables",
    "SteamInMicrowave",
    "SearingMeat",
    "PackIdenticalLunches",
    "PrepareCoffee",
    "DeliverStraw",
    "GetToastedBread",
]

TARGET_COMPOSITE_UNSEEN16_TASKS = [
    "RecycleBottlesByType",
    "WaffleReheat",
    "ArrangeBreadBasket",
    "WeighIngredients",
    "BreadSelection",
    "CuttingToolSelection",
    "GarnishPancake",
    "ArrangeTea",
    "WashFruitColander",
    "MakeIceLemonade",
    "PortionHotDogs",
    "PanTransfer",
    "HeatKebabSandwich",
    "CategorizeCondiments",
    "SeparateFreezerRack",
    "GatherTableware",
]

TARGET_ALL50_TASKS = (
    TARGET_ATOMIC18_TASKS
    + TARGET_COMPOSITE_SEEN16_TASKS
    + TARGET_COMPOSITE_UNSEEN16_TASKS
)

TASK_SETS = {
    "target_atomic18": TARGET_ATOMIC18_TASKS,
    "target_atomic": TARGET_ATOMIC18_TASKS,
    "target_composite_seen16": TARGET_COMPOSITE_SEEN16_TASKS,
    "target_composite_unseen16": TARGET_COMPOSITE_UNSEEN16_TASKS,
    "target_all50": TARGET_ALL50_TASKS,
}

TASK_SUBSET_BY_NAME = {
    **{task: "atomic18" for task in TARGET_ATOMIC18_TASKS},
    **{task: "composite_seen16" for task in TARGET_COMPOSITE_SEEN16_TASKS},
    **{task: "composite_unseen16" for task in TARGET_COMPOSITE_UNSEEN16_TASKS},
}


class DM05InferenceClient:
    """HTTP client for remote DM05 inference service."""

    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        server_url,
        robot_type="panda_omron",
        max_retries=5,
        request_timeout=60,
        retry_backoff=2.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.process_url = f"{self.server_url}/process_frame"
        self.robot_type = robot_type
        self.max_retries = max(1, int(max_retries))
        self.request_timeout = request_timeout
        self.retry_backoff = retry_backoff

        logger.info(f"Connecting to inference server at {self.server_url}...")
        try:
            requests.get(self.server_url, timeout=5)
            logger.info("Server is reachable!")
        except Exception as e:
            logger.warning(
                f"Cannot reach server at {self.server_url}: {e}\n"
                "Make sure the inference service is running on the dexbotic server."
            )

    def _build_request_payload(self, images, state, prompt):
        image_files = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            image_files.append(("image", (f"image_{i}.png", buf, "image/png")))

        form_data = {
            "text": prompt,
            "states": json.dumps(state.tolist()),
            "robot_type": self.robot_type,
        }
        return form_data, image_files

    @staticmethod
    def _close_image_files(image_files):
        for _, file_tuple in image_files:
            try:
                file_tuple[1].close()
            except Exception:
                pass

    def _retry_sleep_seconds(self, attempt_idx):
        # Exponential backoff: 2s, 4s, 8s, ... capped to avoid huge stalls.
        return min(60.0, float(self.retry_backoff) * (2 ** attempt_idx))

    def predict(self, images, state, prompt):
        last_error = None

        for attempt_idx in range(self.max_retries):
            form_data, image_files = self._build_request_payload(images, state, prompt)
            try:
                resp = requests.post(
                    self.process_url,
                    data=form_data,
                    files=image_files,
                    timeout=self.request_timeout,
                )

                if resp.status_code in self.RETRYABLE_STATUS_CODES:
                    body = resp.text[:1000]
                    last_error = requests.exceptions.HTTPError(
                        f"Retryable HTTP {resp.status_code} from inference server: {body}",
                        response=resp,
                    )
                    raise last_error

                resp.raise_for_status()
                result = resp.json()
                if "response" not in result:
                    raise RuntimeError(f"Malformed inference response: {result}")

                actions = np.array(result["response"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.ndim != 2 or actions.shape[0] == 0:
                    raise RuntimeError(f"Invalid action chunk shape: {actions.shape}")
                return actions
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_error = e
                if attempt_idx >= self.max_retries - 1:
                    break

                sleep_s = self._retry_sleep_seconds(attempt_idx)
                logger.warning(
                    "Inference request failed on attempt %d/%d; retrying in %.1fs: %s",
                    attempt_idx + 1,
                    self.max_retries,
                    sleep_s,
                    e,
                )
                time.sleep(sleep_s)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                if status_code not in self.RETRYABLE_STATUS_CODES:
                    logger.error(f"Inference request failed with non-retryable HTTP error: {e}")
                    raise

                last_error = e
                if attempt_idx >= self.max_retries - 1:
                    break

                sleep_s = self._retry_sleep_seconds(attempt_idx)
                logger.warning(
                    "Inference request returned HTTP %s on attempt %d/%d; "
                    "retrying in %.1fs: %s",
                    status_code,
                    attempt_idx + 1,
                    self.max_retries,
                    sleep_s,
                    e,
                )
                time.sleep(sleep_s)
            except Exception as e:
                logger.error(f"Inference request failed with non-retryable error: {e}")
                raise
            finally:
                self._close_image_files(image_files)

        logger.error(
            "Inference request failed after %d attempts: %s",
            self.max_retries,
            last_error,
        )
        raise last_error


def _compact_action_to_native(action):
    """Convert compact 11D PandaOmron action to native 12D if needed."""
    action = np.asarray(action, dtype=np.float64).flatten()
    if action.shape[0] == 12:
        return action
    if action.shape[0] != 11:
        raise ValueError(f"Expected PandaOmron action dim 11 or 12, got {action.shape}")

    native = np.zeros(12, dtype=np.float64)
    native[0:3] = action[0:3]
    native[4] = action[3]
    native[5:8] = action[4:7]
    native[8:11] = action[7:10]
    native[11] = action[10]
    return native


def model_action_to_env_action(model_action):
    """Convert PandaOmron model action to RoboCasa env action dict.

    Hard mode constraint:
      - control_mode >= 0  -> base mode, zero EEF branch
      - control_mode <  0  -> EEF mode, zero base branch
    """
    action = _compact_action_to_native(model_action)

    base_motion = action[0:4].copy()
    control_mode = float(action[4])
    eef_pos = action[5:8].copy()
    eef_rot = action[8:11].copy()
    gripper = action[11:12].copy()


    # 这是原来的，直接硬阈值卡control mode，也没管夹爪
    # if control_mode >= 0:
    #     control_mode = 1.0
    #     eef_pos[:] = 0.0
    #     eef_rot[:] = 0.0
    # else:
    #     control_mode = -1.0
    #     base_motion[:] = 0.0

    # [-1,1] -> [0,1]，因为robocasa365环境内部有阈值判断，并且对于夹爪阈值是0.5
    control_mode_env = (control_mode + 1.0) / 2.0
    gripper_env = (gripper + 1.0) / 2.0

    env_action = {
        "action.base_motion": base_motion,
        "action.control_mode": np.array([control_mode_env], dtype=np.float64),
        "action.end_effector_position": eef_pos,
        "action.end_effector_rotation": eef_rot,
        "action.gripper_close": gripper_env,
    }
    return env_action


def _get_obs_array(obs, candidates, expected_dim=None):
    for key in candidates:
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float32)
            if expected_dim is not None and arr.shape[-1] != expected_dim:
                raise ValueError(
                    f"Observation key {key!r} has dim {arr.shape[-1]}, expected {expected_dim}"
                )
            return arr
    raise KeyError(f"None of the observation keys exist: {candidates}")


def get_state_from_obs(obs):
    """Build compact 14D PandaOmron state from RoboCasa observation."""
    base_pos = _get_obs_array(
        obs,
        ["state.base_position", "robot_state.base_position"],
        expected_dim=3,
    )
    base_rot = _get_obs_array(
        obs,
        ["state.base_rotation", "robot_state.base_rotation"],
        expected_dim=4,
    )
    eef_pos = _get_obs_array(
        obs,
        ["state.end_effector_position_relative"],
        expected_dim=3,
    )
    eef_rot = _get_obs_array(
        obs,
        ["state.end_effector_rotation_relative"],
        expected_dim=4,
    )
    gripper = _get_obs_array(
        obs,
        ["state.gripper_qpos"],
        expected_dim=2,
    )

    state = np.concatenate(
        [
            base_pos,
            base_rot[2:4],
            eef_pos,
            eef_rot,
            gripper,
        ],
        axis=0,
    ).astype(np.float32)

    if state.shape[0] != 14:
        raise ValueError(f"Expected 14D compact PandaOmron state, got {state.shape}")
    return state


def get_images_from_obs(obs):
    camera_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    images = []
    for key in camera_keys:
        img_array = obs[key]
        img = Image.fromarray(np.ascontiguousarray(img_array)).convert("RGB")
        images.append(img)
    return images


def should_save_video(episode_idx, save_video_every):
    if save_video_every <= 0:
        return False
    return ((episode_idx + 1) % save_video_every) == 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_completed_episode_records(path, num_trials):
    if not os.path.exists(path):
        return []

    records_by_episode_idx = {}
    duplicate_episode_indices = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skip malformed jsonl line %s:%d (%s)",
                    path,
                    line_no,
                    exc,
                )
                continue

            episode_idx = rec.get("episode_idx")
            if not isinstance(episode_idx, int):
                continue
            if not (0 <= episode_idx < num_trials):
                continue
            if episode_idx in records_by_episode_idx:
                duplicate_episode_indices.add(episode_idx)

            # Keep the last valid record for each episode_idx. This is safer for
            # resume if the file contains duplicate entries from accidental reruns.
            records_by_episode_idx[episode_idx] = rec

    if duplicate_episode_indices:
        logger.warning(
            "Found duplicated episode records in %s; using the last record for episode_idx=%s",
            path,
            sorted(duplicate_episode_indices),
        )

    return [records_by_episode_idx[idx] for idx in sorted(records_by_episode_idx)]


def _validate_resume_records(
    records,
    task_name,
    split,
    seed,
    run_name,
    robot_type,
    server_url,
    replan_steps,
    max_steps,
):
    for rec in records:
        episode_idx = rec["episode_idx"]
        expected_seed = seed + episode_idx
        mismatches = []

        expected_values = {
            "task": task_name,
            "split": split,
            "seed": expected_seed,
            "run_name": run_name,
            "robot_type": robot_type,
            "server_url": server_url,
            "replan_steps": replan_steps,
            "max_steps": max_steps,
        }

        for key, expected in expected_values.items():
            actual = rec.get(key)
            if actual != expected:
                mismatches.append(f"{key}: expected={expected!r}, found={actual!r}")

        if mismatches:
            mismatch_text = "; ".join(mismatches)
            raise ValueError(
                "Existing resume record is incompatible with current evaluation config for "
                f"task={task_name!r}, episode_idx={episode_idx}: {mismatch_text}. "
                "Use the same --run_name/--video_dir/config, or start a new run, or pass --no_resume."
            )


def evaluate(
    task_name,
    server_url,
    num_trials=10,
    split="target",
    seed=42,
    video_dir="eval_videos",
    run_name=None,
    replan_steps=5,
    max_steps=None,
    save_video_every=1,
    robot_type="panda_omron",
    inference_retries=5,
    inference_timeout=60,
    inference_retry_backoff=2.0,
    resume=True,
):
    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    np.random.seed(seed)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if max_steps is None:
        max_steps = get_task_horizon(task_name)

    output_dir = os.path.join(video_dir, run_name, task_name)
    ensure_dir(output_dir)
    episode_results_path = os.path.join(output_dir, "episode_results.jsonl")

    existing_records = _load_completed_episode_records(episode_results_path, num_trials) if resume else []
    if existing_records:
        _validate_resume_records(
            existing_records,
            task_name=task_name,
            split=split,
            seed=seed,
            run_name=run_name,
            robot_type=robot_type,
            server_url=server_url,
            replan_steps=replan_steps,
            max_steps=max_steps,
        )

    completed_episode_indices = {rec["episode_idx"] for rec in existing_records}
    total_successes = sum(int(bool(rec.get("success", False))) for rec in existing_records)
    total_episodes = len(existing_records)

    logger.info(f"Max steps per episode: {max_steps}")

    if existing_records:
        logger.info("[resume] %s existing episodes: %d/%d", task_name, total_episodes, num_trials)

    if total_episodes >= num_trials:
        final_stats = {
            "task": task_name,
            "task_subset": TASK_SUBSET_BY_NAME.get(task_name, "custom"),
            "split": split,
            "num_episodes": total_episodes,
            "num_successes": total_successes,
            "success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
            "server_url": server_url,
            "inference_retries": inference_retries,
            "inference_timeout": inference_timeout,
            "inference_retry_backoff": inference_retry_backoff,
            "robot_type": robot_type,
            "seed": seed,
            "replan_steps": replan_steps,
            "max_steps": max_steps,
            "save_video_every": save_video_every,
            "run_name": run_name,
            "episode_results_path": episode_results_path,
            "resumed_from_existing": len(existing_records),
        }

        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=2)

        logger.info("[resume] %s already complete, skip env rollout.", task_name)
        return final_stats

    client = DM05InferenceClient(
        server_url,
        robot_type=robot_type,
        max_retries=inference_retries,
        request_timeout=inference_timeout,
        retry_backoff=inference_retry_backoff,
    )

    logger.info(f"Creating environment: robocasa/{task_name}, split={split}")
    env = gym.make(
        f"robocasa/{task_name}",
        split=split,
        seed=seed,
    )

    pending_episode_indices = [episode_idx for episode_idx in range(num_trials) if episode_idx not in completed_episode_indices]

    for episode_idx in tqdm(pending_episode_indices, desc=f"Evaluating {task_name}"):
        record_video = should_save_video(episode_idx, save_video_every)
        replay_images = [] if record_video else None

        obs, info = env.reset(seed=seed + episode_idx)
        task_lang = obs.get("annotation.human.task_description", f"Task: {task_name}")
        logger.info(f"\nEpisode {episode_idx + 1}/{num_trials}: {task_lang}")

        action_plan = collections.deque()
        done = False
        terminated = False
        truncated = False
        inference_error = None
        executed_steps = 0

        for t in range(max_steps):
            if not action_plan:
                images = get_images_from_obs(obs)
                state = get_state_from_obs(obs)

                t0 = time.time()
                try:
                    action_chunk = client.predict(images, state, task_lang)
                except Exception as e:
                    inference_error = str(e)
                    logger.error(
                        "  Inference failed after retries at episode %d step %d; "
                        "marking this episode as failed and continuing: %s",
                        episode_idx + 1,
                        t,
                        e,
                    )
                    break
                dt = time.time() - t0

                if t == 0:
                    logger.info(f"  First inference took {dt:.2f}s, got {len(action_chunk)} actions")

                steps_to_use = min(replan_steps, len(action_chunk))
                for i in range(steps_to_use):
                    action_plan.append(action_chunk[i])

            raw_action = action_plan.popleft()
            env_action = model_action_to_env_action(raw_action)
            obs, reward, terminated, truncated, info = env.step(env_action)
            executed_steps += 1
            done = info.get("success", False)

            if record_video and (t % 2 == 0 or done or t == max_steps - 1):
                try:
                    render_img = env.render()
                    if render_img is not None:
                        replay_images.append(np.ascontiguousarray(render_img))
                except Exception:
                    pass

            if done:
                logger.info(f"  ✅ Success at step {t}!")
                break

            if terminated or truncated:
                logger.info(f"  Episode terminated early at step {t}")
                break

        if not done:
            if inference_error is not None:
                logger.info("  ❌ Failed (inference error after retries)")
            else:
                logger.info(f"  ❌ Failed (reached max steps {max_steps})")

        total_successes += int(done)
        total_episodes += 1

        video_path = None
        if record_video and replay_images:
            suffix = "success" if done else "failure"
            video_path = os.path.join(output_dir, f"episode_{episode_idx + 1:03d}_{suffix}.mp4")
            imageio.mimwrite(video_path, replay_images, fps=20)

        episode_record = {
            "episode_idx": episode_idx,
            "task": task_name,
            "task_subset": TASK_SUBSET_BY_NAME.get(task_name, "custom"),
            "split": split,
            "seed": seed + episode_idx,
            "success": bool(done),
            "success_int": int(done),
            "steps_executed": executed_steps,
            "max_steps": max_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "replan_steps": replan_steps,
            "robot_type": robot_type,
            "server_url": server_url,
            "inference_error": inference_error,
            "run_name": run_name,
        }
        if video_path is not None:
            episode_record["video_path"] = video_path
        _append_jsonl(episode_results_path, episode_record)

        success_rate = total_successes / total_episodes * 100.0
        logger.info(
            f"  Progress: {total_episodes}/{num_trials}, "
            f"Success: {total_successes}/{total_episodes} ({success_rate:.1f}%)"
        )

    final_stats = {
        "task": task_name,
        "task_subset": TASK_SUBSET_BY_NAME.get(task_name, "custom"),
        "split": split,
        "num_episodes": total_episodes,
        "num_successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
        "server_url": server_url,
        "inference_retries": inference_retries,
        "inference_timeout": inference_timeout,
        "inference_retry_backoff": inference_retry_backoff,
        "robot_type": robot_type,
        "seed": seed,
        "replan_steps": replan_steps,
        "max_steps": max_steps,
        "save_video_every": save_video_every,
        "run_name": run_name,
        "episode_results_path": episode_results_path,
        "resumed_from_existing": len(existing_records),
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Final Results for {task_name}:")
    logger.info(f"  Success Rate: {final_stats['success_rate'] * 100:.1f}%")
    logger.info(f"  ({total_successes}/{total_episodes} episodes)")
    logger.info(f"  Outputs saved to: {output_dir}")
    logger.info(f"  Episode records saved to: {episode_results_path}")
    logger.info(f"  Stats saved to: {stats_path}")
    logger.info("=" * 60)

    env.close()
    return final_stats


def evaluate_task_set(
    task_names,
    server_url,
    num_trials=50,
    split="target",
    seed=42,
    video_dir="eval_videos",
    run_name=None,
    #replan_steps=5,
    replan_steps=40,
    max_steps=None,
    save_video_every=10,
    task_set_name="custom",
    robot_type="panda_omron",
    inference_retries=5,
    inference_timeout=60,
    inference_retry_backoff=2.0,
    resume=True,
):
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    ensure_dir(os.path.join(video_dir, run_name))

    all_results = []
    for task_idx, task_name in enumerate(task_names):
        logger.info(
            f"\n{'#' * 70}\n"
            f"Task {task_idx + 1}/{len(task_names)}: {task_name}\n"
            f"{'#' * 70}"
        )

        task_seed = seed + task_idx * 1000
        stats = evaluate(
            task_name=task_name,
            server_url=server_url,
            num_trials=num_trials,
            split=split,
            seed=task_seed,
            video_dir=video_dir,
            run_name=run_name,
            replan_steps=replan_steps,
            max_steps=max_steps,
            save_video_every=save_video_every,
            robot_type=robot_type,
            inference_retries=inference_retries,
            inference_timeout=inference_timeout,
            inference_retry_backoff=inference_retry_backoff,
            resume=resume,
        )
        stats["task_subset"] = TASK_SUBSET_BY_NAME.get(task_name, "custom")
        all_results.append(stats)

    overall_success_rate = (
        float(np.mean([x["success_rate"] for x in all_results])) if all_results else 0.0
    )

    subset_summaries = {}
    subset_names = sorted({x["task_subset"] for x in all_results})
    for subset_name in subset_names:
        subset_results = [x for x in all_results if x["task_subset"] == subset_name]
        subset_summaries[subset_name] = {
            "num_tasks": len(subset_results),
            "overall_success_rate": (
                float(np.mean([x["success_rate"] for x in subset_results]))
                if subset_results else 0.0
            ),
            "tasks": [x["task"] for x in subset_results],
        }

    summary = {
        "task_set_name": task_set_name,
        "split": split,
        "num_tasks": len(task_names),
        "num_trials_per_task": num_trials,
        "overall_success_rate": overall_success_rate,
        "subset_summaries": subset_summaries,
        "server_url": server_url,
        "inference_retries": inference_retries,
        "inference_timeout": inference_timeout,
        "inference_retry_backoff": inference_retry_backoff,
        "robot_type": robot_type,
        "seed": seed,
        "replan_steps": replan_steps,
        "max_steps": max_steps,
        "save_video_every": save_video_every,
        "run_name": run_name,
        "resume": resume,
        "tasks": all_results,
    }

    summary_path = os.path.join(video_dir, run_name, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"Task set finished: {task_set_name}")
    logger.info(f"Overall success rate: {overall_success_rate * 100:.1f}%")
    for subset_name, subset_summary in subset_summaries.items():
        logger.info(
            f"  {subset_name}: {subset_summary['overall_success_rate'] * 100:.1f}% "
            f"over {subset_summary['num_tasks']} tasks"
        )
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)

    return summary


def _parse_task_list(task_arg: str) -> list[str]:
    tasks = [x.strip() for x in task_arg.split(",") if x.strip()]
    if not tasks:
        raise ValueError("--tasks is provided but no valid task names were parsed")
    duplicates = sorted({task for task in tasks if tasks.count(task) > 1})
    if duplicates:
        raise ValueError(f"Duplicated task names in --tasks: {duplicates}")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DM05 model on RoboCasa (client mode)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Single task name (e.g., OpenDrawer, CloseFridge)",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        default=None,
        choices=sorted(TASK_SETS.keys()),
        help="Predefined task set name",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task list, e.g. OpenDrawer,CloseFridge,TurnOnMicrowave",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:7891",
        help="DM05 inference server URL",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="panda_omron",
        help="Robot type/profile passed to the DM05 inference server",
    )
    parser.add_argument(
        "--inference_retries",
        type=int,
        default=5,
        help="Max attempts for each inference HTTP request before failing the episode",
    )
    parser.add_argument(
        "--inference_timeout",
        type=float,
        default=60,
        help="Timeout in seconds for each inference HTTP request attempt",
    )
    parser.add_argument(
        "--inference_retry_backoff",
        type=float,
        default=2.0,
        help="Initial exponential backoff in seconds between inference retries",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of evaluation episodes per task",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["target", "pretrain"],
        help="Environment split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="eval_videos",
        help="Directory to save evaluation videos and stats",
    )
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=5,
        help="Steps to execute before re-querying model",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode (default: task-specific)",
    )
    parser.add_argument(
        "--save_video_every",
        type=int,
        default=1,
        help="Save one video every N episodes; 0 means save no videos",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name; default is timestamp",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable automatic resume from existing episode_results.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    provided_modes = [
        args.task is not None,
        args.tasks is not None,
        args.task_set is not None,
    ]
    if sum(provided_modes) != 1:
        raise ValueError("Provide exactly one of --task, --tasks, or --task_set")

    if args.task is not None:
        evaluate(
            task_name=args.task,
            server_url=args.server_url,
            num_trials=args.num_trials,
            split=args.split,
            seed=args.seed,
            video_dir=args.video_dir,
            run_name=args.run_name,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            save_video_every=args.save_video_every,
            robot_type=args.robot_type,
            inference_retries=args.inference_retries,
            inference_timeout=args.inference_timeout,
            inference_retry_backoff=args.inference_retry_backoff,
            resume=not args.no_resume,
        )
    elif args.tasks is not None:
        evaluate_task_set(
            task_names=_parse_task_list(args.tasks),
            server_url=args.server_url,
            num_trials=args.num_trials,
            split=args.split,
            seed=args.seed,
            video_dir=args.video_dir,
            run_name=args.run_name,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            save_video_every=args.save_video_every,
            task_set_name="custom_tasks",
            robot_type=args.robot_type,
            inference_retries=args.inference_retries,
            inference_timeout=args.inference_timeout,
            inference_retry_backoff=args.inference_retry_backoff,
            resume=not args.no_resume,
        )
    else:
        evaluate_task_set(
            task_names=TASK_SETS[args.task_set],
            server_url=args.server_url,
            num_trials=args.num_trials,
            split=args.split,
            seed=args.seed,
            video_dir=args.video_dir,
            run_name=args.run_name,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            save_video_every=args.save_video_every,
            task_set_name=args.task_set,
            robot_type=args.robot_type,
            inference_retries=args.inference_retries,
            inference_timeout=args.inference_timeout,
            inference_retry_backoff=args.inference_retry_backoff,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()