import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
import json
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _rollout_dir(rollout_root: str, task_name: str, task_config: str) -> Path:
    return Path(str(rollout_root)) / str(task_name) / str(task_config)


def _count_rollout_episodes(rollout_dir: Path) -> int:
    data_dir = rollout_dir / "data"
    if not data_dir.is_dir():
        return 0
    return len(list(data_dir.glob("episode*.hdf5")))


def _load_rollout_resume_state(rollout_root: str, task_name: str, task_config: str) -> tuple[int, int | None]:
    rollout_dir = _rollout_dir(rollout_root, task_name, task_config)
    episode_offset = _count_rollout_episodes(rollout_dir)
    last_seed: int | None = None
    cursor_path = rollout_dir / "eval_seed_cursor.json"
    if cursor_path.exists():
        with open(cursor_path, "r", encoding="utf-8") as f:
            cursor = json.load(f)
        last_seed = int(cursor["last_seed"])
        episode_offset = int(cursor.get("episode_count", episode_offset))
    return episode_offset, last_seed


def _save_rollout_cursor(
    rollout_root: str,
    task_name: str,
    task_config: str,
    *,
    last_seed: int,
    episode_count: int,
) -> None:
    rollout_dir = _rollout_dir(rollout_root, task_name, task_config)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    cursor_path = rollout_dir / "eval_seed_cursor.json"
    with open(cursor_path, "w", encoding="utf-8") as f:
        json.dump(
            {"last_seed": int(last_seed), "episode_count": int(episode_count)},
            f,
            indent=2,
        )


def _init_rollout_recording(TASK_ENV, args, *, episode_idx: int) -> Path | None:
    rollout_root = args.get("rollout_save_root")
    save_rollout = parse_bool(args.get("save_rollout", False))
    if not save_rollout or rollout_root is None or str(rollout_root).strip() == "":
        return None

    rollout_dir = Path(str(rollout_root)) / str(args["task_name"]) / str(args["task_config"])
    rollout_dir.mkdir(parents=True, exist_ok=True)
    TASK_ENV.save_data = True
    TASK_ENV.save_dir = str(rollout_dir)
    TASK_ENV.ep_num = int(episode_idx)
    TASK_ENV.FRAME_IDX = 0
    return rollout_dir


def _record_rollout_observation(TASK_ENV, args) -> None:
    rollout_root = args.get("rollout_save_root")
    if not parse_bool(args.get("save_rollout", False)) or rollout_root is None:
        return
    if not TASK_ENV.save_data:
        return
    TASK_ENV._take_picture()


def _finalize_rollout_recording(
    TASK_ENV,
    args,
    *,
    rollout_dir: Path | None,
    episode_idx: int,
    instruction: str,
    episode_info: dict,
    success: bool,
    env_seed: int | None = None,
) -> None:
    if rollout_dir is None or not TASK_ENV.save_data:
        return
    if parse_bool(args.get("rollout_save_only_success", False)) and not success:
        TASK_ENV.save_data = False
        TASK_ENV.FRAME_IDX = 0
        return
    if TASK_ENV.FRAME_IDX <= 0:
        TASK_ENV.save_data = False
        return

    TASK_ENV.merge_pkl_to_hdf5_video()
    TASK_ENV.remove_data_cache()

    info_path = rollout_dir / "scene_info.json"
    info_db = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info_db = json.load(f)
    entry = {
        "info": episode_info,
        "instruction": instruction,
        "success": bool(success),
    }
    if env_seed is not None:
        entry["env_seed"] = int(env_seed)
    info_db[f"episode_{episode_idx}"] = entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_db, f, ensure_ascii=False, indent=4)

    instructions_dir = rollout_dir / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    with open(instructions_dir / f"episode{episode_idx}.json", "w", encoding="utf-8") as f:
        json.dump({"seen": [instruction], "unseen": []}, f, indent=2, ensure_ascii=False)

    TASK_ENV.save_data = False
    TASK_ENV.FRAME_IDX = 0


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    save_rollout = parse_bool(usr_args.get("save_rollout", False))
    rollout_save_root = usr_args.get("rollout_save_root")
    if save_rollout and (rollout_save_root is None or str(rollout_save_root).strip() == ""):
        raise ValueError("`save_rollout=true` requires non-empty `rollout_save_root`.")
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_rollout"] = save_rollout
    args["rollout_save_root"] = rollout_save_root
    args["rollout_save_only_success"] = parse_bool(usr_args.get("rollout_save_only_success", False))
    rollout_resume = parse_bool(usr_args.get("rollout_resume", False))
    rollout_episode_offset = int(usr_args.get("rollout_episode_offset", 0))
    eval_start_seed = usr_args.get("eval_start_seed")
    if save_rollout and rollout_save_root and rollout_resume:
        auto_offset, auto_seed = _load_rollout_resume_state(rollout_save_root, task_name, task_config)
        if rollout_episode_offset == 0:
            rollout_episode_offset = auto_offset
        if eval_start_seed is None and auto_seed is not None:
            eval_start_seed = auto_seed
    args["rollout_episode_offset"] = rollout_episode_offset

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    if eval_start_seed is not None and str(eval_start_seed).strip() != "":
        st_seed = int(eval_start_seed)
    else:
        st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    # Load policy before sim env so RoboTwin/curobo does not occupy GPU 0 during model init.
    model = get_model(usr_args)
    multi_gpu = _parse_bool_flag(usr_args.get("multi_gpu", False))
    if multi_gpu:
        import gc

        import torch

        if torch.cuda.is_available():
            # cuda:0 is reserved for Sapien rendering + action expert (see EVALUATION.*_device).
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        gc.collect()
    TASK_ENV = class_decorator(args["task_name"])
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan)
    suc_nums.append(suc_num)

    if save_rollout and rollout_save_root:
        _save_rollout_cursor(
            rollout_save_root,
            task_name,
            task_config,
            last_seed=st_seed,
            episode_count=rollout_episode_offset + test_num,
        )

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    try:
        finalize_func = eval_function_decorator(policy_name, "finalize_model_episode")
    except AttributeError:
        finalize_func = None

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except UnStableError as e:
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            TASK_ENV.close_env()
            now_seed += 1
            continue
        except Exception as e:
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            print(" -------------")
            print("Error: ", e)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            TASK_ENV.close_env()
            now_seed += 1
            print("error occurs !")
            continue
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        rollout_episode_offset = int(args.get("rollout_episode_offset", 0))
        episode_idx = TASK_ENV.test_num + rollout_episode_offset
        rollout_dir = _init_rollout_recording(TASK_ENV, args, episode_idx=episode_idx)

        current_video_path = None
        if TASK_ENV.eval_video_path is not None:
            current_video_path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())

            observation = None
            if need_obs:
                observation = TASK_ENV.get_obs()
                _record_rollout_observation(TASK_ENV, args)
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        if finalize_func is not None:
            finalize_func(model, episode_idx=episode_idx, success=succ)
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
            if current_video_path is None or not current_video_path.exists():
                raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
            is_randomized = "randomized" in str(args["task_config"]).lower()
            renamed_video_path = (
                Path(TASK_ENV.eval_video_path)
                / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4"
            )
            current_video_path.rename(renamed_video_path)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        _finalize_rollout_recording(
            TASK_ENV,
            args,
            rollout_dir=rollout_dir,
            episode_idx=episode_idx,
            instruction=instruction,
            episode_info=episode_info_list[0],
            success=succ,
            env_seed=now_seed,
        )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
