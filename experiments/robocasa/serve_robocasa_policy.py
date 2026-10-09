"""RoboCasa365 HTTP inference server for FastWAM checkpoints.

The RoboCasa simulator runs in a separate container and talks to this server
over HTTP (see eval_robocasa_client_old.py). This process loads a trained
FastWAM (fastwam_joint) checkpoint, exposes ``POST /process_frame``, and returns
an action chunk for each observation.

Protocol (fixed by the existing client, do NOT change):
  POST /process_frame
    form fields:
      - text:        task instruction string
      - states:      JSON-encoded list, 14D compact PandaOmron state
      - robot_type:  robot profile string (informational)
    files:
      - image (x3):  PNG frames, in client order
                     [robot0_agentview_left, robot0_agentview_right,
                      robot0_eye_in_hand]
    response JSON:
      { "response": [[a_0...a_10], ...] }  # action chunk [T, 11], denormalized

Alignment with training (verified against configs/data/robocasa365.yaml and
src/fastwam/datasets/lerobot/robot_video_dataset.py):
  * Camera layout: concat_multi_camera="robocasa" -> main(agentview_left) 256x256
    on the left, aux(eye_in_hand) 128x128 stacked over aux(agentview_right)
    128x128 on the right -> 256x384 canvas. The client's camera order differs
    from the training shape_meta order, so we reorder explicitly below.
  * State: the client already sends the 14D compact state, which equals the
    training post-RobocasaSliceTransform layout. We scatter it back to the 16D
    raw layout (the dropped dims are constant zeros) and run the exact same
    processor pipeline (RobocasaSliceTransform.forward + normalizer.forward) so
    normalization is byte-for-byte identical to training.
  * Action: the model outputs 11D normalized action; we denormalize to 11D and
    return it. The client converts 11D -> native 12D itself.
  * Image normalization: x*(2/255)-1 == ToTensor(/255) then Normalize(0.5, 0.5).

Example (run on the inference container):
  python experiments/robocasa/serve_robocasa_policy.py \
    ckpt=runs/robocasa365_join_256x384_1e-4_10task/robocasa365_sft_v1/checkpoints/weights/step_002500.pt \
    gpu_id=0 \
    server.port=7891

Then on the simulator container:
  python eval_robocasa_client_old.py \
    --tasks CloseFridge,NavigateKitchen,... \
    --server_url http://<infer_host>:7891 \
    --split pretrain --num_trials 50 --save_video_every 10
"""


# 入口命令
# cd /mnt/data/dm05/share/wzg/project/gfwam
# python experiments/robocasa/serve_robocasa_policy.py \
#   ckpt=runs/robocasa365_expert_sft_from_3rollout/robocasa365_expert_sft_from_3rollout/checkpoints/weights/step_112500.pt \
#   +EVALUATION.quality_score=5 \
#   gpu_id=0 \
#   server.host=0.0.0.0 \
#   server.port=7891






import inspect
import io
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import build_robotwin_prompt
from fastwam.datasets.lerobot.transforms.robocasa import RobocasaSliceTransform
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_robocasa")

# The client (eval_robocasa_client_old.get_images_from_obs) sends 3 PNGs in this
# fixed order. We must remap them to the training shape_meta camera order.
CLIENT_CAMERA_ORDER = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
# Training shape_meta order (see configs/data/robocasa365.yaml): main first, then
# the two aux cameras that get stacked on the right of the robocasa canvas.
TRAIN_CAMERA_ORDER = [
    "robot0_agentview_left",   # main   -> 256x256 (left)
    "robot0_eye_in_hand",      # aux    -> 128x128 (top-right)
    "robot0_agentview_right",  # aux    -> 128x128 (bottom-right)
]


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


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []
    if not _is_none_like(explicit):
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class RoboCasaPolicy:
    """Model-side inference for RoboCasa365, mirroring the training pipeline."""

    def __init__(
        self,
        *,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        num_video_frames: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        save_predicted_video: bool = False,
        predicted_video_save_root: Optional[Path] = None,
        predicted_video_fps: int = 8,
        predicted_video_save_every: int = 10,
        eval_quality_score: Optional[int] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        self.device = device

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.num_video_frames = int(num_video_frames)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        # Optional quality score appended to the prompt as " Quality: N.".
        # MUST match the score the model saw for high-quality (expert) data
        # during training of a mixed-QUALITY model. Leave None for basic-mixed
        # models (whose prompts have no Quality suffix).
        self.eval_quality_score = (
            int(eval_quality_score) if eval_quality_score is not None else None
        )
        logger.info(
            "[prompt-config] eval_quality_score=%s -> prompts will %s a Quality suffix.",
            self.eval_quality_score,
            ("append 'Quality: %d.'" % self.eval_quality_score)
            if self.eval_quality_score is not None else "NOT append",
        )

        self.save_predicted_video = bool(save_predicted_video)
        self.predicted_video_save_root = predicted_video_save_root
        self.predicted_video_fps = int(predicted_video_fps)
        self.predicted_video_save_every = int(predicted_video_save_every)
        if self.save_predicted_video:
            if self.predicted_video_save_root is None:
                raise ValueError(
                    "`predicted_video_save_root` must be set when save_predicted_video=true."
                )
            self.predicted_video_save_root.mkdir(parents=True, exist_ok=True)
            if self.seed is None:
                # infer_joint's internal action-consistency check requires a seed.
                logger.warning(
                    "save_predicted_video=true but seed is null; using seed=42 for the "
                    "joint sampler's action-consistency check."
                )
                self.seed = 42
        # Global replan counter, used only when the client sends no episode metadata.
        self._replan_counter = 0

        # Validate camera/state/action expectations against the loaded processor.
        image_meta = self.processor.shape_meta["images"]
        if len(image_meta) != 3:
            raise ValueError(
                f"RoboCasa server expects 3 cameras in shape_meta, got {len(image_meta)}."
            )
        train_keys = [m["key"] for m in image_meta]
        if train_keys != TRAIN_CAMERA_ORDER:
            raise ValueError(
                "shape_meta camera order mismatch. "
                f"Expected {TRAIN_CAMERA_ORDER}, got {train_keys}. "
                "Update TRAIN_CAMERA_ORDER / the client mapping if the config changed."
            )

        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("RoboCasa server expects a single merged state key.")
        self._state_key = state_meta[0]["key"]
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("RoboCasa server expects a single merged action key.")
        self._action_key = action_meta[0]["key"]

        # Client -> training camera index remap.
        self._client_to_train_idx = [
            CLIENT_CAMERA_ORDER.index(key) for key in TRAIN_CAMERA_ORDER
        ]

        logger.info(
            "RoboCasaPolicy ready | ckpt=%s | stats=%s | horizon=%d | video_frames=%d | "
            "steps=%d | device=%s | dtype=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.num_video_frames,
            self.num_inference_steps,
            self.device,
            model_dtype,
        )

    def _build_image_tensor(self, images_client_order: list[np.ndarray]) -> torch.Tensor:
        if len(images_client_order) != 3:
            raise ValueError(
                f"Expected 3 camera images, got {len(images_client_order)}."
            )
        # Reorder from client order to training shape_meta order.
        main, aux_top, aux_bottom = (
            images_client_order[i] for i in self._client_to_train_idx
        )
        # robocasa layout: main 256x256 (left); aux 128x128 stacked on the right.
        main = _resize_rgb(main, (256, 256))            # (H=256, W=256)
        aux_top = _resize_rgb(aux_top, (128, 128))      # (H=128, W=128)
        aux_bottom = _resize_rgb(aux_bottom, (128, 128))
        right = np.concatenate([aux_top, aux_bottom], axis=0)  # 256x128
        image = np.concatenate([main, right], axis=1)          # 256x384

        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.model.torch_dtype)
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _normalize_state(self, state_compact_14d: np.ndarray) -> torch.Tensor:
        state_compact_14d = np.asarray(state_compact_14d, dtype=np.float32).reshape(-1)
        if state_compact_14d.shape[0] != len(RobocasaSliceTransform.STATE_KEEP):
            raise ValueError(
                "Expected 14D compact PandaOmron state, got "
                f"{state_compact_14d.shape[0]}D."
            )
        # Scatter the compact 14D back to the 16D raw layout expected by the
        # processor pipeline (dropped base-quat dims are constant zeros).
        raw16 = np.zeros(RobocasaSliceTransform.STATE_RAW_DIM, dtype=np.float32)
        raw16[RobocasaSliceTransform.STATE_KEEP] = state_compact_14d

        state_batch = {
            "state": {
                self._state_key: torch.from_numpy(raw16).unsqueeze(0)  # [1, 16]
            }
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][self._state_key]  # [1, 14]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
        normalizer = self.processor.normalizer.normalizers["action"][self._action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()  # [B, T, 11]

    def _should_save_video(self, save_video_flag: Optional[bool]) -> bool:
        if not self.save_predicted_video:
            return False
        if save_video_flag is not None:
            # Client explicitly controls saving for this replan.
            return bool(save_video_flag)
        # Stateless fallback: save one clip every N replan calls.
        if self.predicted_video_save_every <= 0:
            return True
        return (self._replan_counter % self.predicted_video_save_every) == 0

    def _predicted_video_path(self, episode_idx: Optional[int], step_idx: Optional[int]) -> Path:
        assert self.predicted_video_save_root is not None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if episode_idx is not None:
            step_tag = "" if step_idx is None else f"_step{int(step_idx):04d}"
            name = f"episode{int(episode_idx):04d}{step_tag}_pred.mp4"
        else:
            name = f"replan{self._replan_counter:06d}_{ts}_pred.mp4"
        return self.predicted_video_save_root / name

    @torch.no_grad()
    def predict(
        self,
        images_client_order: list[np.ndarray],
        state_compact_14d: np.ndarray,
        instruction: str,
        *,
        episode_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
        save_video_flag: Optional[bool] = None,
    ) -> np.ndarray:
        image_tensor = self._build_image_tensor(images_client_order)
        proprio = self._normalize_state(state_compact_14d)
        prompt = build_robotwin_prompt(instruction, quality_score=self.eval_quality_score)
        if self._replan_counter < 3:
            logger.info("[prompt-check %d] ACTUAL prompt sent to model: %r",
                        self._replan_counter, prompt)

        want_video = self._should_save_video(save_video_flag)

        t0 = time.perf_counter()
        common_kwargs = {
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

        if want_video:
            joint_kwargs = dict(common_kwargs)
            joint_kwargs["num_video_frames"] = self.num_video_frames
            # FastWAMJoint.infer_joint forces test_action_with_infer_action=False
            # internally, but pass it explicitly for base-class compatibility.
            if "test_action_with_infer_action" in inspect.signature(self.model.infer_joint).parameters:
                joint_kwargs["test_action_with_infer_action"] = False
            pred = self.model.infer_joint(**joint_kwargs)
            video_frames = pred.get("video")
            if video_frames:
                video_path = self._predicted_video_path(episode_idx, step_idx)
                save_mp4(video_frames, str(video_path), fps=self.predicted_video_fps)
                logger.info("Saved predicted video to %s", video_path)
        else:
            action_kwargs = dict(common_kwargs)
            if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
                action_kwargs["num_video_frames"] = self.num_video_frames
            pred = self.model.infer_action(**action_kwargs)

        action_chunk = self._denormalize_action(pred["action"])[0]  # [T, 11]
        action_chunk = action_chunk.astype(np.float32)
        dt = time.perf_counter() - t0

        # Per-request heartbeat so it's clear the server is alive and what it
        # predicts. first_action = the action executed next; gripper is the last
        # dim (native gripper_close is action[-1] after client 11->12 expansion).
        first_action = action_chunk[0]
        with np.printoptions(precision=3, suppress=True):
            logger.info(
                "[replan %06d] %.3fs | mode=%s | horizon=%d | instr=%r | "
                "first_action=%s | chunk_range=[% .3f, % .3f]",
                self._replan_counter,
                dt,
                "joint+video" if want_video else "action",
                action_chunk.shape[0],
                (instruction[:80] + "...") if len(instruction) > 80 else instruction,
                np.array2string(first_action, separator=", "),
                float(action_chunk.min()),
                float(action_chunk.max()),
            )

        self._replan_counter += 1
        return action_chunk


def _read_image_file(file_storage) -> np.ndarray:
    raw = file_storage.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _form_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _form_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool form value: {value!r}")


def build_app(policy: RoboCasaPolicy):
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    # infer_action is not reentrant on a single GPU model; serialize requests.
    infer_lock = threading.Lock()

    @app.get("/")
    def health():
        return jsonify({"status": "ok", "service": "fastwam-robocasa"})

    @app.post("/process_frame")
    def process_frame():
        image_files = request.files.getlist("image")
        if len(image_files) != 3:
            return jsonify({"error": f"expected 3 images, got {len(image_files)}"}), 400

        text = request.form.get("text", "")
        states_json = request.form.get("states", None)
        if states_json is None:
            return jsonify({"error": "missing 'states' form field"}), 400
        try:
            state = np.asarray(json.loads(states_json), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - report parse errors to client
            return jsonify({"error": f"invalid 'states' JSON: {exc}"}), 400

        images = [_read_image_file(f) for f in image_files]

        # Optional per-episode metadata. Old clients omit these; the server then
        # falls back to its global cadence (predicted_video_save_every).
        episode_idx = _form_optional_int(request.form.get("episode_idx"))
        step_idx = _form_optional_int(request.form.get("step_idx"))
        save_video_flag = _form_optional_bool(request.form.get("save_video"))

        try:
            with infer_lock:
                action_chunk = policy.predict(
                    images,
                    state,
                    text,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    save_video_flag=save_video_flag,
                )
        except Exception as exc:  # noqa: BLE001 - surface inference errors as 500
            logger.exception("Inference failed")
            return jsonify({"error": f"inference failed: {exc}"}), 500

        return jsonify({"response": action_chunk.tolist()})

    return app


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robocasa.yaml")
def main(cfg: DictConfig):
    if _is_none_like(cfg.ckpt):
        raise ValueError("`ckpt` must be provided, e.g. ckpt=/path/to/step_XXXXXX.pt")

    ckpt_path = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if str(cfg.gpu_id).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    device = str(cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"

    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    action_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
    if action_horizon is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    num_video_frames = (
        int(cfg.data.train.num_frames) - 1
    ) // int(cfg.data.train.action_video_freq_ratio) + 1

    num_inference_steps = _parse_optional_int(cfg.EVALUATION.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.eval_num_inference_steps)

    save_predicted_video = bool(cfg.EVALUATION.get("save_predicted_video", False))
    predicted_video_save_root: Optional[Path] = None
    if save_predicted_video:
        root_cfg = cfg.EVALUATION.get("predicted_video_save_root")
        if _is_none_like(root_cfg):
            # Default next to the checkpoint's run directory.
            predicted_video_save_root = ckpt_path.parents[2] / "robocasa_eval_predicted_videos"
        else:
            predicted_video_save_root = Path(
                os.path.expanduser(os.path.expandvars(str(root_cfg)))
            )
        logger.info("Predicted videos will be saved under %s", predicted_video_save_root)

    policy = RoboCasaPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(ckpt_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        num_video_frames=num_video_frames,
        num_inference_steps=num_inference_steps,
        sigma_shift=_parse_optional_float(cfg.EVALUATION.get("sigma_shift")),
        seed=_parse_optional_int(cfg.EVALUATION.get("seed")),
        text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        negative_prompt=str(cfg.EVALUATION.get("negative_prompt", "")),
        rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
        tiled=bool(cfg.EVALUATION.get("tiled", False)),
        save_predicted_video=save_predicted_video,
        predicted_video_save_root=predicted_video_save_root,
        predicted_video_fps=int(cfg.EVALUATION.get("predicted_video_fps", 8)),
        predicted_video_save_every=int(cfg.EVALUATION.get("predicted_video_save_every", 10)),
        eval_quality_score=_parse_optional_int(cfg.EVALUATION.get("quality_score")),
    )

    host = str(cfg.server.get("host", "0.0.0.0"))
    port = int(cfg.server.get("port", 7891))
    app = build_app(policy)
    logger.info("Starting RoboCasa inference server on %s:%d", host, port)
    # threaded=False keeps a single worker; the model is not thread-safe and the
    # /process_frame handler already serializes with a lock.
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()