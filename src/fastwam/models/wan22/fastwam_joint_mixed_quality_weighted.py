"""Mixed-quality FastWAM-Joint with quality-weighted action supervision."""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .fastwam_joint_mixed import FastWAMJointMixedQuality
from .joint_mixed_timesteps import beta_sigma_timesteps


def _to_plain_dict(value, *, name: str, required: bool = False):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be dict-like, got {type(value)}")
    return value


class FastWAMJointMixedQualityWeighted(FastWAMJointMixedQuality):
    """Use prompt quality values as per-sample action-loss weights.

    ``supervise_action`` still controls the mixed-quality timestep branch, preserving the
    original expert-vs-low-quality training schedule. The action loss mask is instead
    derived from the resolved ``Quality: ...`` suffix in each prompt.
    """

    _QUALITY_PATTERN = re.compile(r"\bQuality:\s*([A-Za-z]+|\d+(?:\.\d+)?)\s*\.", flags=re.IGNORECASE)
    _QUALITY_LABELS = frozenset({"low", "medium", "high"})
    _NUMERIC_QUALITY_PATTERN = re.compile(r"\d+(?:\.\d+)?")

    @classmethod
    def from_wan22_pretrained(
        cls,
        joint_mixed_training: Optional[dict[str, Any]] = None,
        action_loss_quality_weights: Optional[dict[Any, Any]] = None,
        action_loss_quality_score_linear: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(joint_mixed_training=joint_mixed_training, **kwargs)
        model._action_loss_quality_weights = cls._normalize_quality_weights(action_loss_quality_weights)
        model._action_loss_quality_score_linear = cls._normalize_quality_score_linear(
            action_loss_quality_score_linear
        )
        return model

    @classmethod
    def _normalize_quality_key(cls, raw_value: Any) -> int | str:
        if isinstance(raw_value, bool):
            raise ValueError(f"Invalid quality key: {raw_value!r}")
        if isinstance(raw_value, int):
            score = raw_value
        else:
            text = str(raw_value).strip().lower()
            if text.startswith("quality:"):
                text = text[len("quality:") :].strip()
            text = text.rstrip(".")
            if text.isdigit():
                score = int(text)
            else:
                if text not in cls._QUALITY_LABELS:
                    raise ValueError(
                        f"Invalid quality key {raw_value!r}. "
                        "Expected score 1-5 or one of: high, medium, low."
                    )
                return text
        if score < 1 or score > 5:
            raise ValueError(f"Quality score key must be in [1, 5], got {score}.")
        return score

    @classmethod
    def _normalize_quality_weights(cls, value: Optional[dict[Any, Any]]) -> dict[int | str, float]:
        if value is None:
            value = {
                1: 0.0,
                2: 0.0,
                3: 0.3,
                4: 0.6,
                5: 1.0,
                "low": 0.0,
                "medium": 0.5,
                "high": 1.0,
            }
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise ValueError("`action_loss_quality_weights` must be a mapping from quality value to weight.")

        weights = {}
        for raw_quality, raw_weight in value.items():
            quality = cls._normalize_quality_key(raw_quality)
            weight = float(raw_weight)
            if weight < 0.0:
                raise ValueError(f"`action_loss_quality_weights[{raw_quality}]` must be non-negative, got {weight}.")
            weights[quality] = weight
        if not weights:
            raise ValueError("`action_loss_quality_weights` must not be empty.")
        return weights

    @classmethod
    def _normalize_quality_score_linear(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, float]]:
        """Parse the optional linear score->weight mapping config.

        Expected keys: score_min, score_max, weight_min, weight_max, and optionally
        expert_score / expert_weight for a fixed expert score outside the linear range
        (e.g. real-robot expert trajectories tagged with score 12).
        """
        if value is None:
            return None
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise ValueError(
                "`action_loss_quality_score_linear` must be a mapping with "
                "score_min/score_max/weight_min/weight_max, got "
                f"{type(value).__name__}."
            )
        required = {"score_min", "score_max", "weight_min", "weight_max"}
        missing = required - set(value.keys())
        if missing:
            raise ValueError(
                f"`action_loss_quality_score_linear` missing required keys: {sorted(missing)}."
            )
        cfg = {
            "score_min": float(value["score_min"]),
            "score_max": float(value["score_max"]),
            "weight_min": float(value["weight_min"]),
            "weight_max": float(value["weight_max"]),
            "expert_score": None if value.get("expert_score") is None else float(value["expert_score"]),
            "expert_weight": float(value.get("expert_weight", 1.0)),
        }
        if cfg["score_max"] <= cfg["score_min"]:
            raise ValueError(
                "`action_loss_quality_score_linear.score_max` must be > score_min, got "
                f"[{cfg['score_min']}, {cfg['score_max']}]."
            )
        for key in ("weight_min", "weight_max", "expert_weight"):
            if cfg[key] < 0.0:
                raise ValueError(f"`action_loss_quality_score_linear.{key}` must be non-negative.")
        return cfg

    def _linear_weight_from_score(self, score: float) -> float:
        cfg = self._action_loss_quality_score_linear
        expert_score = cfg["expert_score"]
        if expert_score is not None and abs(score - expert_score) < 1e-6:
            return cfg["expert_weight"]
        t = (score - cfg["score_min"]) / (cfg["score_max"] - cfg["score_min"])
        t = min(max(t, 0.0), 1.0)
        return cfg["weight_min"] + t * (cfg["weight_max"] - cfg["weight_min"])

    @classmethod
    def _raw_quality_from_prompt(cls, prompt: str) -> str:
        match = cls._QUALITY_PATTERN.search(str(prompt))
        if match is None:
            raise ValueError(f"Unable to parse `Quality: ...` from prompt: {prompt!r}")
        return match.group(1)

    @classmethod
    def _quality_key_from_prompt(cls, prompt: str) -> int | str:
        return cls._normalize_quality_key(cls._raw_quality_from_prompt(prompt))

    def _action_loss_weight_from_prompt(self, prompt: str) -> float:
        raw = self._raw_quality_from_prompt(prompt)
        linear_cfg = getattr(self, "_action_loss_quality_score_linear", None)
        if linear_cfg is not None and self._NUMERIC_QUALITY_PATTERN.fullmatch(raw.strip()):
            return self._linear_weight_from_score(float(raw))
        quality = self._normalize_quality_key(raw)
        if quality not in self._action_loss_quality_weights:
            raise ValueError(
                f"`action_loss_quality_weights` has no entry for parsed prompt quality {quality!r}. "
                f"Prompt: {prompt!r}"
            )
        return self._action_loss_quality_weights[quality]

    def _action_loss_mask_from_prompt(self, sample, batch_size: int, *, device, dtype) -> torch.Tensor:
        prompts = sample.get("prompt", None)
        if prompts is None:
            raise ValueError("Quality-weighted action loss requires `prompt` in the batch.")
        if isinstance(prompts, str):
            prompts = [prompts]
        else:
            prompts = list(prompts)
        if len(prompts) != int(batch_size):
            raise ValueError(f"`prompt` batch length mismatch: got {len(prompts)} vs batch_size={batch_size}")

        weights = [self._action_loss_weight_from_prompt(str(prompt)) for prompt in prompts]
        return torch.tensor(weights, device=device, dtype=dtype).flatten()

    def training_loss(self, sample, tiled: bool = False):
        sup = sample.get("supervise_action", None)
        if sup is None:
            return super().training_loss(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        video_context = inputs.get("video_context", inputs["context"])
        video_context_mask = inputs.get("video_context_mask", inputs["context_mask"])
        action_context = inputs.get("action_context", inputs["context"])
        action_context_mask = inputs.get("action_context_mask", inputs["context_mask"])
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        supervise = sup.to(device=self.device, dtype=torch.float32).flatten()
        if int(supervise.numel()) != int(batch_size):
            raise ValueError(
                f"`supervise_action` length mismatch: got {tuple(sup.shape)} vs batch_size={batch_size}"
            )
        expert_m = supervise > 0.5

        noise_video = torch.randn_like(input_latents)
        tv_expert = beta_sigma_timesteps(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
            concentration1=self._jm_expert_video_beta[0],
            concentration0=self._jm_expert_video_beta[1],
            num_train_timesteps=self.train_video_scheduler.num_train_timesteps,
        )
        tv_weak = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        timestep_video = torch.where(expert_m, tv_expert, tv_weak)

        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        ta_expert = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        ta_weak = beta_sigma_timesteps(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
            concentration1=self._jm_weak_action_beta[0],
            concentration0=self._jm_weak_action_beta[1],
            num_train_timesteps=self.train_action_scheduler.num_train_timesteps,
        )
        timestep_action = torch.where(expert_m, ta_expert, ta_weak)

        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (
            loss_video_per_sample.flatten() * video_weight.flatten()
        ).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        mask = self._action_loss_mask_from_prompt(
            sample,
            batch_size,
            device=action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        denom = mask.sum().clamp(min=1.0)
        loss_action = (
            action_loss_per_sample.flatten() * action_weight.flatten() * mask
        ).sum() / denom

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if loss_total.ndim != 0:
            loss_total = loss_total.mean()
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "supervise_action_mean": float(supervise.detach().mean().item()),
            "action_quality_weight_mean": float(mask.detach().mean().item()),
        }
        return loss_total, loss_dict


def create_fastwam_joint_mixed_quality_weighted(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    video_cross_attend_proprio: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    joint_mixed_training: Optional[dict] = None,
    action_loss_quality_weights: Optional[dict] = None,
    action_loss_quality_score_linear: Optional[dict] = None,
):
    video_dit_config = _to_plain_dict(video_dit_config, name="video_dit_config", required=True)
    action_dit_config = _to_plain_dict(action_dit_config, name="action_dit_config")
    video_scheduler = _to_plain_dict(video_scheduler, name="video_scheduler")
    action_scheduler = _to_plain_dict(action_scheduler, name="action_scheduler", required=True)
    loss = _to_plain_dict(loss, name="loss")
    if isinstance(joint_mixed_training, DictConfig):
        joint_mixed_training = OmegaConf.to_container(joint_mixed_training, resolve=True)
    if joint_mixed_training is not None and not isinstance(joint_mixed_training, dict):
        raise ValueError(f"`joint_mixed_training` must be dict-like, got {type(joint_mixed_training)}")

    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    return FastWAMJointMixedQualityWeighted.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        video_cross_attend_proprio=bool(video_cross_attend_proprio),
        joint_mixed_training=joint_mixed_training,
        action_loss_quality_weights=action_loss_quality_weights,
        action_loss_quality_score_linear=action_loss_quality_score_linear,
    )
