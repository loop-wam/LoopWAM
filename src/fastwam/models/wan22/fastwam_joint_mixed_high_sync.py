"""High-quality finetuning variant with synchronized video/action timesteps."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .fastwam_joint_mixed import FastWAMJointMixedQuality


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


class FastWAMJointMixedQualityHighSync(FastWAMJointMixedQuality):
    """High-quality-only finetuning with shared video/action timestep samples.

    The base mixed-quality model samples expert video and action timesteps independently.
    For this finetuning path, action keeps the high-quality action sampler and video reuses
    the same sampled timestep value, so both supervised targets are trained at one noise level.
    """

    def training_loss(self, sample, tiled: bool = False):
        sup = sample.get("supervise_action", None)
        if sup is None:
            return super().training_loss(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        supervise = sup.to(device=self.device, dtype=torch.float32).flatten()
        if int(supervise.numel()) != int(batch_size):
            raise ValueError(
                f"`supervise_action` length mismatch: got {tuple(sup.shape)} vs batch_size={batch_size}"
            )

        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        timestep_video = timestep_action.to(dtype=input_latents.dtype)

        noise_video = torch.randn_like(input_latents)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
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
        loss_video = (loss_video_per_sample.flatten() * video_weight.flatten()).mean()

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
        mask = supervise.to(device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype).flatten()
        denom = mask.sum().clamp(min=1.0)
        loss_action = (action_loss_per_sample.flatten() * action_weight.flatten() * mask).sum() / denom

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if loss_total.ndim != 0:
            loss_total = loss_total.mean()
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "supervise_action_mean": float(supervise.detach().mean().item()),
        }
        return loss_total, loss_dict


def create_fastwam_joint_mixed_high_sync(
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
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    joint_mixed_training: Optional[dict] = None,
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

    return FastWAMJointMixedQualityHighSync.from_wan22_pretrained(
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
        joint_mixed_training=joint_mixed_training,
    )
