"""Optional FastWAM-Joint extension: mixed expert / low-quality data and skewed training timesteps."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .fastwam_joint import FastWAMJoint
from .joint_mixed_timesteps import beta_sigma_timesteps


class FastWAMJointMixedQuality(FastWAMJoint):
    """Joint-mixed training with MoT bidirectional attention on noisy video tokens only.

    Unlike ``FastWAMJoint``, the clean first latent frame does not exchange attention with action;
    all later (noisy) video tokens and action tokens attend each other in MoT.

    - Expert samples (``supervise_action == 1``): video timestep noise ~ Beta(7,1) on sigma; action timestep
      uses the default continuous flow-matching sampler (uniform in shifted time).
    - Low-quality samples (``supervise_action == 0``): video timestep uses the default sampler; action
      timestep is fixed to 0, so action tokens are clean. Action loss is masked out (video-only supervision).

    If ``supervise_action`` is absent, behavior matches ``FastWAMJoint`` / ``FastWAM.training_loss``.
    """

    @classmethod
    def from_wan22_pretrained(cls, joint_mixed_training: Optional[dict[str, Any]] = None, **kwargs):
        model = super().from_wan22_pretrained(**kwargs)
        jm = joint_mixed_training or {}
        ev = jm.get("expert_video_beta", (7.0, 1.0))
        wa = jm.get("weak_action_beta", (1.0, 7.0))
        model._jm_expert_video_beta = (float(ev[0]), float(ev[1]))
        model._jm_weak_action_beta = (float(wa[0]), float(wa[1]))
        return model

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video(query) -> video(key): keep video-only topology from video_expert
        # (for `first_frame_causal`, first-frame queries cannot see later-frame keys).
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action(query) -> action(key): full bidirectional attention within action tokens.
        mask[video_seq_len:, video_seq_len:] = True

        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        if video_seq_len > first_frame_tokens:
            # action(query) -> all-video(key): action can read all video tokens, including first frame.
            mask[video_seq_len:, :video_seq_len] = True
            # mask[video_seq_len:, first_frame_tokens:video_seq_len] = True
            if action_seq_len > 0:
                # noisy-video(query) -> action(key): non-first-frame video can read action tokens.
                # Note: first-frame video(query) -> action(key) remains masked out.
                mask[first_frame_tokens:video_seq_len, video_seq_len:] = True
        return mask

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
        ta_weak = torch.zeros_like(ta_expert)
        timestep_action = torch.where(expert_m, ta_expert, ta_weak)

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
        mask = supervise.to(device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype).flatten()
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
        }
        return loss_total, loss_dict
