"""FastWAM-Joint with bidirectional MoT attention between action and noisy video tokens."""

from __future__ import annotations

import torch

from .fastwam_joint import FastWAMJoint


class FastWAMJointBidirectional(FastWAMJoint):
    """Same training/inference as ``FastWAMJoint``, but noisy video tokens also attend action.

    Compared to ``FastWAMJoint``:
    - action -> all video tokens (unchanged)
    - noisy video tokens (latent frames after the first) -> action (added)
    """

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

        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        
        # action(query) -> all-video(key): action can read all video tokens, including first frame.
        mask[video_seq_len:, :video_seq_len] = True

        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        if video_seq_len > first_frame_tokens and action_seq_len > 0:
            mask[first_frame_tokens:video_seq_len, video_seq_len:] = True
        return mask
