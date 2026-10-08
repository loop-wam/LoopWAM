"""Timestep helpers for mixed-quality joint training (optional code path)."""

from __future__ import annotations

import torch


def beta_sigma_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    concentration1: float,
    concentration0: float,
    num_train_timesteps: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Sample timesteps = k * num_train_timesteps with k ~ Beta(concentration1, concentration0).

    In FastWAM's flow-matching scheduler, ``add_noise`` uses ``sigma = timestep / num_train_timesteps``
    as the noise fraction. So ``k`` is the (scalar) noise level in ``[0, 1]``.

    - Beta(7, 1) skews ``k`` high → high noise (matches expert-video preference in the paper-style recipe).
    - Beta(1, 7) skews ``k`` low → low noise (matches weak-action preference).
    """
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}")
    conc1 = torch.tensor(float(concentration1), device=device, dtype=torch.float32)
    conc0 = torch.tensor(float(concentration0), device=device, dtype=torch.float32)
    dist = torch.distributions.Beta(concentration1=conc1, concentration0=conc0)
    k = dist.sample((batch_size,)).clamp(float(eps), 1.0 - float(eps))
    return (k * float(num_train_timesteps)).to(dtype=dtype)
