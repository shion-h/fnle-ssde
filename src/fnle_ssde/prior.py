"""Prior construction and reproducible initialization utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Gamma, Normal


def make_prior_config(
    theta_dim: int,
    tau2_beta: float | torch.Tensor,
    *,
    theta_loc: float = 0.0,
    theta_scale: float = 10.0,
    tau2_alpha: float = 3.0,
    q_alpha: float = 2.0,
    q_beta: float = 20.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build the prior configuration used by the continuous-time sampler."""
    return {
        "theta_loc": torch.full(
            (theta_dim,), theta_loc, dtype=dtype, device=device
        ),
        "theta_scale": torch.full(
            (theta_dim,), theta_scale, dtype=dtype, device=device
        ),
        "tau2_alpha": tau2_alpha,
        "tau2_beta": torch.as_tensor(tau2_beta, dtype=dtype, device=device),
        "q_alpha": q_alpha,
        "q_beta": q_beta,
    }


def sample_truncated_normal(
    loc: torch.Tensor,
    scale: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    sample_shape: torch.Size = torch.Size(),
) -> torch.Tensor:
    """Sample independent normals restricted componentwise to ``[lower, upper]``."""
    loc, scale, lower, upper = torch.broadcast_tensors(loc, scale, lower, upper)
    if torch.any(scale <= 0) or torch.any(lower >= upper):
        raise ValueError("Normal scales must be positive and lower < upper.")

    normal = Normal(loc, scale)
    cdf_lower = normal.cdf(lower)
    cdf_upper = normal.cdf(upper)
    if torch.any(cdf_upper <= cdf_lower):
        raise ValueError("Truncation interval has zero probability at this precision.")

    shape = sample_shape + loc.shape
    uniform = torch.rand(shape, dtype=loc.dtype, device=loc.device)
    probability = cdf_lower + uniform * (cdf_upper - cdf_lower)
    eps = torch.finfo(loc.dtype).eps
    return normal.icdf(probability.clamp(min=eps, max=1.0 - eps))


def sample_initial_log_theta(
    prior_config: dict[str, Any],
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    switching_mask: torch.Tensor,
    *,
    num_regimes: int,
) -> torch.Tensor:
    """Draw log-theta from its prior truncated to the physical NLE support."""
    lower = torch.log(theta_lower)
    upper = torch.log(theta_upper)
    loc = torch.as_tensor(
        prior_config["theta_loc"], dtype=lower.dtype, device=lower.device
    ).broadcast_to(lower.shape)
    scale = torch.as_tensor(
        prior_config["theta_scale"], dtype=lower.dtype, device=lower.device
    ).broadcast_to(lower.shape)
    switching_mask = switching_mask.to(device=lower.device, dtype=torch.bool)

    theta = sample_truncated_normal(
        loc,
        scale,
        lower,
        upper,
        sample_shape=torch.Size((num_regimes,)),
    )
    # A shared component is one random variable, not one draw per regime.
    theta[:, ~switching_mask] = theta[0, ~switching_mask]
    return theta


def sample_generator_matrix(
    num_regimes: int,
    *,
    q_alpha: float,
    q_beta: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw off-diagonal CTMC rates independently from a Gamma(shape, rate) prior."""
    concentration = torch.as_tensor(q_alpha, dtype=dtype, device=device)
    rate = torch.as_tensor(q_beta, dtype=dtype, device=device)
    rates = Gamma(concentration, rate).sample((num_regimes, num_regimes))
    rates.fill_diagonal_(0.0)
    rates.diagonal().copy_(-rates.sum(dim=1))
    return rates
