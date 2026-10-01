"""Prior construction and reproducible initialization utilities."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.distributions import Distribution, Gamma, Normal


def make_prior_config(
    theta_dim: int,
    tau2_beta: float | torch.Tensor,
    *,
    theta_loc: float = 0.0,
    theta_scale: float = 10.0,
    tau2_alpha: float | torch.Tensor = 3.0,
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


def sample_initial_theta_from_normal_prior(
    prior_config: dict[str, Any],
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    switching_mask: torch.Tensor,
    *,
    dynamics: Any,
    num_regimes: int,
) -> torch.Tensor:
    """Draw legacy-Normal theta inside physical NLE-training bounds.

    The Normal prior is defined in the real-valued NLE coordinate. Physical
    bounds are mapped back through the transform owned by ``dynamics``.
    """
    if theta_lower.shape != theta_upper.shape:
        raise ValueError("theta_lower and theta_upper must have the same shape.")
    if switching_mask.shape != theta_lower.shape:
        raise ValueError("switching_mask must have the same shape as theta bounds.")
    if torch.any(theta_lower >= theta_upper):
        raise ValueError("theta_lower must be strictly less than theta_upper.")
    if num_regimes < 1:
        raise ValueError("num_regimes must be positive.")

    bound_a = dynamics.to_nle_theta(theta_lower)
    bound_b = dynamics.to_nle_theta(theta_upper)
    lower = torch.minimum(bound_a, bound_b)
    upper = torch.maximum(bound_a, bound_b)
    if not torch.all(torch.isfinite(lower)) or not torch.all(torch.isfinite(upper)):
        raise ValueError("Physical theta bounds must map to finite NLE coordinates.")
    if torch.any(lower >= upper):
        raise ValueError("Physical theta bounds map to an empty NLE interval.")

    loc = torch.as_tensor(
        prior_config["theta_loc"], dtype=lower.dtype, device=lower.device
    ).broadcast_to(lower.shape)
    scale = torch.as_tensor(
        prior_config["theta_scale"], dtype=lower.dtype, device=lower.device
    ).broadcast_to(lower.shape)
    theta = sample_truncated_normal(
        loc,
        scale,
        lower,
        upper,
        sample_shape=torch.Size((num_regimes,)),
    )
    switching = switching_mask.to(device=theta.device, dtype=torch.bool)
    theta[:, ~switching] = theta[0, ~switching]
    return theta


def sample_initial_theta_from_prior(
    theta_prior: Distribution,
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    switching_mask: torch.Tensor,
    *,
    dynamics: Any,
    num_regimes: int,
    max_attempts: int = 1_000,
) -> torch.Tensor:
    """Draw NLE-coordinate theta whose physical value lies in NLE support."""
    if not isinstance(theta_prior, Distribution):
        raise TypeError("theta_prior must be a torch Distribution.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    if num_regimes < 1:
        raise ValueError("num_regimes must be positive.")
    if theta_lower.shape != theta_upper.shape:
        raise ValueError("theta_lower and theta_upper must have the same shape.")
    if switching_mask.shape != theta_lower.shape:
        raise ValueError("switching_mask must have the same shape as theta bounds.")
    if torch.any(theta_lower >= theta_upper):
        raise ValueError("theta_lower must be strictly less than theta_upper.")
    if theta_prior.batch_shape != theta_lower.shape:
        raise ValueError(
            "theta_prior batch_shape must match the physical theta bounds."
        )
    if theta_prior.event_shape != torch.Size():
        raise ValueError("theta_prior must have an empty event_shape.")

    for _ in range(max_attempts):
        theta = theta_prior.sample((num_regimes,)).clone()
        switching = switching_mask.to(device=theta.device, dtype=torch.bool)
        theta[:, ~switching] = theta[0, ~switching]
        physical_theta = dynamics.to_physical_theta(theta)
        lower = theta_lower.to(
            device=physical_theta.device,
            dtype=physical_theta.dtype,
        )
        upper = theta_upper.to(
            device=physical_theta.device,
            dtype=physical_theta.dtype,
        )
        if torch.all(torch.isfinite(physical_theta)) and torch.all(
            (physical_theta >= lower) & (physical_theta <= upper)
        ):
            return theta

    raise RuntimeError(
        "Could not draw initial theta inside the physical NLE support after "
        f"{max_attempts} attempts."
    )


def sample_initial_theta_from_truncated_physical_prior(
    *,
    dynamics: Any,
    physical_theta_prior: Distribution | Sequence[Distribution],
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    switching_mask: torch.Tensor,
    num_regimes: int,
) -> torch.Tensor:
    """Draw from a physical prior truncated to NLE support, then transform.

    A sequence of scalar distributions permits different prior families for
    different physical parameters. A single batched distribution retains the
    existing sampling path, including its random-number and dtype behavior.
    """
    if isinstance(physical_theta_prior, Distribution):
        lower_cdf = physical_theta_prior.cdf(theta_lower)
        upper_cdf = physical_theta_prior.cdf(theta_upper)
        probabilities = lower_cdf + torch.rand(
            (num_regimes, theta_lower.numel()),
            dtype=lower_cdf.dtype,
            device=lower_cdf.device,
        ) * (upper_cdf - lower_cdf)
        physical_theta = physical_theta_prior.icdf(probabilities)
    else:
        if len(physical_theta_prior) != theta_lower.numel():
            raise ValueError("One physical prior is required per theta component.")
        lower_cdf = torch.stack([
            prior.cdf(bound.to(torch.float64))
            for prior, bound in zip(physical_theta_prior, theta_lower)
        ])
        upper_cdf = torch.stack([
            prior.cdf(bound.to(torch.float64))
            for prior, bound in zip(physical_theta_prior, theta_upper)
        ])
        probabilities = lower_cdf + torch.rand(
            (num_regimes, theta_lower.numel()),
            dtype=lower_cdf.dtype,
            device=lower_cdf.device,
        ) * (upper_cdf - lower_cdf)
        physical_theta = torch.stack([
            prior.icdf(probabilities[:, index])
            for index, prior in enumerate(physical_theta_prior)
        ], dim=-1)
        physical_theta = physical_theta.to(theta_lower.dtype)
    physical_theta[:, ~switching_mask] = physical_theta[0, ~switching_mask]
    return dynamics.to_nle_theta(physical_theta)


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
