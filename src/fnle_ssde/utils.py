"""Reusable experiment utilities for NLE training and Gibbs sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.distributions import Independent, Uniform

from .common.nle import NLEEstimator
from .prior import (
    make_prior_config,
    sample_generator_matrix,
    sample_initial_log_theta,
)

if TYPE_CHECKING:
    from .continuous import ContinuousTimeAR1HMMSampler


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_spline_nle(
    *,
    dynamics: Any,
    cache_path: Path,
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    obs_times: torch.Tensor,
    x_obs: torch.Tensor,
    reference_times: torch.Tensor,
    n_params: int,
    ref_noise: float,
    max_n_steps: int,
    epochs: int,
    stop_after_epochs: int = 20,
    seed: int = 0,
    noisy_init_strategy: str = "clamp",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
) -> NLEEstimator:
    """Load an NLE cache or train an NSF from a spline through observations."""
    nle = NLEEstimator(
        dynamics=dynamics,
        sampling_dist=Independent(
            Uniform(torch.log(theta_lower), torch.log(theta_upper)), 1
        ),
        device="cpu",
        model_cache_path=cache_path,
        target_type="scaled_dx",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins,
    )
    if nle.estimator is not None:
        return nle

    seed_all(seed)
    print(
        f"training NLE: parameters={n_params}, max_n_steps={max_n_steps}, "
        f"cache={cache_path}"
    )
    nle.train(
        obs_times=obs_times,
        x_obs=x_obs,
        reference_times=reference_times,
        n_params=n_params,
        ref_noize=ref_noise,
        max_n_steps=max_n_steps,
        batch_size=256,
        lr=5e-4,
        epochs=epochs,
        stop_after_epochs=stop_after_epochs,
        samples_per_theta=1,
        state_clamp_bounds=(0.0, 1e4),
        noisy_init_strategy=noisy_init_strategy,
        max_noisy_init_attempts=1_000,
    )
    return nle


def initialize_random_ctmc_path(
    sampler: ContinuousTimeAR1HMMSampler,
    *,
    expected_num_intervals: float = 20.0,
    seed: int = 0,
) -> None:
    """Initialize each series with random jump times and non-self transitions."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for series_index, duration in enumerate(sampler.T_list):
        num_intervals = 1 + int(
            torch.poisson(
                torch.tensor(expected_num_intervals - 1.0), generator=generator
            )
        )
        jump_times = torch.sort(
            torch.rand(num_intervals - 1, generator=generator) * duration
        ).values.to(device=sampler.device, dtype=sampler.dtype)
        states = torch.empty(num_intervals, dtype=torch.long)
        states[0] = torch.randint(sampler.K, (1,), generator=generator)
        for index in range(1, num_intervals):
            candidate = torch.randint(sampler.K - 1, (1,), generator=generator)
            states[index] = candidate + (candidate >= states[index - 1]).long()
        sampler.T_true_list[series_index] = jump_times
        sampler.z_true_list[series_index] = states.to(sampler.device)


def initialize_gibbs_sampler(
    *,
    nle: NLEEstimator,
    x_obs: torch.Tensor,
    obs_times: torch.Tensor,
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    switching_mask: torch.Tensor,
    tau2_beta: float | torch.Tensor,
    tau2_alpha: float = 3.0,
    seed: int = 0,
    Q: torch.Tensor | None = None,
    initial_theta: torch.Tensor | None = None,
    initial_log_tau: torch.Tensor | None = None,
    y0_prior_scale: float | torch.Tensor = 0.5,
    y_tree_depth: int = 3,
    theta_tree_depth: int = 3,
    sir_particles: int = 100,
) -> tuple[ContinuousTimeAR1HMMSampler, dict[str, torch.Tensor]]:
    """Construct the common paper sampler and initialize it reproducibly."""
    from .continuous import ContinuousTimeAR1HMMSampler

    seed_all(seed)
    num_regimes = 2 if Q is None else Q.shape[0]
    prior_config = make_prior_config(
        theta_lower.numel(), tau2_beta, tau2_alpha=tau2_alpha
    )
    if initial_theta is None:
        initial_theta = sample_initial_log_theta(
            prior_config,
            theta_lower,
            theta_upper,
            switching_mask,
            num_regimes=num_regimes,
        )
    if Q is None:
        Q = sample_generator_matrix(
            num_regimes,
            q_alpha=prior_config["q_alpha"],
            q_beta=prior_config["q_beta"],
        )

    sampler = ContinuousTimeAR1HMMSampler(
        Q=Q,
        x_obs=x_obs,
        obs_times=obs_times,
        T=float(obs_times[-1]),
        omega_scale=3.0,
        nle_estimator=nle,
        switching_parameter_mask=switching_mask,
        y_nuts_config={"max_tree_depth": y_tree_depth, "target_accept_prob": 0.8},
        theta_nuts_config={
            "max_tree_depth": theta_tree_depth,
            "target_accept_prob": 0.8,
        },
        sir_config={"num_particles": sir_particles},
        prior_config=prior_config,
        y0_prior_loc=x_obs[0],
        y0_prior_scale=torch.as_tensor(
            y0_prior_scale, dtype=x_obs.dtype, device=x_obs.device
        ).broadcast_to((x_obs.shape[1],)),
        dtype=torch.float32,
        seed=seed,
    )
    sampler.initialize(
        initial_theta=initial_theta,
        initial_log_tau=initial_log_tau,
        initial_state_probs=torch.full((num_regimes,), 1.0 / num_regimes),
    )
    initialize_random_ctmc_path(sampler, seed=seed)
    return sampler, {"initial_Q": Q, "initial_theta": initial_theta}


def run_after_burn_in(
    sampler: ContinuousTimeAR1HMMSampler,
    *,
    num_sweeps: int,
    burn_in: int,
    progress_every: int = 100,
) -> dict[str, list[object]]:
    """Run Gibbs sweeps and retain only draws after burn-in."""
    if not 0 <= burn_in < num_sweeps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < num_sweeps.")
    retained = {key: [] for key in sampler.history}
    for sweep in range(num_sweeps):
        info = sampler.one_sweep()
        if sweep >= burn_in:
            for key in retained:
                retained[key].append(sampler.history[key][-1])
        sampler.history = {key: [] for key in sampler.history}
        if progress_every and (sweep + 1) % progress_every == 0:
            print(
                f"[sweep {sweep + 1:05d}] grid={info['grid_size']}, "
                f"true_segments={info['num_true_segments']}"
            )
    return retained


def posterior_outside_support(
    history: dict[str, list[object]],
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
) -> float:
    """Return the fraction of sweeps with any theta outside the NLE support."""
    theta = torch.stack(history["theta"]).exp()
    outside = ((theta < theta_lower) | (theta > theta_upper)).flatten(1).any(1)
    return float(outside.float().mean())
