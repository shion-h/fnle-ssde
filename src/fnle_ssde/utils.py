"""Reusable experiment utilities for NLE training and Gibbs sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.distributions import Distribution, Independent, Uniform

from .nle import NLEEstimator
from .prior import (
    make_prior_config,
    sample_generator_matrix,
    sample_initial_theta_from_normal_prior,
    sample_initial_theta_from_prior,
)

if TYPE_CHECKING:
    from .sampler import ContinuousTimeAR1HMMSampler


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
    training_data_cache_path: Path | None = None,
) -> NLEEstimator:
    """Load an NLE cache or train an NSF from a spline through observations."""
    if theta_lower.shape != theta_upper.shape:
        raise ValueError("theta_lower and theta_upper must have the same shape.")
    if torch.any(theta_lower >= theta_upper):
        raise ValueError("theta_lower must be strictly less than theta_upper.")
    theta_nle_a = dynamics.to_nle_theta(theta_lower)
    theta_nle_b = dynamics.to_nle_theta(theta_upper)
    theta_nle_lower = torch.minimum(theta_nle_a, theta_nle_b)
    theta_nle_upper = torch.maximum(theta_nle_a, theta_nle_b)
    if not torch.all(torch.isfinite(theta_nle_lower)) or not torch.all(
        torch.isfinite(theta_nle_upper)
    ):
        raise ValueError("Physical theta bounds must map to finite NLE coordinates.")
    if torch.any(theta_nle_lower >= theta_nle_upper):
        raise ValueError("Physical theta bounds map to an empty NLE interval.")
    nle = NLEEstimator(
        dynamics=dynamics,
        sampling_dist=Independent(
            Uniform(theta_nle_lower, theta_nle_upper), 1
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
        training_data_cache_path=training_data_cache_path,
    )
    return nle


def initialize_ctmc_path_from_generator(
    sampler: ContinuousTimeAR1HMMSampler,
    Q: torch.Tensor,
    *,
    seed: int | None = None,
) -> None:
    """Initialize true paths by simulating the supplied CTMC generator exactly."""
    Q = torch.as_tensor(Q, dtype=sampler.dtype, device="cpu")
    if Q.shape != (sampler.K, sampler.K):
        raise ValueError(f"Q must have shape ({sampler.K}, {sampler.K}).")
    off_diagonal = Q - torch.diag(torch.diagonal(Q))
    if torch.any(torch.diagonal(Q) >= 0) or torch.any(off_diagonal < 0):
        raise ValueError(
            "Q must have negative diagonal and nonnegative off-diagonal entries."
        )
    if not torch.allclose(Q.sum(dim=1), torch.zeros(sampler.K), atol=1e-6):
        raise ValueError("Rows of Q must sum to zero.")

    generator = sampler.rng
    if seed is not None:
        generator.manual_seed(seed)
    initial_probs = sampler.initial_state_probs.detach().cpu()

    for series_index, duration in enumerate(sampler.T_list):
        current_time = 0.0
        current_state = int(
            torch.multinomial(initial_probs, 1, generator=generator).item()
        )
        jump_times: list[float] = []
        states = [current_state]

        while True:
            exit_rate = float(-Q[current_state, current_state].item())
            uniform = torch.rand(
                (), generator=generator, dtype=sampler.time_dtype
            ).clamp_min(torch.finfo(sampler.time_dtype).tiny)
            next_time = current_time + float(-torch.log(uniform).item() / exit_rate)
            if next_time >= float(duration.item()):
                break

            transition_rates = Q[current_state].clone()
            transition_rates[current_state] = 0.0
            current_state = int(
                torch.multinomial(
                    transition_rates, 1, generator=generator
                ).item()
            )
            jump_times.append(next_time)
            states.append(current_state)
            current_time = next_time

        sampler.T_true_list[series_index] = torch.tensor(
            jump_times, dtype=sampler.time_dtype, device=sampler.device
        )
        sampler.z_true_list[series_index] = torch.tensor(
            states, dtype=torch.long, device=sampler.device
        )


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
    q_alpha: float = 2.0,
    q_beta: float = 20.0,
    seed: int = 0,
    Q: torch.Tensor | None = None,
    initial_path_Q: torch.Tensor | None = None,
    initial_theta: torch.Tensor | None = None,
    initial_log_tau: torch.Tensor | None = None,
    y0_prior_scale: float | torch.Tensor = 0.5,
    y_mh_config: dict[str, Any] | None = None,
    theta_mh_config: dict[str, Any] | None = None,
    y_tree_depth: int = 3,
    theta_tree_depth: int = 3,
    sir_particles: int = 100,
    use_t_pseudo_in_sir: bool = True,
    latest_sample_path: Path | str | None = None,
    theta_prior: Distribution | None = None,
    time_dtype: torch.dtype = torch.float64,
) -> tuple[ContinuousTimeAR1HMMSampler, dict[str, torch.Tensor]]:
    """Construct the common paper sampler and initialize it reproducibly.

    ``theta_prior`` is a component-wise distribution in the real NLE
    coordinate. Convert a physical-scale prior first with
    ``nle.dynamics.pullback_theta_prior(physical_prior)``.

    ``time_dtype`` controls only observation, jump, candidate, and augmented-grid
    times. Model states, parameters, and NLE calls remain float32.

    The initial discrete path is an exact CTMC draw from ``initial_path_Q``
    when supplied, or from the sampler's initial ``Q`` otherwise.

    Set ``use_t_pseudo_in_sir=False`` to marginalize the previous sweep's pseudo
    points rather than retaining their y values as SIR bridge boundaries.
    """
    from .sampler import ContinuousTimeAR1HMMSampler

    seed_all(seed)
    num_regimes = 2 if Q is None else Q.shape[0]
    prior_config = make_prior_config(
        theta_lower.numel(),
        tau2_beta,
        tau2_alpha=tau2_alpha,
        q_alpha=q_alpha,
        q_beta=q_beta,
    )
    if initial_theta is None:
        if theta_prior is None:
            initial_theta = sample_initial_theta_from_normal_prior(
                prior_config,
                theta_lower,
                theta_upper,
                switching_mask,
                dynamics=nle.dynamics,
                num_regimes=num_regimes,
            )
        else:
            initial_theta = sample_initial_theta_from_prior(
                theta_prior,
                theta_lower,
                theta_upper,
                switching_mask,
                dynamics=nle.dynamics,
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
        y_mh_config=(
            y_mh_config
            if y_mh_config is not None
            else {
                "method": "nuts",
                "max_tree_depth": y_tree_depth,
                "target_accept_prob": 0.8,
            }
        ),
        theta_mh_config=(
            theta_mh_config
            if theta_mh_config is not None
            else {
                "method": "nuts",
                "max_tree_depth": theta_tree_depth,
                "target_accept_prob": 0.8,
            }
        ),
        sir_config={
            "num_particles": sir_particles,
            "use_t_pseudo_in_sir": use_t_pseudo_in_sir,
        },
        prior_config=prior_config,
        theta_prior=theta_prior,
        y0_prior_loc=x_obs[0],
        y0_prior_scale=torch.as_tensor(
            y0_prior_scale, dtype=x_obs.dtype, device=x_obs.device
        ).broadcast_to((x_obs.shape[1],)),
        dtype=torch.float32,
        time_dtype=time_dtype,
        seed=seed,
        latest_sample_path=latest_sample_path,
    )
    sampler.initialize(
        initial_theta=initial_theta,
        initial_log_tau=initial_log_tau,
        initial_state_probs=torch.full((num_regimes,), 1.0 / num_regimes),
    )
    path_generator_Q = Q if initial_path_Q is None else initial_path_Q
    initialize_ctmc_path_from_generator(sampler, path_generator_Q, seed=seed)
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
    *,
    dynamics: Any | None = None,
) -> float:
    """Return the fraction of sweeps with theta outside physical NLE support.

    Omitting ``dynamics`` retains the historical exponential conversion.
    """
    theta_nle = torch.stack(history["theta"])
    theta = (
        theta_nle.exp()
        if dynamics is None
        else dynamics.to_physical_theta(theta_nle)
    )
    outside = ((theta < theta_lower) | (theta > theta_upper)).flatten(1).any(1)
    return float(outside.float().mean())
