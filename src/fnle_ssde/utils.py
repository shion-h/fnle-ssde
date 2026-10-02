"""Reusable experiment utilities for NLE training and MCMC inference."""

from __future__ import annotations

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch
from torch.distributions import Distribution, Independent, Uniform

from .nle import NLEEstimator
from .prior import (
    make_prior_config,
    sample_generator_matrix,
    sample_initial_theta_from_normal_prior,
    sample_initial_theta_from_prior,
    sample_initial_theta_from_truncated_physical_prior,
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
    state_clamp_bounds: tuple[float | None, float | None] | None = (0.0, 1e4),
    spline_clamp_bounds: tuple[float | None, float | None] | None = (0.0, None),
) -> NLEEstimator:
    """Load an NLE cache or train an NSF from a spline through observations."""
    if theta_lower.shape != theta_upper.shape:
        raise ValueError("theta_lower and theta_upper must have the same shape.")
    if torch.any(theta_lower >= theta_upper):
        raise ValueError("theta_lower must be strictly less than theta_upper.")
    theta_nle_lower, theta_nle_upper = dynamics.physical_bounds_to_nle_bounds(
        theta_lower, theta_upper
    )
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
        state_clamp_bounds=state_clamp_bounds,
        spline_clamp_bounds=spline_clamp_bounds,
        noisy_init_strategy=noisy_init_strategy,
        max_noisy_init_attempts=1_000,
        training_data_cache_path=training_data_cache_path,
    )
    return nle


def make_symmetric_generator(
    num_regimes: int,
    jump_rate: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a symmetric generator with total exit rate ``jump_rate``."""
    if num_regimes < 2:
        raise ValueError("num_regimes must be at least two.")
    if jump_rate <= 0:
        raise ValueError("jump_rate must be positive.")
    off_diagonal_rate = jump_rate / (num_regimes - 1)
    Q = torch.full(
        (num_regimes, num_regimes),
        off_diagonal_rate,
        dtype=dtype,
        device=device,
    )
    Q.fill_diagonal_(-jump_rate)
    return Q


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
    initial_probs = sampler.initial_regime_probs.detach().cpu()

    for series_index, duration in enumerate(sampler.T_list):
        current_time = 0.0
        current_regime = int(
            torch.multinomial(initial_probs, 1, generator=generator).item()
        )
        jump_times: list[float] = []
        regimes = [current_regime]

        while True:
            exit_rate = float(-Q[current_regime, current_regime].item())
            uniform = torch.rand(
                (), generator=generator, dtype=sampler.time_dtype
            ).clamp_min(torch.finfo(sampler.time_dtype).tiny)
            next_time = current_time + float(-torch.log(uniform).item() / exit_rate)
            if next_time >= float(duration.item()):
                break

            transition_rates = Q[current_regime].clone()
            transition_rates[current_regime] = 0.0
            current_regime = int(
                torch.multinomial(
                    transition_rates, 1, generator=generator
                ).item()
            )
            jump_times.append(next_time)
            regimes.append(current_regime)
            current_time = next_time

        sampler.T_true_list[series_index] = torch.tensor(
            jump_times, dtype=sampler.time_dtype, device=sampler.device
        )
        sampler.z_true_list[series_index] = torch.tensor(
            regimes, dtype=torch.long, device=sampler.device
        )
        # A replacement true path invalidates any grid built during NUTS
        # initialization. Retain the tuned step sizes, not the stale grid.
        for grids in (
            sampler.T_all_list, sampler.z_aug_list, sampler.is_event_time_list,
            sampler.obs_idx_list, sampler.true_idx_list, sampler.pseudo_idx_list,
            sampler.y_aug_list,
        ):
            grids[series_index] = None


def initialize_mcmc_sampler(
    *,
    nle: NLEEstimator,
    x_obs: torch.Tensor,
    obs_times: torch.Tensor,
    theta_lower: torch.Tensor,
    theta_upper: torch.Tensor,
    switching_mask: torch.Tensor,
    tau2_beta: float | torch.Tensor,
    tau2_alpha: float | torch.Tensor = 3.0,
    q_alpha: float = 2.0,
    q_beta: float = 20.0,
    seed: int = 0,
    num_regimes: int = 2,
    Q: torch.Tensor | None = None,
    initial_path_Q: torch.Tensor | None = None,
    initial_theta: torch.Tensor | None = None,
    initial_log_tau: torch.Tensor | None = None,
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
    num_regimes = num_regimes if Q is None else Q.shape[0]
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
        dtype=torch.float32,
        time_dtype=time_dtype,
        seed=seed,
        latest_sample_path=latest_sample_path,
    )
    sampler.initialize(
        initial_theta=initial_theta,
        initial_log_tau=initial_log_tau,
        initial_regime_probs=torch.full((num_regimes,), 1.0 / num_regimes),
    )
    path_generator_Q = Q if initial_path_Q is None else initial_path_Q
    initialize_ctmc_path_from_generator(sampler, path_generator_Q, seed=seed)
    return sampler, {"initial_Q": Q, "initial_theta": initial_theta}


def _run_and_save_chain(
    *,
    experiment_name: str,
    nle: NLEEstimator,
    x_obs: torch.Tensor,
    obs_times: torch.Tensor,
    physical_theta_prior: Distribution,
    nle_config: dict[str, Any],
    sampler_config: dict[str, Any],
    seed: int,
    chain_id: int,
    num_chains: int,
    num_sweeps: int,
    burn_in: int,
    progress_every: int,
    history_path: Path,
    latest_sample_path: Path | None,
    output_payload: dict[str, Any] | None,
) -> tuple[int, str]:
    """Initialize, run, and save one chain of a configured experiment."""
    theta_lower = nle_config["theta_lower"]
    theta_upper = nle_config["theta_upper"]
    sampler_kwargs = dict(sampler_config)
    num_regimes = int(sampler_kwargs.pop("num_regimes", 2))
    initial_path_jump_rate = float(sampler_kwargs.pop("initial_path_jump_rate"))
    switching_mask = sampler_kwargs["switching_mask"]

    theta_prior = nle.dynamics.pullback_theta_prior(physical_theta_prior)
    initial_path_Q = make_symmetric_generator(
        num_regimes,
        initial_path_jump_rate,
        dtype=theta_lower.dtype,
        device=theta_lower.device,
    )
    seed_all(seed)
    initial_theta = sample_initial_theta_from_truncated_physical_prior(
        dynamics=nle.dynamics,
        physical_theta_prior=physical_theta_prior,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        switching_mask=switching_mask,
        num_regimes=num_regimes,
    )
    sampler, initialization = initialize_mcmc_sampler(
        nle=nle,
        x_obs=x_obs,
        obs_times=obs_times,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        num_regimes=num_regimes,
        initial_path_Q=initial_path_Q,
        initial_theta=initial_theta,
        seed=seed,
        latest_sample_path=latest_sample_path,
        theta_prior=theta_prior,
        **sampler_kwargs,
    )

    if num_chains == 1:
        progress_label = experiment_name
    else:
        progress_label = f"chain {chain_id}, seed {seed}"
    for sweep in range(num_sweeps):
        info = sampler.one_sweep()
        if progress_every and (sweep + 1) % progress_every == 0:
            print(
                f"[{progress_label}, sweep {sweep + 1:05d}] "
                f"grid={info['grid_size']}, "
                f"true_segments={info['num_true_segments']}",
                flush=True,
            )

    history = sampler.history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    saved_payload = {
        "history": history,
        **(output_payload or {}),
        **initialization,
    }
    torch.save(saved_payload, history_path)

    outside = posterior_outside_support(
        {"theta": history["theta"][burn_in:]},
        theta_lower,
        theta_upper,
        dynamics=nle.dynamics,
    )
    print(f"[{progress_label}] initial Q:\n{initialization['initial_Q']}", flush=True)
    print(
        f"[{progress_label}] initial theta:\n"
        f"{nle.dynamics.to_physical_theta(initialization['initial_theta'])}",
        flush=True,
    )
    print(
        f"[{progress_label}] post-burn samples outside NLE support: "
        f"{100 * outside:.2f}%",
        flush=True,
    )
    print(f"[{progress_label}] saved: {history_path}", flush=True)
    return chain_id, str(history_path)


def _configure_chain_worker() -> None:
    """Use deterministic single-threaded tensor operations in each chain worker."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_parallel_chains(
    worker: Callable[..., tuple[int, str | Path]],
    tasks: Sequence[dict[str, Any]],
    *,
    initializer: Callable[[], None] | None = None,
) -> tuple[Path, ...]:
    """Run numbered chain tasks in parallel and return paths in chain order."""
    if not tasks:
        raise ValueError("At least one chain task is required.")
    outputs: list[Path | None] = [None] * len(tasks)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(tasks), mp_context=context, initializer=initializer,
    ) as executor:
        futures = [executor.submit(worker, **task) for task in tasks]
        for future in as_completed(futures):
            chain_id, output = future.result()
            if not 0 <= chain_id < len(tasks) or outputs[chain_id] is not None:
                raise ValueError(f"Invalid or repeated chain_id: {chain_id}")
            outputs[chain_id] = Path(output)
    if any(output is None for output in outputs):
        raise RuntimeError("A chain task did not return a result.")
    return tuple(output for output in outputs if output is not None)


def run_experiment(
    *,
    experiment_name: str,
    dynamics: Any,
    x_obs: torch.Tensor,
    obs_times: torch.Tensor,
    reference_times: torch.Tensor,
    physical_theta_prior: Distribution,
    nle_config: dict[str, Any],
    sampler_config: dict[str, Any],
    chain_config: dict[str, Any],
    output_payload: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Train/load an NLE, run chains, and save histories."""
    seeds = tuple(int(seed) for seed in chain_config["seeds"])
    history_paths = tuple(Path(path) for path in chain_config["history_paths"])
    latest_paths_config = chain_config.get("latest_sample_paths")
    latest_sample_paths = (
        tuple(Path(path) for path in latest_paths_config)
        if latest_paths_config is not None
        else (None,) * len(seeds)
    )
    if not seeds:
        raise ValueError("chain_config['seeds'] must not be empty.")
    if len(history_paths) != len(seeds):
        raise ValueError("history_paths and seeds must have the same length.")
    if len(latest_sample_paths) != len(seeds):
        raise ValueError("latest_sample_paths and seeds must have the same length.")

    num_sweeps = int(chain_config["num_sweeps"])
    burn_in = int(chain_config["burn_in"])
    progress_every = int(chain_config.get("progress_every", 100))
    if not 0 <= burn_in < num_sweeps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < num_sweeps.")

    nle_build_kwargs = {
        "dynamics": dynamics,
        "obs_times": obs_times,
        "x_obs": x_obs,
        "reference_times": reference_times,
        **nle_config,
    }
    # Build or load the NLE once before starting any chain processes. Training
    # tensors remain in the cache file but are not needed during MCMC inference.
    nle = build_spline_nle(**nle_build_kwargs)
    nle.xt_data = None
    nle.ctx_data = None
    common_chain_kwargs = {
        "experiment_name": experiment_name,
        "nle": nle,
        "x_obs": x_obs,
        "obs_times": obs_times,
        "physical_theta_prior": physical_theta_prior,
        "nle_config": nle_config,
        "sampler_config": sampler_config,
        "num_chains": len(seeds),
        "num_sweeps": num_sweeps,
        "burn_in": burn_in,
        "progress_every": progress_every,
        "output_payload": output_payload,
    }

    if len(seeds) == 1:
        _, output = _run_and_save_chain(
            seed=seeds[0],
            chain_id=0,
            history_path=history_paths[0],
            latest_sample_path=latest_sample_paths[0],
            **common_chain_kwargs,
        )
        return (Path(output),)

    tasks = []
    for chain_id, (seed, history_path, latest_sample_path) in enumerate(
        zip(seeds, history_paths, latest_sample_paths, strict=True)
    ):
        tasks.append(
            {
                **common_chain_kwargs,
                "seed": seed,
                "chain_id": chain_id,
                "history_path": history_path,
                "latest_sample_path": latest_sample_path,
            }
        )

    return run_parallel_chains(
        _run_and_save_chain, tasks, initializer=_configure_chain_worker,
    )


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
