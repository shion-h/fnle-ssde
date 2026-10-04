"""Compare exact OU inference with exact-trained and Euler-trained FNLEs.

Run generate_data_1.py first. The two FNLEs share 100,000 training contexts;
then each of the three methods runs 10,000 sweeps in four chains. Each method's
model, training data, and chain histories are saved together under RESULT_PATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.distributions import LogNormal, Normal

from fnle_ssde.dynamics import ExactTransitionOUDynamics, OUDynamics
from fnle_ssde.nle import NLEEstimator
from fnle_ssde.ou_sampler import ExactOUSSDESampler
from fnle_ssde.prior import (
    make_prior_config,
    sample_generator_matrix,
    sample_initial_theta_from_truncated_physical_prior,
)
from fnle_ssde.sampler import SSDESampler
from fnle_ssde.utils import (
    initialize_ctmc_path_from_generator, make_symmetric_generator,
    run_parallel_chains, seed_all,
)
from generate_data_1 import DATA_PATH, DT, ROOT


RESULT_PATH = ROOT / "results/example_1"
METHODS = ("exact", "fnle_exact", "fnle_euler")
SEEDS = (0, 1, 2, 3)
NUM_SWEEPS = 10_000
BURN_IN = 5_000
TRAINING_SIZE = 100_000
MAX_EPOCHS = 200
REF_NOISE = 0.32
Y_STEP_SIZE = 2.0 ** -8
THETA_STEP_SIZE = 2.0 ** -5
INITIAL_PATH_JUMP_RATE = 0.2
THETA_PHYSICAL_LOW = torch.tensor([0.05, -2.0, 0.01])
THETA_PHYSICAL_HIGH = torch.tensor([0.2, 2.0, 0.1])


def method_output_dir(method: str) -> Path:
    if method in METHODS:
        return RESULT_PATH / method
    raise ValueError(f"Unknown method: {method}")


def max_observation_steps(data: dict) -> int:
    """Convert the widest observed time interval to simulation steps."""
    return round(float(data["T_obs"].diff().max()) / DT)


def prepare_paired_training_data(data: dict) -> None:
    """Generate matched contexts and method-specific transition targets."""
    exact_path = method_output_dir("fnle_exact") / "training.pt"
    euler_path = method_output_dir("fnle_euler") / "training.pt"
    if exact_path.exists() and euler_path.exists():
        exact = torch.load(exact_path, map_location="cpu", weights_only=False)
        euler = torch.load(euler_path, map_location="cpu", weights_only=False)
        if not torch.equal(exact["ctx_data"], euler["ctx_data"]):
            raise ValueError("The FNLE training contexts differ between methods")
        print("Reusing both paired OU training datasets", flush=True)
        return
    if exact_path.exists() != euler_path.exists():
        raise FileExistsError("Only one paired training dataset exists")

    exact_dynamics = ExactTransitionOUDynamics(dt=DT, device="cpu")
    euler_dynamics = OUDynamics(dt=DT, device="cpu")
    lower, upper = euler_dynamics.physical_bounds_to_nle_bounds(
        THETA_PHYSICAL_LOW, THETA_PHYSICAL_HIGH,
    )
    reference_times = torch.arange(round(float(data["times"][-1]) / DT) + 1) * DT
    reference = NLEEstimator.build_spline_reference_path(
        data["T_obs"], data["x_obs"], reference_times, spline_clamp_bounds=None,
    )
    context_generator = torch.Generator().manual_seed(0)
    exact_generator = torch.Generator().manual_seed(1)
    euler_generator = torch.Generator().manual_seed(2)
    max_n_steps = max_observation_steps(data)
    contexts, exact_targets, euler_targets = [], [], []
    started = time.monotonic()
    for draw in range(TRAINING_SIZE):
        theta = lower + (upper - lower) * torch.rand(3, generator=context_generator)
        index = int(torch.randint(1, len(reference), (1,), generator=context_generator))
        previous = reference[index - 1] + REF_NOISE * torch.randn(
            reference.shape[1:], generator=context_generator, dtype=reference.dtype,
        )
        steps = int(torch.randint(1, max_n_steps + 1, (1,), generator=context_generator))
        delta = steps * DT
        exact_next = exact_dynamics.sample_transition(
            theta, previous, delta, generator=exact_generator,
        )
        euler_next = euler_dynamics.simulate_n_steps(
            previous, theta, steps, generator=euler_generator,
        )
        contexts.append(torch.cat((theta, previous, previous.new_tensor([steps]))))
        exact_targets.append((exact_next - previous) / delta**0.5)
        euler_targets.append((euler_next - previous) / delta**0.5)
        if (draw + 1) % 10_000 == 0:
            print(
                f"Paired training transitions {draw + 1}/{TRAINING_SIZE}; "
                f"elapsed={(time.monotonic() - started) / 60:.1f} min",
                flush=True,
            )
    context_tensor = torch.stack(contexts)
    exact_path.parent.mkdir(parents=True, exist_ok=True)
    euler_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"ctx_data": context_tensor, "xt_data": torch.stack(exact_targets)}, exact_path,
    )
    torch.save(
        {"ctx_data": context_tensor, "xt_data": torch.stack(euler_targets)}, euler_path,
    )
    print("Saved both paired OU training datasets", flush=True)


def train_fnle(method: str, data: dict) -> None:
    method_dir = method_output_dir(method)
    model_path = method_dir / "nle.pt"
    dynamics = OUDynamics(dt=DT, device="cpu")
    lower, upper = dynamics.physical_bounds_to_nle_bounds(
        THETA_PHYSICAL_LOW, THETA_PHYSICAL_HIGH,
    )
    seed_all(0)
    estimator = NLEEstimator(
        dynamics=dynamics, model_cache_path=model_path, target_type="scaled_dx",
        sampling_dist=torch.distributions.Independent(
            torch.distributions.Uniform(lower, upper), 1,
        ),
    )
    if estimator.estimator is not None:
        print(f"Reusing {method} model: {model_path}", flush=True)
        return
    training_data_path = method_dir / "training.pt"
    if not training_data_path.exists():
        raise FileNotFoundError(
            f"Generate paired OU training data first: {training_data_path}"
        )
    reference_times = torch.arange(round(float(data["times"][-1]) / DT) + 1) * DT
    estimator.train(
        obs_times=data["T_obs"], x_obs=data["x_obs"],
        reference_times=reference_times,
        n_params=TRAINING_SIZE, ref_noize=REF_NOISE,
        max_n_steps=max_observation_steps(data),
        samples_per_theta=1, epochs=MAX_EPOCHS, stop_after_epochs=20,
        batch_size=256, lr=5e-4, state_clamp_bounds=None,
        spline_clamp_bounds=None, noisy_init_strategy="resample",
        training_data_cache_path=training_data_path,
    )


def sample_chain(method: str, chain_id: int) -> tuple[int, Path]:
    """Run one chain; all three methods use identical priors and seeds."""
    torch.set_num_threads(1)
    method_dir = method_output_dir(method)
    method_dir.mkdir(parents=True, exist_ok=True)
    path = method_dir / f"chain{chain_id}.pt"
    if path.exists():
        print(f"Reusing {path}", flush=True)
        return chain_id, path

    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    dynamics_class = ExactTransitionOUDynamics if method == "exact" else OUDynamics
    dynamics = dynamics_class(dt=DT, device="cpu")
    if method == "exact":
        sampler_class = ExactOUSSDESampler
        estimator = None
    else:
        sampler_class = SSDESampler
        estimator = NLEEstimator(
            dynamics=dynamics, model_cache_path=method_dir / "nle.pt",
        )
        estimator.xt_data = estimator.ctx_data = None

    seed_all(SEEDS[chain_id])
    prior = make_prior_config(
        3, torch.tensor([1e-3]), theta_loc=0.0, theta_scale=1.0,
        tau2_alpha=1e-3, q_alpha=2.0, q_beta=20.0,
    )
    switching_mask = torch.tensor([False, True, False])
    physical_theta_priors = (
        LogNormal(prior["theta_loc"][0], prior["theta_scale"][0]),
        Normal(prior["theta_loc"][1], prior["theta_scale"][1]),
        LogNormal(prior["theta_loc"][2], prior["theta_scale"][2]),
    )
    initial_theta = sample_initial_theta_from_truncated_physical_prior(
        dynamics=dynamics, physical_theta_prior=physical_theta_priors,
        theta_lower=THETA_PHYSICAL_LOW, theta_upper=THETA_PHYSICAL_HIGH,
        switching_mask=switching_mask, num_regimes=2,
    )
    initial_q = sample_generator_matrix(2, q_alpha=2.0, q_beta=20.0)
    q_for_initial_path = make_symmetric_generator(2, INITIAL_PATH_JUMP_RATE)
    sampler_kwargs = dict(
        Q=initial_q, x_obs=data["x_obs"], obs_times=data["T_obs"],
        T=float(data["times"][-1]), omega_scale=3.0,
        switching_parameter_mask=switching_mask,
        prior_config=prior,
        y_mh_config=dict(
            method="nuts", max_tree_depth=5, adapt_step_size=False,
            step_size=Y_STEP_SIZE,
        ),
        theta_mh_config=dict(
            method="nuts", max_tree_depth=3, adapt_step_size=False,
            step_size=THETA_STEP_SIZE,
        ),
        sir_config=dict(num_particles=100, use_t_pseudo_in_sir=False),
        seed=SEEDS[chain_id], dtype=torch.float32, time_dtype=torch.float64,
        latest_sample_path=method_dir / f"chain{chain_id}_latest.pt",
    )
    if method == "exact":
        sampler = sampler_class(dynamics=dynamics, **sampler_kwargs)
    else:
        sampler = sampler_class(nle_estimator=estimator, **sampler_kwargs)

    started = time.monotonic()
    sampler.initialize(
        initial_theta=initial_theta,
        initial_regime_probs=torch.full((2,), 0.5),
    )
    initialize_ctmc_path_from_generator(
        sampler, q_for_initial_path, seed=SEEDS[chain_id],
    )
    for sweep in range(NUM_SWEEPS):
        sampler.one_sweep()
        if (sweep + 1) % 100 == 0:
            print(
                f"[{method} chain{chain_id} sweep {sweep + 1:05d}] "
                f"elapsed={(time.monotonic() - started) / 60:.1f} min",
                flush=True,
            )
    payload = dict(
        history=sampler.history, data=data, initial_Q=initial_q,
        initial_path_Q=q_for_initial_path, initial_theta=initial_theta,
        y_nuts_step_sizes=sampler.y_nuts_step_sizes,
        theta_nuts_step_size=sampler.theta_nuts_step_size,
        seconds=time.monotonic() - started,
    )
    torch.save(payload, path)
    print(f"Saved {path}", flush=True)
    return chain_id, path


def run_method(method: str, data: dict) -> None:
    if method != "exact":
        prepare_paired_training_data(data)
        train_fnle(method, data)
    print(f"Starting {method}: {NUM_SWEEPS} sweeps x {len(SEEDS)} chains", flush=True)
    tasks = [dict(method=method, chain_id=chain_id) for chain_id in range(len(SEEDS))]
    outputs = run_parallel_chains(sample_chain, tasks)
    for chain_id, output in enumerate(outputs):
        print(f"Completed {method} chain {chain_id}: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *METHODS), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run scripts/generate_data_1.py first: {DATA_PATH}")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    print(
        f"OU data={DATA_PATH}; theta physical bounds="
        f"{THETA_PHYSICAL_LOW.tolist()} to {THETA_PHYSICAL_HIGH.tolist()}; "
        f"initial path jump rate={INITIAL_PATH_JUMP_RATE}; "
        f"fixed NUTS steps y={Y_STEP_SIZE}, theta={THETA_STEP_SIZE}",
        flush=True,
    )
    methods = METHODS if args.stage == "all" else (args.stage,)
    for method in methods:
        run_method(method, data)


if __name__ == "__main__":
    main()
