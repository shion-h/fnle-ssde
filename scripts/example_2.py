"""Run the synthetic LV, CLE, and SIR inference experiments for Figure 2."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.distributions import LogNormal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnle_ssde.dynamics import (  # noqa: E402
    LotkaVolterraDynamics,
    ReparametrizedGeneExpressionCLEDynamics,
    SIRDynamics,
)
from fnle_ssde.utils import run_experiment  # noqa: E402
from generate_data_2 import DATA_PATHS  # noqa: E402


RESULT_PATH = ROOT / "results/example_2"
NLE_CACHE_PATHS = {
    name: RESULT_PATH / name / "nle.pt" for name in ("lv", "cle", "sir")
}
LV_HISTORY_PATH = RESULT_PATH / "lv/example2_lv_mcmc_chain.pt"
CLE_HISTORY_PATH = RESULT_PATH / "cle/chain.pt"
SIR_HISTORY_PATH = RESULT_PATH / "sir/example2_sir_mcmc_chain.pt"

Q_PRIOR_ALPHA, Q_PRIOR_BETA = 2.0, 20.0
# Settings shared by LV, CLE and SIR. Existing histories are not overwritten.
SEEDS = (0, 1, 2, 3)
NUM_SWEEPS = 10_000
BURN_IN = 5_000
PROGRESS_EVERY = 100
NLE_TRAINING_SAMPLES = 100_000
NLE_SEED = 0


@dataclass(frozen=True)
class Experiment:
    name: str
    dynamics: object
    data_path: Path
    history_path: Path
    nle_cache_path: Path
    nle_epochs: int
    theta_low: torch.Tensor
    theta_high: torch.Tensor
    switching_mask: torch.Tensor
    ref_noise: float
    theta_prior_loc: torch.Tensor
    theta_prior_scale: torch.Tensor
    initial_path_jump_rate: float
    tau2_alpha: float = 1e-3
    tau2_beta: torch.Tensor = field(kw_only=True)
    y_step_size: float | None = 2.0 ** -9
    theta_step_size: float | None = 2.0 ** -8
    y_max_tree_depth: int = 5
    theta_max_tree_depth: int = 3
    # False + step_size=None searches once initially, then fixes the step.
    adapt_step_size: bool = False
    seeds: tuple[int, ...] = SEEDS
    num_sweeps: int = NUM_SWEEPS
    burn_in: int = BURN_IN


def experiments() -> tuple[Experiment, ...]:
    return (
        Experiment(
            name="lv",
            dynamics=LotkaVolterraDynamics(
                dt=0.01,
                device="cpu",
                state_upper_bound=1e4,
            ),
            data_path=DATA_PATHS["lv"],
            history_path=LV_HISTORY_PATH,
            nle_cache_path=NLE_CACHE_PATHS["lv"],
            nle_epochs=100,
            theta_low=torch.tensor([0.01, 0.1, 0.01, 0.1, 0.01, 0.01]),
            theta_high=torch.tensor([2.0, 3.0, 2.0, 3.0, 0.30, 0.20]),
            switching_mask=torch.tensor([True, True, True, True, False, False]),
            ref_noise=0.32,
            theta_prior_loc=torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, -1.0]),
            theta_prior_scale=torch.ones(6),
            initial_path_jump_rate=0.2,
            tau2_beta=torch.full((2,), 1e-3),
        ),
        Experiment(
            name="cle",
            dynamics=ReparametrizedGeneExpressionCLEDynamics(
                dt=0.01,
                device="cpu",
                state_upper_bound=1e4,
            ),
            data_path=DATA_PATHS["cle"],
            history_path=CLE_HISTORY_PATH,
            nle_cache_path=NLE_CACHE_PATHS["cle"],
            nle_epochs=100,
            theta_low=torch.tensor([50.0, 0.5, 0.001, 0.5, 0.01]),
            theta_high=torch.tensor([400.0, 1.5, 0.010, 1.5, 0.10]),
            switching_mask=torch.tensor([True, False, False, False, False]),
            ref_noise=0.32,
            theta_prior_loc=torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0]),
            theta_prior_scale=torch.ones(5),
            initial_path_jump_rate=0.4,
            tau2_beta=torch.full((2,), 1e-3),
        ),
        Experiment(
            name="sir",
            dynamics=SIRDynamics(dt=0.01, device="cpu"),
            data_path=DATA_PATHS["sir"],
            history_path=SIR_HISTORY_PATH,
            nle_cache_path=NLE_CACHE_PATHS["sir"],
            nle_epochs=100,
            theta_low=torch.tensor([0.05, 0.02]),
            theta_high=torch.tensor([2.0, 2.0]),
            switching_mask=torch.tensor([True, True]),
            ref_noise=0.32,
            theta_prior_loc=torch.zeros(2),
            theta_prior_scale=torch.ones(2),
            initial_path_jump_rate=0.5,
            tau2_beta=torch.full((2,), 1e-3),
        ),
    )


def run(experiment: Experiment) -> None:
    """Load one synthetic dataset and run its configured experiment."""
    num_regimes = 2
    if not experiment.data_path.exists():
        raise FileNotFoundError("Run scripts/generate_data_2.py first.")
    data = torch.load(experiment.data_path, map_location="cpu", weights_only=False)
    max_n_steps = int((data["obs_idx"][1:] - data["obs_idx"][:-1]).max())
    physical_theta_prior = LogNormal(
        experiment.theta_prior_loc,
        experiment.theta_prior_scale,
    )
    history_paths = tuple(
        experiment.history_path if len(experiment.seeds) == 1 else
        experiment.history_path.with_stem(f"{experiment.history_path.stem}{i}")
        for i in range(len(experiment.seeds))
    )
    latest_sample_paths = tuple(
        path.with_stem(f"{path.stem}_latest") for path in history_paths
    )
    if all(path.exists() for path in history_paths):
        print(f"[{experiment.name}] existing histories; skipping sampling.", flush=True)
        return
    if any(path.exists() for path in history_paths):
        raise FileExistsError(
            f"Partial {experiment.name} histories exist; refusing to overwrite them."
        )
    y_mh_config = {
        "method": "nuts", "max_tree_depth": experiment.y_max_tree_depth,
        "adapt_step_size": experiment.adapt_step_size,
    }
    theta_mh_config = {
        "method": "nuts", "max_tree_depth": experiment.theta_max_tree_depth,
        "adapt_step_size": experiment.adapt_step_size,
    }
    if experiment.y_step_size is not None:
        y_mh_config["step_size"] = experiment.y_step_size
    if experiment.theta_step_size is not None:
        theta_mh_config["step_size"] = experiment.theta_step_size
    run_experiment(
        experiment_name=experiment.name,
        dynamics=experiment.dynamics,
        x_obs=data["x_obs"],
        obs_times=data["T_obs"],
        reference_times=data["times"],
        physical_theta_prior=physical_theta_prior,
        nle_config={
            "cache_path": experiment.nle_cache_path,
            "theta_lower": experiment.theta_low,
            "theta_upper": experiment.theta_high,
            "n_params": NLE_TRAINING_SAMPLES,
            "seed": NLE_SEED,
            "ref_noise": experiment.ref_noise,
            "max_n_steps": max_n_steps,
            "epochs": experiment.nle_epochs,
            "stop_after_epochs": 20,
            "noisy_init_strategy": "resample",
        },
        sampler_config={
            "num_regimes": num_regimes,
            "switching_mask": experiment.switching_mask,
            "tau2_alpha": experiment.tau2_alpha,
            "tau2_beta": experiment.tau2_beta,
            "q_alpha": Q_PRIOR_ALPHA,
            "q_beta": Q_PRIOR_BETA,
            "initial_path_jump_rate": experiment.initial_path_jump_rate,
            "y_mh_config": y_mh_config,
            "theta_mh_config": theta_mh_config,
            "time_dtype": torch.float64,
            "use_t_pseudo_in_sir": False,
        },
        chain_config={
            "seeds": experiment.seeds,
            "history_paths": history_paths,
            "latest_sample_paths": latest_sample_paths,
            "num_sweeps": experiment.num_sweeps,
            "burn_in": experiment.burn_in,
            "progress_every": PROGRESS_EVERY,
        },
        output_payload={"data": data},
    )


def main() -> None:
    # Match the paper's real-data chains and avoid thread-count-dependent
    # floating-point reductions across repeated runs.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for experiment in experiments():
        run(experiment)


if __name__ == "__main__":
    main()
