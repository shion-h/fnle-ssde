"""Run the synthetic LV, CLE, and SIR inference experiments for Figure 2."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributions import LogNormal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnle_ssde.dynamics import (  # noqa: E402
    GeneExpressionCLEDynamics,
    LotkaVolterraDynamics,
    SIRDynamics,
)
from fnle_ssde.utils import run_experiment  # noqa: E402
from generate_data_2 import DATA_PATHS  # noqa: E402


NLE_CACHE_PATHS = {
    "lv": ROOT / "results/example2_lv_nle.pt",
    "cle": ROOT / "results/example2_cle_nle.pt",
    "sir": ROOT / "results/example2_sir_nle.pt",
}
LV_HISTORY_PATH = ROOT / "results/example2_lv_gs.pt"
CLE_HISTORY_PATH =  ROOT / "results/example2_cle_gs.pt"
SIR_HISTORY_PATH =  ROOT / "results/example2_sir_gs.pt"

Q_PRIOR_ALPHA, Q_PRIOR_BETA = 2.0, 20.0


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
    tau2_beta: torch.Tensor
    ref_noise: float
    theta_prior_loc: torch.Tensor
    theta_prior_scale: torch.Tensor
    initial_path_jump_rate: float


def experiments() -> tuple[Experiment, ...]:
    return (
        Experiment(
            "lv",
            LotkaVolterraDynamics(
                dt=0.01,
                device="cpu",
                state_upper_bound=1e4,
            ),
            DATA_PATHS["lv"],
            LV_HISTORY_PATH,
            NLE_CACHE_PATHS["lv"],
            100,
            torch.tensor([0.01, 0.1, 0.01, 0.1, 0.01, 0.01]),
            torch.tensor([2.0, 3.0, 2.0, 3.0, 0.30, 0.20]),
            torch.tensor([True, True, True, True, False, False]),
            torch.full((2,), 2e-4),
            0.32,
            torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, -1.0]),
            torch.ones(6),
            0.2,
        ),
        Experiment(
            "cle",
            GeneExpressionCLEDynamics(
                dt=0.01,
                device="cpu",
                state_upper_bound=1e4,
            ),
            DATA_PATHS["cle"],
            CLE_HISTORY_PATH,
            NLE_CACHE_PATHS["cle"],
            100,
            torch.tensor([50.0, 0.5, 0.001, 0.5, 0.01]),
            torch.tensor([400.0, 1.5, 0.010, 1.5, 0.10]),
            torch.tensor([True, False, False, False, False]),
            torch.tensor([2.0, 2e-4]),
            0.32,
            torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0]),
            torch.ones(5),
            0.4,
        ),
        Experiment(
            "sir",
            SIRDynamics(dt=0.01, device="cpu"),
            DATA_PATHS["sir"],
            SIR_HISTORY_PATH,
            NLE_CACHE_PATHS["sir"],
            100,
            torch.tensor([0.05, 0.02]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([True, True]),
            torch.full((2,), 0.5),
            0.32,
            torch.zeros(2),
            torch.ones(2),
            0.5,
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
    latest_sample_path = ROOT / (
        f"results/checkpoints/{experiment.history_path.stem}_latest_sample.pt"
    )
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
            "n_params": 100_000,
            "ref_noise": experiment.ref_noise,
            "max_n_steps": max_n_steps,
            "epochs": experiment.nle_epochs,
            "stop_after_epochs": 20,
            "noisy_init_strategy": "resample",
        },
        sampler_config={
            "num_regimes": num_regimes,
            "switching_mask": experiment.switching_mask,
            "tau2_alpha": 2.0,
            "tau2_beta": experiment.tau2_beta,
            "q_alpha": Q_PRIOR_ALPHA,
            "q_beta": Q_PRIOR_BETA,
            "initial_path_jump_rate": experiment.initial_path_jump_rate,
            "y0_prior_scale": 1.0,
            "time_dtype": torch.float64,
            "use_t_pseudo_in_sir": False,
        },
        chain_config={
            "seeds": (0,),
            "history_paths": (experiment.history_path,),
            "latest_sample_paths": (latest_sample_path,),
            "num_sweeps": 1_000,
            "burn_in": 500,
            "progress_every": 100,
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
