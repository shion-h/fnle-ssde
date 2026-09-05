"""Run four process-parallel switching-LV chains for Figure 3."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import LogNormal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnle_ssde.dynamics import LotkaVolterraDynamics  # noqa: E402
from fnle_ssde.utils import run_experiment  # noqa: E402
from generate_data_3 import DATA_PATH  # noqa: E402


NLE_CACHE_PATH = ROOT / "results/example3_nle.pt"

EXPERIMENT_NAME = "example3"
NUM_SWEEPS, BURN_IN = 10_000, 5_000
NUM_CHAINS = 4
NUM_REGIMES = 2
INFERENCE_SEEDS = tuple(range(NUM_CHAINS))
DYNAMICS_DT = 0.01
NLE_NUM_PARAMETERS = 800_000
NLE_SEED = 0
NLE_REFERENCE_NOISE = 0.32
NLE_MAX_EPOCHS = 5_000
NLE_STOP_AFTER_EPOCHS = 20
NLE_HIDDEN_FEATURES = 50
NLE_NUM_TRANSFORMS = 5
NLE_NUM_BINS = 10
Y_TREE_DEPTH = 5
THETA_TREE_DEPTH = 3
TARGET_ACCEPT_PROB = 0.8
SIR_PARTICLES = 100
USE_T_PSEUDO_IN_SIR = False
Q_PRIOR_ALPHA, Q_PRIOR_BETA = 2.0, 20.0
# This auxiliary generator initializes z only. The sampler's Q itself is drawn
# independently from Gamma(Q_PRIOR_ALPHA, Q_PRIOR_BETA).
INITIAL_Z_Q_RATE = 0.5
TAU2_PRIOR_ALPHA = 2.0
TAU2_PRIOR_BETA = torch.full((2,), 2e-4)

# Physical-scale NLE support. The training distribution is uniform after
# taking logarithms of these bounds.
NLE_THETA_LOW = torch.tensor([0.1, 0.01, 0.1, 0.01, 0.01, 0.01])
NLE_THETA_HIGH = torch.tensor([4.0, 6.0, 8.0, 5.0, 0.8, 2.0])

# Physical-scale priors in the order
# [alpha, beta, gamma, delta, sigma_D, sigma_P].
# Drift: LogNormal(0, 1); shared diffusion: LogNormal(-1, 1).
THETA_PRIOR_LOC = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
THETA_PRIOR_SCALE = torch.ones(6)


def history_path(chain_id: int) -> Path:
    """Return the complete-history path for one chain."""
    return ROOT / (
        f"results/{EXPERIMENT_NAME}_gs_chain{chain_id}.pt"
    )


def latest_sample_path(chain_id: int) -> Path:
    """Return the rolling-checkpoint path for one chain."""
    return ROOT / (
        f"results/{EXPERIMENT_NAME}_gs_chain{chain_id}_latest_sample.pt"
    )


HISTORY_PATHS = tuple(
    history_path(chain_id)
    for chain_id in range(NUM_CHAINS)
)
HISTORY_PATH = HISTORY_PATHS[0]


def read_observations() -> tuple[torch.Tensor, torch.Tensor]:
    """Read time and states in model order [didinium, paramecium]."""
    with DATA_PATH.open() as file:
        rows = list(csv.DictReader(file))
    time = np.array([float(row["time"]) for row in rows])
    order = np.argsort(time, kind="stable")
    values = np.column_stack(
        [
            [float(rows[index]["didinium"]) for index in order],
            [float(rows[index]["paramecium"]) for index in order],
        ]
    )
    return torch.tensor(time[order] - time[order][0], dtype=torch.float32), torch.tensor(
        values, dtype=torch.float32
    )


def main() -> None:
    T_obs, x_obs = read_observations()
    dynamics = LotkaVolterraDynamics(
        dt=DYNAMICS_DT,
        device="cpu",
        state_upper_bound=1e4,
    )
    reference_times = torch.arange(
        float(T_obs[0]),
        float(T_obs[-1]) + 0.5 * DYNAMICS_DT,
        DYNAMICS_DT,
        dtype=x_obs.dtype,
    )
    max_n_steps = int(
        torch.ceil(((T_obs[1:] - T_obs[:-1]) / DYNAMICS_DT).max()).item()
    )

    nle_config = {
        "cache_path": NLE_CACHE_PATH,
        "theta_lower": NLE_THETA_LOW,
        "theta_upper": NLE_THETA_HIGH,
        "n_params": NLE_NUM_PARAMETERS,
        "ref_noise": NLE_REFERENCE_NOISE,
        "max_n_steps": max_n_steps,
        "epochs": NLE_MAX_EPOCHS,
        "stop_after_epochs": NLE_STOP_AFTER_EPOCHS,
        "seed": NLE_SEED,
        "noisy_init_strategy": "resample",
        "hidden_features": NLE_HIDDEN_FEATURES,
        "num_transforms": NLE_NUM_TRANSFORMS,
        "num_bins": NLE_NUM_BINS,
    }
    sampler_config = {
        "num_regimes": NUM_REGIMES,
        "switching_mask": torch.tensor(
            [True, True, True, True, False, False]
        ),
        "tau2_alpha": TAU2_PRIOR_ALPHA,
        "tau2_beta": TAU2_PRIOR_BETA,
        "q_alpha": Q_PRIOR_ALPHA,
        "q_beta": Q_PRIOR_BETA,
        "initial_path_jump_rate": INITIAL_Z_Q_RATE,
        "y0_prior_scale": 1.0,
        "y_mh_config": {
            "method": "nuts",
            "max_tree_depth": Y_TREE_DEPTH,
            "target_accept_prob": TARGET_ACCEPT_PROB,
        },
        "theta_mh_config": {
            "method": "nuts",
            "max_tree_depth": THETA_TREE_DEPTH,
            "target_accept_prob": TARGET_ACCEPT_PROB,
        },
        "sir_particles": SIR_PARTICLES,
        "use_t_pseudo_in_sir": USE_T_PSEUDO_IN_SIR,
        "time_dtype": torch.float64,
    }
    chain_config = {
        "seeds": INFERENCE_SEEDS,
        "history_paths": HISTORY_PATHS,
        "latest_sample_paths": tuple(
            latest_sample_path(chain_id)
            for chain_id in range(NUM_CHAINS)
        ),
        "num_sweeps": NUM_SWEEPS,
        "burn_in": BURN_IN,
        "progress_every": 100,
    }

    run_experiment(
        experiment_name=EXPERIMENT_NAME,
        dynamics=dynamics,
        x_obs=x_obs,
        obs_times=T_obs,
        reference_times=reference_times,
        physical_theta_prior=LogNormal(THETA_PRIOR_LOC, THETA_PRIOR_SCALE),
        nle_config=nle_config,
        sampler_config=sampler_config,
        chain_config=chain_config,
    )


if __name__ == "__main__":
    main()
