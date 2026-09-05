"""Generate the synthetic LV, CLE, and SIR datasets used by example_2.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyro.distributions as dist
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnle_ssde.dynamics import (  # noqa: E402
    GeneExpressionCLEDynamics,
    LotkaVolterraDynamics,
    SIRDynamics,
)


DATA_PATHS = {
    "lv": ROOT / "data/example_2_lv.pt",
    "cle": ROOT / "data/example_2_cle.pt",
    "sir": ROOT / "data/example_2_sir.pt",
}


def settings() -> dict[str, dict[str, object]]:
    """Return fixed data-generating settings for all three models."""
    return {
        "lv": {
            "dynamics": LotkaVolterraDynamics(
                dt=0.01,
                device="cpu",
                state_upper_bound=1e4,
            ),
            "theta_true": torch.log(
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
                        [0.5, 0.5, 0.2, 0.2, 0.05, 0.05],
                    ]
                )
            ),
            "Q_true": torch.tensor([[-0.03, 0.03], [0.03, -0.03]]),
            "x0": torch.tensor([1.0, 0.5]),
            "tau_true": torch.tensor([0.01, 0.01]),
            "total_steps": 10_000,
            "obs_interval": 100,
        },
        "cle": {
            "dynamics": GeneExpressionCLEDynamics(
                dt=0.01,
                device="cpu",
                state_upper_bound=1e4,
            ),
            "theta_true": torch.log(
                torch.tensor(
                    [
                        [100.0, 1.0, 0.004, 0.8, 0.03],
                        [240.0, 1.0, 0.004, 0.8, 0.03],
                    ]
                )
            ),
            "Q_true": torch.tensor([[-0.05, 0.05], [0.05, -0.05]]),
            "x0": torch.tensor([100.0, 0.5]),
            "tau_true": torch.tensor([1.0, 0.01]),
            "total_steps": 5_000,
            "obs_interval": 25,
        },
        "sir": {
            "dynamics": SIRDynamics(dt=0.01),
            "theta_true": torch.log(
                torch.tensor([[0.40, 0.10], [0.80, 0.20]])
            ),
            "Q_true": torch.tensor([[-0.025, 0.025], [0.025, -0.025]]),
            # Reduced SIR state: susceptible and recovered. Infected is
            # reconstructed as N - S - R inside SIRDynamics.
            "x0": torch.tensor([990.0, 0.0]),
            "tau_true": torch.tensor([0.5, 0.5]),
            "total_steps": 4_000,
            "obs_interval": 20,
            "forced_jump_time": 20.0,
            "clamp_observations": True,
        },
    }


def simulate(config: dict[str, object]) -> dict[str, torch.Tensor]:
    """Simulate an exact CTMC path and an Euler-Maruyama SDE path."""
    dynamics = config["dynamics"]
    theta, Q = config["theta_true"], config["Q_true"]
    total_steps = config["total_steps"]
    T = total_steps * dynamics.dt

    if "forced_jump_time" in config:
        jump_times = [float(config["forced_jump_time"])]
        path_regimes = [0, 1]
    else:
        regime = dist.Categorical(
            probs=torch.full((Q.shape[0],), 1.0 / Q.shape[0])
        ).sample()
        jump_times, path_regimes = [], [int(regime)]
        current_time = 0.0
        while current_time < T:
            next_time = current_time + float(
                dist.Exponential(-Q[regime, regime]).sample()
            )
            if next_time >= T:
                break
            probabilities = Q[regime].clone()
            probabilities[regime] = 0.0
            regime = dist.Categorical(
                probs=probabilities / probabilities.sum()
            ).sample()
            jump_times.append(next_time)
            path_regimes.append(int(regime))
            current_time = next_time

    T_true = torch.tensor(jump_times, dtype=torch.float32)
    z_true = torch.tensor(path_regimes, dtype=torch.long)
    times = torch.arange(total_steps + 1, dtype=torch.float32) * dynamics.dt
    z_grid = torch.empty(total_steps + 1, dtype=torch.long)
    jump_index = 0
    for index, time in enumerate(times.tolist()):
        while jump_index < T_true.numel() and float(T_true[jump_index]) <= time:
            jump_index += 1
        z_grid[index] = z_true[jump_index]

    y_values = [config["x0"].to(torch.float32)]
    jump_index = 0
    for step in range(total_steps):
        while jump_index < T_true.numel() and float(T_true[jump_index]) <= float(times[step]):
            jump_index += 1
        y_values.append(
            dynamics.simulate_one_step(y_values[-1], theta[z_true[jump_index]])
        )
    return {
        "times": times,
        "y_true": torch.stack(y_values),
        "z_true_grid": z_grid,
        "T_true": T_true,
        "z_true": z_true,
    }


def add_observations(
    data: dict[str, torch.Tensor],
    config: dict[str, object],
) -> None:
    """Add noisy observations to a simulated latent path."""
    obs_idx = torch.arange(0, data["times"].numel(), config["obs_interval"])
    T_obs = data["times"][obs_idx]
    x_obs = data["y_true"][obs_idx] + config["tau_true"] * torch.randn_like(
        data["y_true"][obs_idx]
    )
    if config.get("clamp_observations", False):
        x_obs = x_obs.clamp_min(0.0)
    data.update(
        {
            "obs_idx": obs_idx,
            "T_obs": T_obs,
            "x_obs": x_obs,
            "theta_true": config["theta_true"],
            "Q_true": config["Q_true"],
            "tau_true": config["tau_true"],
        }
    )


def main() -> None:
    for name, config in settings().items():
        torch.manual_seed(0)
        np.random.seed(0)
        data = simulate(config)
        add_observations(data, config)
        path = DATA_PATHS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        print(f"[{name}] saved: {path}")


if __name__ == "__main__":
    main()
