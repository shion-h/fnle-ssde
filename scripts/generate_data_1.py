"""Generate switching OU data directly for the three-method example 1 comparison."""

from pathlib import Path

import torch

from fnle_ssde.dynamics import ExactTransitionOUDynamics


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/example_1_ou_seed0.pt"
DT = 0.01
OBS_DELTA = 1.0
SIMULATION_DELTA = 0.05
END_TIME = 100.0
DATA_SEED = 0
TAU_TRUTH = 0.01
JUMP_TIMES = torch.tensor([30.0, 70.0], dtype=torch.float64)
REGIMES = torch.tensor([0, 1, 0])
PHYSICAL_TRUTH = torch.tensor(
    [[0.15, -1.2, 0.05], [0.15, 1.2, 0.05]],
)


def simulate_path(
    dynamics: ExactTransitionOUDynamics,
    physical_theta: torch.Tensor,
    grid: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Simulate exact OU transitions using the supplied random stream."""
    theta = dynamics.to_nle_theta(physical_theta)
    interval_regimes = REGIMES[torch.searchsorted(JUMP_TIMES, grid[:-1], right=True)]
    values = [torch.zeros(1)]
    for delta, regime in zip(grid.diff(), interval_regimes):
        values.append(dynamics.sample_transition(
            theta[regime], values[-1], delta, generator=generator,
        ))
    return torch.stack(values)


def generate_data() -> dict:
    dynamics = ExactTransitionOUDynamics(dt=DT, device="cpu")
    T_obs = torch.arange(round(END_TIME / OBS_DELTA) + 1, dtype=torch.float64) * OBS_DELTA
    simulation_times = (
        torch.arange(round(END_TIME / SIMULATION_DELTA) + 1, dtype=torch.float64)
        * SIMULATION_DELTA
    )
    grid = torch.unique(torch.cat((T_obs, simulation_times, JUMP_TIMES)), sorted=True)
    obs_idx = torch.searchsorted(grid, T_obs)

    generator = torch.Generator().manual_seed(DATA_SEED)
    y_true = simulate_path(dynamics, PHYSICAL_TRUTH, grid, generator)
    y_at_observations = y_true[obs_idx]
    x_obs = y_at_observations + TAU_TRUTH * torch.randn(
        y_at_observations.shape, generator=generator, dtype=y_at_observations.dtype,
    )

    return dict(
        times=grid, y_true=y_true, obs_idx=obs_idx, T_obs=T_obs, x_obs=x_obs,
        jump_times=JUMP_TIMES.clone(), regimes=REGIMES.clone(),
        theta_true=dynamics.to_nle_theta(PHYSICAL_TRUTH),
        metadata=dict(
            data_seed=DATA_SEED,
            tau_truth=TAU_TRUTH, regime_path="fixed_two_switches", dt=DT,
            obs_delta=OBS_DELTA, simulation_delta=SIMULATION_DELTA,
            observation_noise_stream="same generator, after latent transitions",
        ),
    )


def main() -> None:
    torch.set_num_threads(1)
    data = generate_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, DATA_PATH)
    print(f"Saved {DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
