"""Generate the LV simulator samples used by example_1.py."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from generate_data_2 import DATA_PATHS as EXAMPLE_2_DATA_PATHS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fnle_ssde.dynamics import LotkaVolterraDynamics  # noqa: E402


DATA_PATH = ROOT / "results/data/example_1_simulator.pt"
EXAMPLE_2_LV_DATA_PATH = EXAMPLE_2_DATA_PATHS["lv"]


def contexts() -> torch.Tensor:
    data = torch.load(
        EXAMPLE_2_LV_DATA_PATH,
        map_location="cpu",
        weights_only=False,
    )
    result = []
    for regime, time_index in ((1, 0), (1, 3_000), (0, 6_000)):
        theta = data["theta_true"][regime]
        x_prev = data["y_true"][time_index]
        for n_steps in (25.0, 75.0):
            result.append(torch.cat([theta, x_prev, torch.tensor([n_steps])]))
    return torch.stack(result)


@torch.no_grad()
def main() -> None:
    grid = contexts()
    dynamics = LotkaVolterraDynamics(dt=0.01, state_upper_bound=1e4)
    samples = []
    for index, context in enumerate(grid):
        count = 8_000
        samples.append(
            dynamics.simulate_n_steps(
                context[6:8].expand(count, -1).clone(),
                context[:6].expand(count, -1),
                int(context[8].item()),
                generator=torch.Generator().manual_seed(10 * index),
            )
        )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"contexts": grid, "simulator_samples": samples}, DATA_PATH)
    print(f"saved: {DATA_PATH}")


if __name__ == "__main__":
    main()
