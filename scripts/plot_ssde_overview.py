"""Create an overview figure of a switching Lotka-Volterra SDE.

Usage
-----
python scripts/plot_ssde_overview.py [output_path]

The figure is intentionally inference-free: it shows one simulated continuous
state path, irregular noisy observations, and the underlying CTMC regime path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fnle_ssde.dynamics import LotkaVolterraDynamics  # noqa: E402
from fnle_ssde.visualization import set_figure_font_style  # noqa: E402


FONT_SCALE = 1.0
TEXT_FONT_COEFFICIENT = 15.0
PANEL_TITLE_FONT_COEFFICIENT = 18.0
OVERALL_TITLE_FONT_COEFFICIENT = 21.0  # Reserved if an overall title is added.


def simulate_lv_ssde(
    *,
    seed: int = 0,
    dt: float = 0.01,
    total_time: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float], list[int]]:
    """Simulate an exact CTMC path and an Euler-Maruyama LV path."""
    torch.manual_seed(seed)
    dynamics = LotkaVolterraDynamics(
        dt=dt,
        device="cpu",
        state_upper_bound=1e4,
    )

    # Rows are regimes; columns are alpha, beta, gamma, delta, sigma_D, sigma_P.
    theta = torch.log(
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
                [0.5, 0.5, 0.2, 0.2, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
    )
    Q = torch.tensor([[-0.03, 0.03], [0.03, -0.03]])

    regime = int(dist.Categorical(probs=torch.full((2,), 0.5)).sample())
    jump_times: list[float] = []
    path_regimes = [regime]
    current_time = 0.0
    while current_time < total_time:
        wait = float(dist.Exponential(-Q[regime, regime]).sample())
        next_time = current_time + wait
        if next_time >= total_time:
            break
        regime = 1 - regime
        jump_times.append(next_time)
        path_regimes.append(regime)
        current_time = next_time

    num_steps = round(total_time / dt)
    times = torch.arange(num_steps + 1, dtype=torch.float32) * dt
    z = torch.empty(num_steps + 1, dtype=torch.long)
    y = torch.empty((num_steps + 1, 2), dtype=torch.float32)
    y[0] = torch.tensor([1.0, 0.5])

    jump_index = 0
    for step in range(num_steps + 1):
        left_time = float(times[step])
        while (
            jump_index < len(jump_times)
            and jump_times[jump_index] <= left_time
        ):
            jump_index += 1
        z[step] = path_regimes[jump_index]
        if step < num_steps:
            y[step + 1] = dynamics.simulate_one_step(y[step], theta[z[step]])

    return times, y, z, jump_times, path_regimes


def make_irregular_observations(
    times: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int = 1,
    num_observations: int = 41,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample the latent path irregularly and add Gaussian noise."""
    generator = torch.Generator().manual_seed(seed)
    interior = torch.randperm(times.numel() - 2, generator=generator)[
        : num_observations - 2
    ] + 1
    indices = torch.cat(
        [torch.tensor([0]), interior.sort().values, torch.tensor([times.numel() - 1])]
    )
    tau = torch.tensor([0.1, 0.1])
    noise = torch.randn(y[indices].shape, generator=generator) * tau
    return times[indices], y[indices] + noise


def add_regime_background(
    ax: plt.Axes,
    jump_times: list[float],
    path_regimes: list[int],
    total_time: float,
) -> None:
    """Shade time intervals according to the active discrete regime."""
    colors = ("#dceef2", "#f7e3ca")
    boundaries = [0.0, *jump_times, total_time]
    for left, right, regime in zip(boundaries[:-1], boundaries[1:], path_regimes):
        ax.axvspan(left, right, color=colors[regime], alpha=0.42, linewidth=0)
    for jump_time in jump_times:
        ax.axvline(jump_time, color="#555555", ls=(0, (2, 3)), lw=0.8, alpha=0.55)


def make_figure(output_path: Path) -> None:
    """Generate and save the SSDE overview figure."""
    times, y, z, jump_times, path_regimes = simulate_lv_ssde()
    T_obs, x_obs = make_irregular_observations(times, y)

    set_figure_font_style(
        font_scale=FONT_SCALE,
        text_font_coefficient=TEXT_FONT_COEFFICIENT,
        panel_title_font_coefficient=PANEL_TITLE_FONT_COEFFICIENT,
    )
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, (trajectory_ax, regime_ax) = plt.subplots(
        2,
        1,
        figsize=(10.0, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": (3.4, 1.0), "hspace": 0.08},
        constrained_layout=True,
    )

    add_regime_background(trajectory_ax, jump_times, path_regimes, float(times[-1]))
    add_regime_background(regime_ax, jump_times, path_regimes, float(times[-1]))

    trajectory_colors = ("#155e75", "#c2413b")
    labels = (r"$Y_{t,1}$", r"$Y_{t,2}$")
    observation_labels = (r"$X_{t,1}$", r"$X_{t,2}$")
    for dimension, color in enumerate(trajectory_colors):
        trajectory_ax.plot(
            times,
            y[:, dimension],
            color=color,
            lw=1.8,
            label=labels[dimension],
            zorder=3,
        )
        trajectory_ax.scatter(
            T_obs,
            x_obs[:, dimension],
            s=25,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            label=observation_labels[dimension],
            zorder=4,
        )

    trajectory_ax.set_ylabel(r"Continuous state $Y_t$")
    trajectory_ax.legend(loc="upper left", ncol=2, columnspacing=1.4)
    trajectory_ax.grid(axis="y", alpha=0.18)
    trajectory_ax.text(
        0.99,
        0.96,
        "background color = active regime",
        transform=trajectory_ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
    )

    regime_ax.step(times, z, where="post", color="#222222", lw=2.0, zorder=3)
    regime_ax.set_yticks([0, 1], ["regime 1", "regime 2"])
    regime_ax.set_ylim(-0.35, 1.35)
    regime_ax.set_ylabel(r"Regime $Z_t$")
    regime_ax.set_xlabel("time")
    regime_ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    if pdf_path != output_path:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output_path}")
    if pdf_path != output_path:
        print(f"saved figure: {pdf_path}")
    print(f"jump times: {[round(value, 2) for value in jump_times]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        default=ROOT / "results/figures/fig1_ssde_overview.png",
    )
    args = parser.parse_args()
    make_figure(args.output_path)


if __name__ == "__main__":
    main()
