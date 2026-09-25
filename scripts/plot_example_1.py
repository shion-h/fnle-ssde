"""Plot Figure 1 from the saved simulator/FNLE density grids."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from example_1 import NUMERICAL_RESULT_PATH  # noqa: E402
from generate_data_1 import CONTEXT_REGIMES, DT  # noqa: E402
from fnle_ssde.visualization import (  # noqa: E402
    relative_density_hpd,
    save_figure,
    set_figure_font_style,
)


FIGURE_PATH = ROOT / "results/figures/example_1.png"
PDF_PATH = FIGURE_PATH.with_suffix(".pdf")
DOWNSAMPLE_SIZE = 350
FONT_SCALE = 1
TEXT_FONT_COEFFICIENT = 22.0
PANEL_TITLE_FONT_COEFFICIENT = 24.0
OVERALL_TITLE_FONT_COEFFICIENT = 28.0  # Reserved if an overall title is added.


def load_numerical_rows() -> list[dict[str, Any]]:
    """Load the saved simulator and FNLE density-grid results."""
    if not NUMERICAL_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Run scripts/example_1.py first: {NUMERICAL_RESULT_PATH}"
        )
    return torch.load(NUMERICAL_RESULT_PATH, map_location="cpu", weights_only=False)[
        "rows"
    ]


def plot_density_comparison(
    ax: plt.Axes,
    row: dict[str, Any],
    regime: int,
) -> None:
    """Plot one simulator-versus-FNLE density comparison panel."""
    simulator_density, simulator_levels = relative_density_hpd(row["kde_grid"])
    nle_density, nle_levels = relative_density_hpd(row["nle_grid"])
    ax.contour(
        row["x_mesh"], row["y_mesh"], simulator_density,
        levels=simulator_levels, colors="#2166ac", linewidths=1.8,
    )
    ax.contour(
        row["x_mesh"], row["y_mesh"], nle_density,
        levels=nle_levels, colors="#b2182b", linewidths=1.8, linestyles="--",
    )
    samples = row["simulator_evaluation"]
    sample_index = torch.linspace(
        0,
        samples.shape[0] - 1,
        DOWNSAMPLE_SIZE,
    ).long()
    ax.scatter(
        samples[sample_index, 1],
        samples[sample_index, 0],
        s=5,
        alpha=0.11,
        color="#2166ac",
    )
    context = row["context"]
    ax.scatter(
        float(context[7]),
        float(context[6]),
        marker="x",
        s=55,
        linewidths=2,
        color="black",
    )
    y_prev = context[6:8]
    correlation = row["metrics"]["spearman_log_density"]
    ax.set_title(
        rf"$\theta=\theta_{{{regime + 1}}}$, "
        rf"$\Delta={float(context[8]) * DT:.2f}$"
        "\n"
        rf"$\rho={correlation:.2f}$"
    )
    lower = torch.quantile(samples, 0.001, dim=0)
    upper = torch.quantile(samples, 0.999, dim=0)
    outer_nle_region = nle_density >= min(nle_levels)
    nle_lower = torch.tensor(
        [
            row["y_mesh"][outer_nle_region].min(),
            row["x_mesh"][outer_nle_region].min(),
        ]
    )
    nle_upper = torch.tensor(
        [
            row["y_mesh"][outer_nle_region].max(),
            row["x_mesh"][outer_nle_region].max(),
        ]
    )
    display_lower = torch.minimum(torch.minimum(lower, y_prev), nle_lower)
    display_upper = torch.maximum(torch.maximum(upper, y_prev), nle_upper)
    margin = 0.08 * (display_upper - display_lower).clamp_min(1e-6)
    ax.set(
        xlim=(
            float(display_lower[1] - margin[1]),
            float(display_upper[1] + margin[1]),
        ),
        ylim=(
            float(display_lower[0] - margin[0]),
            float(display_upper[0] + margin[0]),
        ),
        xlabel=r"$Y_{t,2}$",
        ylabel=r"$Y_{t,1}$",
    )
    ax.grid(alpha=0.15)


def add_density_legend(ax: plt.Axes) -> None:
    """Add the shared density-comparison legend to an axis."""
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#2166ac", lw=2, label="Simulator KDE HPD"),
            plt.Line2D([0], [0], color="#b2182b", lw=2, ls="--", label="FNLE HPD"),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                ls="none",
                label=r"$y^{\mathrm{prev}}$",
            ),
        ],
        loc="center",
        ncol=3,
        frameon=False,
    )


def create_density_comparison_figure(
    rows: list[dict[str, Any]],
) -> plt.Figure:
    """Create the complete simulator-versus-FNLE comparison figure."""
    set_figure_font_style(
        font_scale=FONT_SCALE,
        text_font_coefficient=TEXT_FONT_COEFFICIENT,
        panel_title_font_coefficient=PANEL_TITLE_FONT_COEFFICIENT,
    )
    fig = plt.figure(figsize=(12.8, 15.3), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=(0.16, 1, 1, 1))
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    add_density_legend(legend_ax)
    axes = [fig.add_subplot(grid[row, column]) for row in range(1, 4) for column in range(2)]
    for ax, row, regime in zip(axes, rows, CONTEXT_REGIMES, strict=True):
        plot_density_comparison(ax, row, regime)
    return fig


def main() -> None:
    rows = load_numerical_rows()
    fig = create_density_comparison_figure(rows)
    save_figure(
        fig,
        FIGURE_PATH,
        additional_paths=(PDF_PATH,),
    )


if __name__ == "__main__":
    main()
