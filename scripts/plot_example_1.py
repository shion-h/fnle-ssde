"""Plot Figure 1 for the switching OU experiment.

Run generate_data_1.py and example_1.py first. All four chains are included
after burn-in; labels are aligned to the synthetic-data regimes.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from example_1 import (
    BURN_IN, METHODS, NUM_SWEEPS, RESULT_PATH, SEEDS, method_output_dir,
)
from fnle_ssde.dynamics import OUDynamics
from fnle_ssde.visualization import (
    OBSERVATION_COLOR, POSTERIOR_METHOD_STYLES, TRUTH_COLOR,
    interpolated_posterior_y_summary, match_regime_labels, regime_at_times,
    plot_posterior_regime, plot_posterior_y_trajectory,
    save_figure, set_paper_figure_style,
)
from generate_data_1 import DATA_PATH, DT


FIGURE_PATH = RESULT_PATH / "example_1.png"
FONT_SCALE = 1.1
TEXT_FONT_COEFFICIENT = 13.0
PANEL_TITLE_FONT_COEFFICIENT = 13.0
FIGURE_SIZE = (9.6, 12.8)
HISTOGRAM_RANGE_QUANTILES: tuple[float, float] | None = None

METHOD_LABELS = {
    "exact": "Exact MCMC",
    "fnle_exact": "Approx. MCMC (exact FNLE)",
    "fnle_euler": "Approx. MCMC (Euler--Maruyama FNLE)",
}
METHOD_SHORT_LABELS = {
    "exact": "exact MCMC",
    "fnle_exact": "approx. MCMC (exact FNLE)",
    "fnle_euler": "approx. MCMC (Euler--Maruyama FNLE)",
}
METHOD_STYLES = dict(zip(METHODS, POSTERIOR_METHOD_STYLES, strict=True))
PARAMETER_LABELS = (
    r"$\kappa$", r"$\mu_1$", r"$\mu_2$", r"$\sigma$",
)


def load_samples(
    method: str,
    data: dict,
    dynamics: OUDynamics,
    truth_nle: torch.Tensor,
    regime_times: torch.Tensor,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return aligned scalar chains and latent summaries on the full time grid."""
    chains, regime_sum = [], np.zeros(len(regime_times))
    state_history = {"T_all": [], "y_aug": []}
    for chain_id in range(len(SEEDS)):
        path = method_output_dir(method) / f"chain{chain_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Run scripts/example_1.py first: {path}")
        history = torch.load(path, map_location="cpu", weights_only=False)["history"]
        if len(history["theta"]) != NUM_SWEEPS:
            raise ValueError(f"Incomplete history: {path}")

        theta_nle = torch.stack(history["theta"])
        order = match_regime_labels(
            theta_nle[BURN_IN:].mean(0).unsqueeze(0), theta_truth=truth_nle,
        )[0]
        theta = dynamics.to_physical_theta(theta_nle[BURN_IN:, order])
        chains.append(torch.stack((
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 1],
            theta[:, 0, 2],
        ), dim=-1).numpy())
        state_history["T_all"].extend(history["T_all"][BURN_IN:])
        state_history["y_aug"].extend(history["y_aug"][BURN_IN:])

        for draw in range(BURN_IN, NUM_SWEEPS):
            grid = history["T_all"][draw][0]
            observation_idx = torch.searchsorted(grid, data["T_obs"])
            if not torch.allclose(grid[observation_idx], data["T_obs"], atol=1e-8, rtol=0):
                raise ValueError(f"Observation time missing from augmented grid: {path}")
            regimes = regime_at_times(grid, history["z_aug"][draw][0], regime_times)
            regime_sum += (regimes == order[1]).numpy()
        del history

    scalars = np.stack(chains)
    state_summary = interpolated_posterior_y_summary(
        state_history, times=data["times"],
    )
    return (
        scalars,
        {key: value.numpy() for key, value in state_summary.items()},
        regime_sum / len(state_history["T_all"]),
    )


def plot_figure(
    data: dict,
    samples: dict[str, tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]],
    regime_times: torch.Tensor,
    dynamics: OUDynamics,
) -> None:
    """Plot 2-by-2 parameter densities above the state and regime panels."""
    set_paper_figure_style(
        font_scale=FONT_SCALE,
        text_font_coefficient=TEXT_FONT_COEFFICIENT,
        panel_title_font_coefficient=PANEL_TITLE_FONT_COEFFICIENT,
    )
    truth = dynamics.to_physical_theta(data["theta_true"])
    parameter_truth = (
        float(truth[0, 0]), float(truth[0, 1]), float(truth[1, 1]),
        float(truth[0, 2]),
    )

    fig = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.82))
    grid = fig.add_gridspec(
        4, 2,
        height_ratios=(1, 1, 0.7, 0.7),
        hspace=0.08,
        wspace=0.0,
    )
    parameter_axes = [
        fig.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(2)
    ]
    for index, symbol in enumerate(PARAMETER_LABELS):
        ax = parameter_axes[index]
        all_values = np.concatenate([samples[method][0][:, :, index].ravel() for method in METHODS])
        if HISTOGRAM_RANGE_QUANTILES is None:
            low, high = all_values.min(), all_values.max()
        else:
            low, high = np.quantile(all_values, HISTOGRAM_RANGE_QUANTILES)
        edges = np.linspace(low, high, 46) if high > low else 45
        for method, (color, linestyle) in METHOD_STYLES.items():
            values = samples[method][0][:, :, index].ravel()
            ax.hist(
                values, bins=edges, density=True, histtype="step", linewidth=1.8,
                color=color, linestyle=linestyle,
                label=METHOD_LABELS[method],
            )
        if np.isfinite(parameter_truth[index]):
            ax.axvline(parameter_truth[index], color=TRUTH_COLOR, ls="-", lw=1.4)
        ax.set_title(symbol)
        ax.set_ylabel("Density" if index % 2 == 0 else "")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    state_ax = fig.add_subplot(grid[2, :])
    regime_ax = fig.add_subplot(grid[3, :], sharex=state_ax)
    styles = tuple(METHOD_STYLES.values())
    labels = tuple(METHOD_LABELS[method] for method in METHODS)
    plot_posterior_y_trajectory(
        state_ax,
        times=data["times"],
        summaries=tuple(samples[method][1] for method in METHODS),
        styles=styles,
        labels=labels,
        observation_times=data["T_obs"],
        observations=data["x_obs"],
        dimension=0,
        show_truth=True,
        truth_times=data["times"],
        truth=data["y_true"],
        linewidth=1.65,
        truth_on_top=True,
        observation_zorder=10,
        observation_linewidths=0,
    )
    state_ax.set_title(r"Latent state: $Y_t$")
    state_ax.set_ylabel(r"$Y_t$")

    edges = torch.cat((data["times"][:1], data["jump_times"], data["times"][-1:]))
    plot_posterior_regime(
        regime_ax,
        times=regime_times,
        probabilities=tuple(samples[method][2] for method in METHODS),
        styles=styles,
        labels=labels,
        regime=1,
        num_regimes=2,
        truth_times=edges,
        truth_regimes=torch.cat((data["regimes"], data["regimes"][-1:])),
        linewidth=1.65,
        ylim=(-0.04, 1.04),
        yticks=(0, 0.5, 1),
    )

    handles = [
        Line2D(
            [0], [0], color=color, ls=linestyle,
            lw=2, label=METHOD_LABELS[method],
        )
        for method, (color, linestyle) in METHOD_STYLES.items()
    ]
    handles.extend(
        Patch(
            facecolor=METHOD_STYLES[method][0],
            alpha=0.25,
            edgecolor="none",
            label=f"95% predictive interval ({METHOD_SHORT_LABELS[method]})",
        )
        for method in METHODS
    )
    handles.extend((
        Line2D([0], [0], color=TRUTH_COLOR, lw=1.5, label="True values"),
        Line2D([0], [0], color=OBSERVATION_COLOR, marker="o", lw=0, markersize=5,
               label="Observations"),
    ))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=1,
        labelspacing=0.25,
        handletextpad=0.6,
    )
    save_figure(fig, FIGURE_PATH, additional_paths=(FIGURE_PATH.with_suffix(".pdf"),))


def main() -> None:
    torch.set_num_threads(1)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run scripts/generate_data_1.py first: {DATA_PATH}")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    dynamics = OUDynamics(dt=DT, device="cpu")
    truth_nle = data["theta_true"]
    regime_times = data["times"]

    samples = {}
    for method in METHODS:
        samples[method] = load_samples(method, data, dynamics, truth_nle, regime_times)
    plot_figure(data, samples, regime_times, dynamics)


if __name__ == "__main__":
    main()
