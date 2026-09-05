"""Posterior summaries and plotting helpers shared by paper figures."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import torch

from .dynamics import Dynamics


SERIES_HISTORY_KEYS = {"y_aug", "z_aug", "T_all", "T_true", "z_true"}
POSTERIOR_COLOR = "#2364aa"
OBSERVATION_COLOR = "#d1495b"
TRUTH_COLOR = "#1b1b1b"
REGIME_COLOR = "#2a9d8f"


@dataclass(frozen=True)
class ParameterPlotGroup:
    """One parameter subplot in a posterior result figure."""

    title: str
    indices: tuple[int, ...]
    width: float


@dataclass(frozen=True)
class PosteriorFigureCase:
    """Data and display settings for one column of a posterior result figure."""

    title: str
    history: dict[str, list[object]]
    observation_times: torch.Tensor
    observations: torch.Tensor
    y_times: torch.Tensor
    z_times: torch.Tensor
    num_regimes: int
    start: int
    parameter_names: tuple[str, ...]
    parameter_switching: tuple[bool, ...]
    parameter_groups: tuple[ParameterPlotGroup, ...]
    trajectory_dimensions: tuple[int, ...]
    trajectory_labels: tuple[str, ...]
    dynamics: Dynamics
    regime: int = 1
    series_index: int = 0
    show_truth: bool = False
    theta_truth: torch.Tensor | None = None
    y_truth_times: torch.Tensor | None = None
    y_truth: torch.Tensor | None = None
    z_truth_times: torch.Tensor | None = None
    z_truth: torch.Tensor | None = None
    parameter_ylabel: str = ""


def set_paper_figure_style() -> None:
    """Apply the shared Matplotlib style used by the paper figures."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def single_series_history(
    history: dict[str, list[object]],
    *,
    series_index: int = 0,
    start: int = 0,
) -> dict[str, list[object]]:
    """Extract one series and optionally discard leading MCMC draws."""
    return {
        key: [
            value[series_index] if key in SERIES_HISTORY_KEYS else value
            for value in values[start:]
        ]
        for key, values in history.items()
    }


def interpolated_posterior_y_summary(
    history: dict[str, list[object]],
    *,
    times: torch.Tensor,
    start: int = 0,
    series_index: int = 0,
    interval: tuple[float, float] = (0.025, 0.975),
) -> dict[str, torch.Tensor]:
    """Summarize all draws after linear interpolation onto a common grid."""
    num_draws = len(history["y_aug"])
    if not 0 <= start < num_draws:
        raise ValueError(f"start must be in [0, {num_draws}), got {start}.")
    if times.ndim != 1:
        raise ValueError(f"times must be one-dimensional, got shape {times.shape}.")
    if not 0 <= interval[0] < interval[1] <= 1:
        raise ValueError(f"interval must satisfy 0 <= low < high <= 1, got {interval}.")

    interpolated_draws = []
    for draw_index in range(start, num_draws):
        draw_times = history["T_all"][draw_index][series_index]
        draw_y = history["y_aug"][draw_index][series_index]
        evaluation_times = times.to(device=draw_times.device, dtype=draw_times.dtype)
        evaluation_times = evaluation_times.clamp(draw_times[0], draw_times[-1])
        right = torch.searchsorted(draw_times, evaluation_times).clamp(
            min=1, max=draw_times.numel() - 1
        )
        left = right - 1
        weight = (
            (evaluation_times - draw_times[left])
            / (draw_times[right] - draw_times[left])
        ).to(draw_y.dtype)
        interpolated_draws.append(
            draw_y[left] + weight.unsqueeze(-1) * (draw_y[right] - draw_y[left])
        )

    y = torch.stack(interpolated_draws)
    return {
        "mean": y.mean(0),
        "low": torch.quantile(y, interval[0], dim=0),
        "high": torch.quantile(y, interval[1], dim=0),
    }


def regime_at_times(
    T_all: torch.Tensor,
    z_aug: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the right-continuous regime path at arbitrary times."""
    evaluation_times = times.to(device=T_all.device, dtype=T_all.dtype)
    index = torch.searchsorted(T_all, evaluation_times, right=True)
    return z_aug[index.clamp(min=1, max=T_all.numel() - 1)]


def _indices_at_grid_times(
    grid_times: torch.Tensor,
    evaluation_times: torch.Tensor,
) -> torch.Tensor:
    """Return indices of requested grid points, rejecting non-members."""
    evaluation_times = evaluation_times.to(
        device=grid_times.device,
        dtype=grid_times.dtype,
    )
    indices = torch.searchsorted(grid_times, evaluation_times)
    in_bounds = indices < grid_times.numel()
    safe_indices = indices.clamp(max=grid_times.numel() - 1)
    matched = torch.isclose(
        grid_times[safe_indices],
        evaluation_times,
        rtol=4 * torch.finfo(grid_times.dtype).eps,
        atol=4 * torch.finfo(grid_times.dtype).eps,
    )
    if not torch.all(in_bounds & matched):
        missing = evaluation_times[~(in_bounds & matched)]
        raise ValueError(
            "Evaluation times must be present on every augmented grid; "
            f"unmatched values: {missing.detach().cpu().tolist()}."
        )
    return safe_indices


def match_regime_labels(theta: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Match regime labels by mean squared error in the NLE theta coordinate."""
    order = min(
        itertools.permutations(range(truth.shape[0])),
        key=lambda permutation: float(
            ((theta[torch.tensor(permutation)] - truth) ** 2).mean()
        ),
    )
    return torch.tensor(order)


def posterior_summary(
    history: dict[str, list[object]],
    *,
    y_times: torch.Tensor,
    z_times: torch.Tensor,
    num_regimes: int,
    dynamics: Dynamics,
    start: int = 0,
    series_index: int = 0,
    theta_truth: torch.Tensor | None = None,
    y_interval: tuple[float, float] = (0.025, 0.975),
    theta_interval: tuple[float, float] = (0.05, 0.95),
) -> dict[str, torch.Tensor]:
    """Summarize one series at requested y and z evaluation times."""
    selected = single_series_history(
        history,
        series_index=series_index,
        start=start,
    )
    y = torch.stack(
        [
            y_aug[_indices_at_grid_times(T_all, y_times)]
            for T_all, y_aug in zip(selected["T_all"], selected["y_aug"])
        ]
    )
    z = torch.stack(
        [
            regime_at_times(T_all, z_aug, z_times)
            for T_all, z_aug in zip(selected["T_all"], selected["z_aug"])
        ]
    )
    theta_nle = torch.stack(selected["theta"])
    if theta_truth is None:
        order = torch.arange(num_regimes)
    else:
        order = match_regime_labels(theta_nle.mean(0), theta_truth)
    theta_nle = theta_nle[:, order]
    theta = dynamics.to_physical_theta(theta_nle)
    z_prob = torch.nn.functional.one_hot(z, num_classes=num_regimes).float().mean(0)
    z_prob = z_prob[:, order]
    return {
        "y_mean": y.mean(0),
        "y_low": torch.quantile(y, y_interval[0], dim=0),
        "y_high": torch.quantile(y, y_interval[1], dim=0),
        "z_prob": z_prob,
        "theta_mean": theta.mean(0),
        "theta_low": torch.quantile(theta, theta_interval[0], dim=0),
        "theta_high": torch.quantile(theta, theta_interval[1], dim=0),
        "regime_order": order,
    }


def plot_posterior_y_trajectory(
    ax: plt.Axes,
    *,
    times: torch.Tensor,
    mean: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    observation_times: torch.Tensor,
    observations: torch.Tensor,
    dimension: int,
    show_truth: bool = False,
    truth_times: torch.Tensor | None = None,
    truth: torch.Tensor | None = None,
) -> None:
    """Plot one latent dimension using the shared paper-figure design."""
    if show_truth:
        if truth_times is None or truth is None:
            raise ValueError("truth_times and truth are required when show_truth=True.")
        ax.plot(
            truth_times,
            truth[:, dimension],
            color=TRUTH_COLOR,
            lw=1.5,
            label="latent truth",
        )
    ax.fill_between(
        times,
        low[:, dimension],
        high[:, dimension],
        color=POSTERIOR_COLOR,
        alpha=0.25,
        lw=0,
        label="posterior 95% interval (linear interp.)",
    )
    ax.plot(
        times,
        mean[:, dimension],
        color=POSTERIOR_COLOR,
        lw=1.7,
        label="posterior mean",
    )
    ax.scatter(
        observation_times,
        observations[:, dimension],
        s=12,
        color=OBSERVATION_COLOR,
        alpha=0.6,
        label="observed",
    )


def relative_density_hpd(
    log_density: np.ndarray,
    masses: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> tuple[np.ndarray, list[float]]:
    """Normalize a grid density and return levels enclosing HPD masses."""
    density = np.exp(log_density - np.nanmax(log_density))
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = [
        ordered[min(np.searchsorted(cumulative, mass), ordered.size - 1)]
        for mass in reversed(masses)
    ]
    return density, sorted(set(levels))


def plot_parameter_bars(
    ax: plt.Axes,
    *,
    title: str,
    indices: tuple[int, ...],
    names: tuple[str, ...],
    switching: tuple[bool, ...],
    truth: torch.Tensor | None,
    summary: dict[str, torch.Tensor],
    num_regimes: int | None = None,
    latex_labels: bool = False,
) -> None:
    """Plot true and posterior natural-scale parameter values."""
    if num_regimes is None:
        num_regimes = summary["theta_mean"].shape[0]
    entries = [
        (index, regime)
        for index in indices
        for regime in (range(num_regimes) if switching[index] else (0,))
    ]
    x = np.arange(len(entries))
    means = np.array(
        [float(summary["theta_mean"][regime, index]) for index, regime in entries]
    )
    lower = np.array(
        [float(summary["theta_low"][regime, index]) for index, regime in entries]
    )
    upper = np.array(
        [float(summary["theta_high"][regime, index]) for index, regime in entries]
    )
    labels = []
    for index, regime in entries:
        if latex_labels:
            symbol = names[index]
            labels.append(
                rf"${symbol}_{{{regime + 1}}}$"
                if switching[index]
                else rf"${symbol}$"
            )
        else:
            labels.append(
                f"{names[index]}\nregime {regime + 1}"
                if switching[index]
                else f"{names[index]}\nshared"
            )
    posterior_x = x if truth is None else x + 0.18
    posterior_width = 0.60 if truth is None else 0.36
    if truth is not None:
        true_values = np.array(
            [float(truth[regime, index]) for index, regime in entries]
        )
        ax.bar(x - 0.18, true_values, 0.36, color="#444444", label="truth")
    ax.bar(
        posterior_x,
        means,
        posterior_width,
        color=POSTERIOR_COLOR,
        alpha=0.82,
        label="posterior",
    )
    ax.errorbar(
        posterior_x,
        means,
        yerr=np.vstack([means - lower, upper - means]),
        fmt="none",
        ecolor="#1b263b",
        capsize=2,
        label="95% credible interval",
    )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel("natural scale")
    ax.tick_params(axis="x", labelsize=7)


def plot_posterior_figure(
    cases: tuple[PosteriorFigureCase, ...],
    output: Path,
    *,
    figsize: tuple[float, float],
    additional_outputs: tuple[Path, ...] = (),
    y_interval: tuple[float, float] = (0.025, 0.975),
    theta_interval: tuple[float, float] = (0.025, 0.975),
    height_ratios: tuple[float, ...] | None = None,
    title_scale: float = 1.0,
    font_scale: float = 1.0,
) -> None:
    """Create a complete posterior figure from numerical case settings."""
    if not cases:
        raise ValueError("At least one posterior figure case is required.")
    max_trajectories = max(len(case.trajectory_dimensions) for case in cases)
    expected_rows = 5 + max_trajectories
    if height_ratios is None:
        height_ratios = (
            0.12,
            1.10,
            0.15,
            0.22,
            *(1.0 for _ in range(max_trajectories)),
            0.70,
        )
    if len(height_ratios) != expected_rows:
        raise ValueError(
            f"height_ratios must contain {expected_rows} values, "
            f"got {len(height_ratios)}."
        )

    set_paper_figure_style()
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(
        expected_rows,
        len(cases),
        height_ratios=height_ratios,
    )

    for column, case in enumerate(cases):
        if len(case.trajectory_dimensions) != len(case.trajectory_labels):
            raise ValueError(
                "trajectory_dimensions and trajectory_labels must have equal length."
            )
        if case.show_truth and (
            case.theta_truth is None
            or case.y_truth_times is None
            or case.y_truth is None
            or case.z_truth_times is None
            or case.z_truth is None
        ):
            raise ValueError(
                "theta_truth, y_truth_times, y_truth, z_truth_times, and "
                "z_truth are required when show_truth=True."
            )

        title_ax = fig.add_subplot(grid[0, column])
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            case.title,
            ha="center",
            va="center",
            fontsize=12 * title_scale,
        )
        parameter_grid = grid[1, column].subgridspec(
            1,
            len(case.parameter_groups),
            width_ratios=tuple(group.width for group in case.parameter_groups),
            wspace=0.04,
        )
        parameter_axes = [
            fig.add_subplot(parameter_grid[0, index])
            for index in range(len(case.parameter_groups))
        ]

        summary = posterior_summary(
            case.history,
            y_times=case.observation_times,
            z_times=case.z_times,
            num_regimes=case.num_regimes,
            start=case.start,
            series_index=case.series_index,
            theta_truth=case.theta_truth,
            dynamics=case.dynamics,
            y_interval=y_interval,
            theta_interval=theta_interval,
        )
        interpolated_y = interpolated_posterior_y_summary(
            case.history,
            times=case.y_times,
            start=case.start,
            series_index=case.series_index,
            interval=y_interval,
        )
        theta_truth = None
        if case.theta_truth is not None:
            theta_truth = case.dynamics.to_physical_theta(case.theta_truth)
        for ax, group in zip(parameter_axes, case.parameter_groups, strict=True):
            plot_parameter_bars(
                ax,
                title=group.title,
                indices=group.indices,
                names=case.parameter_names,
                switching=case.parameter_switching,
                truth=theta_truth if case.show_truth else None,
                summary=summary,
                num_regimes=case.num_regimes,
                latex_labels=True,
            )
            ax.set_ylabel(case.parameter_ylabel)

        parameter_legend = fig.add_subplot(grid[2, column])
        parameter_legend.axis("off")
        handles, labels = parameter_axes[0].get_legend_handles_labels()
        parameter_legend.legend(
            handles,
            labels,
            loc="center",
            ncol=len(handles),
            fontsize=7 * 1.5,
        )

        trajectory_legend = fig.add_subplot(grid[3, column])
        trajectory_legend.axis("off")
        trajectory_axes = []
        for row, (dimension, label) in enumerate(
            zip(case.trajectory_dimensions, case.trajectory_labels, strict=True)
        ):
            ax = fig.add_subplot(grid[4 + row, column])
            plot_posterior_y_trajectory(
                ax,
                times=case.y_times,
                mean=interpolated_y["mean"],
                low=interpolated_y["low"],
                high=interpolated_y["high"],
                observation_times=case.observation_times,
                observations=case.observations,
                dimension=dimension,
                show_truth=case.show_truth,
                truth_times=case.y_truth_times,
                truth=case.y_truth,
            )
            ax.set_title(f"Trajectory: {label}")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", labelbottom=False)
            trajectory_axes.append(ax)
        handles, labels = trajectory_axes[0].get_legend_handles_labels()
        trajectory_legend.legend(
            handles,
            labels,
            loc="center",
            ncol=min(len(handles), 2),
            fontsize=6.5 * 1.5,
        )
        for row in range(len(trajectory_axes), max_trajectories):
            fig.add_subplot(grid[4 + row, column]).axis("off")

        regime_ax = fig.add_subplot(grid[4 + max_trajectories, column])
        truth_regime_ax = None
        if case.show_truth:
            truth_regime_ax = regime_ax.twinx()
            truth_regime_ax.step(
                case.z_truth_times,
                case.z_truth + 1,
                where="post",
                color=TRUTH_COLOR,
                lw=1.4,
                label=r"true regime $z_t$",
            )
            truth_regime_ax.set(
                ylim=(0.95, case.num_regimes + 0.05),
                yticks=range(1, case.num_regimes + 1),
                ylabel="Regime",
            )
            truth_regime_ax.grid(False)
        regime_ax.plot(
            case.z_times,
            summary["z_prob"][:, case.regime],
            color=REGIME_COLOR,
            lw=1.4,
            label=(
                r"$\Pr(Z_t = "
                + str(case.regime + 1)
                + r" \mid x_{\mathrm{obs}})$"
            ),
        )
        regime_ax.set(
            ylim=(-0.05, 1.05),
            title="Regime posterior",
            xlabel="time",
            ylabel="Posterior probability",
        )
        if truth_regime_ax is not None:
            posterior_handles, posterior_labels = (
                regime_ax.get_legend_handles_labels()
            )
            truth_handles, truth_labels = truth_regime_ax.get_legend_handles_labels()
            regime_ax.legend(
                truth_handles + posterior_handles,
                truth_labels + posterior_labels,
                fontsize=8 * 1.5,
            )

    if title_scale != 1.0:
        for ax in fig.axes:
            for title in (ax.title, ax._left_title, ax._right_title):
                if title.get_text():
                    title.set_fontsize(title.get_fontsize() * title_scale)
    save_figure(
        fig,
        output,
        additional_paths=additional_outputs,
        font_scale=font_scale,
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int = 300,
    additional_paths: tuple[Path, ...] = (),
    font_scale: float = 1.0,
) -> None:
    """Save one figure to one or more paths, then close it."""
    if font_scale != 1.0:
        for text in fig.findobj(Text):
            text.set_fontsize(text.get_fontsize() * font_scale)
    for output in (path, *additional_paths):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"saved: {output}")
    plt.close(fig)
