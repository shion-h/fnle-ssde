"""Posterior summaries and plotting helpers shared by paper figures."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import torch


SERIES_HISTORY_KEYS = {"y_aug", "z_aug", "T_all", "T_true", "z_true"}


def single_series_history(
    history: dict[str, list[object]],
    *,
    series_index: int = 0,
    start: int = 0,
) -> dict[str, list[object]]:
    """Extract one series and optionally discard leading Gibbs draws."""
    return {
        key: [
            value[series_index] if key in SERIES_HISTORY_KEYS else value
            for value in values[start:]
        ]
        for key, values in history.items()
    }


def state_at_times(
    T_all: torch.Tensor,
    z_aug: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Evaluate states using z_aug[j] on [T_all[j-1], T_all[j]]."""
    index = torch.searchsorted(T_all.to(times.dtype), times, right=True)
    return z_aug[index.clamp(min=1, max=T_all.numel() - 1)]


def match_regime_labels(theta: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Match state labels by mean squared error in the NLE theta coordinate."""
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
    start: int = 0,
    theta_truth: torch.Tensor | None = None,
    dynamics: Any | None = None,
    y_interval: tuple[float, float] = (0.025, 0.975),
    theta_interval: tuple[float, float] = (0.05, 0.95),
) -> dict[str, torch.Tensor]:
    """Summarize one series at requested y and z evaluation times.

    ``dynamics`` converts stored NLE-coordinate theta to physical scale. If it
    is omitted, the historical exponential conversion is retained.
    """
    selected = single_series_history(history, start=start)
    y = torch.stack(
        [
            y_aug[torch.searchsorted(T_all.to(y_times.dtype), y_times)]
            for T_all, y_aug in zip(selected["T_all"], selected["y_aug"])
        ]
    )
    z = torch.stack(
        [
            state_at_times(T_all, z_aug, z_times)
            for T_all, z_aug in zip(selected["T_all"], selected["z_aug"])
        ]
    )
    theta_nle = torch.stack(selected["theta"])
    order = (
        match_regime_labels(theta_nle.mean(0), theta_truth)
        if theta_truth is not None
        else torch.arange(num_regimes)
    )
    theta_nle = theta_nle[:, order]
    theta = (
        theta_nle.exp()
        if dynamics is None
        else dynamics.to_physical_theta(theta_nle)
    )
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
        "state_order": order,
    }


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
    truth: torch.Tensor,
    summary: dict[str, torch.Tensor],
    latex_labels: bool = False,
) -> None:
    """Plot true and posterior natural-scale parameter values."""
    entries = [
        (index, state)
        for index in indices
        for state in (range(truth.shape[0]) if switching[index] else (0,))
    ]
    x = np.arange(len(entries))
    true_values = np.array([float(truth[state, index]) for index, state in entries])
    means = np.array([float(summary["theta_mean"][state, index]) for index, state in entries])
    lower = np.array([float(summary["theta_low"][state, index]) for index, state in entries])
    upper = np.array([float(summary["theta_high"][state, index]) for index, state in entries])
    labels = []
    for index, state in entries:
        if latex_labels:
            symbol = names[index]
            labels.append(
                rf"$\{symbol}_{{{state}}}$"
                if switching[index]
                else (rf"$\{symbol}$" if symbol != "c" else r"$c$")
            )
        else:
            labels.append(
                f"{names[index]}\nz={state}"
                if switching[index]
                else f"{names[index]}\nshared"
            )
    ax.bar(x - 0.18, true_values, 0.36, color="#444444", label="truth")
    ax.bar(
        x + 0.18,
        means,
        0.36,
        yerr=np.vstack([means - lower, upper - means]),
        capsize=2,
        color="#2364aa",
        alpha=0.82,
        label="posterior",
    )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel("natural scale")
    ax.tick_params(axis="x", labelsize=7)


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
