"""Plot Figure 3 from the post-burn-in portion of the full MCMC history."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT / "src", SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from example_3 import (  # noqa: E402
    BURN_IN,
    DYNAMICS_DT,
    HISTORY_PATH,
    NUM_SWEEPS,
    read_observations,
)
from fnle_ssde.dynamics import LotkaVolterraDynamics  # noqa: E402
from fnle_ssde.visualization import (  # noqa: E402
    ParameterPlotGroup,
    PosteriorFigureCase,
    plot_posterior_figure,
    save_figure,
)


FIGURE_PATH = ROOT / "results/figures/example_3.png"
FIGURE_PDF_PATH = FIGURE_PATH.with_suffix(".pdf")
TRACE_PATH = ROOT / "results/figures/example_3_trace.png"
TRACE_PDF_PATH = TRACE_PATH.with_suffix(".pdf")
PARAMETER_NAMES = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\sigma_D",
    r"\sigma_P",
)


def plot_result(
    output: Path,
    T_obs: torch.Tensor,
    x_obs: torch.Tensor,
    history: dict[str, list[object]],
    dynamics: LotkaVolterraDynamics,
    additional_outputs: tuple[Path, ...] = (),
    font_scale: float = 1.0,
) -> None:
    """Configure and create the real-data posterior figure."""
    if len(history["T_all"]) != NUM_SWEEPS:
        raise ValueError(
            f"Expected {NUM_SWEEPS} stored sweeps, got {len(history['T_all'])}."
        )
    dense_times = torch.linspace(float(T_obs[0]), float(T_obs[-1]), 1_000)
    case = PosteriorFigureCase(
        title=r"$\mathit{Didinium}$–$\mathit{Paramecium}$",
        history=history,
        observation_times=T_obs,
        observations=x_obs,
        y_times=dense_times,
        z_times=dense_times,
        num_regimes=2,
        start=BURN_IN,
        parameter_names=PARAMETER_NAMES,
        parameter_switching=(True, True, True, True, False, False),
        parameter_groups=(
            ParameterPlotGroup("Switching drift parameters", (0, 1, 2, 3), 4),
            ParameterPlotGroup("Shared diffusion", (4, 5), 1),
        ),
        trajectory_dimensions=(0, 1),
        trajectory_labels=(r"$\mathit{Didinium}$", r"$\mathit{Paramecium}$"),
        dynamics=dynamics,
        parameter_ylabel="physical scale",
    )
    plot_posterior_figure(
        (case,),
        output,
        figsize=(13, 13),
        additional_outputs=additional_outputs,
        height_ratios=(0.12, 0.90, 0.15, 0.22, 1, 1, 1),
        font_scale=font_scale,
    )


def plot_traces(
    output: Path,
    history: dict[str, list[object]],
    dynamics: LotkaVolterraDynamics,
    additional_outputs: tuple[Path, ...] = (),
) -> None:
    """Plot natural-scale parameter traces, retaining the burn-in portion."""
    theta = dynamics.to_physical_theta(torch.stack(history["theta"]))
    Q = torch.stack(history["Q"])
    tau = torch.stack(history["log_tau"]).exp()
    draws = torch.arange(theta.shape[0]) + 1
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)

    for parameter, ax in enumerate(axes.flat[:6]):
        if parameter < 4:
            for regime, color in enumerate(("#2364aa", "#c44900")):
                ax.plot(
                    draws,
                    theta[:, regime, parameter],
                    color=color,
                    lw=0.55,
                    alpha=0.75,
                    label=f"regime {regime + 1}",
                )
        else:
            ax.plot(draws, theta[:, 0, parameter], color="#2f4858", lw=0.55)
        ax.set_title(rf"${PARAMETER_NAMES[parameter]}$")

    axes[2, 0].plot(draws, Q[:, 0, 1], color="#2364aa", lw=0.55)
    axes[2, 0].set_title("q12")
    axes[2, 1].plot(draws, Q[:, 1, 0], color="#c44900", lw=0.55)
    axes[2, 1].set_title("q21")
    axes[2, 2].plot(
        draws,
        tau[:, 0],
        color="#c44900",
        lw=0.55,
        label=r"$\mathit{Didinium}$",
    )
    axes[2, 2].plot(
        draws,
        tau[:, 1],
        color="#2364aa",
        lw=0.55,
        label=r"$\mathit{Paramecium}$",
    )
    axes[2, 2].set_title("observation noise tau")

    for ax in axes.flat:
        ax.axvline(BURN_IN, color="black", lw=1, ls="--", alpha=0.7)
        ax.grid(alpha=0.12)
        ax.set_xlabel("sweep")
    axes[0, 0].legend(fontsize=8)
    axes[2, 2].legend(fontsize=8)
    fig.suptitle(
        "MCMC parameter traces; dashed line = burn-in cutoff",
        fontsize=14,
    )
    save_figure(fig, output, additional_paths=additional_outputs)


def main() -> None:
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Numerical result not found: {HISTORY_PATH}\n"
            "Run scripts/example_3.py first."
        )
    history = torch.load(HISTORY_PATH, map_location="cpu", weights_only=False)
    T_obs, x_obs = read_observations()
    dynamics = LotkaVolterraDynamics(
        dt=DYNAMICS_DT,
        device="cpu",
        state_upper_bound=1e4,
    )
    plot_result(
        FIGURE_PATH,
        T_obs,
        x_obs,
        history,
        dynamics,
        additional_outputs=(FIGURE_PDF_PATH,),
    )
    plot_traces(
        TRACE_PATH,
        history,
        dynamics,
        additional_outputs=(TRACE_PDF_PATH,),
    )


if __name__ == "__main__":
    main()
