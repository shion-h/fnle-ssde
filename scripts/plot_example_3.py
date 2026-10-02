"""Plot Figure 3 from the post-burn-in portion of the full MCMC history."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT / "src", SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from example_3 import (  # noqa: E402
    BURN_IN,
    DYNAMICS_DT,
    HISTORY_PATHS,
    NUM_CHAINS,
    NUM_SWEEPS,
    RESULT_PATH,
)
from fnle_ssde.dynamics import LotkaVolterraDynamics  # noqa: E402
from fnle_ssde.visualization import (  # noqa: E402
    combine_chain_histories,
    match_chain_regime_labels,
    ParameterPlotGroup,
    PosteriorFigureCase,
    plot_multichain_traces,
    plot_posterior_figure,
)


FIGURE_PATH = RESULT_PATH / "posterior.png"
FIGURE_PDF_PATH = FIGURE_PATH.with_suffix(".pdf")
TRACE_PATH = RESULT_PATH / "multichain_trace.png"
TRACE_PDF_PATH = TRACE_PATH.with_suffix(".pdf")
REGIME_ALIGNMENT_PATH = RESULT_PATH / "regime_alignment.csv"
FONT_SCALE = 2.0
TEXT_FONT_COEFFICIENT = 9.0
PANEL_TITLE_FONT_COEFFICIENT = 12.0
OVERALL_TITLE_FONT_COEFFICIENT = 14.0
PARAMETER_NAMES = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\sigma_1",
    r"\sigma_2",
)
PARAMETER_KEYS = ("alpha", "beta", "gamma", "delta", "sigma_1", "sigma_2")
PARAMETER_SWITCHING = (True, True, True, True, False, False)
DENSITY_GROUPS = (
    (r"$\alpha$", ("alpha[1]", "alpha[2]")),
    (r"$\beta$", ("beta[1]", "beta[2]")),
    (r"$\gamma$", ("gamma[1]", "gamma[2]")),
    (r"$\delta$", ("delta[1]", "delta[2]")),
    (r"$\sigma_1$", ("sigma_1",)),
    (r"$\sigma_2$", ("sigma_2",)),
    (r"$q_{12}$", ("q12",)),
    (r"$q_{21}$", ("q21",)),
    (r"$\tau_1$", ("tau_1",)),
    (r"$\tau_2$", ("tau_2",)),
)


def plot_result(
    output: Path,
    T_obs: torch.Tensor,
    x_obs: torch.Tensor,
    history: dict[str, list[object]],
    dynamics: LotkaVolterraDynamics,
    additional_outputs: tuple[Path, ...] = (),
) -> None:
    """Create the real-data posterior figure from combined post-burn-in draws."""
    dense_times = torch.linspace(float(T_obs[0]), float(T_obs[-1]), 1_000)
    case = PosteriorFigureCase(
        title=r"$\mathit{Didinium}$–$\mathit{Paramecium}$",
        history=history,
        observation_times=T_obs,
        observations=x_obs,
        y_times=dense_times,
        z_times=dense_times,
        num_regimes=2,
        start=0,
        parameter_names=PARAMETER_NAMES,
        parameter_switching=PARAMETER_SWITCHING,
        parameter_groups=(
            ParameterPlotGroup("Switching drift parameters", (0, 1, 2, 3), 4),
            ParameterPlotGroup("Shared diffusion", (4, 5), 1),
        ),
        trajectory_dimensions=(0, 1),
        trajectory_labels=(r"$\mathit{Didinium}$", r"$\mathit{Paramecium}$"),
        dynamics=dynamics,
    )
    plot_posterior_figure(
        (case,),
        output,
        figsize=(13, 13),
        additional_outputs=additional_outputs,
        height_ratios=(0.12, 0.90, 0.15, 0.22, 1, 1, 1),
        font_scale=FONT_SCALE,
        text_font_coefficient=TEXT_FONT_COEFFICIENT,
        panel_title_font_coefficient=PANEL_TITLE_FONT_COEFFICIENT,
        overall_title_font_coefficient=OVERALL_TITLE_FONT_COEFFICIENT,
    )


def extract_aligned_scalar_chains(
    histories: tuple[dict[str, list[object]], ...],
    regime_orders: tuple[tuple[int, ...], ...],
    dynamics: LotkaVolterraDynamics,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return aligned natural-scale scalar traces and their display labels."""
    theta = torch.stack(
        [
            dynamics.to_physical_theta(torch.stack(history["theta"]))[:, list(order)]
            for history, order in zip(histories, regime_orders, strict=True)
        ]
    )
    Q = torch.stack(
        [
            torch.stack(history["Q"])[:, list(order)][:, :, list(order)]
            for history, order in zip(histories, regime_orders, strict=True)
        ]
    )
    tau = torch.stack(
        [torch.stack(history["log_tau"]).exp() for history in histories]
    )

    chains: dict[str, torch.Tensor] = {}
    display_names: dict[str, str] = {}
    for parameter, (key, symbol, switching) in enumerate(
        zip(PARAMETER_KEYS, PARAMETER_NAMES, PARAMETER_SWITCHING, strict=True)
    ):
        if switching:
            for regime in range(theta.shape[2]):
                name = f"{key}[{regime + 1}]"
                chains[name] = theta[:, :, regime, parameter]
                display_names[name] = rf"${symbol}_{{{regime + 1}}}$"
        else:
            chains[key] = theta[:, :, 0, parameter]
            display_names[key] = rf"${symbol}$"

    chains["q12"] = Q[:, :, 0, 1]
    chains["q21"] = Q[:, :, 1, 0]
    chains["tau_1"] = tau[:, :, 0]
    chains["tau_2"] = tau[:, :, 1]
    display_names.update(
        {
            "q12": r"$q_{12}$",
            "q21": r"$q_{21}$",
            "tau_1": r"$\tau_1$",
            "tau_2": r"$\tau_2$",
        }
    )
    return chains, display_names


def save_regime_alignment(
    regime_orders: tuple[tuple[int, ...], ...],
) -> None:
    """Record the fixed chain-level regime permutations used in all outputs."""
    REGIME_ALIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "chain": range(len(regime_orders)),
            "reference_chain": 0,
            "regime_order": [
                "-".join(str(regime + 1) for regime in order)
                for order in regime_orders
            ],
        }
    ).to_csv(REGIME_ALIGNMENT_PATH, index=False)
    print(f"saved: {REGIME_ALIGNMENT_PATH}")


def main() -> None:
    missing = [path for path in HISTORY_PATHS if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Numerical results not found:\n{missing_text}\n"
            "Run scripts/example_3.py first."
        )
    payloads = tuple(
        torch.load(path, map_location="cpu", weights_only=False)
        for path in HISTORY_PATHS
    )
    histories = tuple(payload["history"] for payload in payloads)
    data = payloads[0]["data"]
    del payloads
    if len(histories) != NUM_CHAINS:
        raise ValueError(f"Expected {NUM_CHAINS} chains, got {len(histories)}.")
    if any(len(history["theta"]) != NUM_SWEEPS for history in histories):
        raise ValueError(f"Every chain must contain {NUM_SWEEPS} stored sweeps.")
    T_obs, x_obs = data["T_obs"], data["x_obs"]
    dynamics = LotkaVolterraDynamics(
        dt=DYNAMICS_DT,
        device="cpu",
        state_upper_bound=1e4,
    )
    regime_orders = match_chain_regime_labels(
        histories,
        switching_parameter_mask=PARAMETER_SWITCHING,
        start=BURN_IN,
        reference_chain=0,
    )
    chains, display_names = extract_aligned_scalar_chains(
        histories, regime_orders, dynamics
    )
    combined_history = combine_chain_histories(
        histories,
        start=BURN_IN,
        regime_orders=regime_orders,
    )
    del histories
    plot_result(
        FIGURE_PATH,
        T_obs,
        x_obs,
        combined_history,
        dynamics,
        additional_outputs=(FIGURE_PDF_PATH,),
    )
    plot_multichain_traces(
        chains,
        TRACE_PATH,
        burn_in=BURN_IN,
        display_names=display_names,
        density_groups=DENSITY_GROUPS,
        additional_outputs=(TRACE_PDF_PATH,),
        font_scale=FONT_SCALE,
        text_font_coefficient=TEXT_FONT_COEFFICIENT,
        panel_title_font_coefficient=PANEL_TITLE_FONT_COEFFICIENT,
    )
    save_regime_alignment(regime_orders)


if __name__ == "__main__":
    main()
