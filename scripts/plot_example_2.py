"""Plot Figure 2 from the synthetic Gibbs histories."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from example_2 import (  # noqa: E402
    CLE_HISTORY_PATH,
    LV_HISTORY_PATH,
    SIR_HISTORY_PATH,
    experiments,
)
from fnle_ssde.visualization import (  # noqa: E402
    ParameterPlotGroup,
    PosteriorFigureCase,
    plot_posterior_figure,
)


FIGURE_PATH = ROOT / "results/figures/example_2.png"
PDF_PATH = FIGURE_PATH.with_suffix(".pdf")
TITLE_SCALE = 1.8 * (1.8 / 2.0)


def load(path: Path) -> tuple[dict[str, list[object]], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/example_2.py first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["history"], payload["data"]


def main() -> None:
    dynamics_by_model = {
        experiment.name: experiment.dynamics for experiment in experiments()
    }
    results = {
        "lv": load(LV_HISTORY_PATH),
        "cle": load(CLE_HISTORY_PATH),
        "sir": load(SIR_HISTORY_PATH),
    }
    cases = []
    for model, (history, data) in results.items():
        z_truth_times = torch.cat(
            (data["times"][:1], data["T_true"], data["times"][-1:])
        )
        z_truth = torch.cat((data["z_true"], data["z_true"][-1:]))
        if model == "lv":
            title = "Lotka-Volterra"
            groups = (
                ParameterPlotGroup("Drift", (0, 1, 2, 3), 4),
                ParameterPlotGroup("Diffusion", (4, 5), 1.5),
            )
            names = (
                r"\alpha",
                r"\beta",
                r"\gamma",
                r"\delta",
                r"\sigma_1",
                r"\sigma_2",
            )
            switching = (True, True, True, True, False, False)
        elif model == "cle":
            title = "Gene-expression CLE"
            groups = (
                ParameterPlotGroup("Switching", (0,), 2),
                ParameterPlotGroup("Shared", (1, 3), 2),
                ParameterPlotGroup(r"$\gamma$", (2,), 1.4),
                ParameterPlotGroup(r"$c$", (4,), 1.4),
            )
            names = (r"\alpha", r"\beta", r"\gamma", r"\delta", "c")
            switching = (True, False, False, False, False)
        else:
            title = "SIR epidemic model"
            groups = (ParameterPlotGroup("Switching", (0, 1), 1),)
            names = (r"\beta", r"\gamma")
            switching = (True, True)
        dimensions = (0, 1)
        trajectory_labels = (
            ("susceptible", "recovered")
            if model == "sir"
            else tuple(f"y[{dimension}]" for dimension in dimensions)
        )
        cases.append(
            PosteriorFigureCase(
                title=title,
                history=history,
                observation_times=data["T_obs"],
                observations=data["x_obs"],
                y_times=data["times"],
                z_times=data["times"],
                num_regimes=data["theta_true"].shape[0],
                start=500,
                parameter_names=names,
                parameter_switching=switching,
                parameter_groups=groups,
                trajectory_dimensions=dimensions,
                trajectory_labels=trajectory_labels,
                dynamics=dynamics_by_model[model],
                show_truth=True,
                theta_truth=data["theta_true"],
                y_truth_times=data["times"],
                y_truth=data["y_true"],
                z_truth_times=z_truth_times,
                z_truth=z_truth,
            )
        )

    plot_posterior_figure(
        tuple(cases),
        FIGURE_PATH,
        figsize=(18.5, 11.5),
        additional_outputs=(PDF_PATH,),
        height_ratios=(0.12, 1.10, 0.15, 0.22, 1, 1, 0.70),
        title_scale=TITLE_SCALE,
    )


if __name__ == "__main__":
    main()
