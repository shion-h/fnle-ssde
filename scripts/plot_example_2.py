"""Plot Figure 2 from the synthetic MCMC histories."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from example_2 import (  # noqa: E402
    RESULT_PATH,
    experiments,
)
from fnle_ssde.visualization import (  # noqa: E402
    ParameterPlotGroup,
    PosteriorFigureCase,
    combine_chain_histories,
    match_regime_labels,
    plot_posterior_figure,
)


FIGURE_PATH = RESULT_PATH / "example_2.png"
PDF_PATH = FIGURE_PATH.with_suffix(".pdf")
FONT_SCALE = 1.1
TEXT_FONT_COEFFICIENT = 13.0
PANEL_TITLE_FONT_COEFFICIENT = 13.0
OVERALL_TITLE_FONT_COEFFICIENT = 14.0


def main() -> None:
    cases = []
    for experiment in experiments():
        model = experiment.name
        path = experiment.history_path
        paths = tuple(
            path.with_stem(f"{path.stem}{chain_id}")
            for chain_id in range(len(experiment.seeds))
        )
        payloads = tuple(
            torch.load(chain_path, map_location="cpu", weights_only=False)
            for chain_path in paths
        )
        data = payloads[0]["data"]
        histories = tuple(payload["history"] for payload in payloads)
        theta_means = torch.stack([
            torch.stack(history["theta"][experiment.burn_in:]).mean(0)
            for history in histories
        ])
        regime_orders = match_regime_labels(
            theta_means, theta_truth=data["theta_true"],
        ).tolist()
        history = combine_chain_histories(
            histories, start=experiment.burn_in, regime_orders=regime_orders,
        )
        del payloads, histories

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
            names = (r"\rho", r"\beta", r"\gamma", r"\delta", "c")
            switching = (True, False, False, False, False)
        else:
            title = "Susceptible–Infected–Recovered epidemic model"
            groups = (ParameterPlotGroup("Switching", (0, 1), 1),)
            names = (r"\beta", r"\gamma")
            switching = (True, True)
        dimensions = (0, 1)
        trajectory_labels = (r"$Y_{t,1}$", r"$Y_{t,2}$")
        cases.append(
            PosteriorFigureCase(
                title=title,
                history=history,
                observation_times=data["T_obs"],
                observations=data["x_obs"],
                y_times=data["times"],
                z_times=data["times"],
                num_regimes=data["theta_true"].shape[0],
                start=0,
                parameter_names=names,
                parameter_switching=switching,
                parameter_groups=groups,
                trajectory_dimensions=dimensions,
                trajectory_labels=trajectory_labels,
                dynamics=experiment.dynamics,
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
        font_scale=FONT_SCALE,
        text_font_coefficient=TEXT_FONT_COEFFICIENT,
        panel_title_font_coefficient=PANEL_TITLE_FONT_COEFFICIENT,
        overall_title_font_coefficient=OVERALL_TITLE_FONT_COEFFICIENT,
        parameter_section_title="Parameters",
    )


if __name__ == "__main__":
    main()
