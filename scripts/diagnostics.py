"""Write example 1–3 scalar-parameter R-hat and ESS to results/diagnostics.csv.

Run with no arguments after completing example_1.py, example_2.py and
example_3.py. Diagnostics use physical-scale parameters after each experiment's
burn-in and one fixed regime permutation per chain, as in the posterior plots.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import example_1  # noqa: E402
import example_2  # noqa: E402
import example_3  # noqa: E402
from fnle_ssde.dynamics import LotkaVolterraDynamics, OUDynamics  # noqa: E402
from fnle_ssde.visualization import (  # noqa: E402
    match_regime_labels,
)


OUTPUT_PATH = ROOT / "results/diagnostics.csv"
PARAMETER_NAMES = {
    "ou": ("kappa", "mu", "sigma"),
    "lv": ("alpha", "beta", "gamma", "delta", "sigma_1", "sigma_2"),
    "cle": ("alpha_over_beta", "beta", "gamma", "delta", "c"),
    "sir": ("beta", "gamma"),
}


@dataclass(frozen=True)
class Experiment:
    example: int
    model: str
    method: str
    paths: tuple[Path, ...]
    burn_in: int
    num_sweeps: int
    dynamics: object
    switching_mask: tuple[bool, ...]


def experiments() -> tuple[Experiment, ...]:
    """Read history paths and burn-in from the public experiment scripts."""
    ou = tuple(
        Experiment(
            example=1,
            model="ou",
            method=method,
            paths=tuple(
                example_1.method_output_dir(method) / f"chain{chain_id}.pt"
                for chain_id in range(len(example_1.SEEDS))
            ),
            burn_in=example_1.BURN_IN,
            num_sweeps=example_1.NUM_SWEEPS,
            dynamics=OUDynamics(dt=example_1.DT, device="cpu"),
            switching_mask=(False, True, False),
        )
        for method in example_1.METHODS
    )
    synthetic = tuple(
        Experiment(
            example=2,
            model=config.name,
            method="fnle",
            paths=tuple(
                config.history_path if len(config.seeds) == 1 else
                config.history_path.with_stem(f"{config.history_path.stem}{chain_id}")
                for chain_id in range(len(config.seeds))
            ),
            burn_in=config.burn_in,
            num_sweeps=config.num_sweeps,
            dynamics=config.dynamics,
            switching_mask=tuple(config.switching_mask.tolist()),
        )
        for config in example_2.experiments()
    )
    real = Experiment(
        example=3,
        model="lv",
        method="fnle",
        paths=example_3.HISTORY_PATHS,
        burn_in=example_3.BURN_IN,
        num_sweeps=example_3.NUM_SWEEPS,
        dynamics=LotkaVolterraDynamics(dt=example_3.DYNAMICS_DT, device="cpu"),
        switching_mask=(True, True, True, True, False, False),
    )
    return (*ou, *synthetic, real)


def load_chains(experiment: Experiment) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
    """Load only retained scalar draws; release each full history immediately."""
    theta_chains, q_chains, tau_chains = [], [], []
    data = None
    for path in experiment.paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        history = payload["history"]
        for key in ("theta", "Q", "log_tau"):
            if len(history[key]) != experiment.num_sweeps:
                raise ValueError(
                    f"Incomplete history: {path} ({key}: {len(history[key])} draws; "
                    f"expected {experiment.num_sweeps})."
                )
        if not 0 <= experiment.burn_in <= experiment.num_sweeps - 4:
            raise ValueError("At least four retained draws per chain are required.")
        theta_chains.append(torch.stack(history["theta"][experiment.burn_in:]))
        q_chains.append(torch.stack(history["Q"][experiment.burn_in:]))
        tau_chains.append(torch.stack(history["log_tau"][experiment.burn_in:]).exp())
        if data is None:
            data = payload["data"]
        del payload, history
    return torch.stack(theta_chains), torch.stack(q_chains), torch.stack(tau_chains), data


def scalar_chains(experiment: Experiment) -> dict[str, torch.Tensor]:
    """Align chain labels and extract physical theta, off-diagonal Q and tau."""
    theta_nle, q, tau, data = load_chains(experiment)
    truth = data.get("theta_true")

    orders = match_regime_labels(
        theta_nle.mean(1),
        theta_truth=truth,
        switching_parameter_mask=experiment.switching_mask,
        reference_chain=0,
    ).tolist()
    theta = torch.stack([
        experiment.dynamics.to_physical_theta(chain[:, order])
        for chain, order in zip(theta_nle, orders, strict=True)
    ])
    q = torch.stack([
        chain[:, order][:, :, order]
        for chain, order in zip(q, orders, strict=True)
    ])

    scalars = {}
    num_regimes = theta.shape[2]
    for index, (name, switching) in enumerate(zip(
        PARAMETER_NAMES[experiment.model], experiment.switching_mask, strict=True,
    )):
        if switching:
            for regime in range(num_regimes):
                scalars[f"{name}[{regime + 1}]"] = theta[:, :, regime, index]
        else:
            scalars[name] = theta[:, :, 0, index]
    for source in range(num_regimes):
        for target in range(num_regimes):
            if source != target:
                scalars[f"q[{source + 1},{target + 1}]"] = q[:, :, source, target]
    for dimension in range(tau.shape[-1]):
        name = "tau" if tau.shape[-1] == 1 else f"tau[{dimension + 1}]"
        scalars[name] = tau[:, :, dimension]
    return scalars


def diagnostic_rows(experiment: Experiment) -> list[dict[str, object]]:
    """Compute rank-normalized split R-hat and bulk/tail ESS per scalar."""
    rows = []
    for parameter, chain in scalar_chains(experiment).items():
        draws = chain.numpy()
        rows.append({
            "example": experiment.example,
            "model": experiment.model,
            "method": experiment.method,
            "parameter": parameter,
            "num_chains": draws.shape[0],
            "draws_per_chain": draws.shape[1],
            "burn_in": experiment.burn_in,
            "rhat": float(az.rhat(draws, method="rank")),
            "ess_bulk": float(az.ess(draws, method="bulk")),
            "ess_tail": float(az.ess(draws, method="tail")),
        })
    return rows


def main() -> None:
    torch.set_num_threads(1)
    configured = experiments()
    missing = [path for experiment in configured for path in experiment.paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Complete MCMC histories are missing. Run example_1.py–example_3.py "
            "or restore their histories to the configured paths:\n"
            + "\n".join(str(path) for path in missing)
        )
    rows = []
    for experiment in configured:
        print(f"Diagnosing example {experiment.example}: {experiment.model}/{experiment.method}", flush=True)
        rows.extend(diagnostic_rows(experiment))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.DataFrame(rows)
    diagnostics["rhat"] = diagnostics["rhat"].map("{:.3f}".format)
    for column in ("ess_bulk", "ess_tail"):
        diagnostics[column] = diagnostics[column].map("{:.1f}".format)
    diagnostics.to_csv(OUTPUT_PATH, index=False)
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
