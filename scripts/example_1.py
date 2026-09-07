"""Evaluate simulator and NLE transition densities for Figure 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from example_2 import NLE_CACHE_PATHS  # noqa: E402
from fnle_ssde.dynamics import LotkaVolterraDynamics  # noqa: E402
from fnle_ssde.nle import NLEEstimator  # noqa: E402
from generate_data_1 import DATA_PATH as SIMULATOR_DATA_PATH  # noqa: E402


NLE_CACHE_PATH = NLE_CACHE_PATHS["lv"]
NUMERICAL_RESULT_PATH = ROOT / "results/example1.pt"


@torch.no_grad()
def nle_log_prob(
    nle: NLEEstimator,
    values: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    result = []
    for start in range(0, values.shape[0], 4_096):
        batch = values[start : start + 4_096]
        repeated = context.expand(batch.shape[0], -1)
        result.append(
            nle.transition_log_prob(
                batch,
                theta=repeated[:, :6],
                x_prev=repeated[:, 6:8],
                n_steps=repeated[:, 8],
                include_jacobian=True,
            ).reshape(-1).cpu()
        )
    return torch.cat(result)


def kde_log_prob(
    kde: gaussian_kde,
    location: np.ndarray,
    scale: np.ndarray,
    values: torch.Tensor,
) -> torch.Tensor:
    standardized = (values.numpy() - location) / scale
    return torch.from_numpy(kde.logpdf(standardized.T) - np.log(scale).sum())


def evaluate(
    nle: NLEEstimator,
    context: torch.Tensor,
    simulator_samples: torch.Tensor,
    index: int,
) -> dict[str, object]:
    """Compare simulator KDE and NLE density for one fixed context."""
    kde_fit = simulator_samples[:4_000]
    simulator_test = simulator_samples[4_000:]
    location = np.median(kde_fit.numpy(), axis=0)
    scale = np.maximum(np.std(kde_fit.numpy(), axis=0, ddof=1), 1e-8)
    kde = gaussian_kde(((kde_fit.numpy() - location) / scale).T)

    repeated = context.expand(8_000, -1)
    torch.manual_seed(10 * index + 1)
    with torch.no_grad():
        nle_samples = nle.sample_transition(
            theta=repeated[:, :6],
            x_prev=repeated[:, 6:8],
            n_steps=repeated[:, 8],
        ).cpu()

    samples_log_prob_nle = nle_log_prob(nle, simulator_test, context)
    samples_log_prob_kde = kde_log_prob(kde, location, scale, simulator_test)
    lower = torch.quantile(simulator_test, 0.001, dim=0)
    upper = torch.quantile(simulator_test, 0.999, dim=0)
    margin = 0.08 * (upper - lower).clamp_min(1e-6)
    y_axis = np.linspace(float(lower[0] - margin[0]), float(upper[0] + margin[0]), 150)
    x_axis = np.linspace(float(lower[1] - margin[1]), float(upper[1] + margin[1]), 150)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    points = torch.tensor(np.column_stack([y_mesh.ravel(), x_mesh.ravel()]), dtype=torch.float32)

    return {
        "context": context,
        "simulator_evaluation": simulator_test,
        "x_mesh": x_mesh,
        "y_mesh": y_mesh,
        "kde_grid": kde_log_prob(kde, location, scale, points).numpy().reshape(x_mesh.shape),
        "nle_grid": nle_log_prob(nle, points, context).numpy().reshape(x_mesh.shape),
        "metrics": {
            "spearman_log_density": float(
                spearmanr(samples_log_prob_kde, samples_log_prob_nle).statistic
            ),
        },
    }


def main() -> None:
    if NUMERICAL_RESULT_PATH.exists():
        print(f"using cached result: {NUMERICAL_RESULT_PATH}")
        return
    if not SIMULATOR_DATA_PATH.exists():
        raise FileNotFoundError("Run scripts/generate_data_1.py first.")
    if not NLE_CACHE_PATH.exists():
        raise FileNotFoundError(f"Trained NLE not found: {NLE_CACHE_PATH}")

    data = torch.load(SIMULATOR_DATA_PATH, map_location="cpu", weights_only=False)
    nle = NLEEstimator(
        dynamics=LotkaVolterraDynamics(dt=0.01, device="cpu"),
        device="cpu",
        model_cache_path=NLE_CACHE_PATH,
        target_type="scaled_dx",
        hidden_features=50,
        num_transforms=5,
        num_bins=10,
    )
    rows = [
        evaluate(nle, context, simulator_samples, index)
        for index, (context, simulator_samples) in enumerate(
            zip(data["contexts"], data["simulator_samples"], strict=True)
        )
    ]
    NUMERICAL_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"rows": rows, "contexts": data["contexts"]},
        NUMERICAL_RESULT_PATH,
    )
    print(f"saved: {NUMERICAL_RESULT_PATH}")


if __name__ == "__main__":
    main()
