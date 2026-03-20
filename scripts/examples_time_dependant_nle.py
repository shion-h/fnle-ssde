from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyro.distributions as dist
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fnle_ssde.common.dynamics import LotkaVolterraDynamics
from fnle_ssde.common.nle import NLEEstimator
from fnle_ssde.discrete.utils import create_ground_truth_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize whether n_steps conditioning works in NLEEstimator."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-length", type=int, default=250)
    parser.add_argument("--n-params", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ref-noise", type=float, default=0.02)
    parser.add_argument("--max-n-steps", type=int, default=20)
    parser.add_argument(
        "--eval-n-steps",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15, 20],
        help="n_steps values to compare in the visualization.",
    )
    parser.add_argument("--n-eval-contexts", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts" / "examples_time_dependant_nle.png",
    )
    parser.add_argument("--hide", action="store_true")
    return parser.parse_args()


def simulate_reference_trajectory(
    dynamics: LotkaVolterraDynamics,
    theta: torch.Tensor,
    x0: torch.Tensor,
    length: int,
) -> torch.Tensor:
    x = [x0.to(torch.float32)]
    x_curr = x[0]
    for _ in range(length - 1):
        x_curr = dynamics.simulate_one_step(x_curr, theta)
        x_curr = torch.clamp(x_curr, min=0.0, max=1e4)
        x.append(x_curr)
    return torch.stack(x)


def simulate_targets(
    dynamics: LotkaVolterraDynamics,
    x_contexts: torch.Tensor,
    theta: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    targets = []
    for x_init in x_contexts:
        x_curr = x_init.clone()
        for _ in range(n_steps):
            x_curr = dynamics.simulate_one_step(x_curr, theta)
            x_curr = torch.clamp(x_curr, min=0.0, max=1e4)
        targets.append(x_curr)
    return torch.stack(targets)


def build_conditions(
    theta: torch.Tensor,
    x_contexts: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    theta_batch = theta.unsqueeze(0).repeat(x_contexts.shape[0], 1)
    n_steps_batch = torch.full(
        (x_contexts.shape[0], 1),
        float(n_steps),
        dtype=x_contexts.dtype,
        device=x_contexts.device,
    )
    return torch.cat([theta_batch, x_contexts, n_steps_batch], dim=-1)


def average_log_prob(
    nle: NLEEstimator,
    targets: torch.Tensor,
    conditions: torch.Tensor,
) -> float:
    with torch.no_grad():
        log_probs = nle.estimator.log_prob(
            targets.unsqueeze(0), condition=conditions
        )
    return float(log_probs.mean().item())


def plot_results(
    eval_n_steps: list[int],
    avg_log_prob_matrix: np.ndarray,
    output_path: Path,
    hide: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    heatmap = axes[0].imshow(avg_log_prob_matrix, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(eval_n_steps)))
    axes[0].set_yticks(range(len(eval_n_steps)))
    axes[0].set_xticklabels(eval_n_steps)
    axes[0].set_yticklabels(eval_n_steps)
    axes[0].set_xlabel("Conditioned n_steps")
    axes[0].set_ylabel("True n_steps used to generate targets")
    axes[0].set_title("Average log_prob")
    for i in range(len(eval_n_steps)):
        for j in range(len(eval_n_steps)):
            axes[0].text(
                j,
                i,
                f"{avg_log_prob_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)

    for row_idx, true_n_steps in enumerate(eval_n_steps):
        axes[1].plot(
            eval_n_steps,
            avg_log_prob_matrix[row_idx],
            marker="o",
            label=f"true n={true_n_steps}",
        )
    axes[1].set_xlabel("Conditioned n_steps")
    axes[1].set_ylabel("Average log_prob")
    axes[1].set_title("Rows of the heatmap")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    if hide:
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.max_n_steps < 1:
        raise ValueError("--max-n-steps must be positive.")
    if any(n < 1 for n in args.eval_n_steps):
        raise ValueError("--eval-n-steps must all be positive.")

    device = args.device
    dynamics = LotkaVolterraDynamics(device=device)
    hmm_param_true = create_ground_truth_parameters()
    theta_true = hmm_param_true["emissions"][0].to(torch.float32)
    x0 = torch.tensor([1.0, 0.5], dtype=torch.float32)

    print("Generating reference trajectory...")
    x_ref = simulate_reference_trajectory(
        dynamics=dynamics,
        theta=theta_true,
        x0=x0,
        length=args.train_length,
    )

    print("Training NLE with random n_steps conditioning...")
    sampling_dist = dist.MultivariateNormal(
        theta_true,
        0.3 * torch.eye(dynamics.theta_dim),
    )
    nle = NLEEstimator(dynamics, sampling_dist=sampling_dist, device=device)
    nle.train(
        x_ref=x_ref,
        n_params=args.n_params,
        ref_noize=args.ref_noise,
        n_steps=(args.max_n_steps,),
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )

    if not nle.conditions_on_n_steps:
        raise RuntimeError("This script expects n_steps conditioning to be enabled.")

    x_contexts = x_ref[: args.n_eval_contexts].to(torch.float32)
    avg_log_prob_matrix = np.zeros(
        (len(args.eval_n_steps), len(args.eval_n_steps)),
        dtype=np.float64,
    )

    print("Evaluating conditioned log probabilities...")
    for row_idx, true_n_steps in enumerate(args.eval_n_steps):
        targets = simulate_targets(dynamics, x_contexts, theta_true, true_n_steps)
        for col_idx, cond_n_steps in enumerate(args.eval_n_steps):
            conditions = build_conditions(theta_true, x_contexts, cond_n_steps)
            avg_log_prob_matrix[row_idx, col_idx] = average_log_prob(
                nle, targets, conditions
            )

    best_cond_idx = avg_log_prob_matrix.argmax(axis=1)
    print("\nBest conditioned n_steps for each true n_steps:")
    for row_idx, true_n_steps in enumerate(args.eval_n_steps):
        best_n = args.eval_n_steps[int(best_cond_idx[row_idx])]
        row_values = ", ".join(
            f"{cond_n}:{avg_log_prob_matrix[row_idx, col_idx]:.3f}"
            for col_idx, cond_n in enumerate(args.eval_n_steps)
        )
        print(f"  true n={true_n_steps:>2} -> best condition={best_n:>2} | {row_values}")

    plot_results(
        eval_n_steps=args.eval_n_steps,
        avg_log_prob_matrix=avg_log_prob_matrix,
        output_path=args.output,
        hide=args.hide,
    )


if __name__ == "__main__":
    main()
