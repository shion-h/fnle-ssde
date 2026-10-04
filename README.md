# FNLE-SSDE

The paper experiments are implemented in `scripts/example_1.py` through
`scripts/example_3.py`. Data generation and plotting are separate:
`scripts/generate_data_{1,2,3}.py` and `scripts/plot_example_{1,2,3}.py`.

## Reference machine

The runtime estimates below refer to this development PC:

| Component | Specification |
| --- | --- |
| CPU | Intel Core i7-1195G7, 2.90 GHz nominal, 4 physical cores / 8 logical threads |
| Memory | 32 GB RAM (approximately 31 GiB reported by the OS) |
| OS | Ubuntu 22.04.5 LTS, x86-64 |
| Software | Python 3.12.11, PyTorch 2.5.1, Pyro 1.9.1, sbi 0.24.0 |
| Compute device | CPU only; no GPU acceleration |

## Expected runtime

These are wall-clock measurements or estimates from saved 2026 runs, not a
newly timed end-to-end benchmark. Dependency installation is excluded, and
actual times vary with system load and NLE early stopping.

| Experiment | Approximate elapsed time |
| --- | --- |
| Example 1 | About 14 min for exact OU MCMC and 2.3 h for each FNLE MCMC; data generation and FNLE training are additional |
| Example 2 | About 12.5-13 h with trained NLE caches; around 13.5-14 h including NLE preparation |
| Example 3 | 7.5-8.5 h total |

With trained NLE caches available, the Example 1 MCMC stages take about
**4.8 hours** in total. Parallel-chain timings are wall time to finish all
chains, not the sum of their individual times.

For context, the recorded NLE preparation times (simulation plus training)
were about 25 minutes for synthetic LV, 13 minutes for CLE, 13 minutes for
SIR, and 3.1 hours for real-data LV. Real-data MCMC took about 4.3 hours in
one run; related runs took about 5.3 hours.

### Reusing saved results

- To regenerate figures from saved MCMC histories, run only
  `plot_example_{1,2,3}.py`. No NLE training or MCMC is needed.
- `example_1.py` reuses existing OU training data, models, and complete chain
  histories. `example_2.py` and `example_3.py` still rerun MCMC and overwrite
  their history files.

## Reproduce the figures

Run these commands from the repository root, in order. Each example's data
generation, inference, and plotting scripts are grouped together.

```bash
uv run python scripts/generate_data_1.py
uv run python scripts/example_1.py
uv run python scripts/plot_example_1.py
uv run python scripts/generate_data_2.py
uv run python scripts/example_2.py
uv run python scripts/plot_example_2.py
uv run python scripts/generate_data_3.py
uv run python scripts/example_3.py
uv run python scripts/plot_example_3.py
uv run python scripts/diagnostics.py
```
