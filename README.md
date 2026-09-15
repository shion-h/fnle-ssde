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

Example 2 runs the three synthetic experiments sequentially, using one PyTorch
thread. Example 3 runs four GS chains in parallel processes, with one PyTorch
intra-op and inter-op thread per chain. Its NLE is trained once and shared by
the four chains, not trained separately for each chain.

## Expected runtime

**Allow approximately 9-11 hours to reproduce all three experiments from
scratch on this PC**, including training-data generation, NLE training, GS,
density comparison, and plotting. These are wall-clock estimates based on
saved August-September 2026 runs, not a newly timed end-to-end benchmark of
the current scripts. Dependency installation and downloads are excluded.

| Experiment | Workload | Approximate elapsed time |
| --- | --- | --- |
| Example 1 | Simulator/NLE density comparison at six contexts; reuses the LV NLE from Example 2 | A few minutes (less than 0.1 h), excluding the shared NLE training |
| Example 2 | Synthetic LV, CLE, and SIR; 100,000 NLE training transitions per model; one 1,000-sweep GS chain per model | 1.5-2 h total |
| Example 3 | Real-data LV; 800,000 NLE training transitions; four parallel 10,000-sweep GS chains | 7.5-8.5 h total |

The sweep counts include burn-in: 500 sweeps for each synthetic chain and
5,000 for each real-data chain. Example 3 uses NUTS tree depths of 5 for `y`
and 3 for `theta`. Its GS runtime is the elapsed time for all four parallel
chains to finish, not the sum of their individual runtimes.

For context, the recorded NLE preparation times (simulation plus training)
were about 25 minutes for synthetic LV, 13 minutes for CLE, 13 minutes for
SIR, and 3.1 hours for real-data LV. These training measurements used one
PyTorch thread. The real-data NLE stopped after 80 epochs with early-stopping
patience 20, rather than running to the 5,000-epoch limit. Real-data GS took
about 4.3 hours for the seed-0-through-3 run; related runs took about 5.3 hours.
Stopping epochs, NUTS trajectories, thread settings, background CPU load, and
file synchronization can change these times substantially.

### Reusing saved results

- With all four trained NLE caches available, allow roughly **5-7 hours** to
  rerun GS and regenerate the figures; NLE simulation and training are skipped.
- To regenerate figures from saved numerical results and MCMC histories, run
  only `plot_example_{1,2,3}.py`. No NLE training or GS is needed.
- `example_2.py` and `example_3.py` still rerun GS and overwrite their history
  files even when those histories already exist. `example_1.py` instead skips
  density evaluation when `results/example1.pt` exists.

For a fresh reproduction, run `generate_data_2.py` and `example_2.py` before
`generate_data_1.py` and `example_1.py`: Example 1 needs the synthetic LV data
and its trained NLE. The shared LV training cost is counted only once above.

## Reproduce the figures

Run these commands from the repository root, in the order shown:

```bash
uv run python scripts/generate_data_2.py
uv run python scripts/example_2.py
uv run python scripts/plot_example_2.py
uv run python scripts/generate_data_1.py
uv run python scripts/example_1.py
uv run python scripts/plot_example_1.py
uv run python scripts/generate_data_3.py
uv run python scripts/example_3.py
uv run python scripts/plot_example_3.py
```
