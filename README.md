# MDRS-SDE Simulation Code

This repository contains simulation and empirical-analysis code for the
Microstructure-Driven Regime-Switching SDE (MDRS-SDE) paper.

## Repository structure

- `src/simulator.py` — Numba-accelerated simulator for the four-dimensional leaky-extrema MDRS-SDE.
- `run_synthetic_experiments.py` — synthetic figures and tables used for paper reproduction.
- `run_simulation_diagnostics.py` — numerical sanity checks for simulator behavior.
- `src/empirical.py` — real-data empirical calibration utilities.
- `run_empirical_calibration.py` — BTC/ETH main empirical pipeline with XRP/SOL robustness.
- `figures/` — generated figures.
- `tables/` — generated synthetic tables.
- `results/empirical/` — generated empirical CSV tables.

## Data location

The empirical pipeline assumes that raw 5-minute OHLCV CSV files live under the
repository root in `data/`:

```text
data/
  btcusdt_5m.csv
  ethusdt_5m.csv
  xrpusdt_5m.csv
  solusdt_5m.csv
```

Expected CSV columns:

- `Datetime`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

If your filenames differ, pass explicit asset mappings with `--assets`.

## Empirical sample split

The empirical pipeline uses a presample-aware split:

- Presample / burn-in: available 2020 history
- Train: 2021-01-01 to 2023-12-31
- Validation: 2024-01-01 to 2024-12-31
- Test: 2025-01-01 to 2025-12-31
- Recent robustness: 2026-01-01 onward

The presample period is used only for rolling-window initialization, empirical
support/resistance construction, and leaky-state burn-in. It is not used for
parameter fitting, model selection, or headline performance reporting.

SOLUSDT starts later in 2020 than the other assets, but the available presample
history before 2021 is still longer than the 24-hour rolling window used for
signal and extrema construction.

## Run empirical calibration

Default command:

```bash
uv run python run_empirical_calibration.py \
  --data-dir data \
  --out-dir results/empirical
```

If your BTC file is named `btcusdt_5m(1).csv`, use:

```bash
uv run python run_empirical_calibration.py \
  --data-dir data \
  --out-dir results/empirical \
  --assets BTC:btcusdt_5m\(1\).csv ETH:ethusdt_5m.csv XRP:xrpusdt_5m.csv SOL:solusdt_5m.csv
```

## Empirical outputs

The pipeline writes:

- `data_coverage.csv`
- `ou_calibration.csv`
- `ou_oos_diagnostics.csv`
- `leaky_extrema_calibration.csv`
- `leaky_grid_search.csv`
- `volatility_prediction_hac.csv`
- `return_moments_batch_means.csv`
