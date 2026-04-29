# MDRS-SDE Experiments

This repository contains simulation and empirical-analysis code for the paper on a Microstructure-Driven Regime-Switching Stochastic Differential Equation (MDRS-SDE) for cryptocurrency perpetual futures.

The codebase has three roles:

1. Synthetic MDRS-SDE experiments.
2. Numerical simulation diagnostics.
3. Real-data empirical calibration using 5-minute perpetual futures data.

The implementation follows the revised paper framing. In particular, long-run simulations are used as numerical evidence of stability behavior, not as a proof of total-variation geometric ergodicity.

---

## Repository Structure

```text
.
├── data/
│   ├── btcusdt_5m.csv
│   ├── ethusdt_5m.csv
│   ├── xrpusdt_5m.csv
│   └── solusdt_5m.csv
│
├── results/
│   ├── synthetic/
│   │   ├── figures/
│   │   └── tables/
│   ├── diagnostics/
│   │   └── figures/
│   └── empirical/
│
├── scripts/
│   ├── __init__.py
│   ├── synthetic_experiments.py
│   ├── simulation_diagnostics.py
│   └── empirical_calibration.py
│
├── src/
│   ├── simulator.py
│   └── empirical.py
│
├── pyproject.toml
└── README.md
```

The top-level run scripts are exposed through `uv` console entry points:

```text
synthetic_experiments
simulation_diagnostics
empirical_calibration
```

Recommended commands:

```bash
uv run synthetic_experiments
uv run simulation_diagnostics
uv run empirical_calibration --data-dir data --out-dir results/empirical
```

---

## Setup

This project is intended to run with `uv`.

```bash
uv sync
```

Required Python packages include:

```text
numpy
scipy
matplotlib
numba
pandas
statsmodels
```

The `pyproject.toml` should include console entry points similar to:

```toml
[project.scripts]
synthetic_experiments = "scripts.synthetic_experiments:main"
simulation_diagnostics = "scripts.simulation_diagnostics:main"
empirical_calibration = "scripts.empirical_calibration:main"
```

If the environment is not yet configured through `pyproject.toml`, install the dependencies manually:

```bash
uv add numpy scipy matplotlib numba pandas statsmodels
```

---

## Data Requirements

The empirical pipeline expects 5-minute OHLCV CSV files under the root-level `data/` directory.

Default file names:

```text
data/btcusdt_5m.csv
data/ethusdt_5m.csv
data/xrpusdt_5m.csv
data/solusdt_5m.csv
```

Expected CSV columns:

```text
Datetime, Open, High, Low, Close, Volume
```

The `Datetime` column should be parseable as a timestamp. The pipeline sorts by timestamp, removes duplicated timestamps, validates numeric OHLCV fields, and constructs log returns and rolling features internally.

---

## Asset Roles

The empirical section uses BTC and ETH as the main assets and XRP and SOL as robustness assets.

```text
Main empirical assets:
- BTCUSDT
- ETHUSDT

Robustness assets:
- XRPUSDT
- SOLUSDT
```

This split is used only for reporting. The pipeline processes all provided assets and writes results for every asset to CSV.

The console summary prints BTC/ETH as the main block and XRP/SOL as a separate robustness block so that all processed assets are visible.

---

## Empirical Sample Split

The empirical pipeline uses a presample-aware split.

```text
Presample / burn-in: available 2020 history
Train:               2021-01-01 to 2023-12-31
Validation:          2024-01-01 to 2024-12-31
Test:                2025-01-01 to 2025-12-31
Recent robustness:   2026-01-01 onward
```

The 2020 observations are used only for:

```text
- rolling-window initialization,
- empirical support/resistance construction,
- leaky-state burn-in,
- data-quality checks.
```

They are not used for parameter fitting, model selection, validation, or headline test reporting.

Although SOLUSDT starts later in 2020 than BTC, ETH, and XRP, the available SOL presample before 2021 is still much longer than the 24-hour rolling window used for the signal and extrema construction.

---

## Synthetic Experiments

Run:

```bash
uv run synthetic_experiments
```

Generated outputs:

```text
results/synthetic/figures/
results/synthetic/tables/
```

The synthetic experiments include:

1. A trajectory illustration of the four-dimensional MDRS-SDE state.
2. A finite-horizon breakout-time sensitivity experiment.
3. A long-run distribution experiment.
4. Regime-conditional volatility diagnostics.

The synthetic long-run experiment reports batch-means uncertainty for serially dependent simulation output. It does not report iid t-statistics.

Important scope note:

```text
Synthetic long-run simulations illustrate stable numerical behavior.
They do not prove total-variation geometric ergodicity of the full lifted process.
```

---

## Simulation Diagnostics

Run:

```bash
uv run simulation_diagnostics
```

Generated outputs:

```text
results/diagnostics/figures/
```

The diagnostics include numerical sanity checks such as:

```text
- invariant support/resistance ordering,
- drift-dominance parameter checks,
- equilibrium-region stability diagnostics,
- running-mean stabilization diagnostics,
- time-step sensitivity diagnostics,
- finite-horizon exit-frequency diagnostics.
```

These diagnostics are not theorem verification. They are intended to detect implementation issues and to provide numerical sanity checks for the simulator.

---

## Empirical Calibration Pipeline

Run with the default data directory:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical
```

If file names differ from the defaults, provide explicit asset mappings:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical \
  --assets BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv XRP:xrpusdt_5m.csv SOL:solusdt_5m.csv
```

If your BTC file is named `btcusdt_5m.csv`, escape the parentheses in bash:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical \
  --assets BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv XRP:xrpusdt_5m.csv SOL:solusdt_5m.csv
```

Generated empirical outputs:

```text
results/empirical/data_coverage.csv
results/empirical/ou_calibration.csv
results/empirical/ou_oos_diagnostics.csv
results/empirical/leaky_extrema_calibration.csv
results/empirical/leaky_grid_search.csv
results/empirical/volatility_prediction_hac.csv
results/empirical/return_moments_batch_means.csv
```

---

## Empirical Pipeline Details

### 1. Microstructure Signal Construction

The empirical signal is constructed from 5-minute OHLCV data.

Features:

```text
absolute log return magnitude: |r_t|
log volume: log(1 + Volume_t)
```

For each feature, the code computes shifted 24-hour rolling z-scores using a 288-bar window. The shift ensures the signal is past-adapted.

The final signal is a three-bar max-pooled average:

```text
Z_t = (1/3) * sum_{j=0}^{2} max(Z_volume,t-j, Z_return,t-j)
```

The return component uses absolute returns because the signal is intended to measure activation and volatility-expansion magnitude rather than signed price direction.

### 2. OU Calibration

The pipeline fits a one-step AR(1) transition on the training period:

```text
Z_{t+1} = a + b Z_t + eps_t
```

The AR(1) fit is converted to OU parameters:

```text
kappa_Z
long-run mean zbar
sigma_Z
half-life
```

Out-of-sample diagnostics are reported on validation, test, and recent windows.

### 3. Leaky-Extrema Calibration

The pipeline constructs empirical rolling support and resistance levels over quiet regimes and compares them against simulated leaky support/resistance variables.

The leaky-extrema parameters are selected using validation mean absolute error. The final approximation quality is reported separately on train, validation, test, and recent splits.

Reported metrics include:

```text
MAE_R
MAE_S
RMSE_R
RMSE_S
Corr_R
Corr_S
Mean_MAE
```

The 2025 test split should be used for headline out-of-sample results.

### 4. Volatility Prediction with HAC Inference

The pipeline estimates realized-volatility prediction regressions:

```text
RV_future = a + b Z_t + c RV_past + error
```

Horizons:

```text
1h  = 12 bars
4h  = 48 bars
12h = 144 bars
24h = 288 bars
```

HAC standard errors are reported for the coefficient on `Z_t`.

The main empirical claim should be phrased as incremental predictive content for future realized volatility, not high explanatory power.

### 5. Batch-Means Moment Inference

For serially dependent return moments, the pipeline reports batch-means standard errors for:

```text
mean
standard deviation
skewness
kurtosis
```

This avoids iid standard errors for high-frequency crypto time series.

---

## Interpreting the Results

Recommended interpretation:

```text
OU calibration:
The empirical signal exhibits stable mean-reverting transition behavior, supporting the OU specification as a structural ansatz.

Leaky-extrema calibration:
The leaky support/resistance variables closely approximate empirical rolling extrema out of sample.

Volatility prediction:
The microstructure signal has positive incremental predictive content for future realized variance, especially for BTC and ETH.

Return moments:
Batch-means inference should be used for serially dependent moments. Unconditional skewness should not be overstated unless robust uncertainty supports it.
```

Do not claim:

```text
- exact convergence of the empirical signal to an OU process,
- exact Markovian lifting of rolling-window extrema,
- total-variation geometric ergodicity of the full lifted process,
- iid significance of simulation moments,
- universal negative skewness in raw high-frequency returns.
```

---

## Output Hygiene

Generated outputs should remain under `results/`.

Recommended output tree:

```text
results/
  synthetic/
    figures/
    tables/
  diagnostics/
    figures/
  empirical/
```

Avoid committing Python cache files:

```text
__pycache__/
*.pyc
*.pyo
*.nbc
*.nbi
```

A recommended `.gitignore` snippet is:

```gitignore
__pycache__/
*.py[cod]
*.nbc
*.nbi
```

If generated results should not be tracked, also add:

```gitignore
results/
```

If the repository is intended as a paper-replication archive, it is acceptable to include selected generated CSVs and figures under `results/`.

---

## Reproducibility Commands

Run the full synthetic workflow:

```bash
uv run synthetic_experiments
```

Run simulator diagnostics:

```bash
uv run simulation_diagnostics
```

Run empirical calibration:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical
```

Run all three:

```bash
uv run synthetic_experiments
uv run simulation_diagnostics
uv run empirical_calibration --data-dir data --out-dir results/empirical
```

---

## Suggested Citation of Outputs in the Paper

Synthetic outputs:

```text
results/synthetic/figures/fig1_trajectory.png
results/synthetic/figures/fig2_sensitivity.png
results/synthetic/figures/fig3_long_run_distribution.png
results/synthetic/tables/table_long_run_moments_batch_means.csv
results/synthetic/tables/table_threshold_volatility.csv
results/synthetic/tables/table_breakout_sensitivity.csv
```

Diagnostic outputs:

```text
results/diagnostics/figures/check3_stability_diagnostic.png
results/diagnostics/figures/check4_running_mean_stabilization.png
results/diagnostics/figures/check6_dt_sensitivity.png
```

Empirical outputs:

```text
results/empirical/ou_calibration.csv
results/empirical/ou_oos_diagnostics.csv
results/empirical/leaky_extrema_calibration.csv
results/empirical/volatility_prediction_hac.csv
results/empirical/return_moments_batch_means.csv
```

---

## Notes for Paper Claims

The repository follows the revised paper framing:

```text
- Rolling extrema motivate the empirical construction.
- The theoretical state variables are leaky-extrema regularizations.
- The OU signal is a structural ansatz supported by empirical transition diagnostics.
- Viscosity uniqueness is established for the smoothed recalibration problem.
- Stability results are Lyapunov-type and excursion-control results, not full geometric ergodicity.
- Empirical inference uses HAC or batch-means corrections to address serial dependence.
```
