# MDRS-SDE Theory and Empirical Benchmarks

This repository contains simulation, diagnostics, empirical-calibration, and out-of-sample benchmark code for the paper:

> **A Markovian Leaky-Extrema Approach to Path-Dependent Breakouts in Microstructure-Driven SDEs**

The project studies a Microstructure-Driven Regime-Switching Stochastic Differential Equation (MDRS-SDE) for high-frequency cryptocurrency perpetual futures. The codebase supports four roles:

1. synthetic MDRS-SDE experiments;
2. numerical simulation diagnostics;
3. real-data empirical calibration using 5-minute perpetual futures data; and
4. QF-style realized-volatility forecasting benchmarks based on positive log-HAR OOS forecasts.

The implementation follows the revised paper framing. Synthetic long-run simulations are used as numerical evidence of stable behavior, not as a proof of total-variation geometric ergodicity. Empirical regressions are interpreted as evidence of incremental volatility-prediction content, not as universal trading profitability.

Raw exchange data are **not redistributed**. The scripts expect user-provided OHLCV files with the schema described below.

---

## Repository layout

```text
.
├── README.md
├── RELEASE_NOTES_v0.5.0.md
├── pyproject.toml
├── uv.lock
├── data/                         # user-provided; not committed
│   ├── btcusdt_5m.csv
│   ├── ethusdt_5m.csv
│   ├── xrpusdt_5m.csv
│   └── solusdt_5m.csv
├── results/                      # generated; usually not committed
│   ├── synthetic/
│   │   ├── figures/
│   │   └── tables/
│   ├── diagnostics/
│   │   └── figures/
│   ├── empirical/
│   └── oos/
├── scripts/
│   ├── __init__.py
│   ├── run_synthetic_experiments.py
│   ├── run_simulation_diagnostics.py
│   ├── run_empirical_calibration.py
│   └── run_oos_benchmarks.py
└── src/
    ├── __init__.py
    ├── simulator.py
    ├── empirical.py
    └── oos.py
```

The console scripts are exposed through `uv` entry points:

```text
synthetic_experiments
simulation_diagnostics
empirical_calibration
oos_benchmarks
```

---

## Environment

This project uses `uv` and Python `>=3.11,<3.14`.

```bash
uv sync
```

The main dependencies are:

```text
numpy
pandas
scipy
statsmodels
matplotlib
numba
```

If the environment is not yet configured through `pyproject.toml`, install dependencies manually:

```bash
uv add numpy pandas scipy statsmodels matplotlib numba
```

---

## Data requirements

The empirical and QF benchmark pipelines expect 5-minute OHLCV CSV files under the root-level `data/` directory.

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

The `Datetime` column should be parseable as a timestamp. The pipeline sorts by timestamp, removes duplicate timestamps, validates numeric OHLCV fields, and constructs log returns and rolling features internally.

If file names differ from the defaults, pass explicit asset mappings:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical \
  --assets BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv XRP:xrpusdt_5m.csv SOL:solusdt_5m.csv
```

If a filename contains parentheses, escape them in bash:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical \
  --assets BTC:btcusdt_5m\(1\).csv ETH:ethusdt_5m.csv XRP:xrpusdt_5m.csv SOL:solusdt_5m.csv
```

---

## Asset roles

The empirical section uses BTC and ETH as the main assets, and XRP and SOL as robustness assets.

```text
Main empirical assets:
- BTCUSDT
- ETHUSDT

Robustness assets:
- XRPUSDT
- SOLUSDT
```

This split is used only for reporting. The pipeline processes all provided assets and writes results for every asset to CSV.

---

## Empirical sample split

The empirical pipeline uses a presample-aware split:

```text
Presample / burn-in: available 2020 history
Train:               2021-01-01 to 2023-12-31
Validation:          2024-01-01 to 2024-12-31
Test:                2025-01-01 to 2025-12-31
Recent robustness:   2026-01-01 onward, when available
```

The 2020 observations are used only for:

```text
- rolling-window initialization;
- empirical support/resistance construction;
- leaky-state burn-in; and
- data-quality checks.
```

They are not used for headline parameter fitting, model selection, validation, or test reporting. Although SOLUSDT starts later in 2020 than BTC, ETH, and XRP, the available SOL presample before 2021 is still much longer than the 24-hour rolling window used for signal construction.

---

## Synthetic experiments

Run:

```bash
uv run synthetic_experiments --out-dir results/synthetic
```

Generated outputs:

```text
results/synthetic/figures/
results/synthetic/tables/
```

The synthetic experiments include:

1. a representative trajectory illustration of the four-dimensional MDRS-SDE state;
2. finite-horizon breakout-time sensitivity with respect to the activation threshold;
3. a long-run distribution experiment; and
4. regime-conditional volatility diagnostics.

The synthetic long-run experiment reports batch-means uncertainty for serially dependent simulation output. It does not report iid t-statistics.

Important scope note:

```text
Synthetic long-run simulations illustrate stable numerical behavior.
They do not prove total-variation geometric ergodicity of the full lifted process.
```

---

## Simulation diagnostics

Run:

```bash
uv run simulation_diagnostics --out-dir results/diagnostics
```

Generated outputs:

```text
results/diagnostics/figures/
```

The diagnostics include numerical sanity checks such as:

```text
- invariant support/resistance ordering;
- drift-dominance parameter checks;
- equilibrium-region stability diagnostics;
- running-mean stabilization diagnostics;
- time-step sensitivity diagnostics; and
- finite-horizon exit-frequency diagnostics.
```

These diagnostics are not theorem verification. They are intended to detect implementation issues and provide numerical sanity checks for the simulator.

---

## Empirical calibration pipeline

Run:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical
```

Generated empirical outputs:

```text
results/empirical/data_coverage.csv
results/empirical/ou_calibration.csv
results/empirical/ou_oos_diagnostics.csv
results/empirical/ou_acf_diagnostics.csv
results/empirical/ou_acf_summary.csv
results/empirical/leaky_extrema_calibration.csv
results/empirical/leaky_grid_search.csv
results/empirical/volatility_prediction_hac.csv
results/empirical/return_moments_robust_se.csv
```

### 1. Microstructure signal construction

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

### 2. OU calibration

The pipeline fits a one-step AR(1) transition on the training period:

```text
Z_{t+1} = a + b Z_t + eps_t
```

The AR(1) fit is converted to OU-style parameters:

```text
kappa_Z
long-run mean zbar
sigma_Z
half-life
```

Out-of-sample diagnostics are reported on validation, test, and recent windows. The OU specification is treated as a structural ansatz supported by empirical transition and ACF diagnostics, not as a proven weak limit of the discrete signal.

### 3. Leaky-extrema calibration

The pipeline constructs empirical rolling support and resistance levels over quiet regimes and compares them against simulated leaky support/resistance variables.

The leaky-extrema parameters are selected using validation mean absolute error. Final approximation quality is reported separately on train, validation, test, and recent splits.

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

The 2025 test split should be used for headline out-of-sample leaky-extrema results.

### 4. Volatility prediction with HAC inference

The baseline volatility regression is:

```text
RV_{t:t+h} = a_h + b_h Z_t + c_h RV_{t-h:t} + epsilon_{t,h}
```

Horizons:

```text
1h  = 12 bars
4h  = 48 bars
12h = 144 bars
24h = 288 bars
```

HAC standard errors are reported for the coefficient on `Z_t`. The main empirical claim should be phrased as incremental predictive content for future realized volatility, not high explanatory power.

### 5. Robust moment inference

For serially dependent return moments, the pipeline reports robust or batch-means standard errors for:

```text
mean
standard deviation
skewness
kurtosis
```

This avoids iid standard errors for high-frequency crypto time series.

---

## Positive log-HAR OOS benchmarks

Positive log-HAR out-of-sample benchmarks for journal-review robustness.

Run:

```bash
uv run oos_benchmarks \
  --data-dir data \
  --out-dir results/oos
```

Optional asset mapping:

```bash
uv run oos_benchmarks \
  --data-dir data \
  --out-dir results/oos \
  --assets BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv XRP:xrpusdt_5m.csv SOL:solusdt_5m.csv
```

### Benchmark design

The main comparison is:

```text
HAR-logRV baseline:
log RV_{t:t+h} ~ log RV_h + log RV_d + log RV_w + log RV_m

HAR-logRV + Z:
log RV_{t:t+h} ~ Z_t + log RV_h + log RV_d + log RV_w + log RV_m
```

For the 24-hour horizon, duplicate daily regressors are automatically removed. Forecasts are transformed back to strictly positive realized-variance forecasts using a smearing correction, making MSE and QLIKE losses well defined.

Generated files:

```text
results/oos/static_oos_forecast_tests_main_with_bootstrap.csv
results/oos/static_oos_winsorization_robustness.csv
results/oos/static_oos_logrv_spec_robustness.csv
```

These tables report:

```text
- HAR vs HAR+Z MSE and QLIKE losses;
- MSE and QLIKE improvements;
- train-period incremental R²;
- Diebold--Mariano-style HAC tests for loss differences;
- Clark--West nested forecast tests for MSE loss;
- moving-block bootstrap confidence intervals and one-sided p-values;
- robustness to Z_t winsorization; and
- robustness to log-RV/smearing and level-RV positivity specifications.
```

### Recommended interpretation of benchmarks

The log-HAR benchmark is intended as a robustness check for incremental forecasting content. It should be interpreted conservatively:

```text
- BTC and ETH are the primary liquid assets.
- SOL and XRP are robustness assets.
- Evidence should be described as incremental OOS volatility-forecasting content.
- Results should not be described as universal trading profitability.
- QLIKE results should be based on positive forecasts.
- Floor-constrained level-RV forecasts are included only as a robustness check.
```

---

## Interpreting the results

Recommended interpretation:

```text
OU calibration:
The empirical signal exhibits stable mean-reverting transition behavior, supporting the OU specification as a structural ansatz.

Leaky-extrema calibration:
The leaky support/resistance variables closely approximate empirical rolling extrema out of sample.

Volatility prediction:
The microstructure signal has positive incremental predictive content for future realized variance, especially for BTC and ETH.

Positive log-HAR OOS benchmark:
The signal can be evaluated against HAR-style realized-volatility baselines using positive log-RV forecasts and MSE/QLIKE losses.

Return moments:
Robust or batch-means inference should be used for serially dependent moments. Unconditional skewness should not be overstated unless robust uncertainty supports it.
```

Do not claim:

```text
- exact convergence of the empirical signal to an OU process;
- exact Markovian lifting of rolling-window extrema;
- total-variation geometric ergodicity of the full lifted process;
- iid significance of simulation moments;
- universal negative skewness in raw high-frequency returns;
- universal volatility-forecasting improvement across all assets and horizons; or
- direct trading profitability from the forecasting regressions alone.
```

---

## Output hygiene

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
  oos/
```

If the repository is intended as a paper-replication archive, it is acceptable to include selected generated CSVs and figures under `results/`, but raw exchange data should not be redistributed.

---

## Reproducibility commands

Run the full synthetic workflow:

```bash
uv run synthetic_experiments --out-dir results/synthetic
```

Run simulator diagnostics:

```bash
uv run simulation_diagnostics --out-dir results/diagnostics
```

Run empirical calibration:

```bash
uv run empirical_calibration \
  --data-dir data \
  --out-dir results/empirical
```

Run QF-style OOS benchmarks:

```bash
uv run oos_benchmarks \
  --data-dir data \
  --out-dir results/oos
```

Run the main workflows sequentially:

```bash
uv run synthetic_experiments --out-dir results/synthetic
uv run simulation_diagnostics --out-dir results/diagnostics
uv run empirical_calibration --data-dir data --out-dir results/empirical
uv run oos_benchmarks --data-dir data --out-dir results/oos
```

---

## Suggested citation of outputs in the paper

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

Empirical calibration outputs:

```text
results/empirical/ou_calibration.csv
results/empirical/ou_oos_diagnostics.csv
results/empirical/leaky_extrema_calibration.csv
results/empirical/volatility_prediction_hac.csv
results/empirical/return_moments_robust_se.csv
```

QF benchmark outputs:

```text
results/oos/static_oos_forecast_tests_main_with_bootstrap.csv
results/oos/static_oos_winsorization_robustness.csv
results/oos/static_oos_logrv_spec_robustness.csv
```

---

## Notes for paper claims

The repository follows the revised paper framing:

```text
- Rolling extrema motivate the empirical construction.
- The theoretical state variables are leaky-extrema regularizations.
- The OU signal is a structural ansatz supported by empirical transition diagnostics.
- Viscosity uniqueness is established for the smoothed recalibration problem under a standard comparison principle.
- Stability results are Lyapunov-type and excursion-control results, not full geometric ergodicity.
- Empirical inference uses HAC, robust, batch-means, or OOS loss-comparison methods to address serial dependence.
- QF-style OOS benchmarks should be presented as incremental volatility-forecasting evidence, not as a standalone trading strategy.
```

---


## License and data note

This code is provided for research reproducibility. Raw exchange data are not redistributed. Users should obtain OHLCV data from sources they are authorized to use and place the files under `data/` using the expected schema.
