# MDRS-SDE Theory and Empirical Benchmarks

This repository contains simulation, diagnostics, empirical-calibration, and out-of-sample benchmark code for the paper:

> **A Markovian Leaky-Extrema Approach to Path-Dependent Breakouts in Microstructure-Driven SDEs**

The project studies a Microstructure-Driven Regime-Switching Stochastic Differential Equation (MDRS-SDE) for high-frequency cryptocurrency perpetual futures. The codebase supports seven roles:

1. synthetic MDRS-SDE experiments;
2. numerical simulation diagnostics;
3. real-data empirical calibration using 5-minute perpetual futures data;
4. positive log-HAR out-of-sample benchmark tests;
5. paper-ready OOS figure generation for benchmark and activation diagnostics;
6. timeout-safe realized-volatility forecasting diagnostics using stronger HAR-family baselines; and
7. actual funding-rate control diagnostics for perpetual-futures mechanism checks.

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
│   ├── solusdt_5m.csv
│   └── funding_rate/
│       ├── btcusdt.csv
│       ├── ethusdt.csv
│       ├── xrpusdt.csv
│       └── solusdt.csv
├── results/                      # generated; usually not committed
│   ├── synthetic/
│   │   ├── figures/
│   │   └── tables/
│   ├── diagnostics/
│   │   └── figures/
│   ├── empirical/
│   ├── oos/
│   ├── figures/
│   ├── volatility_forecasting/
│   └── funding_controls/
├── scripts/
│   ├── __init__.py
│   ├── run_synthetic_experiments.py
│   ├── run_simulation_diagnostics.py
│   ├── run_empirical_calibration.py
│   ├── run_oos_benchmarks.py
│   ├── plot_oos_figures.py
│   ├── run_volatility_forecasting_experiments.py
│   ├── combine_forecasting_outputs.py
│   └── run_funding_control_experiments.py
└── src/
    ├── __init__.py
    ├── simulator.py
    ├── empirical.py
    ├── oos.py
    ├── volatility_forecasting_fast.py
    └── funding_rate_controls.py
```

The console scripts are exposed through `uv` entry points:

```text
synthetic_experiments
simulation_diagnostics
empirical_calibration
oos_benchmarks
```

The timeout-safe forecasting, funding-control diagnostics, and OOS plotting utilities are provided as standalone Python scripts under `scripts/` and can be run with `uv run python scripts/<script_name>.py`.

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

### Funding-rate data

Funding-control diagnostics additionally expect actual funding-rate CSV files under:

```text
data/funding_rate/btcusdt.csv
data/funding_rate/ethusdt.csv
data/funding_rate/xrpusdt.csv
data/funding_rate/solusdt.csv
```

Expected funding-rate columns:

```text
datetime, calc_time, funding_interval_hours, last_funding_rate
```

The funding-control pipeline uses `last_funding_rate` as the actual funding-rate level and aligns funding observations to 5-minute forecast origins using backward causal as-of merging. Each forecast origin therefore receives only the most recently available funding-rate observation. The funding files are user-provided and are not redistributed with the repository.

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

## OOS figure generation

The repository includes a plotting utility for paper-ready figures based on the positive log-HAR OOS benchmark outputs. It reads the main OOS benchmark CSV and optionally uses raw OHLCV files to generate activation-episode figures.

Run the default BTC/ETH OOS figures and a BTC activation episode figure:

```bash
uv run python scripts/plot_oos_figures.py \
  --results-dir results/oos \
  --data-dir data \
  --out-dir results/figures \
  --assets BTC ETH \
  --activation-assets BTC
```

Optional multi-asset activation examples:

```bash
uv run python scripts/plot_oos_figures.py \
  --results-dir results/oos \
  --data-dir data \
  --out-dir results/figures \
  --assets BTC ETH \
  --activation-assets BTC ETH
```

Expected input:

```text
results/oos/static_oos_forecast_tests_main_with_bootstrap.csv
```

Generated figure files:

```text
results/figures/fig_oos_mse_improvement_btc_eth.pdf
results/figures/fig_oos_qlike_improvement_btc_eth.pdf
results/figures/fig_clark_west_pvalues_btc_eth.pdf
results/figures/fig_activation_episode_BTC.pdf
```

The plotting utility normalizes several possible benchmark-output column names, including `MSE_improvement`, `MSE_improvement_pct`, `QLIKE_improvement`, and Clark--West p-value aliases. The activation-episode figure selects the highest activation point in 2025 when available; otherwise it falls back to the highest activation point in the full available sample.

---

## Timeout-safe volatility forecasting diagnostics

For faster review-oriented diagnostics, the repository includes a lightweight forecasting runner that evaluates stronger HAR-family baselines without running the heavier bootstrap workflow.

Run BTC and ETH:

```bash
uv run python scripts/run_volatility_forecasting_experiments.py \
  --data-dir data \
  --assets BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv \
  --out-dir results/volatility_forecasting \
  --run-validation-selection
```

Run all assets:

```bash
uv run python scripts/run_volatility_forecasting_experiments.py \
  --data-dir data \
  --all-assets \
  --out-dir results/volatility_forecasting \
  --run-validation-selection
```

If asset-level runs are needed to avoid timeouts:

```bash
uv run python scripts/run_volatility_forecasting_experiments.py \
  --data-dir data \
  --assets BTC:btcusdt_5m.csv \
  --out-dir results/volatility_forecasting_btc \
  --run-validation-selection

uv run python scripts/run_volatility_forecasting_experiments.py \
  --data-dir data \
  --assets ETH:ethusdt_5m.csv \
  --out-dir results/volatility_forecasting_eth \
  --run-validation-selection

uv run python scripts/combine_forecasting_outputs.py \
  --input-dirs results/volatility_forecasting_btc results/volatility_forecasting_eth \
  --out-dir results/volatility_forecasting
```

Generated files:

```text
results/volatility_forecasting/stronger_benchmark_tests.csv
results/volatility_forecasting/validation_selected_family_tests.csv
results/volatility_forecasting/scheduled_funding_window_diagnostics.csv
results/volatility_forecasting/high_volatility_ranking_diagnostics.csv
```

These diagnostics compare HAR, HARQ-style, and realized-semivariance HAR families. They are intended to show whether the activation signal remains informative relative to stronger realized-volatility benchmark families.

---

## Actual funding-rate control diagnostics

Actual funding-rate control diagnostics test whether the activation signal is subsumed by observed funding-rate pressure.

Run all assets:

```bash
uv run python scripts/run_funding_control_experiments.py \
  --data-dir data \
  --funding-dir data/funding_rate \
  --all-assets \
  --out-dir results/funding_controls \
  --funding-window-minutes 60
```

Run BTC and ETH only:

```bash
uv run python scripts/run_funding_control_experiments.py \
  --data-dir data \
  --funding-dir data/funding_rate \
  --assets BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv \
  --funding-files BTC:btcusdt.csv ETH:ethusdt.csv \
  --out-dir results/funding_controls \
  --funding-window-minutes 60
```

Generated files:

```text
results/funding_controls/actual_funding_control_tests.csv
results/funding_controls/actual_funding_activation_summary.csv
results/funding_controls/actual_funding_window_loss_diagnostics.csv
```

The main model comparison is:

```text
HAR
HAR+F
HAR+Z
HAR+Z+F
HAR+Z+F+window
```

where `F` denotes actual funding-rate controls and `window` denotes an indicator for observations near actual funding-settlement timestamps. These diagnostics should be interpreted as mechanism checks, not as causal identification of funding-rate effects.

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
  volatility_forecasting/
  funding_controls/
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

Run OOS figure generation:

```bash
uv run python scripts/plot_oos_figures.py \
  --results-dir results/oos \
  --data-dir data \
  --out-dir results/figures \
  --assets BTC ETH \
  --activation-assets BTC
```

Run timeout-safe forecasting diagnostics:

```bash
uv run python scripts/run_volatility_forecasting_experiments.py \
  --data-dir data \
  --all-assets \
  --out-dir results/volatility_forecasting \
  --run-validation-selection
```

Run actual funding-rate control diagnostics:

```bash
uv run python scripts/run_funding_control_experiments.py \
  --data-dir data \
  --funding-dir data/funding_rate \
  --all-assets \
  --out-dir results/funding_controls \
  --funding-window-minutes 60
```

Run the main workflows sequentially:

```bash
uv run synthetic_experiments --out-dir results/synthetic
uv run simulation_diagnostics --out-dir results/diagnostics
uv run empirical_calibration --data-dir data --out-dir results/empirical
uv run oos_benchmarks --data-dir data --out-dir results/oos
uv run python scripts/plot_oos_figures.py --results-dir results/oos --data-dir data --out-dir results/figures --assets BTC ETH --activation-assets BTC
uv run python scripts/run_volatility_forecasting_experiments.py --data-dir data --all-assets --out-dir results/volatility_forecasting --run-validation-selection
uv run python scripts/run_funding_control_experiments.py --data-dir data --funding-dir data/funding_rate --all-assets --out-dir results/funding_controls --funding-window-minutes 60
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

OOS figure outputs:

```text
results/figures/fig_oos_mse_improvement_btc_eth.pdf
results/figures/fig_oos_qlike_improvement_btc_eth.pdf
results/figures/fig_clark_west_pvalues_btc_eth.pdf
results/figures/fig_activation_episode_BTC.pdf
```

Volatility forecasting diagnostic outputs:

```text
results/volatility_forecasting/stronger_benchmark_tests.csv
results/volatility_forecasting/validation_selected_family_tests.csv
results/volatility_forecasting/high_volatility_ranking_diagnostics.csv
results/volatility_forecasting/scheduled_funding_window_diagnostics.csv
```

Funding-control diagnostic outputs:

```text
results/funding_controls/actual_funding_control_tests.csv
results/funding_controls/actual_funding_activation_summary.csv
results/funding_controls/actual_funding_window_loss_diagnostics.csv
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
- OOS figures should be treated as visual summaries of benchmark diagnostics and representative activation episodes, not as separate statistical tests.
- Stronger HAR-family diagnostics should be presented as robustness evidence that the activation signal remains informative beyond the baseline positive log-HAR specification.
- Funding-rate control diagnostics should be interpreted as showing that actual funding-rate levels do not subsume the activation signal; they should not be described as causal identification of funding-rate-driven volatility.
- Funding-window diagnostics are mechanism checks around settlement timing and should not be interpreted as standalone trading signals.
```

---


## License and data note

This code is provided for research reproducibility. Raw exchange data are not redistributed. Users should obtain OHLCV data from sources they are authorized to use and place the files under `data/` using the expected schema.
