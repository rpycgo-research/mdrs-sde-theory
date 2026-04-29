"""
Empirical calibration utilities for MDRS-SDE real-data validation.

This module implements the empirical counterpart of the paper's microstructure
signal, OU ansatz diagnostics, leaky-extrema approximation to rolling
support/resistance, and robust empirical inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BAR_MINUTES = 5
BARS_PER_DAY = 24 * 60 // BAR_MINUTES
ROLLING_WINDOW = BARS_PER_DAY
DT_DAYS = BAR_MINUTES / (24 * 60)
DEFAULT_ZETA = 1.0

SPLITS = {
    # Presample observations are used only for rolling-window initialization,
    # empirical extrema construction, and leaky-state burn-in. They are never
    # used for parameter fitting, validation, or headline test reporting.
    "presample": ("2020-01-01", "2020-12-31 23:59:59"),
    "train": ("2021-01-01", "2023-12-31 23:59:59"),
    "validation": ("2024-01-01", "2024-12-31 23:59:59"),
    "test": ("2025-01-01", "2025-12-31 23:59:59"),
    "recent": ("2026-01-01", "2026-12-31 23:59:59"),
}

HORIZON_LABELS = {
    12: "1h",
    48: "4h",
    144: "12h",
    288: "24h",
}


@dataclass(frozen=True)
class AssetFile:
    """Mapping between an asset label and a CSV filename."""

    asset: str
    filename: str


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load a 5-minute OHLCV CSV and create basic return features."""
    path = Path(path)
    df = pd.read_csv(path)

    if "Datetime" not in df.columns:
        raise ValueError(f"{path} must contain a Datetime column.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.sort_values("Datetime").drop_duplicates("Datetime")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"{path} must contain column {column}.")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Datetime", "Close", "Volume"]).reset_index(drop=True)
    df["log_close"] = np.log(df["Close"])
    df["ret"] = df["log_close"].diff()
    df["abs_ret"] = df["ret"].abs()
    df["log_volume"] = np.log1p(df["Volume"].clip(lower=0))

    return df


def split_sample(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return a named sample split."""
    if split not in SPLITS:
        raise KeyError(f"Unknown split: {split}.")

    start, end = SPLITS[split]
    mask = (df["Datetime"] >= pd.Timestamp(start)) & (
        df["Datetime"] <= pd.Timestamp(end)
    )

    return df.loc[mask].copy()


def data_coverage(df: pd.DataFrame, asset: str) -> dict:
    """Summarize data coverage and split sizes for an asset."""
    gaps = df["Datetime"].diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta(minutes=BAR_MINUTES)]
    missing = int(
        ((large_gaps / pd.Timedelta(minutes=BAR_MINUTES)) - 1).sum()
    )

    row = {
        "Asset": asset,
        "Start": df["Datetime"].min(),
        "End": df["Datetime"].max(),
        "Rows": len(df),
        "Missing_5m_intervals": missing,
        "Zero_volume_bars": int((df["Volume"] == 0).sum()),
    }

    for split in SPLITS:
        row[f"{split}_rows"] = len(split_sample(df, split))

    return row


def add_microstructure_signal(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    ) -> pd.DataFrame:
    """Construct the empirical max-pooled microstructure signal.

    Rolling moments are shifted by one bar, so the signal is past-adapted.
    The return component uses absolute log returns because the signal is meant
    to capture volatility/activation magnitude rather than signed direction.
    """
    df = df.copy()

    for feature, output_column in [("log_volume", "zV"), ("abs_ret", "zR")]:
        rolling = df[feature].rolling(window, min_periods=window)
        mean = rolling.mean().shift(1)
        std = rolling.std(ddof=0).shift(1)
        df[output_column] = (df[feature] - mean) / std.replace(0, np.nan)

    pooled = np.maximum(df["zV"], df["zR"])
    df["Z"] = (pooled + pooled.shift(1) + pooled.shift(2)) / 3.0

    return df


def fit_ou_ar1(z: pd.Series, dt_days: float = DT_DAYS) -> dict:
    """Fit an AR(1) transition and convert it to OU parameters."""
    z = pd.Series(z).dropna().astype(float)
    x = z.iloc[:-1].to_numpy()
    y = z.iloc[1:].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) < 3:
        raise ValueError("Not enough observations to fit AR(1) transition.")

    model = sm.OLS(y, sm.add_constant(x)).fit()
    intercept, slope = model.params
    residual_variance = float(np.mean(model.resid**2))

    if 0 < slope < 1:
        kappa = -np.log(slope) / dt_days
        zbar = intercept / (1 - slope)
        sigma_z = np.sqrt(residual_variance * 2 * kappa / (1 - slope**2))
        half_life_hours = 24 * np.log(2) / kappa
    else:
        kappa = np.nan
        zbar = np.nan
        sigma_z = np.nan
        half_life_hours = np.nan

    return {
        "a": float(intercept),
        "b": float(slope),
        "kappa_per_day": float(kappa),
        "half_life_hours": float(half_life_hours),
        "zbar": float(zbar),
        "sigmaZ_per_sqrt_day": float(sigma_z),
        "r2": float(model.rsquared),
        "n": int(len(y)),
    }


def ou_oos_diagnostics(z: pd.Series, fit: dict) -> dict:
    """Evaluate one-step AR(1)-to-OU transition diagnostics out of sample."""
    z = pd.Series(z).dropna().astype(float)
    x = z.iloc[:-1].to_numpy()
    y = z.iloc[1:].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return {"n": 0, "r2_oos": np.nan, "rmse": np.nan, "corr": np.nan}

    pred = fit["a"] + fit["b"] * x
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))

    return {
        "n": int(len(y)),
        "r2_oos": float(1 - sse / sst) if sst > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "corr": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else np.nan,
    }


def empirical_latched_extrema(
    df: pd.DataFrame,
    zeta: float = DEFAULT_ZETA,
    window: int = ROLLING_WINDOW,
    ) -> tuple[pd.Series, pd.Series]:
    """Construct empirical rolling support/resistance over quiet regimes."""
    price = df["log_close"]
    quiet = df["Z"] < zeta
    price_quiet = price.where(quiet)

    resistance = price_quiet.rolling(window, min_periods=1).max().ffill()
    support = price_quiet.rolling(window, min_periods=1).min().ffill()

    return resistance, support


def simulate_leaky_extrema(
    df: pd.DataFrame,
    fast_hl_bars: int,
    slow_hl_bars: int,
    zeta: float = DEFAULT_ZETA,
    ) -> tuple[pd.Series, pd.Series]:
    """Simulate leaky support/resistance on the observed price path."""
    price = df["log_close"].to_numpy()
    signal = df["Z"].to_numpy()

    beta = 1 - np.exp(-np.log(2) / fast_hl_bars)
    gamma = 1 - np.exp(-np.log(2) / slow_hl_bars)

    resistance = np.full(len(df), np.nan)
    support = np.full(len(df), np.nan)

    valid = np.where(np.isfinite(price) & np.isfinite(signal))[0]
    if len(valid) == 0:
        return pd.Series(resistance, index=df.index), pd.Series(support, index=df.index)

    start_idx = int(valid[0])
    resistance[start_idx] = price[start_idx]
    support[start_idx] = price[start_idx]

    for idx in range(start_idx + 1, len(df)):
        resistance[idx] = resistance[idx - 1]
        support[idx] = support[idx - 1]

        if not np.isfinite(price[idx]) or not np.isfinite(signal[idx]):
            continue

        if signal[idx] < zeta:
            resistance[idx] += beta * max(price[idx] - resistance[idx], 0.0)
            resistance[idx] -= gamma * max(resistance[idx] - price[idx], 0.0)

            support[idx] -= beta * max(support[idx] - price[idx], 0.0)
            support[idx] += gamma * max(price[idx] - support[idx], 0.0)

    return pd.Series(resistance, index=df.index), pd.Series(support, index=df.index)


def evaluate_leaky_extrema(
    df: pd.DataFrame,
    resistance: pd.Series,
    support: pd.Series,
    empirical_resistance: pd.Series,
    empirical_support: pd.Series,
    split: str,
    ) -> dict | None:
    """Evaluate leaky-extrema approximation error on a named split."""
    tmp_df = df.assign(
        R=resistance,
        S=support,
        Rhat=empirical_resistance,
        Shat=empirical_support,
    )
    tmp = split_sample(df=tmp_df, split=split)
    tmp = tmp[["R", "S", "Rhat", "Shat"]]
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) == 0:
        return None

    err_r = tmp["R"] - tmp["Rhat"]
    err_s = tmp["S"] - tmp["Shat"]

    return {
        "N": int(len(tmp)),
        "MAE_R": float(err_r.abs().mean()),
        "MAE_S": float(err_s.abs().mean()),
        "RMSE_R": float(np.sqrt((err_r**2).mean())),
        "RMSE_S": float(np.sqrt((err_s**2).mean())),
        "Corr_R": float(tmp["R"].corr(tmp["Rhat"])),
        "Corr_S": float(tmp["S"].corr(tmp["Shat"])),
        "Mean_MAE": float((err_r.abs().mean() + err_s.abs().mean()) / 2),
    }


def calibrate_leaky_extrema(
    df: pd.DataFrame,
    asset: str,
    zeta: float = DEFAULT_ZETA,
    ) -> tuple[list[dict], list[dict]]:
    """Calibrate leaky-extrema half-lives using validation MAE."""
    empirical_resistance, empirical_support = empirical_latched_extrema(
        df=df,
        zeta=zeta,
    )

    fast_grid = [1, 3, 6, 12, 24]
    slow_grid = [72, 144, 288, 576, 1008, 2016]

    best_params: dict | None = None
    best_score = np.inf
    grid_rows: list[dict] = []

    for fast_hl in fast_grid:
        for slow_hl in slow_grid:
            if slow_hl <= fast_hl:
                continue

            resistance, support = simulate_leaky_extrema(
                df=df,
                fast_hl_bars=fast_hl,
                slow_hl_bars=slow_hl,
                zeta=zeta,
            )
            train_error = evaluate_leaky_extrema(
                df=df,
                resistance=resistance,
                support=support,
                empirical_resistance=empirical_resistance,
                empirical_support=empirical_support,
                split="train",
            )
            validation_error = evaluate_leaky_extrema(
                df=df,
                resistance=resistance,
                support=support,
                empirical_resistance=empirical_resistance,
                empirical_support=empirical_support,
                split="validation",
            )

            if train_error is None or validation_error is None:
                continue

            row = {
                "Asset": asset,
                "fast_hl_bars": fast_hl,
                "slow_hl_bars": slow_hl,
                "fast_hl_hours": fast_hl * BAR_MINUTES / 60,
                "slow_hl_hours": slow_hl * BAR_MINUTES / 60,
                "train_Mean_MAE": train_error["Mean_MAE"],
                "validation_Mean_MAE": validation_error["Mean_MAE"],
            }
            grid_rows.append(row)

            score = row["validation_Mean_MAE"]
            if score < best_score:
                best_score = score
                best_params = row

    if best_params is None:
        raise ValueError(
            f"No valid leaky-extrema calibration candidate for {asset}. "
            "Check split dates, Z construction, and rolling-extrema availability."
        )

    resistance, support = simulate_leaky_extrema(
        df=df,
        fast_hl_bars=int(best_params["fast_hl_bars"]),
        slow_hl_bars=int(best_params["slow_hl_bars"]),
        zeta=zeta,
    )

    rows: list[dict] = []
    for split in ["train", "validation", "test", "recent"]:
        error = evaluate_leaky_extrema(
            df=df,
            resistance=resistance,
            support=support,
            empirical_resistance=empirical_resistance,
            empirical_support=empirical_support,
            split=split,
        )
        if error is not None:
            rows.append({"Asset": asset, "Split": split, **best_params, **error})

    return rows, grid_rows


def add_realized_variance(
    df: pd.DataFrame,
    horizons: Iterable[int] = (12, 48, 144, 288),
    ) -> pd.DataFrame:
    """Add past and future realized-variance columns."""
    df = df.copy()
    r2 = df["ret"] ** 2

    for horizon in horizons:
        df[f"rv_past_{horizon}"] = r2.rolling(
            horizon,
            min_periods=horizon,
        ).sum()

        # Sum of squared returns from t+1 through t+horizon.
        df[f"rv_future_{horizon}"] = (
            r2.shift(-1)
            .rolling(horizon, min_periods=horizon)
            .sum()
            .shift(-(horizon - 1))
        )

    return df


def fit_volatility_prediction(
    df: pd.DataFrame,
    asset: str,
    split: str,
    horizons: Iterable[int] = (12, 48, 144, 288),
    ) -> list[dict]:
    """Fit future realized-variance regressions with HAC inference."""
    sub = split_sample(df=df, split=split)
    rows: list[dict] = []

    for horizon in horizons:
        columns = ["Z", f"rv_past_{horizon}", f"rv_future_{horizon}"]
        tmp = sub[columns].replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) == 0:
            continue

        y = tmp[f"rv_future_{horizon}"]
        X = sm.add_constant(tmp[["Z", f"rv_past_{horizon}"]])
        model = sm.OLS(y, X, missing="drop")
        result = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": min(288, max(1, horizon * 2))},
        )

        rows.append(
            {
                "Asset": asset,
                "Split": split,
                "Horizon_bars": horizon,
                "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                "N": int(result.nobs),
                "coef_Z": float(result.params["Z"]),
                "t_Z_HAC": float(result.tvalues["Z"]),
                "p_Z_HAC": float(result.pvalues["Z"]),
                "coef_past_RV": float(result.params[f"rv_past_{horizon}"]),
                "t_past_RV_HAC": float(result.tvalues[f"rv_past_{horizon}"]),
                "R2": float(result.rsquared),
            }
        )

    return rows


def batch_stat_se(
    x: pd.Series | np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_batches: int = 100,
    ) -> tuple[float, float, int]:
    """Estimate a statistic and batch-means standard error."""
    values = np.asarray(pd.Series(x).dropna(), dtype=float)

    if len(values) == 0:
        return np.nan, np.nan, 0

    n_batches = max(2, min(n_batches, len(values)))
    batch_size = len(values) // n_batches
    values = values[: batch_size * n_batches]

    batch_values = []
    for idx in range(n_batches):
        chunk = values[idx * batch_size : (idx + 1) * batch_size]
        batch_values.append(float(stat_fn(chunk)))

    batch_values = np.asarray(batch_values)
    se = float(batch_values.std(ddof=1) / np.sqrt(n_batches))

    return float(batch_values.mean()), se, int(n_batches)


def calendar_block_bootstrap_se(
    df: pd.DataFrame,
    value_col: str,
    stat_fn: Callable[[np.ndarray], float],
    *,
    block_freq: str = "1D",
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """Estimate a statistic SE using calendar-block bootstrap.

    The bootstrap resamples whole calendar blocks, such as days or weeks,
    preserving within-block intraday dependence and seasonality.
    """
    if "Datetime" not in df.columns:
        raise ValueError("calendar_block_bootstrap_se requires a Datetime column.")

    if value_col not in df.columns:
        raise ValueError(f"calendar_block_bootstrap_se requires {value_col}.")

    tmp = df[["Datetime", value_col]].copy()
    tmp["Datetime"] = pd.to_datetime(tmp["Datetime"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.empty:
        return {
            "estimate": np.nan,
            "bootstrap_mean": np.nan,
            "bootstrap_se": np.nan,
            "n_boot": int(n_boot),
            "n_blocks": 0,
        }

    tmp = tmp.sort_values("Datetime")
    blocks = [
        block[value_col].to_numpy(dtype=float)
        for _, block in tmp.groupby(pd.Grouper(key="Datetime", freq=block_freq))
        if len(block) > 0
    ]

    if not blocks:
        return {
            "estimate": np.nan,
            "bootstrap_mean": np.nan,
            "bootstrap_se": np.nan,
            "n_boot": int(n_boot),
            "n_blocks": 0,
        }

    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    values = np.concatenate(blocks)
    estimate = float(stat_fn(values))

    boot_stats = np.empty(n_boot, dtype=float)
    for boot_idx in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([blocks[idx] for idx in chosen])
        boot_stats[boot_idx] = float(stat_fn(sample))

    return {
        "estimate": estimate,
        "bootstrap_mean": float(np.mean(boot_stats)),
        "bootstrap_se": float(np.std(boot_stats, ddof=1)),
        "n_boot": int(n_boot),
        "n_blocks": int(n_blocks),
    }

def real_data_moments(
    df: pd.DataFrame,
    asset: str,
    split: str,
    n_batches: int = 100,
    n_boot: int = 500,
) -> list[dict]:
    """Compute robust moment estimates for real-data returns.

    Daily calendar-block bootstrap SE is the main real-data uncertainty
    estimate. Weekly calendar-block bootstrap and batch-means SE are reported
    as robustness checks.
    """
    sub = split_sample(df=df, split=split)
    returns = sub["ret"].replace([np.inf, -np.inf], np.nan).dropna()

    stat_functions: dict[str, Callable[[np.ndarray], float]] = {
        "mean": np.mean,
        "std": np.std,
        "skewness": stats.skew,
        "kurtosis": lambda x: stats.kurtosis(x, fisher=False),
    }

    rows: list[dict] = []
    for name, stat_fn in stat_functions.items():
        values = returns.to_numpy(dtype=float)
        value = float(stat_fn(values)) if len(values) else np.nan
        _, batch_se, n_batches_used = batch_stat_se(
            returns,
            stat_fn,
            n_batches=n_batches,
        )
        daily = calendar_block_bootstrap_se(
            sub,
            "ret",
            stat_fn,
            block_freq="1D",
            n_boot=n_boot,
            seed=42,
        )
        weekly = calendar_block_bootstrap_se(
            sub,
            "ret",
            stat_fn,
            block_freq="7D",
            n_boot=n_boot,
            seed=43,
        )
        rows.append(
            {
                "Asset": asset,
                "Split": split,
                "Statistic": name,
                "Value": value,
                "DailyBlockBootstrap_SE": daily["bootstrap_se"],
                "DailyBlockBootstrap_Blocks": daily["n_blocks"],
                "WeeklyBlockBootstrap_SE": weekly["bootstrap_se"],
                "WeeklyBlockBootstrap_Blocks": weekly["n_blocks"],
                "Batch_SE": batch_se,
                "Batches": n_batches_used,
                "Bootstrap_Replications": n_boot,
            }
        )

    return rows
