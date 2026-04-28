"""
Empirical calibration utilities for MDRS-SDE real-data validation.

This module implements the empirical counterpart of the paper's microstructure
signal, OU ansatz diagnostics, leaky-extrema approximation to rolling
support/resistance, and robust empirical inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
