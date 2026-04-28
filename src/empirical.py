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
