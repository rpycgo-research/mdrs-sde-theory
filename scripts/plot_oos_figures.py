#!/usr/bin/env python
"""
Generate QF-style figures for the MDRS-SDE / log-HAR OOS paper.

Expected inputs
---------------
1. QF OOS result CSVs, e.g.
   results/oos/static_oos_forecast_tests_main_with_bootstrap.csv
   results/oos/static_oos_winsorization_robustness.csv
   results/oos/static_oos_logrv_spec_robustness.csv

2. Optional raw OHLCV files for activation-episode figures, e.g.
   data/btcusdt_5m.csv
   data/ethusdt_5m.csv

Example
-------
uv run python scripts/plot_oos_figures.py \
  --results-dir results/oos \
  --data-dir data \
  --out-dir results/figures \
  --assets BTC ETH
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import matplotlib.pyplot as plt


HORIZON_ORDER = ["1h", "4h", "12h", "24h"]


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"None of these columns found: {list(candidates)}")


def _asset_to_filename(asset: str) -> list[str]:
    a = asset.lower()
    return [
        f"{a}usdt_5m.csv",
        f"{a}_5m.csv",
        f"{a}usdt.csv",
        f"{a}.csv",
    ]


def load_oos_results(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "static_oos_forecast_tests_main_with_bootstrap.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the QF OOS benchmark pipeline first."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    required_raw = {"Asset", "Horizon"}
    missing_raw = required_raw.difference(df.columns)
    if missing_raw:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_raw)}")

    # Normalize current CSV column names into plotting-friendly aliases.
    if "MSE_improve_pct" not in df.columns:
        if "MSE_improvement" in df.columns:
            # Current benchmark CSV stores MSE improvement as a ratio, not percent.
            df["MSE_improve_pct"] = 100.0 * pd.to_numeric(
                df["MSE_improvement"], errors="coerce"
            )
        elif "MSE_improvement_pct" in df.columns:
            df["MSE_improve_pct"] = pd.to_numeric(
                df["MSE_improvement_pct"], errors="coerce"
            )
        else:
            raise ValueError(
                f"{path} is missing MSE improvement column. "
                "Expected one of: MSE_improve_pct, MSE_improvement, "
                "MSE_improvement_pct."
            )

    if "QLIKE_improve" not in df.columns:
        if "QLIKE_improvement" in df.columns:
            df["QLIKE_improve"] = pd.to_numeric(
                df["QLIKE_improvement"], errors="coerce"
            )
        else:
            raise ValueError(
                f"{path} is missing QLIKE improvement column. "
                "Expected one of: QLIKE_improve, QLIKE_improvement."
            )

    if "CW_MSE_p_one" not in df.columns:
        if "CW_MSE_p_one_sided" in df.columns:
            df["CW_MSE_p_one"] = pd.to_numeric(
                df["CW_MSE_p_one_sided"], errors="coerce"
            )
        elif "CW_MSE_p_one-sided" in df.columns:
            df["CW_MSE_p_one"] = pd.to_numeric(
                df["CW_MSE_p_one-sided"], errors="coerce"
            )
        else:
            raise ValueError(
                f"{path} is missing Clark-West one-sided p-value column. "
                "Expected one of: CW_MSE_p_one, CW_MSE_p_one_sided."
            )

    # Optional aliases for future tables/labels.
    if "Delta_R2" not in df.columns and "Delta_train_R2" in df.columns:
        df["Delta_R2"] = pd.to_numeric(df["Delta_train_R2"], errors="coerce")

    df["Asset"] = df["Asset"].astype(str).str.upper()
    df["Horizon"] = df["Horizon"].astype(str)

    df["Horizon"] = pd.Categorical(
        df["Horizon"],
        categories=HORIZON_ORDER,
        ordered=True,
    )

    required_final = {
        "Asset",
        "Horizon",
        "MSE_improve_pct",
        "QLIKE_improve",
        "CW_MSE_p_one",
    }
    missing_final = required_final.difference(df.columns)
    if missing_final:
        raise ValueError(
            f"{path} is missing normalized required columns: {sorted(missing_final)}"
        )

    return df.sort_values(["Asset", "Horizon"])


def plot_oos_metric(
    df: pd.DataFrame,
    assets: list[str],
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    sub = df[df["Asset"].isin(assets)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for assets={assets}")

    pivot = (
        sub.pivot_table(index="Horizon", columns="Asset", values=metric, observed=False)
        .reindex(HORIZON_ORDER)
    )

    ax = pivot.plot(kind="bar", figsize=(8, 4.8), rot=0)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    # ax.set_title(ylabel + " from adding microstructure activation")
    ax.legend(title="Asset")
    ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_cw_pvalues(df: pd.DataFrame, assets: list[str], out_path: Path) -> None:
    sub = df[df["Asset"].isin(assets)].copy()
    sub["minus_log10_p"] = -np.log10(sub["CW_MSE_p_one"].clip(lower=1e-12))
    pivot = (
        sub.pivot_table(
            index="Horizon", columns="Asset", values="minus_log10_p", observed=False
        )
        .reindex(HORIZON_ORDER)
    )

    ax = pivot.plot(kind="bar", figsize=(8, 4.8), rot=0)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.axhline(-np.log10(0.10), linestyle=":", linewidth=1)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"$-\log_{10}$ one-sided Clark--West $p$-value")
    # ax.set_title("Nested forecast-comparison evidence")
    ax.legend(title="Asset")
    ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def load_raw_ohlcv(data_dir: Path, asset: str) -> pd.DataFrame | None:
    for fname in _asset_to_filename(asset):
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            return df
    return None


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    time_col = None
    for c in ["timestamp", "Timestamp", "time", "Time", "datetime", "Date", "open_time"]:
        if c in df.columns:
            time_col = c
            break

    close_col = _find_column(df, ["Close", "close", "c"])
    volume_col = _find_column(df, ["Volume", "volume", "vol", "v"])

    out = df.copy()
    if time_col is not None:
        out["timestamp"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    else:
        out["timestamp"] = pd.RangeIndex(len(out))

    out = out.dropna(subset=[close_col, volume_col]).copy()
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    out["volume"] = pd.to_numeric(out[volume_col], errors="coerce")
    out = out[(out["close"] > 0) & (out["volume"].notna())].copy()

    out["log_close"] = np.log(out["close"])
    out["ret"] = out["log_close"].diff()
    out["log_volume"] = np.log1p(out["volume"].clip(lower=0))
    return out.dropna(subset=["ret"])


def rolling_z(x: pd.Series, window: int, floor: float = 1e-12) -> pd.Series:
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std(ddof=0).clip(lower=floor)
    return (x - mu) / sd


def add_activation_signal(df: pd.DataFrame, window_bars: int = 288) -> pd.DataFrame:
    out = df.copy()
    z_v = rolling_z(out["log_volume"], window_bars)
    z_r = rolling_z(out["ret"].abs(), window_bars)
    out["Z_raw"] = np.maximum(z_v, z_r)
    out["Z"] = out["Z_raw"].rolling(3, min_periods=3).mean()
    out["rv_24h_forward"] = (
        out["ret"].pow(2)
        .rolling(288, min_periods=288)
        .sum()
        .shift(-288)
    )
    return out.dropna(subset=["Z", "rv_24h_forward"])


def plot_activation_episode(
    data_dir: Path,
    asset: str,
    out_path: Path,
    window_bars: int = 288,
    episode_bars: int = 288 * 3,
) -> bool:
    raw = load_raw_ohlcv(data_dir, asset)
    if raw is None:
        return False

    df = add_activation_signal(prepare_ohlcv(raw), window_bars=window_bars)
    if df.empty:
        return False

    # Pick an interpretable episode: highest activation point in 2025 if available;
    # otherwise highest activation point in the full sample.
    if is_datetime64_any_dtype(df["timestamp"]):
        df_2025 = df[df["timestamp"].dt.year == 2025]
    else:
        df_2025 = pd.DataFrame()

    base = df_2025 if not df_2025.empty else df
    center_idx = base["Z"].idxmax()
    loc = df.index.get_loc(center_idx)
    lo = max(0, loc - episode_bars // 2)
    hi = min(len(df), loc + episode_bars // 2)
    ep = df.iloc[lo:hi].copy()

    x = ep["timestamp"] if "timestamp" in ep else np.arange(len(ep))

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(x, ep["log_close"], label="log price")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("log price")

    ax2 = ax1.twinx()
    ax2.plot(x, ep["Z"], linestyle="--", label="activation signal")
    ax2.set_ylabel("activation signal")

    # ax1.set_title(f"{asset}: activation episode and log-price path")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    ax1.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--activation-assets", nargs="+", default=["BTC"])
    args = parser.parse_args()

    df = load_oos_results(args.results_dir)

    plot_oos_metric(
        df,
        assets=args.assets,
        metric="MSE_improve_pct",
        ylabel="MSE improvement (%)",
        out_path=args.out_dir / "fig_oos_mse_improvement_btc_eth.pdf",
    )
    plot_oos_metric(
        df,
        assets=args.assets,
        metric="QLIKE_improve",
        ylabel="QLIKE improvement",
        out_path=args.out_dir / "fig_oos_qlike_improvement_btc_eth.pdf",
    )
    plot_cw_pvalues(
        df,
        assets=args.assets,
        out_path=args.out_dir / "fig_clark_west_pvalues_btc_eth.pdf",
    )

    for asset in args.activation_assets:
        ok = plot_activation_episode(
            args.data_dir,
            asset=asset.upper(),
            out_path=args.out_dir / f"fig_activation_episode_{asset.upper()}.pdf",
        )
        if not ok:
            print(f"[warn] Could not create activation episode figure for {asset}")

    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
