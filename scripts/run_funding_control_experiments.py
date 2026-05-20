"""Run actual funding-rate robustness tests for the volatility-forecasting manuscript.

This script expects 5-minute OHLCV files and separate funding-rate CSVs with
columns such as datetime, calc_time, funding_interval_hours, last_funding_rate.
It writes timeout-safe CSV outputs that can be committed directly.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_empirical_calibration import DEFAULT_FILES, parse_asset_files
from src.empirical import AssetFile
from src.volatility_forecasting_fast import load_prepared_asset
from src.funding_rate_controls import (
    FundingConfig,
    evaluate_actual_funding_controls,
    load_funding_rates,
    merge_actual_funding_features,
    summarize_actual_funding_activation,
    summarize_actual_funding_window_losses,
)

DEFAULT_FUNDING_FILES = {
    "BTC": "btcusdt_funding.csv",
    "ETH": "ethusdt_funding.csv",
    "XRP": "xrpusdt_funding.csv",
    "SOL": "solusdt_funding.csv",
}


def parse_funding_files(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_FUNDING_FILES)
    out: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Funding mapping must look like ASSET:file.csv, got {item!r}")
        asset, filename = item.split(":", 1)
        out[asset.upper()] = filename
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run actual funding-rate volatility-forecasting diagnostics.")
    parser.add_argument("--data-dir", default="data", help="Directory containing OHLCV CSV files.")
    parser.add_argument("--funding-dir", default="data/funding_rate", help="Directory containing funding-rate CSV files.")
    parser.add_argument("--out-dir", default="results/funding_controls", help="Output directory.")
    parser.add_argument("--assets", nargs="*", help="OHLCV mappings like BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv.")
    parser.add_argument("--funding-files", nargs="*", help="Funding mappings like BTC:btcusdt_funding.csv.")
    parser.add_argument("--all-assets", action="store_true", help="Run all default assets instead of BTC/ETH only.")
    parser.add_argument("--funding-history-events", type=int, default=90)
    parser.add_argument("--funding-window-minutes", type=int, default=60)
    return parser.parse_args()


def _asset_files(args: argparse.Namespace) -> list[AssetFile]:
    if args.assets:
        return parse_asset_files(args.assets)
    if args.all_assets:
        return DEFAULT_FILES
    return [item for item in DEFAULT_FILES if item.asset in {"BTC", "ETH"}]


def _append_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    if df.empty:
        return
    path = out_dir / filename
    write_header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=write_header)
    print(f"[save] {path} (+{len(df)} rows)", flush=True)


def _reset_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "actual_funding_control_tests.csv",
        "actual_funding_activation_summary.csv",
        "actual_funding_window_loss_diagnostics.csv",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    funding_dir = Path(args.funding_dir)
    out_dir = Path(args.out_dir)
    _reset_outputs(out_dir)

    funding_files = parse_funding_files(args.funding_files)
    config = FundingConfig(
        funding_history_events=args.funding_history_events,
        funding_window_minutes=args.funding_window_minutes,
    )

    all_rows = []
    activation_rows = []
    for asset_file in _asset_files(args):
        ohlcv_path = data_dir / asset_file.filename
        funding_name = funding_files.get(asset_file.asset)
        if funding_name is None:
            print(f"[skip] {asset_file.asset}: no funding file mapping", flush=True)
            continue
        funding_path = funding_dir / funding_name
        if not ohlcv_path.exists():
            print(f"[skip] {asset_file.asset}: missing OHLCV file {ohlcv_path}", flush=True)
            continue
        if not funding_path.exists():
            print(f"[skip] {asset_file.asset}: missing funding file {funding_path}", flush=True)
            continue

        print(f"[load] {asset_file.asset}: {ohlcv_path} + {funding_path}", flush=True)
        df = load_prepared_asset(ohlcv_path)
        funding = load_funding_rates(funding_path, history_events=args.funding_history_events)
        df = merge_actual_funding_features(df, funding, funding_window_minutes=args.funding_window_minutes)

        activation = pd.DataFrame([summarize_actual_funding_activation(df, asset_file.asset, config)])
        rows, detail = evaluate_actual_funding_controls(df, asset_file.asset, config)
        window_losses = summarize_actual_funding_window_losses(detail, model_col_prefix="har_z")

        _append_csv(rows, out_dir, "actual_funding_control_tests.csv")
        _append_csv(activation, out_dir, "actual_funding_activation_summary.csv")
        _append_csv(window_losses, out_dir, "actual_funding_window_loss_diagnostics.csv")
        all_rows.append(rows)
        activation_rows.append(activation)

        del df, funding, rows, detail, activation, window_losses
        gc.collect()

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        print("\n=== Actual funding-control summary ===", flush=True)
        cols = ["Asset", "Horizon", "Model", "MSE_improvement_vs_HAR_pct", "QLIKE_improvement_vs_HAR"]
        print(summary[cols].to_string(index=False), flush=True)
    if activation_rows:
        activation = pd.concat(activation_rows, ignore_index=True)
        print("\n=== Funding activation summary ===", flush=True)
        cols = ["Asset", "Corr_Z_abs_funding_rate_z", "Mean_Z_all", "Mean_Z_in_actual_funding_window", "Mean_Z_outside_actual_funding_window"]
        print(activation[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
