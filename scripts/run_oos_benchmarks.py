"""Run QF-style HAR/log-HAR OOS benchmark checks.

The script expects the same OHLCV CSV schema as run_empirical_calibration.py:
Datetime, Open, High, Low, Close, Volume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.empirical import AssetFile, add_microstructure_signal, add_realized_variance, load_ohlcv
from src.oos import (
    ForecastConfig,
    add_har_realized_variance_features,
    evaluate_specification_robustness,
    evaluate_static_oos,
    evaluate_winsorization_robustness,
)
from scripts.run_empirical_calibration import DEFAULT_FILES, parse_asset_files


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run positive log-HAR OOS benchmarks against HAR+Z forecasts.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing OHLCV CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/oos",
        help="Directory for generated OOS benchmark CSV outputs.",
    )
    parser.add_argument(
        "--assets",
        nargs="*",
        help=(
            "Optional asset mappings, for example "
            "BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv."
        ),
    )
    parser.add_argument(
        "--bootstrap-replications",
        type=int,
        default=1000,
        help="Moving-block bootstrap replications for loss-difference CIs.",
    )
    parser.add_argument(
        "--bootstrap-block-length",
        type=int,
        default=24,
        help="Moving-block bootstrap block length in forecast-origin observations.",
    )
    return parser.parse_args()


def load_asset_frame(data_dir: Path, asset_file: AssetFile) -> pd.DataFrame | None:
    """Load one asset and create Z and realized-variance features."""
    path = data_dir / asset_file.filename
    if not path.exists():
        print(f"[skip] {asset_file.asset}: missing file {path}")
        return None

    print(f"[load] {asset_file.asset}: {path}")
    df = load_ohlcv(path)
    df = add_microstructure_signal(df)
    df = add_realized_variance(df)
    df = add_har_realized_variance_features(df)

    return df


def save_table(rows: list[dict], out_dir: Path, filename: str) -> None:
    """Save a list of rows as a CSV table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] {path} ({len(rows)} rows)")


def main() -> None:
    """Run all QF-style OOS benchmark screens."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    asset_files = parse_asset_files(args.assets) if args.assets else DEFAULT_FILES

    main_rows: list[dict] = []
    winsor_rows: list[dict] = []
    spec_rows: list[dict] = []

    base_config = ForecastConfig(
        bootstrap_replications=args.bootstrap_replications,
        bootstrap_block_length=args.bootstrap_block_length,
    )

    for asset_file in asset_files:
        df = load_asset_frame(data_dir, asset_file)
        if df is None:
            continue

        print(f"[run] {asset_file.asset}: main OOS tests")
        main_rows.extend(evaluate_static_oos(df, asset_file.asset, base_config))
        print(f"[run] {asset_file.asset}: Z winsorization robustness")
        winsor_rows.extend(evaluate_winsorization_robustness(df,
                                                             asset_file.asset,
                                                             base_config=base_config))
        print(f"[run] {asset_file.asset}: log-RV specification robustness")
        spec_rows.extend(evaluate_specification_robustness(df,
                                                           asset_file.asset,
                                                           base_config=base_config))

    save_table(main_rows, out_dir, "static_oos_forecast_tests_main_with_bootstrap.csv")
    save_table(winsor_rows, out_dir, "static_oos_winsorization_robustness.csv")
    save_table(spec_rows, out_dir, "static_oos_logrv_spec_robustness.csv")

    if main_rows:
        summary = pd.DataFrame(main_rows)
        keep = [
            "Asset",
            "Horizon",
            "MSE_improvement",
            "QLIKE_improvement",
            "Delta_train_R2",
            "CW_MSE_p_one_sided",
            "DM_QLIKE_p_one_sided",
            "Boot_MSE_p_one_sided",
            "Boot_QLIKE_p_one_sided",
        ]
        print("\n=== Positive log-HAR OOS summary ===")
        print(summary[keep].to_string(index=False))


if __name__ == "__main__":
    main()
