"""Run timeout-safe volatility-forecasting experiments.

Outputs
-------
1. stronger_benchmark_tests.csv
   HAR/HARQ/RSV-HAR baselines compared with +Z.
2. validation_selected_family_tests.csv
   Optional: 2024 validation-selected family evaluated on 2025 test.
3. scheduled_funding_window_diagnostics.csv
   Loss improvements inside/outside scheduled 00/08/16 UTC funding windows.
4. high_volatility_ranking_diagnostics.csv
   Top-volatility ranking and underprediction diagnostics.

The default runs BTC and ETH only.  It writes each asset's output incrementally
and releases the large OHLCV feature frame before loading the next asset, which
prevents notebook/CI timeouts caused by holding all assets in memory.
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

from scripts.run_empirical_calibration import DEFAULT_FILES, parse_asset_files  # noqa: E402
from src.empirical import AssetFile  # noqa: E402
from src.volatility_forecasting_fast import (  # noqa: E402
    ForecastingExperimentConfig,
    evaluate_stronger_benchmarks,
    evaluate_validation_selected_family,
    load_prepared_asset,
    summarize_funding_window,
    summarize_high_volatility_ranking,
)

DEFAULT_MAIN_FILES = [item for item in DEFAULT_FILES if item.asset in {"BTC", "ETH"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast volatility-forecasting diagnostics.")
    parser.add_argument("--data-dir", default="data", help="Directory containing 5-minute OHLCV CSV files.")
    parser.add_argument("--out-dir", default="results/volatility_forecasting", help="Output directory for CSV diagnostics.")
    parser.add_argument(
        "--assets",
        nargs="*",
        help="Optional asset mappings such as BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv.",
    )
    parser.add_argument("--all-assets", action="store_true", help="Run all default assets instead of BTC/ETH only.")
    parser.add_argument("--funding-window-minutes", type=int, default=60, help="Minutes around 00/08/16 UTC settlements.")
    parser.add_argument("--top-quantile", type=float, default=0.90, help="Quantile for high-volatility ranking diagnostics.")
    parser.add_argument(
        "--selection-loss",
        choices=["MSE", "QLIKE"],
        default="QLIKE",
        help="Validation loss used to select the strongest HAR-family baseline.",
    )
    parser.add_argument(
        "--run-validation-selection",
        action="store_true",
        help="Also run 2024 validation-selected best-family tests. Slower; off by default to avoid timeouts.",
    )
    return parser.parse_args()


def _asset_files(args: argparse.Namespace) -> list[AssetFile]:
    if args.assets:
        return parse_asset_files(args.assets)
    if args.all_assets:
        return DEFAULT_FILES
    return DEFAULT_MAIN_FILES


def _append_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    write_header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=write_header)
    print(f"[save] {path} (+{len(df)} rows)", flush=True)


def _reset_outputs(out_dir: Path, filenames: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        path = out_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    config = ForecastingExperimentConfig(
        funding_window_minutes=args.funding_window_minutes,
        top_quantile=args.top_quantile,
        selection_loss=args.selection_loss,
    )

    output_files = [
        "stronger_benchmark_tests.csv",
        "scheduled_funding_window_diagnostics.csv",
        "high_volatility_ranking_diagnostics.csv",
    ]
    if args.run_validation_selection:
        output_files.append("validation_selected_family_tests.csv")
    _reset_outputs(out_dir, output_files)

    summary_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []

    for asset_file in _asset_files(args):
        path = data_dir / asset_file.filename
        if not path.exists():
            print(f"[skip] {asset_file.asset}: missing file {path}", flush=True)
            continue

        print(f"[load] {asset_file.asset}: {path}", flush=True)
        df = load_prepared_asset(path)

        print(f"[run] {asset_file.asset}: stronger HAR-family benchmarks", flush=True)
        benchmark_rows, detail = evaluate_stronger_benchmarks(df, asset_file.asset, config)
        _append_csv(benchmark_rows, out_dir, "stronger_benchmark_tests.csv")
        summary_rows.append(benchmark_rows)

        if not detail.empty:
            funding = summarize_funding_window(detail, window_minutes=args.funding_window_minutes)
            ranking = summarize_high_volatility_ranking(detail, top_quantile=args.top_quantile)
            _append_csv(funding, out_dir, "scheduled_funding_window_diagnostics.csv")
            _append_csv(ranking, out_dir, "high_volatility_ranking_diagnostics.csv")

        if args.run_validation_selection:
            print(f"[run] {asset_file.asset}: validation-selected best HAR-family + Z", flush=True)
            selected = evaluate_validation_selected_family(df, asset_file.asset, config)
            _append_csv(selected, out_dir, "validation_selected_family_tests.csv")
            selected_rows.append(selected)

        del df, benchmark_rows, detail
        gc.collect()

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        print("\n=== Stronger HAR-family benchmark summary ===", flush=True)
        cols = [
            "Asset",
            "Horizon",
            "Family",
            "MSE_improvement_pct",
            "QLIKE_improvement",
            "MSE_diff_p_one_sided_NW",
            "QLIKE_diff_p_one_sided_NW",
        ]
        print(summary[cols].to_string(index=False), flush=True)

    if selected_rows:
        selected_summary = pd.concat(selected_rows, ignore_index=True)
        print("\n=== Validation-selected family summary ===", flush=True)
        cols = ["Asset", "Horizon", "Family", "Validation_baseline_loss", "MSE_improvement_pct", "QLIKE_improvement"]
        print(selected_summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
