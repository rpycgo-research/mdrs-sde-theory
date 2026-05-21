"""Run forecasting robustness diagnostics for the volatility-forecasting paper.

Outputs
-------
- ex_ante_liquidity_2024.csv
- har_per_origin_loss_details.csv
- pooled_loss_tests.csv
- hyperparameter_sensitivity_har.csv
- funding_window_effect_size.csv
- margin_capital_efficiency_diagnostics.csv
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.empirical import AssetFile
from src.forecasting_robustness import (
    DEFAULT_FUNDING_FILES,
    DEFAULT_OHLCV_FILES,
    ForecastingRobustnessConfig,
    compute_ex_ante_liquidity,
    compute_funding_window_effect_size,
    compute_har_per_origin_details,
    compute_margin_capital_efficiency,
    compute_pooled_loss_tests,
    evaluate_hyperparameter_sensitivity,
)


def parse_asset_files(items: list[str] | None) -> list[AssetFile]:
    if not items:
        return DEFAULT_OHLCV_FILES
    out = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Asset mapping must be ASSET:file.csv, got {item}")
        asset, filename = item.split(":", 1)
        out.append(AssetFile(asset.upper(), filename))
    return out


def parse_funding_files(items: list[str] | None) -> dict[str, str]:
    mapping = dict(DEFAULT_FUNDING_FILES)
    if not items:
        return mapping
    for item in items:
        if ":" not in item:
            raise ValueError(f"Funding mapping must be ASSET:file.csv, got {item}")
        asset, filename = item.split(":", 1)
        mapping[asset.upper()] = filename
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forecasting robustness diagnostics.")
    parser.add_argument("--data-dir", default="data", help="Directory containing OHLCV files.")
    parser.add_argument("--funding-dir", default="data/funding_rate", help="Directory containing actual funding-rate files.")
    parser.add_argument("--out-dir", default="results/forecasting_robustness", help="Output directory.")
    parser.add_argument("--assets", nargs="*", help="Optional ASSET:file.csv mappings.")
    parser.add_argument("--funding-files", nargs="*", help="Optional ASSET:file.csv funding mappings.")
    parser.add_argument("--funding-window-minutes", type=int, default=60)
    parser.add_argument("--margin-targets", nargs="*", type=float, default=[0.05, 0.10])
    parser.add_argument("--skip-hyperparameter", action="store_true")
    parser.add_argument("--skip-margin", action="store_true")
    parser.add_argument("--skip-pooled", action="store_true")
    parser.add_argument("--skip-funding-effect", action="store_true")
    parser.add_argument("--skip-liquidity", action="store_true")
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] {path} ({len(df)} rows)", flush=True)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    funding_dir = Path(args.funding_dir)
    out_dir = Path(args.out_dir)
    assets = parse_asset_files(args.assets)
    funding_files = parse_funding_files(args.funding_files)
    config = ForecastingRobustnessConfig(
        funding_window_minutes=args.funding_window_minutes,
        margin_target_exceedance_rates=tuple(args.margin_targets),
    )

    if not args.skip_liquidity:
        print("[run] ex-ante liquidity ranking", flush=True)
        liquidity = compute_ex_ante_liquidity(data_dir, assets)
        save(liquidity, out_dir / "ex_ante_liquidity_2024.csv")

    detail = pd.DataFrame()
    if not args.skip_pooled:
        print("[run] HAR per-origin details and pooled loss tests", flush=True)
        detail = compute_har_per_origin_details(data_dir, assets, config=config)
        save(detail, out_dir / "har_per_origin_loss_details.csv")
        pooled = compute_pooled_loss_tests(detail, hac_lags=config.hac_lags)
        save(pooled, out_dir / "pooled_loss_tests.csv")
        del pooled
        gc.collect()

    if not args.skip_hyperparameter:
        print("[run] hyperparameter sensitivity", flush=True)
        hyper = evaluate_hyperparameter_sensitivity(data_dir, assets, config=config)
        save(hyper, out_dir / "hyperparameter_sensitivity_har.csv")
        del hyper
        gc.collect()

    if not args.skip_funding_effect:
        print("[run] actual funding-window effect sizes", flush=True)
        funding_effect = compute_funding_window_effect_size(
            data_dir,
            funding_dir,
            assets,
            funding_files,
            window_minutes=args.funding_window_minutes,
        )
        save(funding_effect, out_dir / "funding_window_effect_size.csv")

    if not args.skip_margin:
        print("[run] margin capital-efficiency diagnostics", flush=True)
        margin = compute_margin_capital_efficiency(data_dir, assets, config=config)
        save(margin, out_dir / "margin_capital_efficiency_diagnostics.csv")

    print("[done] forecasting robustness diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
