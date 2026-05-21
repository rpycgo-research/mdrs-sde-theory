"""Combine per-asset forecasting robustness diagnostic outputs.

This helper is useful when full all-asset runs are split to avoid local timeout
or memory issues.  It concatenates per-asset outputs and recomputes pooled loss
statistics from the combined per-origin HAR loss details.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.forecasting_robustness import compute_pooled_loss_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine forecasting robustness diagnostic outputs.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Per-asset result directories.")
    parser.add_argument("--out-dir", required=True, help="Combined output directory.")
    return parser.parse_args()


def _read_existing(input_dirs: list[Path], filename: str) -> list[pd.DataFrame]:
    frames = []
    for d in input_dirs:
        p = d / filename
        if p.exists():
            frames.append(pd.read_csv(p))
    return frames


def _save(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"[save] {path} ({len(df)} rows)", flush=True)


def main() -> None:
    args = parse_args()
    input_dirs = [Path(x) for x in args.input_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    liquidity_frames = _read_existing(input_dirs, "ex_ante_liquidity_2024.csv")
    if liquidity_frames:
        liquidity = pd.concat(liquidity_frames, ignore_index=True).drop_duplicates("Asset")
        liquidity = liquidity.sort_values("Avg_daily_notional", ascending=False).reset_index(drop=True)
        liquidity["Liquidity_rank"] = range(1, len(liquidity) + 1)
        liquidity["Primary_asset_by_ex_ante_top2"] = liquidity["Liquidity_rank"] <= 2
        _save(liquidity, out_dir, "ex_ex_ante_liquidity_2024.csv".replace("ex_ex", "ex"))

    detail_frames = _read_existing(input_dirs, "har_per_origin_loss_details.csv")
    if detail_frames:
        detail = pd.concat(detail_frames, ignore_index=True)
        _save(detail, out_dir, "har_per_origin_loss_details.csv")
        pooled = compute_pooled_loss_tests(detail)
        _save(pooled, out_dir, "pooled_loss_tests.csv")

    for filename in [
        "funding_window_effect_size.csv",
        "margin_capital_efficiency_diagnostics.csv",
        "hyperparameter_sensitivity_har.csv",
    ]:
        frames = _read_existing(input_dirs, filename)
        if frames:
            _save(pd.concat(frames, ignore_index=True), out_dir, filename)


if __name__ == "__main__":
    main()
