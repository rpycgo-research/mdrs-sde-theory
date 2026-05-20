"""Combine per-asset fast volatility-forecasting experiment outputs.

This helper is intended for timeout-limited environments: run each asset into a
separate output directory, then combine the CSVs for manuscript tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_FILES = [
    "stronger_benchmark_tests.csv",
    "validation_selected_family_tests.csv",
    "scheduled_funding_window_diagnostics.csv",
    "high_volatility_ranking_diagnostics.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-asset fast volatility-forecasting output CSVs.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Per-asset result directories.")
    parser.add_argument("--out-dir", required=True, help="Combined output directory.")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="CSV filenames to combine.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in args.files:
        frames = []
        for directory in args.input_dirs:
            path = Path(directory) / filename
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            print(f"[skip] {filename}: not found in input dirs")
            continue
        combined = pd.concat(frames, ignore_index=True)
        out_path = out_dir / filename
        combined.to_csv(out_path, index=False)
        print(f"[save] {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
