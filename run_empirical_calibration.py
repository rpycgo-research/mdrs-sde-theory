"""
Run the MDRS-SDE empirical calibration pipeline.

Expected CSV columns
--------------------
Datetime, Open, High, Low, Close, Volume

Default asset filenames assume the files are located under the repository's
root-level data/ directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.empirical import (
    AssetFile,
    add_microstructure_signal,
    data_coverage,
    fit_ou_ar1,
    load_ohlcv,
    ou_oos_diagnostics,
    split_sample,
)

DEFAULT_FILES = [
    AssetFile("BTC", "btcusdt_5m.csv"),
    AssetFile("ETH", "ethusdt_5m.csv"),
    AssetFile("XRP", "xrpusdt_5m.csv"),
    AssetFile("SOL", "solusdt_5m.csv"),
]


def parse_asset_files(items: list[str] | None) -> list[AssetFile]:
    """Parse optional ASSET:filename command-line mappings."""
    if not items:
        return DEFAULT_FILES

    parsed = []
    for item in items:
        if ":" not in item:
            raise ValueError(
                "Asset mapping must be ASSET:filename, "
                "for example BTC:btcusdt_5m.csv."
            )

        asset, filename = item.split(":", 1)
        parsed.append(AssetFile(asset.upper(), filename))

    return parsed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run MDRS-SDE empirical calibration pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing OHLCV CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/empirical",
        help="Directory for generated CSV outputs.",
    )
    parser.add_argument(
        "--assets",
        nargs="*",
        help=(
            "Optional asset mappings, for example "
            "BTC:btcusdt_5m.csv ETH:ethusdt_5m.csv."
        ),
    )
    return parser.parse_args()


def process_asset(
    *,
    asset_file: AssetFile,
    data_dir: Path,
    coverage_rows: list[dict],
    ou_rows: list[dict],
    ou_oos_rows: list[dict],
    ) -> None:
    """Load one asset and append coverage and OU results."""
    path = data_dir / asset_file.filename
    if not path.exists():
        print(f"[skip] {asset_file.asset}: missing file {path}")
        return

    print(f"[load] {asset_file.asset}: {path}")
    df = load_ohlcv(path)
    df = add_microstructure_signal(df)

    coverage_rows.append(data_coverage(df, asset_file.asset))
    fit = fit_ou_ar1(split_sample(df, "train")["Z"])
    ou_rows.append(
        {
            "Asset": asset_file.asset,
            **{
                key: fit[key]
                for key in [
                    "b",
                    "kappa_per_day",
                    "half_life_hours",
                    "zbar",
                    "sigmaZ_per_sqrt_day",
                    "r2",
                    "n",
                ]
            },
        }
    )

    for split in ["train", "validation", "test", "recent"]:
        diagnostics = ou_oos_diagnostics(split_sample(df, split)["Z"], fit)
        ou_oos_rows.append(
            {
                "Asset": asset_file.asset,
                "Split": split,
                **diagnostics,
            }
        )


def main():
    """Run coverage and OU calibration outputs."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows: list[dict] = []
    ou_rows: list[dict] = []
    ou_oos_rows: list[dict] = []

    for asset_file in parse_asset_files(args.assets):
        process_asset(
            asset_file=asset_file,
            data_dir=data_dir,
            coverage_rows=coverage_rows,
            ou_rows=ou_rows,
            ou_oos_rows=ou_oos_rows,
        )

    pd.DataFrame(coverage_rows).to_csv(out_dir / "data_coverage.csv", index=False)
    pd.DataFrame(ou_rows).to_csv(out_dir / "ou_calibration.csv", index=False)
    pd.DataFrame(ou_oos_rows).to_csv(
        out_dir / "ou_oos_diagnostics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
