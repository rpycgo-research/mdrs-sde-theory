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
    calibrate_leaky_extrema,
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

OUTPUT_TABLES = {
    "data_coverage.csv": "coverage_rows",
    "ou_calibration.csv": "ou_rows",
    "ou_oos_diagnostics.csv": "ou_oos_rows",
    "leaky_extrema_calibration.csv": "leaky_rows",
    "leaky_grid_search.csv": "leaky_grid_rows",
}


def empty_outputs() -> dict[str, list[dict]]:
    """Initialize output containers."""
    return {
        "coverage_rows": [],
        "ou_rows": [],
        "ou_oos_rows": [],
        "leaky_rows": [],
        "leaky_grid_rows": [],
    }


def process_asset(
    *,
    asset_file: AssetFile,
    data_dir: Path,
    outputs: dict[str, list[dict]],
    ) -> None:
    """Load one asset and append coverage, OU, and leaky-extrema results."""
    path = data_dir / asset_file.filename
    if not path.exists():
        print(f"[skip] {asset_file.asset}: missing file {path}")
        return

    print(f"[load] {asset_file.asset}: {path}")
    df = load_ohlcv(path)
    df = add_microstructure_signal(df)

    outputs["coverage_rows"].append(data_coverage(df, asset_file.asset))

    fit = fit_ou_ar1(split_sample(df, "train")["Z"])
    outputs["ou_rows"].append({"Asset": asset_file.asset, **fit})

    for split in ["train", "validation", "test", "recent"]:
        diagnostics = ou_oos_diagnostics(split_sample(df, split)["Z"], fit)
        outputs["ou_oos_rows"].append(
            {"Asset": asset_file.asset, "Split": split, **diagnostics}
        )

    rows, grid = calibrate_leaky_extrema(df, asset_file.asset)
    outputs["leaky_rows"].extend(rows)
    outputs["leaky_grid_rows"].extend(grid)


def save_outputs(
    *,
    outputs: dict[str, list[dict]],
    out_dir: Path,
    ) -> None:
    """Write output tables to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, key in OUTPUT_TABLES.items():
        output_path = out_dir / filename
        rows = outputs[key]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"[save] {output_path} ({len(rows)} rows)")


def main():
    """Run the empirical calibration pipeline."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    outputs = empty_outputs()

    for asset_file in parse_asset_files(args.assets):
        process_asset(
            asset_file=asset_file,
            data_dir=data_dir,
            outputs=outputs,
        )

    save_outputs(outputs=outputs, out_dir=out_dir)


if __name__ == "__main__":
    main()
