"""
Run the MDRS-SDE empirical calibration pipeline.

Expected CSV columns
--------------------
Datetime, Open, High, Low, Close, Volume

Default asset filenames match the 5-minute perpetual futures files used in the
paper experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.empirical import (
    AssetFile,
    add_microstructure_signal,
    add_realized_variance,
    calibrate_leaky_extrema,
    data_coverage,
    fit_ou_ar1,
    fit_volatility_prediction,
    load_ohlcv,
    ou_oos_diagnostics,
    real_data_moments,
    split_sample,
)

DEFAULT_FILES = [
    AssetFile("BTC", "btcusdt_5m.csv"),
    AssetFile("ETH", "ethusdt_5m.csv"),
    AssetFile("XRP", "xrpusdt_5m.csv"),
    AssetFile("SOL", "solusdt_5m.csv"),
]

OUTPUT_TABLES = {
    "data_coverage.csv": "coverage_rows",
    "ou_calibration.csv": "ou_rows",
    "ou_oos_diagnostics.csv": "ou_oos_rows",
    "leaky_extrema_calibration.csv": "leaky_rows",
    "leaky_grid_search.csv": "leaky_grid_rows",
    "volatility_prediction_hac.csv": "vol_rows",
    "return_moments_batch_means.csv": "moment_rows",
}

MAIN_ASSETS = ("BTC", "ETH")
ROBUSTNESS_ASSETS = ("XRP", "SOL")


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


def empty_outputs() -> dict[str, list[dict]]:
    """Initialize the output table containers."""
    return {
        "coverage_rows": [],
        "ou_rows": [],
        "ou_oos_rows": [],
        "leaky_rows": [],
        "leaky_grid_rows": [],
        "vol_rows": [],
        "moment_rows": [],
    }


def process_asset(
    *,
    asset_file: AssetFile,
    data_dir: Path,
    outputs: dict[str, list[dict]],
    ) -> None:
    """Load one asset and append its empirical results to output containers."""
    path = data_dir / asset_file.filename
    if not path.exists():
        print(f"[skip] {asset_file.asset}: missing file {path}")
        return

    print(f"[load] {asset_file.asset}: {path}")
    df = load_ohlcv(path)
    df = add_microstructure_signal(df)
    df = add_realized_variance(df)

    outputs["coverage_rows"].append(data_coverage(df, asset_file.asset))

    append_ou_results(
        df=df,
        asset=asset_file.asset,
        outputs=outputs,
    )
    append_leaky_extrema_results(
        df=df,
        asset=asset_file.asset,
        outputs=outputs,
    )
    append_volatility_prediction_results(
        df=df,
        asset=asset_file.asset,
        outputs=outputs,
    )
    append_moment_results(
        df=df,
        asset=asset_file.asset,
        outputs=outputs,
    )


def append_ou_results(
    *,
    df: pd.DataFrame,
    asset: str,
    outputs: dict[str, list[dict]],
    ) -> None:
    """Fit AR(1)-to-OU parameters and append OOS diagnostics."""
    fit = fit_ou_ar1(split_sample(df, "train")["Z"])
    outputs["ou_rows"].append(
        {
            "Asset": asset,
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
        outputs["ou_oos_rows"].append(
            {
                "Asset": asset,
                "Split": split,
                **diagnostics,
            }
        )


def append_leaky_extrema_results(
    *,
    df: pd.DataFrame,
    asset: str,
    outputs: dict[str, list[dict]],
    ) -> None:
    """Calibrate and append leaky-extrema approximation results."""
    rows, grid = calibrate_leaky_extrema(df, asset)
    outputs["leaky_rows"].extend(rows)
    outputs["leaky_grid_rows"].extend(grid)


def append_volatility_prediction_results(
    *,
    df: pd.DataFrame,
    asset: str,
    outputs: dict[str, list[dict]],
    ) -> None:
    """Append HAC volatility-prediction regression results."""
    for split in ["validation", "test", "recent"]:
        rows = fit_volatility_prediction(df, asset, split=split)
        outputs["vol_rows"].extend(rows)


def append_moment_results(
    *,
    df: pd.DataFrame,
    asset: str,
    outputs: dict[str, list[dict]],
) -> None:
    """Append batch-means return-moment diagnostics."""
    for split in ["train", "validation", "test", "recent"]:
        rows = real_data_moments(df, asset, split=split)
        outputs["moment_rows"].extend(rows)


def save_outputs(
    *,
    outputs: dict[str, list[dict]],
    out_dir: Path,
    ) -> None:
    """Write all output tables to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, key in OUTPUT_TABLES.items():
        rows = outputs[key]
        output_path = out_dir / filename
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"[save] {output_path} ({len(rows)} rows)")


def print_table_block(
    *,
    title: str,
    rows: list[dict],
    assets: tuple[str, ...],
    columns: list[str] | None = None,
    query: str | None = None,
) -> None:
    """Print a filtered table block for selected assets."""
    if not rows:
        print(f"\n{title}: no rows")
        return

    table = pd.DataFrame(rows)
    table = table[table["Asset"].isin(assets)]

    if query:
        table = table.query(query)

    if columns:
        table = table[columns]

    print(f"\n{title}:")
    if table.empty:
        print("No rows")
    else:
        print(table.to_string(index=False))


def print_compact_summary(outputs: dict[str, list[dict]]) -> None:
    """Print compact main-asset and robustness summaries."""
    print("\n=== Main empirical assets: BTC / ETH ===")

    print_table_block(
        title="OU calibration summary",
        rows=outputs["ou_rows"],
        assets=MAIN_ASSETS,
    )

    print_table_block(
        title="Leaky-extrema test summary",
        rows=outputs["leaky_rows"],
        assets=MAIN_ASSETS,
        query="Split == 'test'",
        columns=[
            "Asset",
            "Split",
            "fast_hl_hours",
            "slow_hl_hours",
            "Mean_MAE",
            "Corr_R",
            "Corr_S",
        ],
    )

    print_table_block(
        title="Volatility-prediction test summary",
        rows=outputs["vol_rows"],
        assets=MAIN_ASSETS,
        query="Split == 'test'",
        columns=[
            "Asset",
            "Split",
            "Horizon",
            "coef_Z",
            "t_Z_HAC",
            "p_Z_HAC",
            "R2",
        ],
    )

    print("\n=== Robustness assets: XRP / SOL ===")

    print_table_block(
        title="OU calibration robustness summary",
        rows=outputs["ou_rows"],
        assets=ROBUSTNESS_ASSETS,
    )

    print_table_block(
        title="Leaky-extrema test robustness summary",
        rows=outputs["leaky_rows"],
        assets=ROBUSTNESS_ASSETS,
        query="Split == 'test'",
        columns=[
            "Asset",
            "Split",
            "fast_hl_hours",
            "slow_hl_hours",
            "Mean_MAE",
            "Corr_R",
            "Corr_S",
        ],
    )

    print_table_block(
        title="Volatility-prediction test robustness summary",
        rows=outputs["vol_rows"],
        assets=ROBUSTNESS_ASSETS,
        query="Split == 'test'",
        columns=[
            "Asset",
            "Split",
            "Horizon",
            "coef_Z",
            "t_Z_HAC",
            "p_Z_HAC",
            "R2",
        ],
    )


def main() -> None:
    """Run the full empirical calibration pipeline."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    asset_files = parse_asset_files(args.assets)
    outputs = empty_outputs()

    for asset_file in asset_files:
        process_asset(
            asset_file=asset_file,
            data_dir=data_dir,
            outputs=outputs,
        )

    save_outputs(outputs=outputs, out_dir=out_dir)
    print_compact_summary(outputs)


if __name__ == "__main__":
    main()
