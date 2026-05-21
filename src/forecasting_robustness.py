"""Forecasting robustness diagnostics for the volatility forecasting workflow.

This module adds lightweight diagnostics for ex-ante asset selection,
pooled cross-asset loss tests, activation hyperparameter sensitivity,
funding-window effect sizes, and margin-style capital-efficiency checks.
It deliberately excludes return-based GARCH benchmarks, which can be added
separately if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.empirical import AssetFile, BARS_PER_DAY, HORIZON_LABELS, add_realized_variance, load_ohlcv, split_sample
from src.oos import HORIZONS, RV_EPS
from src.volatility_forecasting_fast import (
    ForecastingExperimentConfig,
    _combine_splits,
    _fit_predict_log_smear,
    _model_comparison_row,
    _prepare_frame,
    _select_nonoverlap,
    _z_clip_bounds,
    add_forecasting_features,
)

DEFAULT_OHLCV_FILES = [
    AssetFile("BTC", "btcusdt_5m.csv"),
    AssetFile("ETH", "ethusdt_5m.csv"),
    AssetFile("XRP", "xrpusdt_5m.csv"),
    AssetFile("SOL", "solusdt_5m.csv"),
]

DEFAULT_FUNDING_FILES = {
    "BTC": "btcusdt.csv",
    "ETH": "ethusdt.csv",
    "XRP": "xrpusdt.csv",
    "SOL": "solusdt.csv",
}

LossName = Literal["MSE", "QLIKE"]


@dataclass(frozen=True)
class ForecastingRobustnessConfig:
    """Configuration for forecasting robustness diagnostics."""

    horizons: tuple[int, ...] = HORIZONS
    train_splits: tuple[str, ...] = ("train", "validation")
    test_split: str = "test"
    validation_split: str = "validation"
    z_clip_quantiles: tuple[float, float] | None = (0.005, 0.995)
    hac_lags: int | None = None
    funding_window_minutes: int = 60
    margin_target_exceedance_rates: tuple[float, ...] = (0.05, 0.10)


def _hac_mean_test(values: Iterable[float], max_lags: int | None = None) -> dict[str, float]:
    """Newey-West one-sided mean test for positive mean values."""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {"N": n, "mean": np.nan, "se": np.nan, "t": np.nan, "p_one_sided": np.nan, "max_lags": 0}
    mean = float(np.mean(x))
    centered = x - mean
    if max_lags is None:
        max_lags = int(np.floor(np.sqrt(n)))
    max_lags = max(0, min(int(max_lags), n - 1))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for lag in range(1, max_lags + 1):
        cov = float(np.dot(centered[:-lag], centered[lag:]) / n)
        weight = 1.0 - lag / (max_lags + 1.0)
        long_run += 2.0 * weight * cov
    se = float(np.sqrt(max(long_run / n, 0.0)))
    if se <= 0:
        return {"N": n, "mean": mean, "se": se, "t": np.nan, "p_one_sided": np.nan, "max_lags": max_lags}
    t_value = mean / se
    p_one = float(1.0 - stats.t.cdf(t_value, df=n - 1))
    return {"N": n, "mean": mean, "se": se, "t": float(t_value), "p_one_sided": p_one, "max_lags": max_lags}


def compute_ex_ante_liquidity(
    data_dir: str | Path,
    asset_files: list[AssetFile] = DEFAULT_OHLCV_FILES,
    *,
    validation_year: int = 2024,
) -> pd.DataFrame:
    """Compute pre-test liquidity ranking from validation-year notional volume."""
    rows: list[dict] = []
    data_dir = Path(data_dir)
    start = pd.Timestamp(f"{validation_year}-01-01")
    end = pd.Timestamp(f"{validation_year}-12-31 23:59:59")
    for item in asset_files:
        path = data_dir / item.filename
        df = load_ohlcv(path)
        val = df[(df["Datetime"] >= start) & (df["Datetime"] <= end)].copy()
        val["date"] = val["Datetime"].dt.date
        val["notional"] = val["Close"] * val["Volume"]
        daily = val.groupby("date", observed=True)["notional"].sum()
        rows.append(
            {
                "Asset": item.asset,
                "Validation_year": validation_year,
                "Start": str(val["Datetime"].min()),
                "End": str(val["Datetime"].max()),
                "N_5m_bars": int(len(val)),
                "N_days": int(daily.shape[0]),
                "Avg_daily_notional": float(daily.mean()),
                "Median_daily_notional": float(daily.median()),
                "Total_notional": float(daily.sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("Avg_daily_notional", ascending=False).reset_index(drop=True)
    out.insert(0, "Liquidity_rank", np.arange(1, len(out) + 1))
    out["Primary_asset_by_ex_ante_top2"] = out["Liquidity_rank"] <= 2
    return out


def add_microstructure_signal_custom(
    df: pd.DataFrame,
    *,
    normalization_window_bars: int = BARS_PER_DAY,
    smoothing_bars: int = 3,
    pooling: Literal["max", "mean", "sum"] = "max",
) -> pd.DataFrame:
    """Construct a configurable activation signal for hyperparameter checks."""
    out = df.copy()
    for feature, output_column in [("log_volume", "zV"), ("abs_ret", "zR")]:
        rolling = out[feature].rolling(normalization_window_bars, min_periods=normalization_window_bars)
        mean = rolling.mean().shift(1)
        std = rolling.std(ddof=0).shift(1).replace(0.0, np.nan)
        out[output_column] = (out[feature] - mean) / std
    if pooling == "max":
        pooled = np.maximum(out["zV"], out["zR"])
    elif pooling == "mean":
        pooled = 0.5 * (out["zV"] + out["zR"])
    elif pooling == "sum":
        pooled = out["zV"] + out["zR"]
    else:
        raise ValueError(f"Unknown pooling rule: {pooling}")
    out["Z_raw_custom"] = pooled
    out["Z"] = pooled.rolling(int(smoothing_bars), min_periods=int(smoothing_bars)).mean()
    return out


def load_prepared_asset_custom_signal(
    path: str | Path,
    *,
    horizons: Iterable[int] = HORIZONS,
    normalization_window_bars: int = BARS_PER_DAY,
    smoothing_bars: int = 3,
    pooling: Literal["max", "mean", "sum"] = "max",
) -> pd.DataFrame:
    """Load OHLCV data and construct features with a configurable signal."""
    df = load_ohlcv(path)
    df = add_microstructure_signal_custom(
        df,
        normalization_window_bars=normalization_window_bars,
        smoothing_bars=smoothing_bars,
        pooling=pooling,
    )
    df = add_realized_variance(df, horizons=horizons)
    return add_forecasting_features(df, horizons=horizons)


def evaluate_hyperparameter_sensitivity(
    data_dir: str | Path,
    asset_files: list[AssetFile] = DEFAULT_OHLCV_FILES,
    *,
    normalization_hours: tuple[int, ...] = (12, 24, 48),
    smoothing_bars_values: tuple[int, ...] = (1, 3, 6),
    pooling_rules: tuple[str, ...] = ("max", "mean"),
    config: ForecastingRobustnessConfig = ForecastingRobustnessConfig(),
) -> pd.DataFrame:
    """Run HAR+Z sensitivity to normalization window, smoothing, and pooling."""
    data_dir = Path(data_dir)
    rows: list[dict] = []
    fconfig = ForecastingExperimentConfig(
        horizons=config.horizons,
        train_splits=config.train_splits,
        test_split=config.test_split,
        z_clip_quantiles=config.z_clip_quantiles,
        hac_lags=config.hac_lags,
    )
    for item in asset_files:
        path = data_dir / item.filename
        for hours in normalization_hours:
            window_bars = int(hours * 60 / 5)
            for smoothing_bars in smoothing_bars_values:
                for pooling in pooling_rules:
                    df = load_prepared_asset_custom_signal(
                        path,
                        horizons=config.horizons,
                        normalization_window_bars=window_bars,
                        smoothing_bars=smoothing_bars,
                        pooling=pooling,  # type: ignore[arg-type]
                    )
                    raw_train = _combine_splits(df, config.train_splits)
                    raw_test = split_sample(df, config.test_split)
                    z_bounds = _z_clip_bounds(raw_train, config.z_clip_quantiles)
                    summary_rows = []
                    for horizon in config.horizons:
                        train = _prepare_frame(raw_train, horizon, "HAR", include_z=True, z_clip_bounds=z_bounds)
                        test = _prepare_frame(raw_test, horizon, "HAR", include_z=True, z_clip_bounds=z_bounds)
                        test = _select_nonoverlap(test, horizon)
                        if len(train) < 100 or len(test) < 20:
                            continue
                        row, _ = _model_comparison_row(
                            asset=item.asset,
                            horizon=horizon,
                            family="HAR",
                            train=train,
                            test=test,
                            hac_lags=config.hac_lags,
                            return_detail=False,
                        )
                        summary_rows.append(row)
                    if not summary_rows:
                        del df
                        continue
                    summary = pd.DataFrame(summary_rows)
                    summary["Normalization_hours"] = hours
                    summary["Smoothing_bars"] = smoothing_bars
                    summary["Smoothing_minutes"] = int(smoothing_bars * 5)
                    summary["Pooling"] = pooling
                    rows.append(summary)
                    del df, summary
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    preferred_cols = [
        "Asset",
        "Horizon_bars",
        "Horizon",
        "Normalization_hours",
        "Smoothing_bars",
        "Smoothing_minutes",
        "Pooling",
        "MSE_improvement_pct",
        "QLIKE_improvement",
        "MSE_diff_t_NW",
        "MSE_diff_p_one_sided_NW",
        "QLIKE_diff_t_NW",
        "QLIKE_diff_p_one_sided_NW",
        "N_test",
    ]
    return out[preferred_cols].sort_values(["Asset", "Horizon_bars", "Pooling", "Normalization_hours", "Smoothing_bars"])


def _pooled_time_series(detail: pd.DataFrame, *, group_assets: set[str], horizon: str, loss: LossName) -> pd.Series:
    """Create time-indexed pooled normalized loss-difference series."""
    sub = detail[(detail["Asset"].isin(group_assets)) & (detail["Horizon"] == horizon)].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    if loss == "MSE":
        # Normalize by asset-horizon baseline loss so the pooled test is not
        # dominated by the most volatile contract.
        denom = sub.groupby(["Asset", "Horizon"], observed=True)["baseline_mse"].transform("mean")
        sub["norm_diff"] = sub["mse_diff"] / denom.replace(0.0, np.nan)
    else:
        sub["norm_diff"] = sub["qlike_diff"]
    sub["Datetime"] = pd.to_datetime(sub["Datetime"])
    # Average across assets at the same forecast timestamp to reduce the effect
    # of cross-asset contemporaneous dependence.
    return sub.groupby("Datetime", observed=True)["norm_diff"].mean().sort_index()


def compute_pooled_loss_tests(detail: pd.DataFrame, *, hac_lags: int | None = None) -> pd.DataFrame:
    """Compute pooled BTC/ETH, XRP/SOL, and all-asset loss-difference tests."""
    rows: list[dict] = []
    pools = {
        "Primary_BTC_ETH": {"BTC", "ETH"},
        "Robustness_XRP_SOL": {"XRP", "SOL"},
        "All_assets": {"BTC", "ETH", "XRP", "SOL"},
    }
    for pool_name, assets in pools.items():
        for horizon in sorted(detail["Horizon"].dropna().unique(), key=lambda x: {"1h": 1, "4h": 4, "12h": 12, "24h": 24}.get(str(x), 999)):
            sub = detail[(detail["Asset"].isin(assets)) & (detail["Horizon"] == horizon)]
            if sub.empty:
                continue
            for loss in ("MSE", "QLIKE"):
                ts = _pooled_time_series(detail, group_assets=assets, horizon=horizon, loss=loss)  # type: ignore[arg-type]
                test = _hac_mean_test(ts.to_numpy(dtype=float), max_lags=hac_lags)
                if loss == "MSE":
                    base = float(sub["baseline_mse"].mean())
                    aug = float(sub["augmented_mse"].mean())
                    raw_improvement = 100.0 * (base - aug) / base if base > 0 else np.nan
                    norm_imp = 100.0 * test["mean"]
                else:
                    base = float(sub["baseline_qlike"].mean())
                    aug = float(sub["augmented_qlike"].mean())
                    raw_improvement = base - aug
                    norm_imp = test["mean"]
                rows.append(
                    {
                        "Pool": pool_name,
                        "Assets": "+".join(sorted(assets)),
                        "Horizon": horizon,
                        "Loss": loss,
                        "N_origin_rows": int(len(sub)),
                        "N_time_points": int(test["N"]),
                        "Raw_pooled_improvement": raw_improvement,
                        "Time_avg_normalized_improvement": norm_imp,
                        "Mean_loss_diff_time_avg": test["mean"],
                        "NW_t": test["t"],
                        "NW_p_one_sided": test["p_one_sided"],
                        "NW_lags": test["max_lags"],
                    }
                )
    return pd.DataFrame(rows)


def compute_har_per_origin_details(
    data_dir: str | Path,
    asset_files: list[AssetFile] = DEFAULT_OHLCV_FILES,
    *,
    config: ForecastingRobustnessConfig = ForecastingRobustnessConfig(),
) -> pd.DataFrame:
    """Compute HAR vs HAR+Z per-origin loss details for all assets."""
    data_dir = Path(data_dir)
    details: list[pd.DataFrame] = []
    fconfig = ForecastingExperimentConfig(
        horizons=config.horizons,
        train_splits=config.train_splits,
        test_split=config.test_split,
        z_clip_quantiles=config.z_clip_quantiles,
        hac_lags=config.hac_lags,
    )
    for item in asset_files:
        df = load_ohlcv(data_dir / item.filename)
        from src.empirical import add_microstructure_signal
        df = add_microstructure_signal(df)
        df = add_realized_variance(df, horizons=config.horizons)
        df = add_forecasting_features(df, horizons=config.horizons)
        raw_train = _combine_splits(df, config.train_splits)
        raw_test = split_sample(df, config.test_split)
        z_bounds = _z_clip_bounds(raw_train, config.z_clip_quantiles)
        for horizon in config.horizons:
            train = _prepare_frame(raw_train, horizon, "HAR", include_z=True, z_clip_bounds=z_bounds)
            test = _prepare_frame(raw_test, horizon, "HAR", include_z=True, z_clip_bounds=z_bounds)
            test = _select_nonoverlap(test, horizon)
            if len(train) < 100 or len(test) < 20:
                continue
            _, detail = _model_comparison_row(
                asset=item.asset,
                horizon=horizon,
                family="HAR",
                train=train,
                test=test,
                hac_lags=config.hac_lags,
                return_detail=True,
            )
            if not detail.empty:
                details.append(detail)
        del df
    return pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def _load_funding_file(path: str | Path) -> pd.DataFrame:
    funding = pd.read_csv(path)
    if "datetime" not in funding.columns or "last_funding_rate" not in funding.columns:
        raise ValueError(f"{path} must contain datetime and last_funding_rate columns.")
    funding = funding.copy()
    funding["funding_time"] = pd.to_datetime(funding["datetime"], errors="coerce")
    funding["last_funding_rate"] = pd.to_numeric(funding["last_funding_rate"], errors="coerce")
    funding = funding.dropna(subset=["funding_time", "last_funding_rate"]).sort_values("funding_time")
    funding = funding.drop_duplicates("funding_time").reset_index(drop=True)
    funding["funding_rate_bps"] = 10000.0 * funding["last_funding_rate"]
    funding["abs_funding_rate_bps"] = funding["funding_rate_bps"].abs()
    rolling = funding["abs_funding_rate_bps"].rolling(90, min_periods=20)
    mean = rolling.mean().shift(1)
    std = rolling.std(ddof=0).shift(1).replace(0.0, np.nan)
    funding["abs_funding_rate_z"] = (funding["abs_funding_rate_bps"] - mean) / std
    return funding[["funding_time", "last_funding_rate", "funding_rate_bps", "abs_funding_rate_bps", "abs_funding_rate_z"]]


def compute_funding_window_effect_size(
    data_dir: str | Path,
    funding_dir: str | Path,
    asset_files: list[AssetFile] = DEFAULT_OHLCV_FILES,
    funding_files: dict[str, str] = DEFAULT_FUNDING_FILES,
    *,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """Compute inside-vs-outside actual funding-window effect sizes for Z."""
    rows: list[dict] = []
    data_dir = Path(data_dir)
    funding_dir = Path(funding_dir)
    for item in asset_files:
        df = load_ohlcv(data_dir / item.filename)
        from src.empirical import add_microstructure_signal
        df = add_microstructure_signal(df)
        df = split_sample(df, "test")
        funding = _load_funding_file(funding_dir / funding_files[item.asset])
        nearest = pd.merge_asof(
            df[["Datetime"]].sort_values("Datetime"),
            funding[["funding_time"]].sort_values("funding_time"),
            left_on="Datetime",
            right_on="funding_time",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=int(window_minutes)),
            allow_exact_matches=True,
        )
        df = df.copy()
        df["actual_funding_window"] = nearest["funding_time"].notna().to_numpy()
        # Most recent observed funding z-score for correlation diagnostic.
        merged = pd.merge_asof(
            df.sort_values("Datetime"),
            funding.sort_values("funding_time"),
            left_on="Datetime",
            right_on="funding_time",
            direction="backward",
            allow_exact_matches=True,
        )
        z = merged["Z"].replace([np.inf, -np.inf], np.nan)
        inside = merged.loc[merged["actual_funding_window"], "Z"].dropna().to_numpy(dtype=float)
        outside = merged.loc[~merged["actual_funding_window"], "Z"].dropna().to_numpy(dtype=float)
        diff = float(np.mean(inside) - np.mean(outside))
        pooled_sd = float(np.sqrt(((len(inside) - 1) * np.var(inside, ddof=1) + (len(outside) - 1) * np.var(outside, ddof=1)) / max(1, len(inside) + len(outside) - 2)))
        d = diff / pooled_sd if pooled_sd > 0 else np.nan
        x = merged["Z"].where(merged["actual_funding_window"], np.nan)
        # Standard two-sample Welch statistics are sufficient for this diagnostic
        # table; the paper interprets the effect size as small but systematic,
        # rather than as a causal funding-window estimate.
        welch = stats.ttest_ind(inside, outside, equal_var=False, nan_policy="omit") if len(inside) > 2 and len(outside) > 2 else None
        corr = merged[["Z", "abs_funding_rate_z"]].replace([np.inf, -np.inf], np.nan).dropna().corr().iloc[0, 1]
        rows.append(
            {
                "Asset": item.asset,
                "Window_minutes": int(window_minutes),
                "N_inside": int(len(inside)),
                "N_outside": int(len(outside)),
                "Mean_Z_all": float(z.mean()),
                "Mean_Z_inside": float(np.mean(inside)),
                "Mean_Z_outside": float(np.mean(outside)),
                "Mean_Z_difference_inside_minus_outside": diff,
                "Cohen_d": d,
                "Welch_t": float(welch.statistic) if welch is not None else np.nan,
                "Welch_p_two_sided": float(welch.pvalue) if welch is not None else np.nan,
                "Corr_Z_abs_funding_rate_z": float(corr),
            }
        )
    return pd.DataFrame(rows).sort_values("Asset")


def _forecast_validation_and_test_for_margin(
    df: pd.DataFrame,
    asset: str,
    horizon: int,
    config: ForecastingRobustnessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return validation and test per-origin forecasts for HAR and HAR+Z."""
    raw_train = split_sample(df, "train")
    raw_validation = split_sample(df, "validation")
    raw_final_train = _combine_splits(df, config.train_splits)
    raw_test = split_sample(df, config.test_split)
    z_bounds = _z_clip_bounds(raw_final_train, config.z_clip_quantiles)

    def make_detail(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
        train = _prepare_frame(train_raw, horizon, "HAR", include_z=True, z_clip_bounds=z_bounds)
        eval_frame = _prepare_frame(eval_raw, horizon, "HAR", include_z=True, z_clip_bounds=z_bounds)
        eval_frame = _select_nonoverlap(eval_frame, horizon)
        base = _fit_predict_log_smear(train, eval_frame, "HAR", horizon, include_z=False)
        aug = _fit_predict_log_smear(train, eval_frame, "HAR", horizon, include_z=True)
        y = eval_frame["rv_future"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "Asset": asset,
                "Split": split_name,
                "Horizon_bars": horizon,
                "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                "Datetime": eval_frame["Datetime"].to_numpy(),
                "rv_future": y,
                "HAR_forecast": base.forecast,
                "HARZ_forecast": aug.forecast,
            }
        )

    return make_detail(raw_train, raw_validation, "validation"), make_detail(raw_final_train, raw_test, "test")


def compute_margin_capital_efficiency(
    data_dir: str | Path,
    asset_files: list[AssetFile] = DEFAULT_OHLCV_FILES,
    *,
    config: ForecastingRobustnessConfig = ForecastingRobustnessConfig(),
) -> pd.DataFrame:
    """Margin-style diagnostic: capital required at matched validation coverage.

    For each asset/horizon/model, choose a multiplier on the validation period so
    that ``rv_future <= multiplier * forecast`` at the target rate.  Apply that
    multiplier once to the 2025 test period and compare realized exceedance and
    average margin.  This is not a trading strategy; it is a risk-management
    capital-efficiency diagnostic.
    """
    rows: list[dict] = []
    data_dir = Path(data_dir)
    for item in asset_files:
        df = load_ohlcv(data_dir / item.filename)
        from src.empirical import add_microstructure_signal
        df = add_microstructure_signal(df)
        df = add_realized_variance(df, horizons=config.horizons)
        df = add_forecasting_features(df, horizons=config.horizons)
        for horizon in config.horizons:
            val, test = _forecast_validation_and_test_for_margin(df, item.asset, horizon, config)
            if val.empty or test.empty:
                continue
            for alpha in config.margin_target_exceedance_rates:
                model_rows = []
                for model_name, col in [("HAR", "HAR_forecast"), ("HAR+Z", "HARZ_forecast")]:
                    ratio_val = np.maximum(val["rv_future"].to_numpy(dtype=float), RV_EPS) / np.maximum(val[col].to_numpy(dtype=float), RV_EPS)
                    multiplier = float(np.quantile(ratio_val[np.isfinite(ratio_val)], 1.0 - alpha))
                    margin = multiplier * np.maximum(test[col].to_numpy(dtype=float), RV_EPS)
                    rv = np.maximum(test["rv_future"].to_numpy(dtype=float), RV_EPS)
                    exceed = rv > margin
                    row = {
                        "Asset": item.asset,
                        "Horizon_bars": horizon,
                        "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                        "Target_exceedance_rate": float(alpha),
                        "Model": model_name,
                        "Validation_multiplier": multiplier,
                        "Test_exceedance_rate": float(np.mean(exceed)),
                        "Test_coverage_rate": float(1.0 - np.mean(exceed)),
                        "Avg_margin": float(np.mean(margin)),
                        "Median_margin": float(np.median(margin)),
                        "Avg_margin_over_mean_RV": float(np.mean(margin) / np.mean(rv)),
                        "N_test": int(len(test)),
                    }
                    rows.append(row)
                    model_rows.append(row)
                if len(model_rows) == 2:
                    har = model_rows[0]
                    harz = model_rows[1]
                    rows.append(
                        {
                            "Asset": item.asset,
                            "Horizon_bars": horizon,
                            "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                            "Target_exceedance_rate": float(alpha),
                            "Model": "HAR+Z vs HAR",
                            "Validation_multiplier": np.nan,
                            "Test_exceedance_rate": harz["Test_exceedance_rate"] - har["Test_exceedance_rate"],
                            "Test_coverage_rate": harz["Test_coverage_rate"] - har["Test_coverage_rate"],
                            "Avg_margin": harz["Avg_margin"] - har["Avg_margin"],
                            "Median_margin": harz["Median_margin"] - har["Median_margin"],
                            "Avg_margin_over_mean_RV": harz["Avg_margin_over_mean_RV"] - har["Avg_margin_over_mean_RV"],
                            "Avg_margin_reduction_pct_vs_HAR": 100.0 * (har["Avg_margin"] - harz["Avg_margin"]) / har["Avg_margin"] if har["Avg_margin"] > 0 else np.nan,
                            "N_test": har["N_test"],
                        }
                    )
    return pd.DataFrame(rows)
