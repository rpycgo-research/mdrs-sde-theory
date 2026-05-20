"""Fast volatility-forecasting diagnostics.

This module adds lightweight, commit-ready diagnostics for the perpetual-futures volatility-forecasting paper.  The design deliberately
uses non-overlapping OOS forecast origins and NumPy least-squares fits so the
full BTC/ETH experiment finishes in seconds rather than timing out in repeated
bootstrap/HAC loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.empirical import (
    BARS_PER_DAY,
    HORIZON_LABELS,
    add_microstructure_signal,
    add_realized_variance,
    load_ohlcv,
    split_sample,
)
from src.oos import HORIZONS, RV_EPS, add_har_realized_variance_features

ModelFamily = Literal["HAR", "HARQ", "RSV-HAR"]
LossName = Literal["MSE", "QLIKE"]


@dataclass(frozen=True)
class ForecastingExperimentConfig:
    """Configuration for fast volatility-forecasting diagnostics."""

    horizons: tuple[int, ...] = HORIZONS
    train_splits: tuple[str, ...] = ("train", "validation")
    selection_train_split: str = "train"
    selection_validation_split: str = "validation"
    test_split: str = "test"
    z_clip_quantiles: tuple[float, float] | None = (0.005, 0.995)
    selection_loss: LossName = "QLIKE"
    funding_window_minutes: int = 60
    hac_lags: int | None = None
    top_quantile: float = 0.90


@dataclass(frozen=True)
class FitResult:
    """Container for fitted forecasts and basic in-sample diagnostics."""

    forecast: np.ndarray
    train_r2: float
    smear: float


def _safe_log(values: pd.Series | np.ndarray, eps: float = RV_EPS) -> np.ndarray:
    """Numerically safe log transform."""
    arr = np.asarray(values, dtype=float)
    return np.log(np.maximum(arr, eps))


def _combine_splits(df: pd.DataFrame, splits: Iterable[str]) -> pd.DataFrame:
    """Concatenate named sample splits in chronological order."""
    parts = [split_sample(df, split) for split in splits]
    return pd.concat(parts, axis=0).sort_values("Datetime").drop_duplicates("Datetime")


def add_forecasting_features(df: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Add stronger HAR-family features used in the volatility-forecasting diagnostics.

    Features include realized semivariance and realized quarticity over the
    horizon-matched, daily, weekly, and monthly windows.  These are computable
    from the existing 5-minute OHLCV files and avoid any proprietary order-book
    or liquidation data dependency.
    """
    df = add_har_realized_variance_features(df).copy()
    r = pd.to_numeric(df["ret"], errors="coerce")
    r2 = r**2
    r4 = r**4
    neg_r2 = r2.where(r < 0, 0.0)
    pos_r2 = r2.where(r >= 0, 0.0)

    windows = sorted(set(tuple(horizons) + (BARS_PER_DAY, 7 * BARS_PER_DAY, 30 * BARS_PER_DAY)))
    for window in windows:
        minp = int(window)
        df[f"rsv_neg_past_{window}"] = neg_r2.rolling(window, min_periods=minp).sum()
        df[f"rsv_pos_past_{window}"] = pos_r2.rolling(window, min_periods=minp).sum()
        # Realized-quarticity scale.  The exact scale is less important than
        # using a consistent measurement-error proxy in the log-HAR comparison.
        df[f"rq_past_{window}"] = (window / 3.0) * r4.rolling(window, min_periods=minp).sum()

    alias_map = {
        "1d": BARS_PER_DAY,
        "1w": 7 * BARS_PER_DAY,
        "1m": 30 * BARS_PER_DAY,
    }
    for label, window in alias_map.items():
        df[f"rsv_neg_past_{label}"] = df[f"rsv_neg_past_{window}"]
        df[f"rsv_pos_past_{label}"] = df[f"rsv_pos_past_{window}"]
        df[f"rq_past_{label}"] = df[f"rq_past_{window}"]

    return df


def load_prepared_asset(path: str | Path, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Load OHLCV data and construct all features needed by fast diagnostics."""
    df = load_ohlcv(path)
    df = add_microstructure_signal(df)
    df = add_realized_variance(df, horizons=horizons)
    return add_forecasting_features(df, horizons=horizons)


def _z_clip_bounds(train: pd.DataFrame, quantiles: tuple[float, float] | None) -> tuple[float, float] | None:
    """Compute train-only winsorization bounds for Z."""
    if quantiles is None:
        return None
    z = train["Z"].replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty:
        return None
    return float(z.quantile(quantiles[0])), float(z.quantile(quantiles[1]))


def _feature_specs(family: ModelFamily, horizon: int, include_z: bool) -> list[tuple[str, str]]:
    """Return (output-name, source-column) pairs for one model family."""
    base: list[tuple[str, str]]
    if horizon == BARS_PER_DAY:
        base = [("log_rv_d", "rv_past_1d"), ("log_rv_w", "rv_past_1w"), ("log_rv_m", "rv_past_1m")]
    else:
        base = [
            ("log_rv_h", f"rv_past_{horizon}"),
            ("log_rv_d", "rv_past_1d"),
            ("log_rv_w", "rv_past_1w"),
            ("log_rv_m", "rv_past_1m"),
        ]

    if family == "HARQ":
        if horizon == BARS_PER_DAY:
            extra = [("log_rq_d", "rq_past_1d"), ("log_rq_w", "rq_past_1w"), ("log_rq_m", "rq_past_1m")]
        else:
            extra = [
                ("log_rq_h", f"rq_past_{horizon}"),
                ("log_rq_d", "rq_past_1d"),
                ("log_rq_w", "rq_past_1w"),
                ("log_rq_m", "rq_past_1m"),
            ]
        base = base + extra
    elif family == "RSV-HAR":
        if horizon == BARS_PER_DAY:
            windows = [("d", "1d"), ("w", "1w"), ("m", "1m")]
        else:
            windows = [("h", str(horizon)), ("d", "1d"), ("w", "1w"), ("m", "1m")]
        extra = []
        for short, suffix in windows:
            extra.append((f"log_rsv_neg_{short}", f"rsv_neg_past_{suffix}"))
            extra.append((f"log_rsv_pos_{short}", f"rsv_pos_past_{suffix}"))
        base = base + extra

    if include_z:
        return [("Z", "Z")] + base
    return base


def _prepare_frame(
    df: pd.DataFrame,
    horizon: int,
    family: ModelFamily,
    *,
    include_z: bool,
    z_clip_bounds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Create a clean modeling frame for one horizon/family."""
    specs = _feature_specs(family, horizon, include_z=include_z)
    needed = ["Datetime", f"rv_future_{horizon}"] + [source for _, source in specs]
    tmp = df.loc[:, needed].copy()
    tmp = tmp.rename(columns={f"rv_future_{horizon}": "rv_future"})

    for out_name, source in specs:
        if out_name != source:
            tmp[out_name] = tmp[source]
        if source != out_name and source in tmp.columns:
            tmp = tmp.drop(columns=[source])

    if include_z and z_clip_bounds is not None:
        lo, hi = z_clip_bounds
        tmp["Z"] = tmp["Z"].clip(lower=lo, upper=hi)

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    tmp = tmp[tmp["rv_future"] > 0]
    return tmp.sort_values("Datetime").reset_index(drop=True)


def _select_nonoverlap(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Select non-overlapping forecast origins."""
    if frame.empty:
        return frame
    return frame.sort_values("Datetime").reset_index(drop=True).iloc[::horizon].copy()


def _design_matrix(frame: pd.DataFrame, family: ModelFamily, horizon: int, include_z: bool) -> np.ndarray:
    """Build a dense NumPy design matrix for positive log-RV forecasts."""
    names = [name for name, _ in _feature_specs(family, horizon, include_z=include_z)]
    cols = []
    for name in names:
        if name == "Z":
            cols.append(frame[name].to_numpy(dtype=float))
        else:
            cols.append(_safe_log(frame[name]))
    if cols:
        X = np.column_stack(cols)
        return np.column_stack([np.ones(len(frame)), X])
    return np.ones((len(frame), 1))


def _fit_predict_log_smear(
    train: pd.DataFrame,
    test: pd.DataFrame,
    family: ModelFamily,
    horizon: int,
    *,
    include_z: bool,
) -> FitResult:
    """Fit log-RV least squares and return positive smearing-corrected forecasts."""
    X_train = _design_matrix(train, family, horizon, include_z)
    y_train = _safe_log(train["rv_future"])
    X_test = _design_matrix(test, family, horizon, include_z)

    mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]
    xtx = X_train.T @ X_train
    xty = X_train.T @ y_train
    ridge = 1e-10 * np.eye(xtx.shape[0])
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(xtx + ridge, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx + ridge) @ xty

    fitted = X_train @ beta
    resid = y_train - fitted
    pred = X_test @ beta
    smear = float(np.mean(np.exp(resid))) if len(resid) else 1.0
    forecast = np.maximum(np.exp(pred) * smear, RV_EPS)

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FitResult(forecast=forecast, train_r2=float(r2), smear=smear)


def _loss_arrays(y_true: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return MSE and QLIKE arrays."""
    y = np.maximum(np.asarray(y_true, dtype=float), RV_EPS)
    f = np.maximum(np.asarray(forecast, dtype=float), RV_EPS)
    ratio = y / f
    return (y - f) ** 2, ratio - np.log(ratio) - 1.0


def _hac_mean_test(values: np.ndarray, max_lags: int | None = None) -> dict[str, float]:
    """Fast Newey-West one-sided mean test for positive loss differences."""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": np.nan, "t": np.nan, "p_one_sided": np.nan, "se": np.nan, "max_lags": 0}
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
    var_mean = max(long_run / n, 0.0)
    se = float(np.sqrt(var_mean))
    if se <= 0:
        return {"mean": mean, "t": np.nan, "p_one_sided": np.nan, "se": se, "max_lags": max_lags}
    t_value = mean / se
    p_one = float(1.0 - stats.t.cdf(t_value, df=n - 1))
    return {"mean": mean, "t": float(t_value), "p_one_sided": p_one, "se": se, "max_lags": max_lags}


def _model_comparison_row(
    *,
    asset: str,
    horizon: int,
    family: ModelFamily,
    train: pd.DataFrame,
    test: pd.DataFrame,
    hac_lags: int | None,
    return_detail: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Fit baseline and +Z models, then return summary row and per-origin losses."""
    base = _fit_predict_log_smear(train, test, family, horizon, include_z=False)
    aug = _fit_predict_log_smear(train, test, family, horizon, include_z=True)
    y = test["rv_future"].to_numpy(dtype=float)
    base_mse, base_qlike = _loss_arrays(y, base.forecast)
    aug_mse, aug_qlike = _loss_arrays(y, aug.forecast)
    mse_diff = base_mse - aug_mse
    qlike_diff = base_qlike - aug_qlike

    mse_hac = _hac_mean_test(mse_diff, max_lags=hac_lags)
    qlike_hac = _hac_mean_test(qlike_diff, max_lags=hac_lags)
    mse_base = float(np.mean(base_mse))
    mse_aug = float(np.mean(aug_mse))
    qlike_base = float(np.mean(base_qlike))
    qlike_aug = float(np.mean(aug_qlike))

    row = {
        "Asset": asset,
        "Horizon_bars": horizon,
        "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
        "Family": family,
        "N_train": int(len(train)),
        "N_test": int(len(test)),
        "Baseline_train_R2": base.train_r2,
        "Augmented_train_R2": aug.train_r2,
        "Delta_train_R2": aug.train_r2 - base.train_r2,
        "Baseline_MSE": mse_base,
        "Augmented_MSE": mse_aug,
        "MSE_improvement_pct": 100.0 * (mse_base - mse_aug) / mse_base if mse_base > 0 else np.nan,
        "Baseline_QLIKE": qlike_base,
        "Augmented_QLIKE": qlike_aug,
        "QLIKE_improvement": qlike_base - qlike_aug,
        "MSE_diff_mean": mse_hac["mean"],
        "MSE_diff_t_NW": mse_hac["t"],
        "MSE_diff_p_one_sided_NW": mse_hac["p_one_sided"],
        "QLIKE_diff_mean": qlike_hac["mean"],
        "QLIKE_diff_t_NW": qlike_hac["t"],
        "QLIKE_diff_p_one_sided_NW": qlike_hac["p_one_sided"],
        "NW_lags": mse_hac["max_lags"],
    }

    if not return_detail:
        return row, pd.DataFrame()

    detail = pd.DataFrame(
        {
            "Asset": asset,
            "Horizon_bars": horizon,
            "Horizon": row["Horizon"],
            "Family": family,
            "Datetime": test["Datetime"].to_numpy(),
            "rv_future": y,
            "Z": test["Z"].to_numpy(dtype=float) if "Z" in test.columns else np.nan,
            "baseline_forecast": base.forecast,
            "augmented_forecast": aug.forecast,
            "baseline_mse": base_mse,
            "augmented_mse": aug_mse,
            "baseline_qlike": base_qlike,
            "augmented_qlike": aug_qlike,
            "mse_diff": mse_diff,
            "qlike_diff": qlike_diff,
        }
    )
    return row, detail


def evaluate_stronger_benchmarks(
    df: pd.DataFrame,
    asset: str,
    config: ForecastingExperimentConfig = ForecastingExperimentConfig(),
    detail_families: tuple[ModelFamily, ...] = ("HAR",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate HAR, HARQ, and RSV-HAR baselines with and without Z.

    Per-origin detail rows are retained only for ``detail_families`` so the
    default path stays memory- and timeout-safe.
    """
    raw_train = _combine_splits(df, config.train_splits)
    raw_test = split_sample(df, config.test_split)
    z_bounds = _z_clip_bounds(raw_train, config.z_clip_quantiles)
    rows: list[dict] = []
    details: list[pd.DataFrame] = []

    for horizon in config.horizons:
        for family in ("HAR", "HARQ", "RSV-HAR"):
            train = _prepare_frame(raw_train, horizon, family, include_z=True, z_clip_bounds=z_bounds)
            test = _prepare_frame(raw_test, horizon, family, include_z=True, z_clip_bounds=z_bounds)
            test = _select_nonoverlap(test, horizon)
            if len(train) < 100 or len(test) < 20:
                continue
            keep_detail = family in detail_families
            row, detail = _model_comparison_row(
                asset=asset,
                horizon=horizon,
                family=family,
                train=train,
                test=test,
                hac_lags=config.hac_lags,
                return_detail=keep_detail,
            )
            rows.append(row)
            if keep_detail and not detail.empty:
                details.append(detail)

    return pd.DataFrame(rows), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def evaluate_validation_selected_family(df: pd.DataFrame, asset: str, config: ForecastingExperimentConfig = ForecastingExperimentConfig()) -> pd.DataFrame:
    """Select the strongest HAR-family baseline on validation, then test +Z.

    Selection uses 2021--2023 training and 2024 validation only.  The selected
    family is then refit on 2021--2024 and evaluated once on the 2025 test set.
    """
    raw_select_train = split_sample(df, config.selection_train_split)
    raw_validation = split_sample(df, config.selection_validation_split)
    raw_final_train = _combine_splits(df, config.train_splits)
    raw_test = split_sample(df, config.test_split)
    z_bounds = _z_clip_bounds(raw_final_train, config.z_clip_quantiles)
    rows: list[dict] = []

    for horizon in config.horizons:
        validation_losses: list[tuple[float, ModelFamily]] = []
        for family in ("HAR", "HARQ", "RSV-HAR"):
            tr = _prepare_frame(raw_select_train, horizon, family, include_z=False)
            val = _prepare_frame(raw_validation, horizon, family, include_z=False)
            val = _select_nonoverlap(val, horizon)
            if len(tr) < 100 or len(val) < 20:
                continue
            fit = _fit_predict_log_smear(tr, val, family, horizon, include_z=False)
            y_val = val["rv_future"].to_numpy(dtype=float)
            mse, qlike = _loss_arrays(y_val, fit.forecast)
            loss = float(np.mean(qlike if config.selection_loss == "QLIKE" else mse))
            validation_losses.append((loss, family))
        if not validation_losses:
            continue
        selected_loss, selected_family = min(validation_losses, key=lambda x: x[0])
        train = _prepare_frame(raw_final_train, horizon, selected_family, include_z=True, z_clip_bounds=z_bounds)
        test = _prepare_frame(raw_test, horizon, selected_family, include_z=True, z_clip_bounds=z_bounds)
        test = _select_nonoverlap(test, horizon)
        row, _ = _model_comparison_row(asset=asset, horizon=horizon, family=selected_family, train=train, test=test, hac_lags=config.hac_lags)
        row["Selection_loss"] = config.selection_loss
        row["Validation_baseline_loss"] = selected_loss
        rows.append(row)

    return pd.DataFrame(rows)


def add_funding_window_flags(detail: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """Flag forecast origins near scheduled 00/08/16 UTC funding settlements."""
    out = detail.copy()
    dt = pd.to_datetime(out["Datetime"])
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    settlements = np.array([0, 8 * 60, 16 * 60, 24 * 60], dtype=int)
    minute_values = minute_of_day.to_numpy(dtype=int)
    distances = np.min(np.abs(minute_values[:, None] - settlements[None, :]), axis=1)
    distances = np.minimum(distances, 24 * 60 - distances)
    out["minutes_to_scheduled_funding"] = distances
    out["funding_window"] = distances <= int(window_minutes)
    return out


def summarize_funding_window(detail: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """Summarize loss improvements inside and outside scheduled funding windows."""
    flagged = add_funding_window_flags(detail, window_minutes=window_minutes)
    rows: list[dict] = []
    group_cols = ["Asset", "Horizon_bars", "Horizon", "Family", "funding_window"]
    for keys, sub in flagged.groupby(group_cols, dropna=False):
        asset, horizon_bars, horizon, family, in_window = keys
        base_mse = float(sub["baseline_mse"].mean())
        aug_mse = float(sub["augmented_mse"].mean())
        base_q = float(sub["baseline_qlike"].mean())
        aug_q = float(sub["augmented_qlike"].mean())
        rows.append(
            {
                "Asset": asset,
                "Horizon_bars": int(horizon_bars),
                "Horizon": horizon,
                "Family": family,
                "Window_minutes": int(window_minutes),
                "Funding_window": bool(in_window),
                "N": int(len(sub)),
                "Mean_Z": float(sub["Z"].mean()),
                "Mean_future_RV": float(sub["rv_future"].mean()),
                "MSE_improvement_pct": 100.0 * (base_mse - aug_mse) / base_mse if base_mse > 0 else np.nan,
                "QLIKE_improvement": base_q - aug_q,
                "MSE_diff_mean": float(sub["mse_diff"].mean()),
                "QLIKE_diff_mean": float(sub["qlike_diff"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["Asset", "Horizon_bars", "Family", "Funding_window"])


def summarize_high_volatility_ranking(detail: pd.DataFrame, top_quantile: float = 0.90) -> pd.DataFrame:
    """Summarize top-volatility ranking and underprediction diagnostics."""
    rows: list[dict] = []
    for keys, sub in detail.groupby(["Asset", "Horizon_bars", "Horizon", "Family"], dropna=False):
        asset, horizon_bars, horizon, family = keys
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["rv_future", "baseline_forecast", "augmented_forecast"])
        if len(sub) < 20:
            continue
        realized_cut = float(sub["rv_future"].quantile(top_quantile))
        realized_top = sub["rv_future"] >= realized_cut
        unconditional_mean = float(sub["rv_future"].mean())
        for label, forecast_col in [("Baseline", "baseline_forecast"), ("Augmented", "augmented_forecast")]:
            pred_cut = float(sub[forecast_col].quantile(top_quantile))
            pred_top = sub[forecast_col] >= pred_cut
            top_realized_mean = float(sub.loc[pred_top, "rv_future"].mean()) if pred_top.any() else np.nan
            recall = float((pred_top & realized_top).sum() / max(1, realized_top.sum()))
            under_ratio = np.maximum(sub["rv_future"].to_numpy(dtype=float) / np.maximum(sub[forecast_col].to_numpy(dtype=float), RV_EPS) - 1.0, 0.0)
            tail_under_ratio = under_ratio[realized_top.to_numpy()]
            rows.append(
                {
                    "Asset": asset,
                    "Horizon_bars": int(horizon_bars),
                    "Horizon": horizon,
                    "Family": family,
                    "Model": label,
                    "Top_quantile": float(top_quantile),
                    "N": int(len(sub)),
                    "Top_pred_realized_RV_over_all": top_realized_mean / unconditional_mean if unconditional_mean > 0 else np.nan,
                    "Recall_realized_top_decile": recall,
                    "Mean_underprediction_ratio": float(np.mean(under_ratio)),
                    "Tail_mean_underprediction_ratio": float(np.mean(tail_under_ratio)) if len(tail_under_ratio) else np.nan,
                }
            )
    return pd.DataFrame(rows)
