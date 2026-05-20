"""Actual funding-rate diagnostics for the volatility-forecasting paper.

This module uses actual Binance funding-rate CSV files and merges them
causally into the 5-minute OHLCV feature frame.  It is intentionally separate
from ``volatility_forecasting_fast.py`` so the timeout-safe benchmark path remains lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.empirical import BARS_PER_DAY, HORIZON_LABELS, split_sample
from src.volatility_forecasting_fast import (
    _combine_splits,
    _hac_mean_test,
    _loss_arrays,
    _safe_log,
    _select_nonoverlap,
    _z_clip_bounds,
)
from src.oos import HORIZONS, RV_EPS

FundingModel = Literal["HAR", "HAR+F", "HAR+Z", "HAR+Z+F", "HAR+Z+F+window"]


@dataclass(frozen=True)
class FundingConfig:
    """Configuration for actual funding-rate diagnostics."""

    horizons: tuple[int, ...] = HORIZONS
    train_splits: tuple[str, ...] = ("train", "validation")
    test_split: str = "test"
    z_clip_quantiles: tuple[float, float] | None = (0.005, 0.995)
    funding_history_events: int = 90
    funding_window_minutes: int = 60
    hac_lags: int | None = None


FUNDING_FEATURES = ["abs_funding_rate_z", "funding_rate_change_bps"]
WINDOW_FEATURE = "actual_funding_window"


def load_funding_rates(path: str | Path, history_events: int = 90) -> pd.DataFrame:
    """Load funding-rate data and construct causal event-level controls.

    Expected columns are ``datetime``, ``calc_time``, ``funding_interval_hours``,
    and ``last_funding_rate``.  The timestamp is the funding-settlement timestamp.
    Rolling standardization is shifted by one funding event, so the z-score at
    event k uses only events strictly before k as its normalization reference.
    """
    path = Path(path)
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
    funding["funding_rate_change_bps"] = funding["funding_rate_bps"].diff()

    rolling_abs = funding["abs_funding_rate_bps"].rolling(history_events, min_periods=max(20, history_events // 4))
    abs_mean = rolling_abs.mean().shift(1)
    abs_std = rolling_abs.std(ddof=0).shift(1).replace(0.0, np.nan)
    funding["abs_funding_rate_z"] = (funding["abs_funding_rate_bps"] - abs_mean) / abs_std

    rolling_change = funding["funding_rate_change_bps"].rolling(history_events, min_periods=max(20, history_events // 4))
    change_mean = rolling_change.mean().shift(1)
    change_std = rolling_change.std(ddof=0).shift(1).replace(0.0, np.nan)
    funding["funding_rate_change_z"] = (funding["funding_rate_change_bps"] - change_mean) / change_std

    return funding[
        [
            "funding_time",
            "last_funding_rate",
            "funding_rate_bps",
            "abs_funding_rate_bps",
            "funding_rate_change_bps",
            "abs_funding_rate_z",
            "funding_rate_change_z",
            "funding_interval_hours",
        ]
        if "funding_interval_hours" in funding.columns
        else [
            "funding_time",
            "last_funding_rate",
            "funding_rate_bps",
            "abs_funding_rate_bps",
            "funding_rate_change_bps",
            "abs_funding_rate_z",
            "funding_rate_change_z",
        ]
    ].copy()


def merge_actual_funding_features(
    df: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    funding_window_minutes: int = 60,
) -> pd.DataFrame:
    """Merge actual funding-rate controls into the 5-minute OHLCV frame.

    Funding-rate values are joined with ``merge_asof(..., direction='backward')``.
    Thus, at a forecast origin, the model only sees the most recent observed
    funding settlement and its already-realized funding rate.  The window flag
    uses the known settlement calendar and measures proximity to the nearest
    actual funding timestamp within ``funding_window_minutes``.
    """
    out = df.copy().sort_values("Datetime")
    funding = funding.copy().sort_values("funding_time")

    out = pd.merge_asof(
        out,
        funding,
        left_on="Datetime",
        right_on="funding_time",
        direction="backward",
        allow_exact_matches=True,
    )

    nearest = pd.merge_asof(
        df[["Datetime"]].sort_values("Datetime"),
        funding[["funding_time"]].sort_values("funding_time"),
        left_on="Datetime",
        right_on="funding_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=int(funding_window_minutes)),
        allow_exact_matches=True,
    )
    distance = (nearest["Datetime"] - nearest["funding_time"]).abs() / pd.Timedelta(minutes=1)
    out["minutes_to_actual_funding"] = distance.to_numpy(dtype=float)
    out["actual_funding_window"] = np.isfinite(out["minutes_to_actual_funding"].to_numpy(dtype=float))

    # A fallback raw level is useful in summaries even where early rolling z-scores
    # are unavailable.  Models below intentionally use the standardized pressure
    # plus funding-rate changes.
    for col in ["funding_rate_change_bps", "abs_funding_rate_z", "funding_rate_change_z"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _har_feature_specs(horizon: int) -> list[tuple[str, str]]:
    if horizon == BARS_PER_DAY:
        return [("log_rv_d", "rv_past_1d"), ("log_rv_w", "rv_past_1w"), ("log_rv_m", "rv_past_1m")]
    return [
        ("log_rv_h", f"rv_past_{horizon}"),
        ("log_rv_d", "rv_past_1d"),
        ("log_rv_w", "rv_past_1w"),
        ("log_rv_m", "rv_past_1m"),
    ]


def _model_feature_names(model: FundingModel, horizon: int) -> list[str]:
    names = [name for name, _ in _har_feature_specs(horizon)]
    if "+Z" in model:
        names = ["Z", *names]
    if "+F" in model:
        names = [*names, *FUNDING_FEATURES]
    if "window" in model:
        names = [*names, WINDOW_FEATURE]
    return names


def _prepare_funding_frame(
    df: pd.DataFrame,
    horizon: int,
    *,
    z_clip_bounds: tuple[float, float] | None,
) -> pd.DataFrame:
    specs = _har_feature_specs(horizon)
    needed = [
        "Datetime",
        "Z",
        f"rv_future_{horizon}",
        *[source for _, source in specs],
        *FUNDING_FEATURES,
        WINDOW_FEATURE,
        "last_funding_rate",
        "funding_rate_bps",
        "abs_funding_rate_bps",
        "minutes_to_actual_funding",
    ]
    available = [c for c in needed if c in df.columns]
    tmp = df.loc[:, available].copy()
    tmp = tmp.rename(columns={f"rv_future_{horizon}": "rv_future"})
    for out_name, source in specs:
        tmp[out_name] = tmp[source]
        if source != out_name:
            tmp = tmp.drop(columns=[source])
    if z_clip_bounds is not None:
        lo, hi = z_clip_bounds
        tmp["Z"] = tmp["Z"].clip(lower=lo, upper=hi)
    tmp[WINDOW_FEATURE] = tmp[WINDOW_FEATURE].astype(float)
    tmp = tmp.replace([np.inf, -np.inf], np.nan)

    # Do not drop rows just because ``minutes_to_actual_funding`` is NaN.
    # That column is NaN by design outside the +/- funding-window tolerance.
    # Dropping it would silently restrict all forecast origins to settlement
    # windows and make the window dummy constant.
    required_for_model = [
        "Datetime",
        "rv_future",
        "Z",
        *[name for name, _ in specs],
        *FUNDING_FEATURES,
        WINDOW_FEATURE,
        "last_funding_rate",
        "funding_rate_bps",
        "abs_funding_rate_bps",
    ]
    required_for_model = [c for c in required_for_model if c in tmp.columns]
    tmp = tmp.dropna(subset=required_for_model)
    tmp = tmp[tmp["rv_future"] > 0]
    return tmp.sort_values("Datetime").reset_index(drop=True)


def _design_matrix(frame: pd.DataFrame, model: FundingModel, horizon: int) -> np.ndarray:
    cols = []
    for name in _model_feature_names(model, horizon):
        if name.startswith("log_rv"):
            cols.append(_safe_log(frame[name]))
        else:
            cols.append(frame[name].to_numpy(dtype=float))
    if cols:
        return np.column_stack([np.ones(len(frame)), np.column_stack(cols)])
    return np.ones((len(frame), 1))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: FundingModel, horizon: int) -> tuple[np.ndarray, float]:
    X_train = _design_matrix(train, model, horizon)
    y_train = _safe_log(train["rv_future"])
    X_test = _design_matrix(test, model, horizon)
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
    return forecast, float(r2)


def evaluate_actual_funding_controls(
    df: pd.DataFrame,
    asset: str,
    config: FundingConfig = FundingConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate HAR, HAR+F, HAR+Z, HAR+Z+F, and HAR+Z+F+window."""
    raw_train = _combine_splits(df, config.train_splits)
    raw_test = split_sample(df, config.test_split)
    z_bounds = _z_clip_bounds(raw_train, config.z_clip_quantiles)
    rows: list[dict] = []
    details: list[pd.DataFrame] = []
    models: tuple[FundingModel, ...] = ("HAR", "HAR+F", "HAR+Z", "HAR+Z+F", "HAR+Z+F+window")

    for horizon in config.horizons:
        train = _prepare_funding_frame(raw_train, horizon, z_clip_bounds=z_bounds)
        test = _prepare_funding_frame(raw_test, horizon, z_clip_bounds=z_bounds)
        test = _select_nonoverlap(test, horizon)
        if len(train) < 100 or len(test) < 20:
            continue

        forecasts: dict[str, np.ndarray] = {}
        r2s: dict[str, float] = {}
        losses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        y = test["rv_future"].to_numpy(dtype=float)
        for model in models:
            forecast, r2 = _fit_predict(train, test, model, horizon)
            forecasts[model] = forecast
            r2s[model] = r2
            losses[model] = _loss_arrays(y, forecast)

        base_mse, base_q = losses["HAR"]
        harf_mse, harf_q = losses["HAR+F"]
        harz_mse, harz_q = losses["HAR+Z"]

        for model in models:
            mse, q = losses[model]
            mse_diff_har = base_mse - mse
            q_diff_har = base_q - q
            mse_hac = _hac_mean_test(mse_diff_har, max_lags=config.hac_lags)
            q_hac = _hac_mean_test(q_diff_har, max_lags=config.hac_lags)
            row = {
                "Asset": asset,
                "Horizon_bars": horizon,
                "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                "Model": model,
                "N_train": int(len(train)),
                "N_test": int(len(test)),
                "Train_R2": r2s[model],
                "Baseline_HAR_MSE": float(np.mean(base_mse)),
                "Model_MSE": float(np.mean(mse)),
                "MSE_improvement_vs_HAR_pct": 100.0 * (np.mean(base_mse) - np.mean(mse)) / np.mean(base_mse),
                "Baseline_HAR_QLIKE": float(np.mean(base_q)),
                "Model_QLIKE": float(np.mean(q)),
                "QLIKE_improvement_vs_HAR": float(np.mean(base_q) - np.mean(q)),
                "MSE_diff_vs_HAR_mean": mse_hac["mean"],
                "MSE_diff_vs_HAR_t_NW": mse_hac["t"],
                "MSE_diff_vs_HAR_p_one_sided_NW": mse_hac["p_one_sided"],
                "QLIKE_diff_vs_HAR_mean": q_hac["mean"],
                "QLIKE_diff_vs_HAR_t_NW": q_hac["t"],
                "QLIKE_diff_vs_HAR_p_one_sided_NW": q_hac["p_one_sided"],
                "NW_lags": mse_hac["max_lags"],
            }
            if model in {"HAR+Z+F", "HAR+Z+F+window"}:
                row["MSE_improvement_vs_HARF_pct"] = 100.0 * (np.mean(harf_mse) - np.mean(mse)) / np.mean(harf_mse)
                row["QLIKE_improvement_vs_HARF"] = float(np.mean(harf_q) - np.mean(q))
                row["MSE_improvement_vs_HARZ_pct"] = 100.0 * (np.mean(harz_mse) - np.mean(mse)) / np.mean(harz_mse)
                row["QLIKE_improvement_vs_HARZ"] = float(np.mean(harz_q) - np.mean(q))
            rows.append(row)

        details.append(
            pd.DataFrame(
                {
                    "Asset": asset,
                    "Horizon_bars": horizon,
                    "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                    "Datetime": test["Datetime"].to_numpy(),
                    "rv_future": y,
                    "Z": test["Z"].to_numpy(dtype=float),
                    "last_funding_rate": test["last_funding_rate"].to_numpy(dtype=float),
                    "funding_rate_bps": test["funding_rate_bps"].to_numpy(dtype=float),
                    "abs_funding_rate_bps": test["abs_funding_rate_bps"].to_numpy(dtype=float),
                    "abs_funding_rate_z": test["abs_funding_rate_z"].to_numpy(dtype=float),
                    "funding_rate_change_bps": test["funding_rate_change_bps"].to_numpy(dtype=float),
                    "actual_funding_window": test[WINDOW_FEATURE].to_numpy(dtype=bool),
                    "minutes_to_actual_funding": test["minutes_to_actual_funding"].to_numpy(dtype=float),
                    "har_forecast": forecasts["HAR"],
                    "har_z_forecast": forecasts["HAR+Z"],
                    "har_f_forecast": forecasts["HAR+F"],
                    "har_z_f_forecast": forecasts["HAR+Z+F"],
                    "har_z_f_window_forecast": forecasts["HAR+Z+F+window"],
                    "har_mse": base_mse,
                    "har_z_mse": losses["HAR+Z"][0],
                    "har_f_mse": losses["HAR+F"][0],
                    "har_z_f_mse": losses["HAR+Z+F"][0],
                    "har_z_f_window_mse": losses["HAR+Z+F+window"][0],
                    "har_qlike": base_q,
                    "har_z_qlike": losses["HAR+Z"][1],
                    "har_f_qlike": losses["HAR+F"][1],
                    "har_z_f_qlike": losses["HAR+Z+F"][1],
                    "har_z_f_window_qlike": losses["HAR+Z+F+window"][1],
                }
            )
        )

    return pd.DataFrame(rows), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def summarize_actual_funding_activation(df: pd.DataFrame, asset: str, config: FundingConfig = FundingConfig()) -> dict:
    """Summarize actual funding-rate correlation and funding-window activation."""
    test = split_sample(df, config.test_split).replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Z", "abs_funding_rate_z", "last_funding_rate", WINDOW_FEATURE]
    )
    if test.empty:
        return {"Asset": asset, "N_test_bars": 0}
    in_window = test[WINDOW_FEATURE].astype(bool)
    corr_absz = float(test[["Z", "abs_funding_rate_z"]].corr().iloc[0, 1])
    corr_abs_level = float(test[["Z", "abs_funding_rate_bps"]].corr().iloc[0, 1]) if "abs_funding_rate_bps" in test else np.nan
    return {
        "Asset": asset,
        "N_test_bars": int(len(test)),
        "Funding_window_minutes": int(config.funding_window_minutes),
        "Corr_Z_abs_funding_rate_z": corr_absz,
        "Corr_Z_abs_funding_rate_bps": corr_abs_level,
        "Mean_Z_all": float(test["Z"].mean()),
        "Mean_Z_in_actual_funding_window": float(test.loc[in_window, "Z"].mean()) if in_window.any() else np.nan,
        "Mean_Z_outside_actual_funding_window": float(test.loc[~in_window, "Z"].mean()) if (~in_window).any() else np.nan,
        "Funding_window_bar_share": float(in_window.mean()),
        "Mean_abs_funding_bps_all": float(test["abs_funding_rate_bps"].mean()),
        "Mean_abs_funding_bps_in_window": float(test.loc[in_window, "abs_funding_rate_bps"].mean()) if in_window.any() else np.nan,
    }


def summarize_actual_funding_window_losses(detail: pd.DataFrame, model_col_prefix: str = "har_z") -> pd.DataFrame:
    """Summarize HAR vs HAR+Z loss improvements inside/outside actual funding windows."""
    rows: list[dict] = []
    for keys, sub in detail.groupby(["Asset", "Horizon_bars", "Horizon", "actual_funding_window"], dropna=False):
        asset, horizon_bars, horizon, in_window = keys
        base_mse = float(sub["har_mse"].mean())
        aug_mse = float(sub[f"{model_col_prefix}_mse"].mean())
        base_q = float(sub["har_qlike"].mean())
        aug_q = float(sub[f"{model_col_prefix}_qlike"].mean())
        rows.append(
            {
                "Asset": asset,
                "Horizon_bars": int(horizon_bars),
                "Horizon": horizon,
                "Model": model_col_prefix.upper().replace("_", "+"),
                "Actual_funding_window": bool(in_window),
                "N": int(len(sub)),
                "Mean_Z": float(sub["Z"].mean()),
                "Mean_abs_funding_bps": float(sub["abs_funding_rate_bps"].mean()),
                "Mean_future_RV": float(sub["rv_future"].mean()),
                "MSE_improvement_vs_HAR_pct": 100.0 * (base_mse - aug_mse) / base_mse if base_mse > 0 else np.nan,
                "QLIKE_improvement_vs_HAR": base_q - aug_q,
            }
        )
    return pd.DataFrame(rows).sort_values(["Asset", "Horizon_bars", "Actual_funding_window"])
