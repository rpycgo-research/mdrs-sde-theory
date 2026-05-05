"""QF-style HAR/log-HAR out-of-sample benchmark utilities.

The routines in this module compare HAR-style realized-volatility forecasts
against HAR+Z forecasts, where Z is the past-adapted microstructure activation
signal used in the MDRS-SDE paper.  The default specification uses a positive
log-HAR forecast with smearing correction, which avoids negative realized-
variance forecasts and makes QLIKE loss well defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.empirical import BARS_PER_DAY, HORIZON_LABELS, split_sample

HORIZONS: tuple[int, ...] = (12, 48, 144, 288)
RV_EPS = 1e-12

LossName = Literal["mse", "qlike"]
ModelSpec = Literal["log_smear", "log_no_smear", "level_floor"]


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for a HAR/log-HAR OOS comparison."""

    horizons: tuple[int, ...] = HORIZONS
    train_splits: tuple[str, ...] = ("train", "validation")
    test_split: str = "test"
    model_spec: ModelSpec = "log_smear"
    z_clip_quantiles: tuple[float, float] | None = (0.005, 0.995)
    nonoverlap: bool = True
    bootstrap_replications: int = 1000
    bootstrap_block_length: int = 24
    bootstrap_seed: int = 42
    run_tests: bool = True


def add_har_realized_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily, weekly, and monthly realized-variance features once."""
    df = df.copy()
    r2 = df["ret"] ** 2
    df["rv_past_1d"] = df.get("rv_past_288", r2.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY).sum())
    df["rv_past_1w"] = r2.rolling(7 * BARS_PER_DAY, min_periods=7 * BARS_PER_DAY).sum()
    df["rv_past_1m"] = r2.rolling(30 * BARS_PER_DAY, min_periods=30 * BARS_PER_DAY).sum()

    return df


def _safe_log(series: pd.Series | np.ndarray, eps: float = RV_EPS) -> np.ndarray:
    """Log-transform with a positive floor."""
    values = np.asarray(series, dtype=float)

    return np.log(np.maximum(values, eps))


def _combine_splits(df: pd.DataFrame, splits: Iterable[str]) -> pd.DataFrame:
    """Concatenate named sample splits in chronological order."""
    parts = [split_sample(df, split) for split in splits]
    out = pd.concat(parts, axis=0).sort_values("Datetime")

    return out.drop_duplicates("Datetime")


def _prepare_har_frame(
    df: pd.DataFrame,
    horizon: int,
    *,
    z_clip_bounds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Create a clean frame containing target and HAR predictors."""
    columns = [
        "Datetime",
        "Z",
        f"rv_past_{horizon}",
        "rv_past_1d",
        "rv_past_1w",
        "rv_past_1m",
        f"rv_future_{horizon}",
    ]
    tmp = df[columns].copy()
    tmp = tmp.rename(
        columns={
            f"rv_past_{horizon}": "rv_h",
            "rv_past_1d": "rv_d",
            "rv_past_1w": "rv_w",
            "rv_past_1m": "rv_m",
            f"rv_future_{horizon}": "rv_future",
        }
    )

    if z_clip_bounds is not None:
        lower, upper = z_clip_bounds
        tmp["Z"] = tmp["Z"].clip(lower=lower, upper=upper)

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    tmp = tmp[tmp["rv_future"] > 0]

    return tmp


def _feature_columns(horizon: int, *, include_z: bool) -> list[str]:
    """Return HAR predictor columns, avoiding exact duplicates."""
    cols = ["rv_h", "rv_d", "rv_w", "rv_m"]
    if horizon == BARS_PER_DAY:
        # rv_h and rv_d are both 24-hour realized variance.
        cols = ["rv_d", "rv_w", "rv_m"]
    if include_z:
        cols = ["Z", *cols]

    return cols


def _design_matrix(
    frame: pd.DataFrame,
    horizon: int,
    *,
    include_z: bool,
    model_spec: ModelSpec,
) -> pd.DataFrame:
    """Build the regression design matrix for the requested specification."""
    cols = _feature_columns(horizon, include_z=include_z)
    X = frame[cols].copy()

    if model_spec in {"log_smear", "log_no_smear"}:
        for col in cols:
            if col != "Z":
                X[col] = _safe_log(X[col])

    return sm.add_constant(X, has_constant="add")


def _target(frame: pd.DataFrame, *, model_spec: ModelSpec) -> np.ndarray:
    """Return regression target for the requested model specification."""
    if model_spec in {"log_smear", "log_no_smear"}:
        return _safe_log(frame["rv_future"])

    return frame["rv_future"].to_numpy(dtype=float)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    *,
    include_z: bool,
    model_spec: ModelSpec,
) -> tuple[np.ndarray, float, float]:
    """Fit one model and return positive forecasts, train R2, and smearing.

    This uses NumPy least squares rather than statsmodels for speed because the
    OOS benchmark repeatedly fits small linear models to long 5-minute samples.
    HAC inference is applied later only to the forecast loss differentials.
    """
    X_train_df = _design_matrix(train, horizon, include_z=include_z, model_spec=model_spec)
    y_train = _target(train, model_spec=model_spec)
    X_test_df = _design_matrix(test, horizon, include_z=include_z, model_spec=model_spec)

    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    mask = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    xtx = X_train.T @ X_train
    xty = X_train.T @ y_train
    ridge = 1e-12 * np.eye(xtx.shape[0])
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(xtx + ridge, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx + ridge) @ xty
    fitted = X_train @ beta
    pred = X_test @ beta
    resid = y_train - fitted

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if model_spec == "log_smear":
        smear = float(np.mean(np.exp(resid))) if len(resid) else 1.0
        forecast = np.exp(pred) * smear
    elif model_spec == "log_no_smear":
        smear = 1.0
        forecast = np.exp(pred)
    else:
        smear = np.nan
        floor = max(float(np.nanpercentile(train["rv_future"], 0.5)), RV_EPS)
        forecast = np.maximum(pred, floor)

    forecast = np.maximum(forecast, RV_EPS)

    return forecast, float(train_r2), smear


def _losses(y_true: np.ndarray, forecast: np.ndarray) -> dict[str, np.ndarray]:
    """Return MSE and QLIKE losses."""
    y = np.maximum(np.asarray(y_true, dtype=float), RV_EPS)
    f = np.maximum(np.asarray(forecast, dtype=float), RV_EPS)
    ratio = y / f

    return {
        "mse": (y - f) ** 2,
        "qlike": ratio - np.log(ratio) - 1.0,
    }


def _hac_mean_test(values: np.ndarray, *, maxlags: int = 0) -> dict[str, float]:
    """One-sided HAC t-test for positive mean."""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(x) < 3 or np.allclose(x.std(ddof=1), 0.0):
        return {"mean": float(np.mean(x)) if len(x) else np.nan, "t": np.nan, "p_one_sided": np.nan}

    X = np.ones((len(x), 1))
    result = sm.OLS(x, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    t_value = float(result.tvalues[0])
    p_two = float(result.pvalues[0])
    p_one = p_two / 2 if t_value > 0 else 1 - p_two / 2

    return {"mean": float(result.params[0]), "t": t_value, "p_one_sided": float(p_one)}


def _moving_block_bootstrap(
    values: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Moving-block bootstrap confidence interval and one-sided p-value."""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    n = len(x)

    if n_boot <= 0:
        return {
            "boot_mean": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_p_one_sided": np.nan,
            "boot_block_length": int(block_length),
            "boot_replications": int(n_boot),
        }
    if n == 0:
        return {
            "boot_mean": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_p_one_sided": np.nan,
            "boot_block_length": int(block_length),
            "boot_replications": int(n_boot),
        }

    block_length = max(1, min(int(block_length), n))
    starts = np.arange(0, n - block_length + 1)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    n_blocks = int(np.ceil(n / block_length))

    for boot_idx in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([x[s : s + block_length] for s in chosen])[:n]
        boot_means[boot_idx] = float(np.mean(sample))

    return {
        "boot_mean": float(np.mean(boot_means)),
        "boot_ci_low": float(np.quantile(boot_means, 0.025)),
        "boot_ci_high": float(np.quantile(boot_means, 0.975)),
        "boot_p_one_sided": float(np.mean(boot_means <= 0.0)),
        "boot_block_length": int(block_length),
        "boot_replications": int(n_boot),
    }


def _select_nonoverlap(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Select non-overlapping forecast origins within a sorted frame."""
    if frame.empty:
        return frame
    frame = frame.sort_values("Datetime").reset_index(drop=True)

    return frame.iloc[::horizon].copy()


def _z_clip_bounds(train: pd.DataFrame, quantiles: tuple[float, float] | None) -> tuple[float, float] | None:
    """Compute train-period Z winsorization bounds."""
    if quantiles is None:
        return None
    q_low, q_high = quantiles

    z = train["Z"].replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty:
        return None

    return float(z.quantile(q_low)), float(z.quantile(q_high))


def evaluate_static_oos(
    df: pd.DataFrame,
    asset: str,
    config: ForecastConfig = ForecastConfig(),
) -> list[dict]:
    """Evaluate static train-to-test HAR vs HAR+Z OOS forecasts."""
    raw_train = _combine_splits(df, config.train_splits)
    raw_test = split_sample(df, config.test_split)
    bounds = _z_clip_bounds(raw_train, config.z_clip_quantiles)

    rows: list[dict] = []
    for horizon in config.horizons:
        train = _prepare_har_frame(raw_train, horizon, z_clip_bounds=bounds)
        test = _prepare_har_frame(raw_test, horizon, z_clip_bounds=bounds)
        if config.nonoverlap:
            test = _select_nonoverlap(test, horizon)
        if len(train) < 50 or len(test) < 10:
            continue

        y = test["rv_future"].to_numpy(dtype=float)
        har_pred, har_train_r2, har_smear = _fit_predict(
            train, test, horizon, include_z=False, model_spec=config.model_spec
        )
        z_pred, z_train_r2, z_smear = _fit_predict(
            train, test, horizon, include_z=True, model_spec=config.model_spec
        )

        har_losses = _losses(y, har_pred)
        z_losses = _losses(y, z_pred)
        mse_har = float(np.mean(har_losses["mse"]))
        mse_z = float(np.mean(z_losses["mse"]))
        qlike_har = float(np.mean(har_losses["qlike"]))
        qlike_z = float(np.mean(z_losses["qlike"]))

        mse_diff = har_losses["mse"] - z_losses["mse"]
        qlike_diff = har_losses["qlike"] - z_losses["qlike"]
        cw_diff = (y - har_pred) ** 2 - ((y - z_pred) ** 2 - (har_pred - z_pred) ** 2)
        if config.run_tests:
            hac_lags = max(0, int(np.floor(np.sqrt(len(test)))))
            dm_mse = _hac_mean_test(mse_diff, maxlags=hac_lags)
            dm_qlike = _hac_mean_test(qlike_diff, maxlags=hac_lags)
            cw_mse = _hac_mean_test(cw_diff, maxlags=hac_lags)
            boot_mse = _moving_block_bootstrap(
                mse_diff,
                block_length=config.bootstrap_block_length,
                n_boot=config.bootstrap_replications,
                seed=config.bootstrap_seed + horizon,
            )
            boot_qlike = _moving_block_bootstrap(
                qlike_diff,
                block_length=config.bootstrap_block_length,
                n_boot=config.bootstrap_replications,
                seed=config.bootstrap_seed + 10_000 + horizon,
            )
        else:
            dm_mse = {"mean": float(np.mean(mse_diff)), "t": np.nan, "p_one_sided": np.nan}
            dm_qlike = {"mean": float(np.mean(qlike_diff)), "t": np.nan, "p_one_sided": np.nan}
            cw_mse = {"mean": float(np.mean(cw_diff)), "t": np.nan, "p_one_sided": np.nan}
            boot_mse = {
                "boot_mean": np.nan,
                "boot_ci_low": np.nan,
                "boot_ci_high": np.nan,
                "boot_p_one_sided": np.nan,
            }
            boot_qlike = {
                "boot_mean": np.nan,
                "boot_ci_low": np.nan,
                "boot_ci_high": np.nan,
                "boot_p_one_sided": np.nan,
            }

        rows.append(
            {
                "Asset": asset,
                "Horizon_bars": horizon,
                "Horizon": HORIZON_LABELS.get(horizon, f"{horizon} bars"),
                "ModelSpec": config.model_spec,
                "ZClip": "none" if config.z_clip_quantiles is None else f"{config.z_clip_quantiles[0]}-{config.z_clip_quantiles[1]}",
                "NonOverlap": bool(config.nonoverlap),
                "N_train": int(len(train)),
                "N_test": int(len(test)),
                "HAR_train_R2": har_train_r2,
                "HARZ_train_R2": z_train_r2,
                "Delta_train_R2": z_train_r2 - har_train_r2,
                "HAR_smear": har_smear,
                "HARZ_smear": z_smear,
                "HAR_MSE": mse_har,
                "HARZ_MSE": mse_z,
                "MSE_improvement": (mse_har - mse_z) / mse_har if mse_har > 0 else np.nan,
                "HAR_QLIKE": qlike_har,
                "HARZ_QLIKE": qlike_z,
                "QLIKE_improvement": qlike_har - qlike_z,
                "DM_MSE_mean_diff": dm_mse["mean"],
                "DM_MSE_t": dm_mse["t"],
                "DM_MSE_p_one_sided": dm_mse["p_one_sided"],
                "CW_MSE_mean_diff": cw_mse["mean"],
                "CW_MSE_t": cw_mse["t"],
                "CW_MSE_p_one_sided": cw_mse["p_one_sided"],
                "DM_QLIKE_mean_diff": dm_qlike["mean"],
                "DM_QLIKE_t": dm_qlike["t"],
                "DM_QLIKE_p_one_sided": dm_qlike["p_one_sided"],
                "Boot_MSE_mean": boot_mse["boot_mean"],
                "Boot_MSE_CI_low": boot_mse["boot_ci_low"],
                "Boot_MSE_CI_high": boot_mse["boot_ci_high"],
                "Boot_MSE_p_one_sided": boot_mse["boot_p_one_sided"],
                "Boot_QLIKE_mean": boot_qlike["boot_mean"],
                "Boot_QLIKE_CI_low": boot_qlike["boot_ci_low"],
                "Boot_QLIKE_CI_high": boot_qlike["boot_ci_high"],
                "Boot_QLIKE_p_one_sided": boot_qlike["boot_p_one_sided"],
            }
        )

    return rows


def evaluate_winsorization_robustness(
    df: pd.DataFrame,
    asset: str,
    quantile_grid: Iterable[tuple[float, float] | None] = (None, (0.005, 0.995), (0.01, 0.99), (0.025, 0.975)),
    base_config: ForecastConfig = ForecastConfig(),
) -> list[dict]:
    """Evaluate robustness to Z winsorization thresholds."""
    rows: list[dict] = []
    for quantiles in quantile_grid:
        config = ForecastConfig(
            horizons=base_config.horizons,
            train_splits=base_config.train_splits,
            test_split=base_config.test_split,
            model_spec=base_config.model_spec,
            z_clip_quantiles=quantiles,
            nonoverlap=base_config.nonoverlap,
            bootstrap_replications=0,
            bootstrap_block_length=base_config.bootstrap_block_length,
            bootstrap_seed=base_config.bootstrap_seed,
            run_tests=False,
        )
        rows.extend(evaluate_static_oos(df, asset, config))

    return rows


def evaluate_specification_robustness(
    df: pd.DataFrame,
    asset: str,
    specs: Iterable[ModelSpec] = ("log_smear", "log_no_smear", "level_floor"),
    base_config: ForecastConfig = ForecastConfig(),
) -> list[dict]:
    """Evaluate robustness to log-RV and positivity specifications."""
    rows: list[dict] = []
    for spec in specs:
        config = ForecastConfig(
            horizons=base_config.horizons,
            train_splits=base_config.train_splits,
            test_split=base_config.test_split,
            model_spec=spec,
            z_clip_quantiles=base_config.z_clip_quantiles,
            nonoverlap=base_config.nonoverlap,
            bootstrap_replications=0,
            bootstrap_block_length=base_config.bootstrap_block_length,
            bootstrap_seed=base_config.bootstrap_seed,
            run_tests=False,
        )
        rows.extend(evaluate_static_oos(df, asset, config))

    return rows
