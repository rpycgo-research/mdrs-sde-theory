"""
Synthetic reproduction experiments for the MDRS-SDE paper.

  Experiment 1 — Figure 1: 4D trajectory with leaky extrema
  Experiment 2 — Figure 2: E[τ_r*] sensitivity to γ  (IC-Alpha resolution)
  Experiment 3 — Figure 3 + Tables 1-2: long-run stability and volatility identification

Scope note
----------
These experiments provide numerical illustrations and stability evidence. They do
not constitute a proof of total-variation geometric ergodicity or breakout-time
integrability.
"""
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.simulator import MDRSSimulator

RESULTS_DIR = Path("results")
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
FIGURE_DIR = SYNTHETIC_DIR / "figures"
TABLE_DIR = SYNTHETIC_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def batch_stat_se(x: np.ndarray, stat_fn: Callable[[np.ndarray], float], n_batches: int = 100) -> Tuple[float, float, int]:
    """Estimate a statistic and its SE using contiguous batch means.

    This is used instead of iid t-statistics because the simulated path is a
    serially dependent Markov trajectory. It also works for nonlinear statistics
    such as skewness and kurtosis.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, 0

    n_batches = max(2, min(n_batches, len(x)))
    batch_size = len(x) // n_batches
    trimmed = x[: batch_size * n_batches]

    vals = []
    for k in range(n_batches):
        chunk = trimmed[k * batch_size : (k + 1) * batch_size]
        vals.append(float(stat_fn(chunk)))
    vals = np.asarray(vals)

    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n_batches)), int(n_batches)


# ===================================================================
# Experiment 1: Trajectory with latched extrema  (Figure 1)
# ===================================================================
def experiment_1_trajectory(sim: MDRSSimulator) -> None:
    print("\n=== Experiment 1: Trajectory (Figure 1) ===")
    out = sim.run(num_paths=500, num_steps=500, dt=0.01, seed=42)

    # Select a path that clearly shows leaky-extrema tracking and freezing
    best, best_score = 0, -1.0
    for i in range(out["signal"].shape[1]):
        sig = out["signal"][:, i]
        t_above = np.sum(sig > sim.zeta)
        mx = np.max(sig)
        if t_above > 40 and mx > sim.gamma:
            sc = t_above + mx * 10
            if sc > best_score:
                best_score, best = sc, i
    print(f"  Selected path {best} (score {best_score:.1f})")

    t = np.arange(500) * 0.01
    p = out["price"][:, best]
    r = out["resistance"][:, best]
    s = out["support"][:, best]
    z = out["signal"][:, best]
    w = out["weight"][:, best]
    momentum_regime = z >= sim.zeta

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, p, "k", lw=1.2, label=r"Price ($p_t$)")
    axes[0].plot(t, r, "r--", label=r"Resistance ($R_t$)")
    axes[0].plot(t, s, "b--", label=r"Support ($S_t$)")
    axes[0].set_ylabel("Price Level")
    axes[0].set_title("MDRS-SDE Dynamics: Leaky Extrema during Breakouts")

    axes[1].plot(t, z, "purple", label=r"Signal ($Z_t$)")
    axes[1].axhline(sim.zeta, color="gray", ls=":", label=r"Quiet Threshold ($\zeta$)")
    axes[1].axhline(sim.gamma, color="orange", ls=":", label=r"Activation ($\gamma$)")
    axes[1].set_ylabel("Signal Z-Score")

    axes[2].plot(t, w, "green", label=r"Regime Weight ($w_t$)")
    axes[2].set_ylabel("Momentum Weight")
    axes[2].set_xlabel("Time (t)")

    for idx, ax in enumerate(axes):
        lo, hi = ax.get_ylim()
        lab = r"Frozen-extrema regime ($Z_t \geq \zeta$)" if idx == 0 else ""
        ax.fill_between(t, lo, hi, where=momentum_regime, color="gray", alpha=0.2, label=lab)
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1_trajectory.png")
    print("  Saved fig1_trajectory.png")


# ===================================================================
# Experiment 2: Sensitivity of finite-horizon breakout time to gamma
# ===================================================================
def experiment_2_sensitivity(sim: MDRSSimulator, num_paths: int = 2000) -> None:
    print("\n=== Experiment 2: Finite-Horizon Breakout Sensitivity (Figure 2) ===")
    print(f"  Using {num_paths} paths per gamma value")

    gammas = np.linspace(0.5, 3.5, 12)
    means, ses, censor_rates = [], [], []
    r_star = 0.1
    num_steps = 1000
    dt = 0.01

    for g in gammas:
        out = sim.run(num_paths=num_paths, num_steps=num_steps, dt=dt, gamma_override=float(g), seed=42)
        prices = out["price"]
        hit = np.abs(prices) >= r_star
        ever_hit = hit.any(axis=0)
        idx = np.argmax(hit, axis=0)
        idx[~ever_hit] = prices.shape[0] - 1
        tau = idx * dt
        m, se = tau.mean(), tau.std(ddof=1) / np.sqrt(len(tau))
        censor = 1.0 - ever_hit.mean()
        means.append(m)
        ses.append(se)
        censor_rates.append(censor)
        print(f"  gamma={g:.2f}  finite-horizon E[tau]={m:.4f}  SE={se:.4f}  censor={censor:.2%}")

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        gammas,
        means,
        yerr=1.96 * np.array(ses),
        marker="o",
        color="darkblue",
        lw=2,
        capsize=4,
        label=r"Finite-horizon $\mathbb{E}[\tau_{r^*}] \pm 1.96\,SE$",
    )
    plt.title("Finite-Horizon Breakout-Time Sensitivity")
    plt.xlabel(r"Activation Threshold ($\gamma$)")
    plt.ylabel(r"Estimated First-Passage Time")
    plt.grid(True, ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig2_sensitivity.png")
    print("  Saved fig2_sensitivity.png")

    np.savetxt(
        TABLE_DIR / "table_breakout_sensitivity.csv",
        np.column_stack([gammas, means, ses, censor_rates]),
        delimiter=",",
        header="gamma,finite_horizon_mean_tau,se,censor_rate",
        comments="",
    )


# ===================================================================
# Experiment 3: Long-run stability + volatility identification
# ===================================================================
def experiment_3_long_run_stability(sim: MDRSSimulator, num_steps: int = 2_000_000) -> None:
    print(f"\n=== Experiment 3: Long-Run Stability (N={num_steps:,}) ===")

    out = sim.run(num_paths=1, num_steps=num_steps, dt=0.01, seed=42)
    p = out["price"][:, 0]
    z = out["signal"][:, 0]

    # --- Table 1: Batch-means moments ---
    mean_bm, mean_se, k_batches = batch_stat_se(p, np.mean)
    std_bm, std_se, _ = batch_stat_se(p, np.std)
    skew_bm, skew_se, _ = batch_stat_se(p, stats.skew)
    kurt_bm, kurt_se, _ = batch_stat_se(p, lambda x: stats.kurtosis(x, fisher=False))

    print("\n  [Table 1] Long-Run Moments with Batch-Means SE")
    print(f"  Batches   : {k_batches}")
    print(f"  Mean      : {mean_bm:.6f}  (BM SE = {mean_se:.6f})")
    print(f"  Std. Dev. : {std_bm:.6f}  (BM SE = {std_se:.6f})")
    print(f"  Skewness  : {skew_bm:.6f}  (BM SE = {skew_se:.6f})")
    print(f"  Kurtosis  : {kurt_bm:.4f}  (BM SE = {kurt_se:.4f})")

    np.savetxt(
        TABLE_DIR / "table_long_run_moments_batch_means.csv",
        np.array([
            [mean_bm, mean_se],
            [std_bm, std_se],
            [skew_bm, skew_se],
            [kurt_bm, kurt_se],
        ]),
        delimiter=",",
        header="estimate,batch_means_se",
        comments="",
    )

    # --- Table 2: Threshold-moving volatility ---
    dp = np.diff(p)
    z_lag = z[:-1]

    print("\n  [Table 2] Threshold-Moving Volatility Recovery")
    print(f"  {'Threshold':<15} {'N':<10} {'sigma_hat':<12} {'|sigma_hat-sigma_1|':<20} {'|sigma_hat-sigma_1_eff|'}")
    sigma_1_eff = (1 - sim.w_max) * sim.sigma_0 + sim.w_max * sim.sigma_1
    vol_rows = []
    for thr in [1.0, 1.25, 1.5, 1.75, 2.0]:
        mask = z_lag >= thr
        n_mask = int(mask.sum())
        if n_mask > 0:
            rv = np.sum(dp[mask] ** 2) / (n_mask * 0.01)
            sv = np.sqrt(rv)
            vol_rows.append([thr, n_mask, sv, abs(sv - sim.sigma_1), abs(sv - sigma_1_eff)])
            print(f"  Z>={thr:<12.2f} {n_mask:<10d} {sv:<12.4f} {abs(sv - sim.sigma_1):<20.4f} {abs(sv - sigma_1_eff):.4f}")

    if vol_rows:
        np.savetxt(
            TABLE_DIR / "table_threshold_volatility.csv",
            np.asarray(vol_rows),
            delimiter=",",
            header="threshold,n_obs,sigma_hat,abs_error_sigma1,abs_error_sigma1_eff",
            comments="",
        )

    # --- Figure 3: Log-density histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(p, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(0, color="red", ls="--", lw=2, label="Fundamental Equilibrium (p=0)")
    plt.yscale("log")
    plt.title(f"Long-Run Distribution (T = {num_steps * 0.01:,.0f})")
    plt.xlabel(r"Price Level $p_t$")
    plt.ylabel(r"Log-Density")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig3_long_run_distribution.png")
    print("  Saved fig3_long_run_distribution.png")


def main():
    sim = MDRSSimulator()

    experiment_1_trajectory(sim)
    experiment_2_sensitivity(sim)
    experiment_3_long_run_stability(sim, num_steps=2_000_000)

    print("\n=== All synthetic reproduction experiments complete. ===")


# ===================================================================
if __name__ == "__main__":
    main()
