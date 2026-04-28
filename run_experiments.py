"""
Implements the three main experiments from Section 4 of the paper.

  Experiment 1 — Figure 1: 4D trajectory with leaky/latched extrema
  Experiment 2 — Figure 2: E[τ_r*] sensitivity to γ  (IC-Alpha resolution)
  Experiment 3 — Figure 3 + Tables 1-2: long-run stability and volatility identification
"""
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.simulator import MDRSSimulator

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


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
    latched = z >= sim.zeta

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
        lab = r"Latched Regime ($Z_t \geq \zeta$)" if idx == 0 else ""
        ax.fill_between(t, lo, hi, where=latched, color="gray", alpha=0.2, label=lab)
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig1_trajectory.png"), dpi=300)
    print("  Saved fig1_trajectory.png")


# ===================================================================
# Experiment 2: Sensitivity of E[τ_r*] to γ  (Figure 2)
# ===================================================================
def experiment_2_sensitivity(sim: MDRSSimulator, num_paths: int = 2000) -> None:
    print("\n=== Experiment 2: IC-Alpha Sensitivity (Figure 2) ===")
    print(f"  (Using {num_paths} paths per γ value)")

    gammas = np.linspace(0.5, 3.5, 12)
    means, ses = [], []
    r_star = 0.1

    for g in gammas:
        out = sim.run(num_paths=num_paths, num_steps=1000, dt=0.01, gamma_override=g, seed=42)
        prices = out["price"]
        hit = np.abs(prices) >= r_star
        idx = np.argmax(hit, axis=0)
        idx[~hit.any(axis=0)] = prices.shape[0] - 1
        tau = idx * 0.01
        m, se = tau.mean(), tau.std() / np.sqrt(len(tau))
        means.append(m)
        ses.append(se)
        print(f"  γ={g:.2f}  E[τ]={m:.4f}  SE={se:.4f}")

    plt.figure(figsize=(8, 5))
    plt.errorbar(gammas, means, yerr=1.96 * np.array(ses),
                 marker="o", color="darkblue", lw=2, capsize=4,
                 label=r"$\mathbb{E}[\tau_{r^*}] \pm 1.96\,SE$")
    plt.title("Expected Breakout Time Sensitivity (Resolution of IC-Alpha Paradox)")
    plt.xlabel(r"Activation Threshold ($\gamma$)")
    plt.ylabel(r"Expected First-Passage Time $\mathbb{E}[\tau_{r^*}]$")
    plt.grid(True, ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig2_sensitivity.png"), dpi=300)
    print("  Saved fig2_sensitivity.png")


# ===================================================================
# Experiment 3: Long-run stability + volatility identification
# ===================================================================
def experiment_3_long_run_stability(sim: MDRSSimulator, num_steps: int = 2_000_000) -> None:
    print(f"\n=== Experiment 3: Long-Run Stability (N={num_steps:,}) ===")

    out = sim.run(num_paths=1, num_steps=num_steps, dt=0.01, seed=42)
    p = out["price"][:, 0]
    z = out["signal"][:, 0]

    # --- Table 1: Moments ---
    n = len(p)
    mu, sd = p.mean(), p.std()
    sk = stats.skew(p)
    ku = stats.kurtosis(p, fisher=False)
    t_mu = mu / (sd / np.sqrt(n))
    t_sk = sk / np.sqrt(6.0 / n)

    print("\n  [Table 1] Empirical Moments")
    print(f"  Mean      : {mu:.6f}  (t = {t_mu:.2f})")
    print(f"  Std. Dev. : {sd:.6f}")
    print(f"  Skewness  : {sk:.6f}  (t = {t_sk:.2f})")
    print(f"  Kurtosis  : {ku:.4f}")

    # --- Table 2: Threshold-moving volatility ---
    dp = np.diff(p)
    z_lag = z[:-1]

    print("\n  [Table 2] Threshold-Moving Volatility Recovery")
    print(f"  {'Threshold':<15} {'σ_hat':<12} {'|σ_hat - σ_1|':<15} {'|σ_hat - σ_1_eff|'}")
    sigma_1_eff = (1 - sim.w_max) * sim.sigma_0 + sim.w_max * sim.sigma_1
    for thr in [1.0, 1.25, 1.5, 1.75, 2.0]:
        mask = z_lag >= thr
        if mask.sum() > 0:
            rv = np.sum(dp[mask] ** 2) / (mask.sum() * 0.01)
            sv = np.sqrt(rv)
            print(f"  Z≥{thr:<12.2f} {sv:<12.4f} {abs(sv - sim.sigma_1):<15.4f} {abs(sv - sigma_1_eff):.4f}")

    # --- Figure 3: Log-density histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(p, bins=200, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(0, color="red", ls="--", lw=2, label="Fundamental Equilibrium (p=0)")
    plt.yscale("log")
    plt.title(f"Empirical Stationary Distribution (T = {num_steps * 0.01:,.0f})")
    plt.xlabel(r"Price Level $p_t$")
    plt.ylabel(r"Log-Density")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig3_long_run_distribution.png"), dpi=300)
    print("  Saved fig3_long_run_distribution.png")


# ===================================================================
if __name__ == "__main__":
    sim = MDRSSimulator()

    experiment_1_trajectory(sim)
    experiment_2_sensitivity(sim)
    experiment_3_long_run_stability(sim, num_steps=2_000_000)

    print("\n=== All paper reproduction experiments complete. ===")
