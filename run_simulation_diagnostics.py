"""
Numerical sanity-check suite for the MDRS-SDE simulator.

These checks are diagnostic only. They do not constitute numerical proofs of
geometric ergodicity, Harris recurrence, or breakout-time integrability.

  Check 1 — Invariant ordering: S_t <= R_t
  Check 2 — Drift-dominance parameter sanity check
  Check 3 — Equilibrium-region stability diagnostic
  Check 4 — Running-mean stabilization diagnostic
  Check 5 — Correlation robustness rho != 0
  Check 6 — Implicit scheme dt convergence
  Check 7 — Finite-horizon exit-frequency diagnostic
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.simulator import MDRSSimulator, DEFAULT_PARAMS

RESULTS_DIR = Path("results")
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
FIGURE_DIR = DIAGNOSTICS_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PASS = "\033[92mPASS\033[0m"
WARN = "\033[93mWARN\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ===================================================================
# Check 1: Invariant ordering S_t <= R_t
# ===================================================================
def check_invariant_ordering() -> bool:
    print("\n=== Check 1: Invariant Ordering S_t <= R_t ===")

    simulator = MDRSSimulator()
    out = simulator.run(num_paths=50, num_steps=50_000, dt=0.01, seed=42)
    violations = int(np.sum(out["support"] > out["resistance"] + 1e-12))
    total = out["support"].size
    ok = violations == 0

    print(f"  Total state evaluations: {total:,}")
    print(f"  Violations (S_t > R_t): {violations}")
    print(f"  Result: {PASS if ok else FAIL}")
    return ok


# ===================================================================
# Check 2: Drift-dominance condition sanity check
# ===================================================================
def check_drift_dominance() -> bool:
    print("\n=== Check 2: Drift-Dominance Parameter Sanity Check ===")

    p = DEFAULT_PARAMS
    w_zeta = p["w_max"] / (1.0 + np.exp(-p["k"] * (p["zeta"] - p["gamma"])))
    lhs = (1.0 - w_zeta) * p["kappa"]
    rhs = 4.0 * w_zeta * (p["alpha_long"] + p["alpha_short"])
    margin = lhs - rhs

    print(f"  w(zeta) = {w_zeta:.6f}")
    print(f"  LHS = (1 - w_zeta) kappa = {lhs:.6f}")
    print(f"  RHS = 4 w_zeta (alpha_L + alpha_S) = {rhs:.6f}")
    print(f"  Margin = {margin:.6f}")

    ok = margin > 0
    print(f"  Result: {PASS if ok else WARN}")
    return ok


# ===================================================================
# Check 3: Equilibrium-region stability diagnostic
# ===================================================================
def check_equilibrium_region_stability() -> bool:
    print("\n=== Check 3: Equilibrium-Region Stability Diagnostic ===")
    sim = MDRSSimulator()
    n_steps = 500_000
    out = sim.run(num_paths=1, num_steps=n_steps, dt=0.01, seed=42)
    p_cfg = DEFAULT_PARAMS

    # Weight used in the paper's equilibrium-region drift-dominance condition.
    w_zeta = p_cfg["w_max"] / (1.0 + np.exp(-p_cfg["k"] * (p_cfg["zeta"] - p_cfg["gamma"])))
    eta = w_zeta * (p_cfg["alpha_long"] + p_cfg["alpha_short"])
    lambda_p = 2.0 * (1.0 - w_zeta) * p_cfg["kappa"] - 4.0 * eta
    c_pR = p_cfg["beta_r"] ** 2 / p_cfg["gamma_r"]
    c_pS = p_cfg["beta_s"] ** 2 / p_cfg["gamma_s"]
    A_p = (c_pR + c_pS + 0.1) / max(lambda_p, 1e-12)

    p = out["price"][:, 0]
    z = out["signal"][:, 0]
    r = out["resistance"][:, 0]
    s = out["support"][:, 0]
    V = 1.0 + A_p * p**2 + z**2 + r**2 + s**2

    window = 10_000
    V_avg = np.convolve(V, np.ones(window) / window, mode="valid")
    last_quarter = V_avg[len(V_avg) * 3 // 4 :]
    first_quarter = V_avg[: len(V_avg) // 4]
    ratio = float(np.mean(last_quarter) / max(np.mean(first_quarter), 1e-12))
    ok = ratio < 2.0

    print(f"  A_p = {A_p:.4f}")
    print(f"  Mean V (first quarter): {np.mean(first_quarter):.4f}")
    print(f"  Mean V (last quarter):  {np.mean(last_quarter):.4f}")
    print(f"  Ratio: {ratio:.4f} (diagnostic threshold < 2.0)")
    print(f"  Result: {PASS if ok else WARN}")

    t = np.arange(len(V_avg)) * 0.01
    plt.figure(figsize=(10, 4))
    plt.plot(t[::100], V_avg[::100], lw=0.5, color="steelblue")
    plt.xlabel("Time")
    plt.ylabel(r"$V(\mathbf{Y}_t)$ running average")
    plt.title("Equilibrium-Region Stability Diagnostic")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "check3_stability_diagnostic.png")
    print("  Saved check3_stability_diagnostic.png")
    return ok


# ===================================================================
# Check 4: Running-mean stabilization diagnostic
# ===================================================================
def check_running_mean_stabilization() -> bool:
    print("\n=== Check 4: Running-Mean Stabilization Diagnostic ===")

    simulator = MDRSSimulator()
    n_steps = 1_000_000
    out = simulator.run(num_paths=1, num_steps=n_steps, dt=0.01, seed=42)
    p = out["price"][:, 0]

    checkpoints = np.logspace(3, np.log10(n_steps), 30, dtype=int)
    checkpoints = np.unique(checkpoints)
    running_means = np.array([np.mean(p[:cp]) for cp in checkpoints])
    final_mean = float(np.mean(p))
    errors = np.abs(running_means - final_mean)

    # This is only a stabilization diagnostic, not an ergodicity-rate estimate.
    early = float(np.median(errors[: max(3, len(errors) // 5)]))
    late = float(np.median(errors[-max(3, len(errors) // 5) :]))
    ok = late < early

    print(f"  Median early error: {early:.6e}")
    print(f"  Median late error : {late:.6e}")
    print(f"  Result: {PASS if ok else WARN} (diagnostic only)")

    T = checkpoints * 0.01
    plt.figure(figsize=(8, 5))
    plt.semilogy(T, np.maximum(errors, 1e-12), "o-", ms=3, color="darkblue")
    plt.xlabel("Simulation Time T")
    plt.ylabel(r"$|\bar{p}_T - \bar{p}_{final}|$")
    plt.title("Running-Mean Stabilization Diagnostic")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "check4_running_mean_stabilization.png")
    print("  Saved check4_running_mean_stabilization.png")
    return ok


# ===================================================================
# Check 5: Correlation robustness rho != 0
# ===================================================================
def check_rho_robustness() -> bool:
    from scipy.stats import kurtosis, skew

    print("\n=== Check 5: Robustness to rho != 0 ===")

    rhos = [-0.5, 0.0, 0.3, 0.7]
    results = {}

    for rho_val in rhos:
        sim = MDRSSimulator()
        out = sim.run(num_paths=1, num_steps=500_000, dt=0.01, rho_override=rho_val, seed=42)
        p = out["price"][:, 0]
        results[rho_val] = {
            "mean": float(p.mean()),
            "std": float(p.std()),
            "skew": float(skew(p)),
            "kurt": float(kurtosis(p, fisher=False)),
        }
        print(
            f"  rho={rho_val:+.1f}: mean={p.mean():.5f}  std={p.std():.4f}"
            f"  skew={skew(p):.4f}  kurt={kurtosis(p, fisher=False):.2f}"
        )

    all_finite = all(v["std"] < 1.0 for v in results.values())
    all_centered = all(abs(v["mean"]) < 0.02 for v in results.values())
    all_lepto = all(v["kurt"] > 3.0 for v in results.values())
    ok = all_finite and all_centered and all_lepto

    print(f"  All finite std (<1): {all_finite}")
    print(f"  All near-zero mean: {all_centered}")
    print(f"  All leptokurtic (>3): {all_lepto}")
    print("  Note: Skewness sign may depend on rho; this is expected model behavior.")
    print(f"  Result: {PASS if ok else WARN}")
    return ok


# ===================================================================
# Check 6: dt convergence of implicit scheme
# ===================================================================
def check_dt_convergence() -> bool:
    print("\n=== Check 6: dt Convergence Diagnostic (Implicit Scheme) ===")
    print("  Using time-average E_T[p^2] as a time-step sensitivity diagnostic")

    dts = [0.05, 0.02, 0.01, 0.005]
    total_time = 5000.0
    ref_dt = 0.002
    ref_steps = int(total_time / ref_dt)
    sim = MDRSSimulator()
    ref_out = sim.run(num_paths=1, num_steps=ref_steps, dt=ref_dt, seed=42)
    ref_stat = float(np.mean(ref_out["price"][:, 0] ** 2))
    print(f"  Reference (dt={ref_dt}, T={total_time}): <p^2>_T = {ref_stat:.6f}")

    errors = []
    for dt in dts:
        steps = int(total_time / dt)
        out = sim.run(num_paths=1, num_steps=steps, dt=dt, seed=42)
        stat = float(np.mean(out["price"][:, 0] ** 2))
        err = abs(stat - ref_stat)
        errors.append(err)
        print(f"  dt={dt:.4f} ({steps:>7} steps): <p^2>_T = {stat:.6f}  |err| = {err:.6f}")

    decreasing_count = sum(1 for i in range(len(errors) - 1) if errors[i] > errors[i + 1])
    ok = errors[0] > errors[-1] and decreasing_count >= len(errors) // 2
    print(f"  Coarsest error > finest error: {errors[0] > errors[-1]}")
    print(f"  Decreasing pairs: {decreasing_count}/{len(errors)-1}")
    print(f"  Result: {PASS if ok else WARN}")

    plt.figure(figsize=(7, 5))
    plt.loglog(dts, errors, "o-", color="darkred", lw=2)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$|\langle p^2 \rangle_T - \langle p^2 \rangle_{ref}|$")
    plt.title(r"Time-Step Sensitivity Diagnostic: $\langle p^2 \rangle_T$ vs $\Delta t$")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "check6_dt_convergence.png")
    print("  Saved check6_dt_convergence.png")
    return ok


# ===================================================================
# Check 7: Finite-horizon exit-frequency diagnostic
# ===================================================================
def check_finite_horizon_exit_frequency() -> bool:
    print("\n=== Check 7: Finite-Horizon Exit-Frequency Diagnostic ===")

    simulator = MDRSSimulator()
    r_star = 0.1
    n_paths = 500
    n_steps = 10_000
    dt = 0.01

    out = simulator.run(num_paths=n_paths, num_steps=n_steps, dt=dt, seed=42)
    prices = out["price"]
    hit = np.abs(prices) >= r_star
    ever_hit = hit.any(axis=0)
    frac = float(ever_hit.mean())
    idx = np.argmax(hit, axis=0)
    idx[~ever_hit] = n_steps - 1
    tau = idx * dt
    tau_hit = tau[ever_hit]

    print(f"  Barrier r* = {r_star}")
    print(f"  Fraction hitting barrier in T={n_steps * dt:.0f}: {frac:.4f} ({ever_hit.sum()}/{n_paths})")
    if len(tau_hit) > 0:
        print(f"  E[tau | hit] = {tau_hit.mean():.4f}")
        print(f"  max(tau | hit) = {tau_hit.max():.4f}")
    print("  Scope: diagnostic only; this is not a proof of almost-sure finiteness.")

    ok = frac > 0.90
    print(f"  Result: {PASS if ok else WARN} (diagnostic threshold >90% hitting)")
    return ok


# ===================================================================
def main():
    results = {
        "1_invariant_ordering": check_invariant_ordering(),
        "2_drift_dominance": check_drift_dominance(),
        "3_equilibrium_region_stability": check_equilibrium_region_stability(),
        "4_running_mean_stabilization": check_running_mean_stabilization(),
        "5_rho_robustness": check_rho_robustness(),
        "6_dt_convergence": check_dt_convergence(),
        "7_finite_horizon_exit_frequency": check_finite_horizon_exit_frequency(),
    }

    print("\n" + "=" * 72)
    print("NUMERICAL SANITY-CHECK SUMMARY")
    print("=" * 72)
    all_ok = True
    for name, ok in results.items():
        status = PASS if ok else WARN
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n  All diagnostics passed their configured thresholds.")
    else:
        print("\n  Some diagnostics did not pass their configured thresholds. Review output above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
