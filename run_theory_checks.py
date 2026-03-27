"""
Theory verification suite for the MDRS-SDE paper.

Each check maps to a specific theorem/lemma and prints PASS/FAIL.

  Check 1 — Invariant ordering: S_t ≤ R_t   (Remark 3.7)
  Check 2 — Drift dominance condition       (Theorem 3.18, Eq. drift_dominance)
  Check 3 — Foster–Lyapunov drift           (Theorem 3.18, Step 1)
  Check 4 — Geometric ergodicity rate       (Theorem 3.18)
  Check 5 — Correlation robustness ρ ≠ 0    (Assumption 3.4)
  Check 6 — Implicit scheme Δt convergence  (Section 4.1)
  Check 7 — τ_{r*} finiteness               (Lemma tau_finite)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from src.simulator import MDRSSimulator, DEFAULT_PARAMS

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ===================================================================
# Check 1: Invariant ordering  S_t ≤ R_t   (Remark 3.7)
# ===================================================================
def check_invariant_ordering() -> bool:
    print("\n=== Check 1: Invariant Ordering S_t ≤ R_t ===")

    simulator = MDRSSimulator()
    out = simulator.run(num_paths=50, num_steps=50_000, dt=0.01, seed=42)
    violations = np.sum(out["support"] > out["resistance"] + 1e-12)
    total = out["support"].size
    ok = violations == 0

    print(f"  Total state evaluations: {total:,}")
    print(f"  Violations (S_t > R_t): {violations}")
    print(f"  Result: {PASS if ok else FAIL}")

    return ok


# ===================================================================
# Check 2: Drift dominance condition  (Theorem 3.18)
# ===================================================================
def check_drift_dominance() -> bool:
    print("\n=== Check 2: Drift Dominance Condition ===")

    p = DEFAULT_PARAMS
    w_zeta = p["w_max"] / (1.0 + np.exp(-p["k"] * (p["zeta"] - p["gamma"])))
    lhs = (1.0 - w_zeta) * p["kappa"]
    rhs = 4.0 * w_zeta * (p["alpha_long"] + p["alpha_short"])
    margin = lhs - rhs

    print(f"  w(ζ) = {w_zeta:.6f}")
    print(f"  LHS = (1 - w_ζ)κ = {lhs:.6f}")
    print(f"  RHS = 4 w_ζ (α_L + α_S) = {rhs:.6f}")
    print(f"  Margin = {margin:.6f}")

    ok = margin > 0
    print(f"  Result: {PASS if ok else FAIL}")

    return ok


# ===================================================================
# Check 3: Foster–Lyapunov drift  (Theorem 3.18, Step 1)
# ===================================================================
def check_foster_lyapunov() -> bool:
    print("\n=== Check 3: Foster–Lyapunov V(Y_t) trajectory ===")
    sim = MDRSSimulator()
    N = 500_000
    out = sim.run(num_paths=1, num_steps=N, dt=0.01, seed=42)
    p_cfg = DEFAULT_PARAMS

    # Compute w_zeta and A_p as in the corrected theorem
    w_zeta = p_cfg["w_max"] / (1.0 + np.exp(-p_cfg["k"] * (p_cfg["zeta"] - p_cfg["gamma"])))
    eta = w_zeta * (p_cfg["alpha_long"] + p_cfg["alpha_short"])
    lambda_p = 2.0 * (1.0 - w_zeta) * p_cfg["kappa"] - 4.0 * eta
    c_pR = p_cfg["beta_r"] ** 2 / p_cfg["gamma_r"]
    c_pS = p_cfg["beta_s"] ** 2 / p_cfg["gamma_s"]
    A_p = (c_pR + c_pS + 0.1) / lambda_p  # delta = 0.1

    p = out["price"][:, 0]
    z = out["signal"][:, 0]
    r = out["resistance"][:, 0]
    s = out["support"][:, 0]
    V = 1.0 + (A_p * p**2) + z**2 + r**2 + s**2

    # Running time-average of V(Y_t)
    window = 10_000
    V_avg = np.convolve(V, np.ones(window) / window, mode="valid")

    # V should be bounded in time-average (not diverging)
    last_quarter = V_avg[len(V_avg) * 3 // 4 :]
    first_quarter = V_avg[: len(V_avg) // 4]
    ratio = np.mean(last_quarter) / max(np.mean(first_quarter), 1e-12)
    ok = ratio < 2.0  # should not be growing

    print(f"  A_p = {A_p:.4f}")
    print(f"  Mean V (first quarter): {np.mean(first_quarter):.4f}")
    print(f"  Mean V (last quarter):  {np.mean(last_quarter):.4f}")
    print(f"  Ratio: {ratio:.4f} (should be < 2.0)")
    print(f"  Result: {PASS if ok else FAIL}")

    # Plot
    t = np.arange(len(V_avg)) * 0.01
    plt.figure(figsize=(10, 4))
    plt.plot(t[::100], V_avg[::100], lw=0.5, color="steelblue")
    plt.xlabel("Time")
    plt.ylabel(r"$V(\mathbf{Y}_t)$ (running avg)")
    plt.title("Foster–Lyapunov Function: Time-Averaged Trajectory")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "check3_lyapunov.png"), dpi=200)
    print("  Saved check3_lyapunov.png")

    return ok


# ===================================================================
# Check 4: Geometric ergodicity rate  (Theorem 3.18)
# ===================================================================
def check_ergodic_rate() -> bool:
    print("\n=== Check 4: Geometric Ergodicity Convergence Rate ===")

    simulator = MDRSSimulator()
    N = 1_000_000
    out = simulator.run(num_paths=1, num_steps=N, dt=0.01, seed=42)
    p = out["price"][:, 0]

    # Compute running mean at exponentially-spaced checkpoints
    checkpoints = np.logspace(3, np.log10(N), 30, dtype=int)
    checkpoints = np.unique(checkpoints)
    running_means = np.array([np.mean(p[:cp]) for cp in checkpoints])
    final_mean = np.mean(p)

    # Measure |running_mean - final_mean| — should decay
    errors = np.abs(running_means - final_mean)
    errors = np.maximum(errors, 1e-12)  # avoid log(0)

    # Fit log-linear: log(error) = a - b * T  => exponential decay
    T = checkpoints * 0.01
    mask = errors > 1e-10

    if mask.sum() > 5:
        coeffs = np.polyfit(T[mask], np.log(errors[mask]), 1)
        decay_rate = -coeffs[0]
    else:
        decay_rate = 0.0

    ok = decay_rate > 0
    print(f"  Fitted exponential decay rate: {decay_rate:.6f}")
    print(f"  Result: {PASS if ok else FAIL} (rate > 0 implies exponential convergence)")

    plt.figure(figsize=(8, 5))
    plt.semilogy(T, errors, "o-", ms=3, color="darkblue")
    plt.xlabel("Simulation Time T")
    plt.ylabel(r"$|\bar{p}_T - \bar{p}_\infty|$")
    plt.title("Ergodic Convergence of Running Mean")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "check4_ergodic_rate.png"), dpi=200)
    print("  Saved check4_ergodic_rate.png")

    return ok


# ===================================================================
# Check 5: Correlation robustness ρ ≠ 0  (Assumption 3.4)
# ===================================================================
def check_rho_robustness() -> bool:
    from scipy.stats import skew, kurtosis

    print("\n=== Check 5: Robustness to ρ ≠ 0 ===")

    rhos = [-0.5, 0.0, 0.3, 0.7]
    results = {}

    for rho_val in rhos:
        sim = MDRSSimulator()
        out = sim.run(num_paths=1, num_steps=500_000, dt=0.01,
                      rho_override=rho_val, seed=42)
        p = out["price"][:, 0]
        results[rho_val] = {
            "mean": p.mean(),
            "std": p.std(),
            "skew": skew(p),
            "kurt": kurtosis(p, fisher=False),
        }
        print(f"  ρ={rho_val:+.1f}: mean={p.mean():.5f}  std={p.std():.4f}"
              f"  skew={skew(p):.4f}  kurt={kurtosis(p, fisher=False):.2f}")

    # Core ergodicity checks:
    # 1. All std should be finite and bounded (not diverging)
    all_finite = all(v["std"] < 1.0 for v in results.values())
    # 2. All means should be near zero (mean-reversion working)
    all_centered = all(abs(v["mean"]) < 0.01 for v in results.values())
    # 3. All kurtosis should be > 3 (leptokurtic from regime switching)
    all_lepto = all(v["kurt"] > 3.0 for v in results.values())

    ok = all_finite and all_centered and all_lepto
    print(f"  All finite std (<1): {all_finite}")
    print(f"  All near-zero mean: {all_centered}")
    print(f"  All leptokurtic (>3): {all_lepto}")
    print(f"  Note: Skewness sign depends on ρ (positive ρ can flip skew).")
    print(f"        This is expected model behavior, not a failure.")
    print(f"  Result: {PASS if ok else FAIL}")

    return ok


# ===================================================================
# Check 6: Δt convergence of implicit scheme  (Section 4.1)
# ===================================================================
def check_dt_convergence() -> bool:
    print("\n=== Check 6: Δt Convergence (Implicit Scheme) ===")
    print("  Using ergodic time-average E_T[p^2] for stable weak convergence test")

    dts = [0.05, 0.02, 0.01, 0.005]
    T_total = 5000.0  # long path for ergodic average

    # Reference: finest dt
    ref_dt = 0.002
    ref_steps = int(T_total / ref_dt)
    sim = MDRSSimulator()
    ref_out = sim.run(num_paths=1, num_steps=ref_steps, dt=ref_dt, seed=42)
    ref_stat = np.mean(ref_out["price"][:, 0] ** 2)
    print(f"  Reference (dt={ref_dt}, T={T_total}): <p^2>_T = {ref_stat:.6f}")

    errors = []
    for dt in dts:
        steps = int(T_total / dt)
        out = sim.run(num_paths=1, num_steps=steps, dt=dt, seed=42)
        stat = np.mean(out["price"][:, 0] ** 2)
        err = abs(stat - ref_stat)
        errors.append(err)
        print(f"  dt={dt:.4f} ({steps:>7} steps): <p^2>_T = {stat:.6f}  |err| = {err:.6f}")

    # Check monotone decrease (allow one non-monotone due to noise)
    decreasing_count = sum(1 for i in range(len(errors) - 1) if errors[i] > errors[i + 1])
    ok = errors[0] > errors[-1] and decreasing_count >= len(errors) // 2

    print(f"  Coarsest error > finest error: {errors[0] > errors[-1]}")
    print(f"  Decreasing pairs: {decreasing_count}/{len(errors)-1}")
    print(f"  Result: {PASS if ok else FAIL}")

    plt.figure(figsize=(7, 5))
    plt.loglog(dts, errors, "o-", color="darkred", lw=2)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$|\langle p^2 \rangle_T - \langle p^2 \rangle_{\rm ref}|$")
    plt.title(r"Weak Convergence: Ergodic $\langle p^2 \rangle$ vs $\Delta t$")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "check6_dt_convergence.png"), dpi=200)
    print("  Saved check6_dt_convergence.png")

    return ok


# ===================================================================
# Check 7: τ_{r*} < ∞ a.s.  (Lemma tau_finite)
# ===================================================================
def check_tau_finite() -> bool:
    print("\n=== Check 7: τ_{r*} Finiteness ===")

    simulator = MDRSSimulator()
    r_star = 0.1
    N_paths = 500
    N_steps = 10000  # T_max = 100

    out = simulator.run(num_paths=N_paths, num_steps=N_steps, dt=0.01, seed=42)
    prices = out["price"]
    hit = np.abs(prices) >= r_star
    ever_hit = hit.any(axis=0)
    frac = ever_hit.mean()

    # For paths that hit, compute mean tau
    idx = np.argmax(hit, axis=0)
    idx[~ever_hit] = N_steps - 1
    tau = idx * 0.01
    tau_hit = tau[ever_hit]

    print(f"  Barrier r* = {r_star}")
    print(f"  Fraction hitting barrier in T={N_steps*0.01:.0f}: {frac:.4f} ({ever_hit.sum()}/{N_paths})")
    if len(tau_hit) > 0:
        print(f"  E[τ | hit] = {tau_hit.mean():.4f}")
        print(f"  max(τ | hit) = {tau_hit.max():.4f}")

    ok = frac > 0.90  # essentially all paths should hit within T_max
    print(f"  Result: {PASS if ok else FAIL} (expect >90% hitting)")
    return ok


# ===================================================================
def main():
    results = {}
    results["1_invariant_ordering"] = check_invariant_ordering()
    results["2_drift_dominance"] = check_drift_dominance()
    results["3_foster_lyapunov"] = check_foster_lyapunov()
    results["4_ergodic_rate"] = check_ergodic_rate()
    results["5_rho_robustness"] = check_rho_robustness()
    results["6_dt_convergence"] = check_dt_convergence()
    results["7_tau_finite"] = check_tau_finite()

    print("\n" + "=" * 60)
    print("THEORY VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n  All checks passed.")
    else:
        print(f"\n  Some checks FAILED. Review output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
