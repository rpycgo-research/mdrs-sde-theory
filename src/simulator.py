"""
Core 4D MDRS-SDE simulator (Numba-accelerated).

Implements the fractional-step Euler-Maruyama scheme for the lifted system
(p_t, Z_t, R_t, S_t) defined by Eqs. (3.5)-(3.8) of the paper.

CRITICAL FIX: The breakout depth D_t^+ = (p_t - R_t)^+ is implemented as a
continuous positive-part function, matching Definition 3.3. The original
codebase incorrectly used a binary indicator (1 if p > R else 0).
"""
import numpy as np
from typing import Dict, Tuple

try:
    from numba import njit
except ImportError:
    # Fallback: plain function (no JIT)
    def njit(**_kw):  # type: ignore[override]
        def _id(fn):
            return fn
        return _id

# ---------------------------------------------------------------------------
# Default parameters (Section 4.1)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "kappa": 1.0,
    "alpha_long": 0.5,
    "alpha_short": 0.6,
    "sigma_0": 0.05,
    "sigma_1": 0.50,
    "w_max": 0.9,
    "k": 10.0,
    "gamma": 1.5,
    "kappa_z": 2.0,
    "z_bar": 0.0,
    "sigma_z": 1.0,
    "rho": 0.0,
    "beta_r": 5.0,
    "gamma_r": 0.1,
    "beta_s": 5.0,
    "gamma_s": 0.1,
    "zeta": 1.0,
}


@njit(cache=True)
def _simulate_core(
    num_paths: int,
    num_steps: int,
    dt: float,
    gamma_val: float,
    kappa: float,
    alpha_l: float,
    alpha_s: float,
    sigma_0: float,
    sigma_1: float,
    w_max: float,
    k: float,
    kappa_z: float,
    z_bar: float,
    sigma_z: float,
    rho: float,
    beta_r: float,
    gamma_r: float,
    beta_s: float,
    gamma_s: float,
    zeta: float,
    dW_p: np.ndarray,
    dW_ind: np.ndarray,
    p0: float,
    z0: float,
    r0: float,
    s0: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-compiled inner loop for the 4D MDRS-SDE."""
    price = np.zeros((num_steps, num_paths))
    signal = np.zeros((num_steps, num_paths))
    resistance = np.zeros((num_steps, num_paths))
    support = np.zeros((num_steps, num_paths))
    weight = np.zeros((num_steps, num_paths))

    sqrt_dt = np.sqrt(dt)
    rho_c = np.sqrt(1.0 - rho * rho)

    for i in range(num_paths):
        p = p0
        z = z0
        r = r0
        s = s0

        for t in range(num_steps):
            # --- Record current state ---
            price[t, i] = p
            signal[t, i] = z
            resistance[t, i] = r
            support[t, i] = s

            # --- Regime weight (Eq. 3.3) ---
            w = w_max / (1.0 + np.exp(-k * (z - gamma_val)))
            weight[t, i] = w

            # --- Brownian increments with correlation ---
            dw_p = dW_p[t, i] * sqrt_dt
            dw_z = (rho * dW_p[t, i] + rho_c * dW_ind[t, i]) * sqrt_dt

            # --- Breakout depth: CONTINUOUS positive part (Def 3.3) ---
            d_plus = max(p - r, 0.0)   # (p_t - R_t)^+
            d_minus = max(s - p, 0.0)  # (S_t - p_t)^+

            # --- Price SDE (Eq. 3.5) ---
            drift_p = -(1.0 - w) * kappa * p + w * (alpha_l * d_plus - alpha_s * d_minus)
            diff_p = (1.0 - w) * sigma_0 + w * sigma_1
            p_new = p + drift_p * dt + diff_p * dw_p

            # --- Signal OU (Eq. 3.6) ---
            z_new = z + kappa_z * (z_bar - z) * dt + sigma_z * dw_z

            # --- Extrema: implicit step (Eqs. 3.7-3.8) ---
            if z < zeta:
                # R_t update
                if p > r:
                    r_new = (r + beta_r * p * dt) / (1.0 + beta_r * dt)
                else:
                    r_new = (r + gamma_r * p * dt) / (1.0 + gamma_r * dt)
                # S_t update
                if s > p:
                    s_new = (s + beta_s * p * dt) / (1.0 + beta_s * dt)
                else:
                    s_new = (s + gamma_s * p * dt) / (1.0 + gamma_s * dt)
            else:
                r_new = r
                s_new = s

            p = p_new
            z = z_new
            r = r_new
            s = s_new

    return price, signal, resistance, support, weight


class MDRSSimulator:
    """Monte Carlo simulator for the 4D lifted MDRS-SDE."""
    def __init__(self, params: Dict[str, float] | None = None):
        cfg = {**DEFAULT_PARAMS, **(params or {})}
        for key, val in cfg.items():
            setattr(self, key, val)

    def run(
        self,
        num_paths: int = 1,
        num_steps: int = 500,
        dt: float = 0.01,
        *,
        gamma_override: float | None = None,
        rho_override: float | None = None,
        seed: int | None = 42,
        p0: float = 0.0,
        z0: float = 0.0,
        r0: float = 0.1,
        s0: float = -0.1,
        ) -> Dict[str, np.ndarray]:
        """
        Run simulation and return dict of arrays.

        Returns
        -------
        dict with keys: price, signal, resistance, support, weight
            Each array has shape (num_steps, num_paths).
        """
        if seed is not None:
            np.random.seed(seed)

        gamma_val = gamma_override if gamma_override is not None else self.gamma
        rho_val = rho_override if rho_override is not None else self.rho


        if num_paths <= 0:
            raise ValueError("num_paths must be positive.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if not (-1.0 < rho_val < 1.0):
            raise ValueError("rho must lie in (-1, 1).")
        if self.sigma_0 <= 0 or self.sigma_1 <= 0 or self.sigma_z <= 0:
            raise ValueError("sigma_0, sigma_1, and sigma_z must be positive.")
        if not (0.0 < self.w_max < 1.0):
            raise ValueError("w_max must lie in (0, 1).")
        if s0 > r0:
            raise ValueError("Initial support must not exceed initial resistance: require s0 <= r0.")

        dW_p = np.random.standard_normal((num_steps, num_paths))
        dW_ind = np.random.standard_normal((num_steps, num_paths))

        p, z, r, s, w = _simulate_core(
            num_paths, num_steps, dt, gamma_val,
            self.kappa, self.alpha_long, self.alpha_short,
            self.sigma_0, self.sigma_1, self.w_max, self.k,
            self.kappa_z, self.z_bar, self.sigma_z, rho_val,
            self.beta_r, self.gamma_r, self.beta_s, self.gamma_s,
            self.zeta, dW_p, dW_ind, p0, z0, r0, s0,
        )

        return dict(price=p, signal=z, resistance=r, support=s, weight=w)
