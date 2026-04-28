# MDRS-SDE Simulation Code

This repository contains simulation and numerical diagnostic code for the
Microstructure-Driven Regime-Switching SDE (MDRS-SDE) paper.

## Repository structure

- `src/simulator.py` — Numba-accelerated simulator for the four-dimensional leaky-extrema MDRS-SDE.
- `run_experiments.py` — synthetic figures and tables used for paper reproduction.
- `run_theory_checks.py` — numerical sanity checks for simulator behavior.
- `figures/` — generated figures.
- `tables/` — generated synthetic tables.

## Scope note

The long-run simulation workflow provides numerical evidence of stable behavior.
It does **not** prove total-variation geometric ergodicity, Harris recurrence, or
breakout-time integrability. Those mathematical claims are handled separately in
the paper and, where appropriate, stated as assumptions or future work.

## Synthetic experiments

`run_experiments.py` generates three synthetic experiments:

1. trajectory and leaky/latched extrema illustration;
2. finite-horizon breakout-time sensitivity to the activation threshold;
3. long-run distribution and threshold-moving volatility identification.

The long-run moment table reports batch-means standard errors rather than iid
standard errors, because a single simulated Markov path is serially dependent.

## Numerical sanity checks

`run_theory_checks.py` runs diagnostics for:

1. invariant ordering of support and resistance;
2. drift-dominance parameter sanity;
3. equilibrium-region stability;
4. running-mean stabilization;
5. robustness to nonzero Brownian correlation;
6. weak convergence of the implicit scheme as `dt` changes;
7. finite-horizon exit frequency.

These are diagnostics, not theorem-verification tests.

## Usage

```bash
uv run python run_experiments.py
uv run python run_theory_checks.py
```

If you are not using `uv`, install the dependencies in `pyproject.toml` and run
with your Python environment of choice.
