"""Leaky noisy-OR combination of contributing factors (paper Eq. combine,
revised 2026-09-14 after review).

The combination rule for a failure mode Y with direct causal parents
X_1..X_k is

    P(Y | O) = 1 - (1 - l) * prod_i (1 - pi_i(O) * p_i),

where pi_i(O) is the occurrence probability of parent i at operating point O
(physics where a model exists, corpus base rate otherwise), p_i is the
probability that parent i alone produces Y, and l is the leak, the
probability that Y occurs with none of the listed parents present. Both l
and the p_i are fitted by maximum likelihood on the binary presence table
(one row per report), so p_i carries the noisy-OR semantics rather than the
marginal conditional P(Y | X_i) that the earlier version used.

The leading-factor set L(Y) (paper Eq. leading) is the set of direct parents
whose edge support s_iY reaches a fraction RHO of the records that carry Y,
s_iY >= RHO * s_Y; no lift threshold is applied, and lift is reported as a
diagnostic. RHO = 0.02 gives 14 parents for loss of control (s_Y = 9,813)
and 9 for engine failure (s_Y = 16,406). A sweep over absolute thresholds
(200 to 500) showed that the fitted leak is stable except when a near-
ubiquitous weak parent enters, which the relative rule avoids. Parents whose within-accident lift is below one
receive a fitted p_i near zero, which the paper reads as a collider effect of
selection on the accident together with mediation through parents already in
the set.

Per-factor contribution: Delta_i = P(Y | O) - P(Y | O, pi_i = 0).

Used by loc_risk_model.py, ef_risk_model.py and risk_uncertainty.py.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

KG = "event_extraction/out/causation_kg"
RHO = 0.02  # leading-factor rule: edge support >= RHO * (records carrying Y)

OUTCOMES = {"GROUND_IMPACT", "WATER_IMPACT", "MIDAIR_COLLISION",
            "GROUND_COLLISION", "RUNWAY_EXCURSION_OR_OVERRUN",
            "INFLIGHT_BREAKUP", "EMERGENCY_LANDING", "SUCCESSFUL_RECOVERY",
            "INJURY_OR_FATALITY"}

# Occurrence-model tier per factor, a property of the factor, not of the
# failure study: 'physics' has a first-principles occurrence model,
# 'surrogate' a reliability or hazard-rate stand-in, 'data' the corpus rate.
TIER = {
    "STALL": ("physics", "6-DOF"),
    "AIRFRAME_STRUCTURAL_FAILURE": ("physics", "V--n"),
    "TURBULENCE_ENCOUNTER": ("physics", "gust"),
    "WIND_SHEAR_OR_GUST": ("physics", "gust"),
    "CARBURETOR_OR_INDUCTION_ICING": ("physics", "icing"),
    "FUEL_EXHAUSTION_OR_STARVATION": ("physics", "endurance"),
    "ENGINE_FAILURE": ("surrogate", "hazard"),
    "CONTROL_SURFACE_ANOMALY": ("surrogate", "reliability"),
    "FUEL_SYSTEM_ANOMALY": ("surrogate", "reliability"),
    "FUEL_CONTAMINATION": ("surrogate", "reliability"),
}


def tier(name: str) -> tuple[str, str | None]:
    return TIER.get(name, ("data", None))


def load_edges(kg: str = KG) -> list[dict]:
    with open(os.path.join(kg, "causation_edges.csv")) as f:
        return [{"src": r["src"], "dst": r["dst"], "support": int(r["support"]),
                 "q": float(r["p_dst_given_src"]), "lift": float(r["lift"])}
                for r in csv.DictReader(f)]


def leading_factors(edges: list[dict], target: str, n_target: int,
                    rho: float = RHO) -> list[dict]:
    """Direct parents of `target` with edge support >= rho * n_target."""
    s0 = rho * n_target
    rows = [e for e in edges if e["dst"] == target and e["src"] not in OUTCOMES
            and e["support"] >= s0]
    return sorted(rows, key=lambda e: -e["support"])


def load_presence(kg: str = KG) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(kg, "factor_vectors.parquet"))


# --- maximum-likelihood fit ---------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _nll_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray):
    """Negative log-likelihood and gradient in logit parameters.

    theta[0] is logit(leak), theta[1:] are logit(p_i). With
    s_r = log(1 - l) + sum_i x_ri log(1 - p_i), P(y_r = 1) = 1 - exp(s_r).
    """
    l = _sigmoid(theta[0])
    p = _sigmoid(theta[1:])
    log1m = np.log1p(-p)
    s = np.log1p(-l) + X @ log1m
    es = np.exp(s)
    p1 = np.clip(1.0 - es, 1e-12, 1.0)
    nll = -(y * np.log(p1) + (1.0 - y) * s).sum()
    # d nll / d s_r
    ds = y * es / p1 - (1.0 - y)
    g = np.empty_like(theta)
    g[0] = (ds * (-l)).sum()
    g[1:] = (ds[:, None] * (-X * p)).sum(axis=0)
    return nll, g


def fit_leaky_noisy_or(X: np.ndarray, y: np.ndarray,
                       theta0: np.ndarray | None = None) -> dict:
    """MLE of the leak and the per-parent strengths on a presence table."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    k = X.shape[1]
    if theta0 is None:
        theta0 = np.concatenate([[-2.5], np.full(k, -1.0)])
    res = minimize(_nll_grad, theta0, args=(X, y), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    leak = float(_sigmoid(res.x[0]))
    p = _sigmoid(res.x[1:])
    return {"leak": leak, "p": p, "theta": res.x, "nll": float(res.fun),
            "nll_per_record": float(res.fun / len(y)), "converged": bool(res.success)}


def combine(pi: np.ndarray, p: np.ndarray, leak: float) -> np.ndarray:
    """1 - (1 - leak) prod(1 - pi p) along the last axis."""
    return 1.0 - (1.0 - leak) * np.prod(1.0 - pi * p, axis=-1)


def contributions(pi: np.ndarray, p: np.ndarray, leak: float) -> np.ndarray:
    """Delta_i = P - P(pi_i = 0), for a one-dimensional pi."""
    base = combine(pi, p, leak)
    out = np.empty(len(pi))
    for i in range(len(pi)):
        pj = pi.copy()
        pj[i] = 0.0
        out[i] = base - combine(pj, p, leak)
    return out


def bootstrap_fit(X: np.ndarray, y: np.ndarray, n_boot: int, seed: int = 0,
                  theta0: np.ndarray | None = None):
    """Refit on bootstrap resamples of the records.

    Returns (leak_draws [n_boot], p_draws [n_boot, k], pi_draws [n_boot, k]),
    where pi_draws are the resampled corpus base rates of the parents.
    """
    rng = np.random.default_rng(seed)
    n, k = X.shape
    leaks = np.empty(n_boot)
    ps = np.empty((n_boot, k))
    pis = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        fit = fit_leaky_noisy_or(Xb, yb, theta0)
        leaks[b] = fit["leak"]
        ps[b] = fit["p"]
        pis[b] = Xb.mean(axis=0)
    return leaks, ps, pis


if __name__ == "__main__":
    edges = load_edges()
    df = load_presence()
    for target in ("LOSS_OF_CONTROL_INFLIGHT", "ENGINE_FAILURE"):
        y = df[target].to_numpy(dtype=np.int8)
        lf = leading_factors(edges, target, int(y.sum()))
        names = [e["src"] for e in lf]
        X = df[names].to_numpy(dtype=np.int8)
        fit = fit_leaky_noisy_or(X, y)
        pi = X.mean(axis=0)
        P = combine(pi, fit["p"], fit["leak"])
        print(f"\n=== {target}: {len(names)} parents at support >= "
              f"{RHO:.2f} x {int(y.sum())}, "
              f"leak {fit['leak']:.4f}, fitted mean {P:.4f}, "
              f"empirical {y.mean():.4f}, nll/record {fit['nll_per_record']:.4f}")
        d = contributions(pi, fit["p"], fit["leak"])
        for e, pi_i, p_i, d_i in zip(lf, pi, fit["p"], d):
            print(f"  {e['src']:38s} s={e['support']:5d} q={e['q']:.3f} "
                  f"lift={e['lift']:.2f} base={pi_i:.3f} p={p_i:.3f} "
                  f"Delta={d_i:.4f} {tier(e['src'])[0]}")
