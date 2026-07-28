"""Uncertainty on the combined failure-mode risk for both case studies.

Section 'Fusion and Tiered Calibration' states that uncertainty propagates by
carrying tier-specific distributions on the two terms and sampling from them.
This script performs that sampling for the noisy-OR combination, so every
combined risk, per-factor contribution and counterfactual in the manuscript can
be quoted with an interval instead of a bare point estimate.

Two sources of uncertainty enter:

  pi_i   occurrence probability. Data-tier factors take the corpus base rate,
         resampled by a nonparametric bootstrap over records.
  q_i    triggering strength P(Y | X_i) read off the causal edge, drawn from
         the Beta posterior implied by the edge support, q_i ~ Beta(s_ij + 1,
         s_i - s_ij + 1), which is the Jeffreys-style interval for a binomial
         proportion observed s_ij times out of s_i.

The physics-tier occurrence probabilities used in the scenarios are held at
their model values, so the reported intervals describe uncertainty in the
corpus-estimated terms. That is the honest scope: the physics term carries
model-form uncertainty that a resampling scheme cannot express, and the
sensitivity analysis in sensitivity_carb_icing.py addresses it separately.

Output: physics/out/risk_uncertainty.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import P_ENGINE_FAILURE_GIVEN_ICE, p_carb_icing  # noqa: E402
from physics.ef_risk_model import p_fuel_exhaustion  # noqa: E402

KG = "event_extraction/out/causation_kg"
N_BOOT = 2000
SEED = 0

LOC_TARGET = "LOSS_OF_CONTROL_INFLIGHT"
EF_TARGET = "ENGINE_FAILURE"

LOC_FACTORS = [
    ("STALL", "physics"), ("CONTROL_INPUT_IMPROPER", "data"),
    ("ENGINE_FAILURE", "data"), ("DECISION_INAPPROPRIATE", "data"),
    ("CONTROL_SURFACE_ANOMALY", "surrogate"),
    ("AIRFRAME_STRUCTURAL_FAILURE", "physics"),
    ("SPATIAL_DISORIENTATION", "data"),
    ("PILOT_INCAPACITATION_OR_IMPAIRMENT", "data"),
    ("TURBULENCE_ENCOUNTER", "physics"), ("WIND_SHEAR_OR_GUST", "physics"),
]
EF_FACTORS = [
    ("FUEL_EXHAUSTION_OR_STARVATION", "physics"),
    ("LATENT_MECHANICAL_DEFECT", "data"), ("PROCEDURE_NOT_FOLLOWED", "data"),
    ("OTHER_SYSTEM_FAILURE", "data"),
    ("CARBURETOR_OR_INDUCTION_ICING", "physics"),
    ("FUEL_SYSTEM_ANOMALY", "surrogate"), ("FUEL_CONTAMINATION", "surrogate"),
    ("MAINTENANCE_INADEQUATE", "data"),
]


def load_edges(target: str) -> dict[str, tuple[int, float]]:
    """src -> (edge support s_ij, P(dst|src)) for edges into the target."""
    out = {}
    with open(os.path.join(KG, "causation_edges.csv")) as f:
        for r in csv.DictReader(f):
            if r["dst"] == target:
                out[r["src"]] = (int(r["support"]), float(r["p_dst_given_src"]))
    return out


def noisy_or(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """1 - prod(1 - pi*q) along the factor axis (last)."""
    return 1.0 - np.prod(1.0 - pi * q, axis=-1)


def ci(v: np.ndarray) -> list[float]:
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


def run(target: str, factors: list[tuple[str, str]], df: pd.DataFrame,
        scenario_pi: dict[str, float], cf_specs: list[tuple[str, str]]) -> dict:
    rng = np.random.default_rng(SEED)
    names = [n for n, _ in factors]
    edges = load_edges(target)

    # s_i, the number of reports carrying the source factor, recovers the
    # binomial denominator behind q_i = s_ij / s_i.
    s_ij = np.array([edges[n][0] for n in names], dtype=float)
    q_hat = np.array([edges[n][1] for n in names], dtype=float)
    s_i = np.maximum(s_ij / np.clip(q_hat, 1e-9, None), s_ij)

    present = df[names].to_numpy(dtype=np.int8)
    n = len(df)

    # --- bootstrap draws -------------------------------------------------
    pi_draws = np.empty((N_BOOT, len(names)))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        pi_draws[b] = present[idx].mean(axis=0)
    q_draws = rng.beta(s_ij + 1.0, np.maximum(s_i - s_ij, 0.0) + 1.0,
                       size=(N_BOOT, len(names)))

    base = noisy_or(pi_draws, q_draws)

    # per-factor contribution: drop when pi_i is set to zero
    contrib = np.empty((N_BOOT, len(names)))
    for j in range(len(names)):
        pj = pi_draws.copy()
        pj[:, j] = 0.0
        contrib[:, j] = base - noisy_or(pj, q_draws)

    # --- scenario and counterfactuals -----------------------------------
    pi_scn = pi_draws.copy()
    for name, val in scenario_pi.items():
        pi_scn[:, names.index(name)] = val
    scn = noisy_or(pi_scn, q_draws)

    # A counterfactual spec maps factor names to the intervened occurrence
    # probability. A value of None restores the corpus base rate, so removal of
    # a physics-elevated factor returns it to the population level; a numeric
    # value is a physics-level intervention that sets the driver directly.
    cfs = {}
    for overrides, label in cf_specs:
        pc = pi_scn.copy()
        for name, val in overrides.items():
            j = names.index(name)
            pc[:, j] = pi_draws[:, j] if val is None else val
        v = noisy_or(pc, q_draws)
        cfs[label] = {"mean": round(float(v.mean()), 4), "ci95": ci(v)}

    return {
        "target": target,
        "n_records": int(n),
        "n_boot": N_BOOT,
        "empirical_prevalence": round(float(df[target].mean()), 4),
        "combined_baseline": {"mean": round(float(base.mean()), 4), "ci95": ci(base)},
        "scenario": {"pi_overrides": scenario_pi,
                     "combined": {"mean": round(float(scn.mean()), 4), "ci95": ci(scn)},
                     "counterfactuals": cfs},
        "factors": [
            {"factor": names[j], "tier": factors[j][1],
             "pi_base": round(float(pi_draws[:, j].mean()), 4),
             "pi_ci95": ci(pi_draws[:, j]),
             "q": round(float(q_draws[:, j].mean()), 4),
             "q_ci95": ci(q_draws[:, j]),
             "contribution": round(float(contrib[:, j].mean()), 4),
             "contribution_ci95": ci(contrib[:, j])}
            for j in range(len(names))],
    }


def main() -> None:
    os.makedirs("physics/out", exist_ok=True)
    df = pd.read_parquet(os.path.join(KG, "factor_vectors.parquet"))

    # LOC-I scenario: maneuvering flight in icing conditions.
    try:
        from physics.jsbsim_stall import p_stall
        pi_stall = float(p_stall("c172x", 0.0, "maneuvering", seed=2).p_stall)
    except Exception:
        pi_stall = 0.335
    ice = p_carb_icing(5.0, 3.0, "descent").p_ice
    pi_ef = min(0.95, float(df["ENGINE_FAILURE"].mean())
                + ice * P_ENGINE_FAILURE_GIVEN_ICE)
    loc = run(LOC_TARGET, LOC_FACTORS, df,
              {"STALL": pi_stall, "ENGINE_FAILURE": pi_ef},
              [({"STALL": None}, "do(recover stall margin)"),
               ({"ENGINE_FAILURE": None}, "do(carburetor heat)")])

    # SCF-PP scenario: serious icing weather with a thin fuel reserve.
    ef = run(EF_TARGET, EF_FACTORS, df,
             {"CARBURETOR_OR_INDUCTION_ICING": p_carb_icing(13.0, 12.0, "descent").p_ice,
              "FUEL_EXHAUSTION_OR_STARVATION": p_fuel_exhaustion(0.5)},
             [({"CARBURETOR_OR_INDUCTION_ICING": None}, "do(carburetor heat)"),
              ({"FUEL_EXHAUSTION_OR_STARVATION": p_fuel_exhaustion(2.0)},
               "do(restore 2 h reserve)"),
              ({"CARBURETOR_OR_INDUCTION_ICING": None,
                "FUEL_EXHAUSTION_OR_STARVATION": p_fuel_exhaustion(2.0)},
               "do(both interventions)")])

    out = {"loss_of_control": loc, "engine_failure": ef}
    with open("physics/out/risk_uncertainty.json", "w") as f:
        json.dump(out, f, indent=2)

    for k, r in out.items():
        print(f"\n=== {k}: empirical {r['empirical_prevalence']}, "
              f"combined {r['combined_baseline']['mean']} "
              f"{r['combined_baseline']['ci95']} ===")
        print(f"{'factor':38s} {'pi':>7} {'q':>7} {'contrib':>8}  ci95")
        for f_ in r["factors"]:
            print(f"{f_['factor']:38s} {f_['pi_base']:>7.3f} {f_['q']:>7.3f} "
                  f"{f_['contribution']:>8.4f}  {f_['contribution_ci95']}")
        s = r["scenario"]
        print(f"scenario combined {s['combined']['mean']} {s['combined']['ci95']}")
        for lab, v in s["counterfactuals"].items():
            print(f"  {lab:34s} {v['mean']} {v['ci95']}")


if __name__ == "__main__":
    main()
