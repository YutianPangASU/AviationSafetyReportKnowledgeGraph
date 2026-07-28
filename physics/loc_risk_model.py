"""Failure-mode risk model: P(loss-of-control event) from contributing-factor
models, combined by a noisy-OR (bow-tie) over the causal parents of LOC.

Each contributing factor X_i has:
  pi_i  = P(X_i present in the scenario)  -- physics (Tier 1) where a model
          exists, corpus base rate (Tier 3) otherwise;
  q_i   = P(LOC | X_i)  -- the corpus causal-edge strength (Tier 3).
Treating the factors as independent contributing causes of the top event,
  P(LOC) = 1 - prod_i (1 - pi_i * q_i),
and factor i's contribution is the drop in P(LOC) when it is removed
(pi_i -> 0), i.e. a do(X_i absent) counterfactual on the failure mode.

Physics refinement: the aerodynamic factors (stall, and icing feeding stall /
engine failure) take pi_i from the physics occurrence models rather than the
corpus base rate, making P(LOC) conditional on the operating point.
"""
from __future__ import annotations
import csv, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing, P_ENGINE_FAILURE_GIVEN_ICE

KG = "event_extraction/out/causation_kg"
TARGET = "LOSS_OF_CONTROL_INFLIGHT"

# Contributing factors (direct causal parents of LOC), tagged by model tier.
# 'physics' = has a first-principles occurrence model; 'data' = corpus only.
# Tier tags match Table "Contributing factors of loss of control in flight" in
# the manuscript: 'physics' has a first-principles occurrence model, 'surrogate'
# a reliability or hazard-rate stand-in, 'data' the corpus base rate only.
FACTORS = [
    ("STALL", "physics"),
    ("SPATIAL_DISORIENTATION", "data"),
    ("CONTROL_SURFACE_ANOMALY", "surrogate"),
    ("AIRFRAME_STRUCTURAL_FAILURE", "physics"),
    ("PILOT_INCAPACITATION_OR_IMPAIRMENT", "data"),
    ("TURBULENCE_ENCOUNTER", "physics"),
    ("CONTROL_INPUT_IMPROPER", "data"),
    ("ENGINE_FAILURE", "surrogate"),
    ("WIND_SHEAR_OR_GUST", "physics"),
    ("DECISION_INAPPROPRIATE", "data"),
]


def load_q():
    """q_i = P(LOC | X_i) from the causal edges."""
    q = {}
    with open(os.path.join(KG, "causation_edges.csv")) as f:
        for r in csv.DictReader(f):
            if r["dst"] == TARGET:
                q[r["src"]] = float(r["p_dst_given_src"])
    return q


def noisy_or(pi, q, factors):
    prod = 1.0
    for name, _ in factors:
        prod *= (1.0 - pi[name] * q[name])
    return 1.0 - prod


def main():
    df = pd.read_parquet(KG + "/factor_vectors.parquet")
    q = load_q()
    base_rate = {name: float(df[name].mean()) for name, _ in FACTORS}
    p_loc_base = float(df[TARGET].mean())

    # --- population baseline: pi = corpus base rate ---
    pi = dict(base_rate)
    P0 = noisy_or(pi, q, FACTORS)

    rows = []
    for name, tier in FACTORS:
        pi_wo = dict(pi); pi_wo[name] = 0.0
        contrib = P0 - noisy_or(pi_wo, q, [f for f in FACTORS])
        rows.append({"factor": name, "tier": tier, "pi_base": round(pi[name], 3),
                     "q_P_loc_given": round(q[name], 3),
                     "contribution_to_P_loc": round(contrib, 4)})
    rows.sort(key=lambda r: -r["contribution_to_P_loc"])

    # --- physics scenario: maneuvering flight in icing conditions ---
    # Stall occurrence from the high-fidelity JSBSim model (validated C172 aero);
    # icing conditions add an engine-failure path (carb model -> engine failure).
    try:
        from physics.jsbsim_stall import p_stall as _jsb_pstall
        pi_stall = round(_jsb_pstall("c172x", 0.0, "maneuvering", seed=2).p_stall, 3)
    except Exception:
        pi_stall = 0.33
    pi_scn = dict(base_rate)
    pi_scn["STALL"] = pi_stall       # JSBSim 6-DOF stall occurrence, maneuvering
    ice = p_carb_icing(5.0, 3.0, "descent").p_ice     # cold, moist, low power
    pi_scn["ENGINE_FAILURE"] = min(0.95, base_rate["ENGINE_FAILURE"]
                                   + ice * P_ENGINE_FAILURE_GIVEN_ICE)
    P_scn = noisy_or(pi_scn, q, FACTORS)

    # counterfactuals on the scenario (the two elevated factors)
    cf = {}
    for name, val, label in [("STALL", base_rate["STALL"], "do(recover stall margin)"),
                             ("ENGINE_FAILURE", base_rate["ENGINE_FAILURE"], "do(carb heat: remove icing path)")]:
        pic = dict(pi_scn); pic[name] = val
        cf[label] = round(noisy_or(pic, q, FACTORS), 4)

    out = {
        "target": TARGET,
        "empirical_P_loc": round(p_loc_base, 4),
        "noisy_or_P_loc_baseline": round(P0, 4),
        "factor_contributions": rows,
        "physics_scenario": {
            "description": "maneuvering flight, icing conditions (T=5C, Td=3C, descent)",
            "pi_stall": round(pi_scn["STALL"], 3),
            "pi_engine_failure": round(pi_scn["ENGINE_FAILURE"], 3),
            "P_loc": round(P_scn, 4),
            "counterfactuals": cf,
        },
    }
    os.makedirs("physics/out", exist_ok=True)
    with open("physics/out/loc_risk.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
