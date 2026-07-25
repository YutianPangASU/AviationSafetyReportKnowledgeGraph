"""Failure-mode risk model for Case Study 2: ENGINE FAILURE (SCF-PP).

Identical machinery to loc_risk_model.py, applied to the engine-failure node:
contributing factors read from the causation KG, each carrying an occurrence
probability pi_i (physics where a mechanism exists, corpus base rate
otherwise) and a triggering strength q_i = P(EF | X_i) from the causal edge,
combined noisy-OR:

    P(EF | O) = 1 - prod_i (1 - pi_i(O) * q_i).

Physics-tier factors:
  * CARBURETOR_OR_INDUCTION_ICING -- physics/carb_icing_model.py (worked
    factor of Case Study 1, reused verbatim: the case studies interlock).
  * FUEL_EXHAUSTION_OR_STARVATION -- endurance mass balance: usable fuel m_f
    at fuel flow mdot gives endurance E = m_f / mdot; exhaustion occurs when
    flight time exceeds it, pi_fuel = P(t + reserve < 0) with a Gaussian
    margin absorbing flow/leaning/quantity uncertainty.
Surrogate tier: fuel system anomaly, fuel contamination (component rates).
Data tier: latent defect, maintenance, procedure, other system failure.

Outputs physics/out/ef_risk.json (drives fig_ef.pdf and the paper numbers).
"""
from __future__ import annotations
import csv, json, os, sys
from math import erf, sqrt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing

KG = "event_extraction/out/causation_kg"
TARGET = "ENGINE_FAILURE"

# Contributing factors (causal parents of EF), tagged by model tier.
FACTORS = [
    ("FUEL_EXHAUSTION_OR_STARVATION", "physics"),
    ("LATENT_MECHANICAL_DEFECT", "data"),
    ("PROCEDURE_NOT_FOLLOWED", "data"),
    ("OTHER_SYSTEM_FAILURE", "data"),
    ("CARBURETOR_OR_INDUCTION_ICING", "physics"),
    ("FUEL_SYSTEM_ANOMALY", "surrogate"),
    ("FUEL_CONTAMINATION", "surrogate"),
    ("MAINTENANCE_INADEQUATE", "data"),
]


def p_fuel_exhaustion(reserve_hr: float, sigma_hr: float = 0.75) -> float:
    """P(endurance shortfall): Gaussian margin on planned fuel reserve.

    E = m_fuel / mdot sets the endurance; the reserve is the planned margin
    E - t_flight, and sigma absorbs fuel-flow (leaning), quantity-indication
    and wind/rerouting uncertainty.
    """
    return 0.5 * (1.0 - erf(reserve_hr / (max(sigma_hr, 1e-6) * sqrt(2.0))))


def load_q():
    q = {}
    with open(os.path.join(KG, "causation_edges.csv")) as f:
        for r in csv.DictReader(f):
            if r["dst"] == TARGET:
                q[r["src"]] = (float(r["p_dst_given_src"]), int(r["support"]),
                               float(r["lift"]))
    return q


def noisy_or(pi, q, factors):
    prod = 1.0
    for name, _ in factors:
        prod *= (1.0 - pi[name] * q[name][0])
    return 1.0 - prod


def main():
    df = pd.read_parquet(KG + "/factor_vectors.parquet")
    q = load_q()
    base_rate = {name: float(df[name].mean()) for name, _ in FACTORS}
    p_ef_emp = float(df[TARGET].mean())

    pi = dict(base_rate)
    P0 = noisy_or(pi, q, FACTORS)

    rows = []
    for name, tier in FACTORS:
        pi_wo = dict(pi); pi_wo[name] = 0.0
        contrib = P0 - noisy_or(pi_wo, q, FACTORS)
        rows.append({"factor": name, "tier": tier,
                     "pi_base": round(pi[name], 4),
                     "q": round(q[name][0], 3), "support": q[name][1],
                     "lift": round(q[name][2], 2),
                     "contribution": round(contrib, 4)})
    rows.sort(key=lambda r: -r["contribution"])

    # --- physics scenario: icing weather + thin fuel planning ---------------
    ice = p_carb_icing(13.0, 12.0, "descent").p_ice     # serious icing wx
    pi_scn = dict(base_rate)
    pi_scn["CARBURETOR_OR_INDUCTION_ICING"] = ice
    pi_scn["FUEL_EXHAUSTION_OR_STARVATION"] = p_fuel_exhaustion(0.5)
    P_scn = noisy_or(pi_scn, q, FACTORS)

    cf = {}
    for label, name, val in [
        ("do(carb heat)", "CARBURETOR_OR_INDUCTION_ICING",
         base_rate["CARBURETOR_OR_INDUCTION_ICING"]),
        ("do(fuel reserve 2 h)", "FUEL_EXHAUSTION_OR_STARVATION",
         p_fuel_exhaustion(2.0)),
    ]:
        pic = dict(pi_scn); pic[name] = val
        cf[label] = round(noisy_or(pic, q, FACTORS), 4)
    pic = dict(pi_scn)
    pic["CARBURETOR_OR_INDUCTION_ICING"] = base_rate["CARBURETOR_OR_INDUCTION_ICING"]
    pic["FUEL_EXHAUSTION_OR_STARVATION"] = p_fuel_exhaustion(2.0)
    cf["do(both)"] = round(noisy_or(pic, q, FACTORS), 4)

    out = {
        "target": TARGET,
        "empirical_P_ef": round(p_ef_emp, 4),
        "noisy_or_P_ef_baseline": round(P0, 4),
        "factor_contributions": rows,
        "fuel_physics": {
            "pi_fuel(reserve=0.5h)": round(p_fuel_exhaustion(0.5), 4),
            "pi_fuel(reserve=1h)": round(p_fuel_exhaustion(1.0), 4),
            "pi_fuel(reserve=2h)": round(p_fuel_exhaustion(2.0), 4),
        },
        "physics_scenario": {
            "description": "serious icing weather (T=13C,Td=12C, descent) + 0.5-h fuel reserve",
            "pi_carb_ice": round(ice, 3),
            "pi_fuel": round(pi_scn["FUEL_EXHAUSTION_OR_STARVATION"], 3),
            "P_ef": round(P_scn, 4),
            "counterfactuals": cf,
        },
    }
    os.makedirs("physics/out", exist_ok=True)
    with open("physics/out/ef_risk.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
