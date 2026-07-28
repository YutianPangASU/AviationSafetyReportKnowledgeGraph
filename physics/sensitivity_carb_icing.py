"""Sensitivity of the carburetor-icing results to the two free constants.

The occurrence model carries two constants that were set to reproduce the
published induction icing chart rather than fitted to accident data: the
power-dependent venturi cooling dT(power) and the exposure constant K in
p_ice = 1 - exp(-K * w). A reader is entitled to ask whether the reported
discrimination and the reported counterfactual reductions survive a different
choice of those constants.

This script perturbs each constant over +/- 25 % and recomputes

  (a) the AUC separating narrative-attributed icing accidents from the rest,
  (b) the worked counterfactual R_ice = p_ice * P(engine failure | ice) at
      T = 13 C, Td = 12 C, descent power, together with the carburetor-heat
      and drier-air interventions.

K enters p_ice through a monotone transform of the ice index, so it cannot
change the ranking of reports and leaves the AUC exactly invariant; only the
absolute risk level moves. dT changes which reports reach the sub-freezing
region at all and so does move the AUC. Reporting both makes clear which
conclusions rest on a tuned constant and which do not.

Output: physics/out/carb_icing_sensitivity.json
"""
from __future__ import annotations

import json
import os
import sys
from math import exp

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics import carb_icing_model as cim  # noqa: E402
from physics.ceiling_carb_icing import load_joined  # noqa: E402
from physics.validate_carb_icing import FACTOR  # noqa: E402

OUT = "physics/out"
SCALES = [0.75, 0.9, 1.0, 1.1, 1.25]
SCENARIO = {"temp_c": 13.0, "dew_c": 12.0, "power": "descent"}
CARB_HEAT_RISE_C = 30.0   # heat restores this much to the intake charge
DRY_AIR_DEWPOINT_C = -5.0  # the drier-atmosphere counterfactual


def p_ice_with(temp_c: float, dew_c: float, dt: float, k: float) -> float:
    """p_carb_icing with the two constants supplied explicitly."""
    dew_c = min(dew_c, temp_c)
    throat = temp_c - dt
    if throat >= 0.0:
        return 0.0
    upper = min(dew_c, 0.0)
    w = max(0.0, cim.sat_vapor_pressure(upper) - cim.sat_vapor_pressure(throat))
    return 1.0 - exp(-k * w)


def scenario_risk(dt: float, k: float) -> dict:
    """Worked counterfactual of Section 'Counterfactual Results' under (dt, k)."""
    q = cim.P_ENGINE_FAILURE_GIVEN_ICE
    T, Td = SCENARIO["temp_c"], SCENARIO["dew_c"]
    base = p_ice_with(T, Td, dt, k)
    heat = p_ice_with(T + CARB_HEAT_RISE_C, Td, dt, k)
    dry = p_ice_with(T, DRY_AIR_DEWPOINT_C, dt, k)
    return {
        "p_ice": round(base, 4),
        "R_ice": round(base * q, 4),
        "R_ice_do_carb_heat": round(heat * q, 4),
        "R_ice_do_dry_air": round(dry * q, 4),
        "ranking_preserved": bool(heat * q < dry * q < base * q),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_joined()
    y = df[FACTOR].to_numpy(dtype=int)
    T = df["temp_c"].to_numpy(dtype=float)
    Td = df["dew_c"].to_numpy(dtype=float)

    dt0 = cim.VENTURI_COOLING_C["descent"]
    k0 = cim._K_EXPOSURE

    out = {
        "n_total": int(len(df)),
        "n_icing": int(y.sum()),
        "baseline_constants": {"dT_descent_C": dt0, "K": k0},
        "scenario": SCENARIO | {"carb_heat_rise_C": CARB_HEAT_RISE_C,
                                "dry_air_dewpoint_C": DRY_AIR_DEWPOINT_C},
        "vary_dT": [],
        "vary_K": [],
    }

    for s in SCALES:
        dt = dt0 * s
        score = np.array([p_ice_with(t, d, dt, k0) for t, d in zip(T, Td)])
        out["vary_dT"].append({
            "scale": s, "dT_descent_C": round(dt, 2),
            "auc": round(float(roc_auc_score(y, score)), 4),
            **scenario_risk(dt, k0),
        })

    for s in SCALES:
        k = k0 * s
        score = np.array([p_ice_with(t, d, dt0, k) for t, d in zip(T, Td)])
        out["vary_K"].append({
            "scale": s, "K": round(k, 4),
            "auc": round(float(roc_auc_score(y, score)), 4),
            **scenario_risk(dt0, k),
        })

    aucs_dt = [r["auc"] for r in out["vary_dT"]]
    aucs_k = [r["auc"] for r in out["vary_K"]]
    out["summary"] = {
        "auc_range_over_dT": [min(aucs_dt), max(aucs_dt)],
        "auc_range_over_K": [min(aucs_k), max(aucs_k)],
        "auc_invariant_to_K": bool(max(aucs_k) - min(aucs_k) < 1e-9),
        "counterfactual_ranking_preserved_everywhere": bool(
            all(r["ranking_preserved"] for r in out["vary_dT"] + out["vary_K"])),
    }

    with open(os.path.join(OUT, "carb_icing_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'scale':>6} {'dT':>6} {'AUC':>7} {'R_ice':>7} {'do(heat)':>9} {'do(dry)':>8}")
    for r in out["vary_dT"]:
        print(f"{r['scale']:>6.2f} {r['dT_descent_C']:>6.1f} {r['auc']:>7.4f} "
              f"{r['R_ice']:>7.3f} {r['R_ice_do_carb_heat']:>9.3f} "
              f"{r['R_ice_do_dry_air']:>8.3f}")
    print()
    print(f"{'scale':>6} {'K':>6} {'AUC':>7} {'R_ice':>7} {'do(heat)':>9} {'do(dry)':>8}")
    for r in out["vary_K"]:
        print(f"{r['scale']:>6.2f} {r['K']:>6.3f} {r['auc']:>7.4f} "
              f"{r['R_ice']:>7.3f} {r['R_ice_do_carb_heat']:>9.3f} "
              f"{r['R_ice_do_dry_air']:>8.3f}")
    print()
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
