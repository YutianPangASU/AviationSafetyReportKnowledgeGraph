"""Failure Study I: P(loss of control in flight) from its contributing factors.

Revised 2026-09-14 after review. The combination is the leaky noisy-OR of
leaky_noisy_or.py, fitted by maximum likelihood on the presence table over
the leading-factor set of Eq. (leading) (edge support at least RHO times the
records carrying the mode). Each parent enters with an occurrence
probability pi_i: the corpus base rate for data-tier parents and, in a
scenario, the calibrated physics value of calibrate_physics.py for the
physics-tier parents. Every calibrated value refers to the accident
population ("given a reportable accident"), so the scenario and the
baseline share one reference class and their ratio is meaningful.

Scenario: low-speed maneuvering in a 35 kt surface wind. Stall takes the
phase-stratified calibrated value for maneuvering flight under the default
load-factor prior; turbulence takes the calibrated Dryden exceedance at the
recorded wind. Interventions act on physical drivers: a coordinated turn at
mean load factor 1.2 (stall margin), and a 10 kt higher airspeed against the
gust (gust margin).

Note on the icing path: the edge engine failure -> loss of control has lift
0.35 within the accident population and the fitted p is zero, so carburetor
icing no longer enters this study through engine failure. The icing physics
is exercised in the engine-failure study (ef_risk_model.py).

Output: physics/out/loc_risk.json (drives fig_loc.pdf, fig_loc_kg.pdf and
the numbers in the manuscript).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.leaky_noisy_or import (RHO, combine, contributions,  # noqa: E402
                                    fit_leaky_noisy_or, leading_factors,
                                    load_edges, load_presence, tier)

TARGET = "LOSS_OF_CONTROL_INFLIGHT"
CAL = "physics/out/calibration.json"
OUT = "physics/out/loc_risk.json"
SCENARIO_WIND_KTS = "35"


def main() -> None:
    df = load_presence()
    edges = load_edges()
    cal = json.load(open(CAL))["factors"]

    y = df[TARGET].to_numpy(dtype=np.int8)
    lf = leading_factors(edges, TARGET, int(y.sum()))
    names = [e["src"] for e in lf]
    X = df[names].to_numpy(dtype=np.int8)
    fit = fit_leaky_noisy_or(X, y)
    leak, p = fit["leak"], fit["p"]
    base = X.mean(axis=0)
    P0 = float(combine(base, p, leak))
    delta = contributions(base, p, leak)

    rows = []
    for e, b, pi_, d in zip(lf, base, p, delta):
        t, tag = tier(e["src"])
        rows.append({"factor": e["src"], "tier": t, "model": tag,
                     "support": e["support"], "q": round(e["q"], 4),
                     "lift": round(e["lift"], 3), "base_rate": round(float(b), 4),
                     "p": round(float(pi_), 4), "contribution": round(float(d), 4)})

    # --- scenario: low-speed maneuvering in a 35 kt surface wind -------------
    st = cal["STALL"]["scenario_maneuvering"]
    tb = cal["TURBULENCE_ENCOUNTER"]["scenario_by_wind_kts"][SCENARIO_WIND_KTS]
    i_st, i_tb = names.index("STALL"), names.index("TURBULENCE_ENCOUNTER")

    def evaluate(overrides: dict[str, float]) -> dict:
        pi = base.copy()
        for k, v in overrides.items():
            pi[names.index(k)] = v
        P = float(combine(pi, p, leak))
        return {"pi_overrides": {k: round(v, 4) for k, v in overrides.items()},
                "P": round(P, 4), "ratio_vs_baseline": round(P / P0, 3)}

    scn_over = {"STALL": st["default"]["calibrated"], "TURBULENCE_ENCOUNTER": tb["calibrated"]}
    scenario = evaluate(scn_over)
    cfs = {
        "do(coordinated turn, load factor 1.2)": evaluate(
            {**scn_over, "STALL": st["gentle"]["calibrated"]}),
        "do(10 kt gust margin)": evaluate(
            {**scn_over, "TURBULENCE_ENCOUNTER": tb["calibrated_plus10kt"]}),
        "do(both)": evaluate({"STALL": st["gentle"]["calibrated"],
                              "TURBULENCE_ENCOUNTER": tb["calibrated_plus10kt"]}),
    }
    for k, v in cfs.items():
        v["ratio_vs_scenario"] = round(v["P"] / scenario["P"], 3)
    variants = {
        "maneuvering only (calm wind)": evaluate({"STALL": st["default"]["calibrated"]}),
        "aggressive turn, load factor 1.8, 35 kt wind": evaluate(
            {**scn_over, "STALL": st["aggressive"]["calibrated"]}),
        "35 kt wind only": evaluate({"TURBULENCE_ENCOUNTER": tb["calibrated"]}),
    }

    out = {
        "target": TARGET,
        "reference_class": "given a reportable accident",
        "leading_factor_rule": {"rho": RHO, "records_with_target": int(y.sum()),
                                "support_threshold": round(RHO * int(y.sum()), 1),
                                "n_parents": len(names)},
        "leak": round(leak, 4),
        "fitted_mean": round(P0, 4),
        "empirical_prevalence": round(float(y.mean()), 4),
        "nll_per_record": round(fit["nll_per_record"], 4),
        "factors": rows,
        "scenario": {"description": "low-speed maneuvering in a 35 kt surface wind",
                     "stall_source": "phase-stratified calibration, maneuvering, "
                                     "load factor Normal(1.55, 0.45)",
                     "turbulence_source": "isotonic link on the Dryden exceedance at 35 kt",
                     **scenario, "counterfactuals": cfs},
        "variants": variants,
    }
    os.makedirs("physics/out", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "factors"}, indent=2))
    for r in rows:
        print(f"  {r['factor']:38s} n={r['support']:5d} q={r['q']:.3f} lift={r['lift']:.2f} "
              f"base={r['base_rate']:.3f} p={r['p']:.3f} Delta={r['contribution']:.4f} {r['tier']}")


if __name__ == "__main__":
    main()
