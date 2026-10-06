"""Failure Study II: P(engine failure) from its contributing factors.

Same machinery as loc_risk_model.py (leaky noisy-OR fitted by maximum
likelihood over the leading-factor set of Eq. (leading)), applied to the
engine-failure node. Physics-tier parents:

  * CARBURETOR_OR_INDUCTION_ICING: the icing physics of carb_icing_model.py,
    entering through the calibrated link of calibrate_physics.py (an
    isotonic link on the condensable water w, fitted on the weather-recorded
    NTSB accidents). The link is rescaled to the corpus denominator
    (calibrated_corpus = corpus base rate x link / prevalence in D), so the
    scenario value sits on the same 55,940-record population as the base
    rates of the other parents (revised 2026-09-30).
  * FUEL_EXHAUSTION_OR_STARVATION: endurance margin. Usable fuel m_f at fuel
    flow mdot gives an endurance E = m_f / mdot; exhaustion occurs when the
    flight outlasts it. With a planned reserve r = E - t_flight and a
    Gaussian margin of scale sigma, pi_fuel = Phi(-r / sigma). The scale is
    multiplicative in flight length, sigma = c t_flight (review 2026-09-14),
    with c = 0.25 so that a 3 h flight has sigma = 0.75 h. NOT calibrated:
    fuel on board and flight time are not coded fields, so Eq. (calib) has no
    operating-point distribution to average over, and the fuel term enters
    at its physics level. The manuscript reports the fuel scenario as a
    ratio and labels the term uncalibrated.

Output: physics/out/ef_risk.json (drives fig_ef.pdf, fig_ef_kg.pdf and the
manuscript numbers).
"""
from __future__ import annotations

import json
import os
import sys
from math import erf, sqrt

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.leaky_noisy_or import (RHO, combine, contributions,  # noqa: E402
                                    fit_leaky_noisy_or, leading_factors,
                                    load_edges, load_presence, tier)

TARGET = "ENGINE_FAILURE"
ICE = "CARBURETOR_OR_INDUCTION_ICING"
FUEL = "FUEL_EXHAUSTION_OR_STARVATION"
CAL = "physics/out/calibration.json"
OUT = "physics/out/ef_risk.json"
FUEL_SIGMA_FRACTION = 0.25   # sigma = 0.25 x flight time
T_FLIGHT_HR = 3.0


def fuel_margin_hr(t_flight_hr: float, endurance_hr: float) -> float:
    """Guard margin (paper Table tbl:guards): t - t_end in hours."""
    return t_flight_hr - endurance_hr


def p_fuel_exhaustion(reserve_hr: float, t_flight_hr: float = T_FLIGHT_HR,
                      c: float = FUEL_SIGMA_FRACTION) -> float:
    """Availability of the fuel-exhaustion guard, P(t - t_end >= 0), with a
    Gaussian margin whose scale grows with the flight length."""
    sigma = max(c * t_flight_hr, 1e-6)
    return 0.5 * (1.0 - erf(reserve_hr / (sigma * sqrt(2.0))))


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
    P0 = float(combine(base, p, leak))          # Eq. (combine) at the base rates
    delta = contributions(base, p, leak)
    # mean over records of the fitted P(y | x_r); differs from P0 because the
    # parents are not independent in the presence table
    fitted_mean_records = float(np.mean(1.0 - (1.0 - leak) * np.exp(X @ np.log1p(-p))))

    # chain-level parent coverage per record: records whose EF node has an
    # explicit caused_by parent in the leading-factor set (review 2026-09-30).
    # 369 chains repeat the EF node, so the parents of all EF nodes in a record
    # are pooled and the record counts once (review 2026-10-06).
    cover = {"records": 0, "root": 0, "parent_in_set": 0, "parents_outside_set": 0}
    with open(os.path.join("event_extraction/out/causation_kg", "per_accident_chains.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            nodes = {n["idx"]: n for n in rec["chain"]}
            ef_nodes = [n for n in rec["chain"] if n["factor_type"] == TARGET]
            if not ef_nodes:
                continue
            par = {nodes[e["src"]]["factor_type"] for n in ef_nodes
                   for e in n.get("caused_by", []) if e["src"] in nodes} - {TARGET}
            cover["records"] += 1
            if not par:
                cover["root"] += 1
            elif par & set(names):
                cover["parent_in_set"] += 1
            else:
                cover["parents_outside_set"] += 1

    rows = []
    for e, b, pi_, d in zip(lf, base, p, delta):
        t, tag = tier(e["src"])
        rows.append({"factor": e["src"], "tier": t, "model": tag,
                     "support": e["support"], "q": round(e["q"], 4),
                     "lift": round(e["lift"], 3), "base_rate": round(float(b), 4),
                     "p": round(float(pi_), 4), "contribution": round(float(d), 4)})

    ice_pts = cal[ICE]["points"]
    pi_ice = ice_pts["worked_T13_Td12_descent"]["calibrated_corpus"]
    pi_ice_heat = ice_pts["heat_T43_Td12_descent"]["calibrated_corpus"]
    pi_ice_drier = ice_pts["drier_T13_Tdm5_descent"]["calibrated_corpus"]
    pi_fuel_thin, pi_fuel_ok = p_fuel_exhaustion(0.5), p_fuel_exhaustion(2.0)

    def evaluate(overrides: dict[str, float]) -> dict:
        pi = base.copy()
        for k, v in overrides.items():
            pi[names.index(k)] = v
        P = float(combine(pi, p, leak))
        return {"pi_overrides": {k: round(v, 4) for k, v in overrides.items()},
                "P": round(P, 4), "ratio_vs_baseline": round(P / P0, 3)}

    ice_only = evaluate({ICE: pi_ice})
    scn_over = {ICE: pi_ice, FUEL: pi_fuel_thin}
    scenario = evaluate(scn_over)
    cfs = {
        "do(carburetor heat)": evaluate({**scn_over, ICE: pi_ice_heat}),
        "do(restore 2 h reserve)": evaluate({**scn_over, FUEL: pi_fuel_ok}),
        "do(both)": evaluate({ICE: pi_ice_heat, FUEL: pi_fuel_ok}),
    }
    for v in cfs.values():
        v["ratio_vs_scenario"] = round(v["P"] / scenario["P"], 3)
    ice_cfs = {
        "do(carburetor heat)": evaluate({ICE: pi_ice_heat}),
        "do(drier air, dewpoint -5 C)": evaluate({ICE: pi_ice_drier}),
    }
    for v in ice_cfs.values():
        v["ratio_vs_icing_scenario"] = round(v["P"] / ice_only["P"], 3)

    out = {
        "target": TARGET,
        "reference_class": "given a reportable accident; the fuel term is uncalibrated",
        "leading_factor_rule": {"rho": RHO, "records_with_target": int(y.sum()),
                                "support_threshold": round(RHO * int(y.sum()), 1),
                                "n_parents": len(names)},
        "leak": round(leak, 4),
        "fitted_mean": round(P0, 4),
        "population_value_note": "fitted_mean is Eq. (combine) evaluated at the corpus base "
                                 "rates; fitted_mean_over_records averages the fitted P(y | x_r)",
        "fitted_mean_over_records": round(fitted_mean_records, 4),
        "empirical_prevalence": round(float(y.mean()), 4),
        "parent_coverage": {**cover, "share_parent_in_set": round(cover["parent_in_set"] / cover["records"], 4),
                            "share_root": round(cover["root"] / cover["records"], 4)},
        "nll_per_record": round(fit["nll_per_record"], 4),
        "factors": rows,
        "icing_term": {"chart_pi_worked_point": round(ice_pts["worked_T13_Td12_descent"]["chart"], 4),
                       "link_on_D_worked_point": round(ice_pts["worked_T13_Td12_descent"]["calibrated"], 4),
                       "prevalence_in_D": round(cal[ICE]["prevalence_in_D"], 4),
                       "calibrated_worked_point": round(pi_ice, 4),
                       "calibrated_with_heat": round(pi_ice_heat, 4),
                       "calibrated_drier": round(pi_ice_drier, 4),
                       "base_rate": round(float(base[names.index(ICE)]), 4)},
        "fuel_term": {"sigma_fraction": FUEL_SIGMA_FRACTION, "t_flight_hr": T_FLIGHT_HR,
                      "pi_fuel(reserve=0.5h)": round(p_fuel_exhaustion(0.5), 4),
                      "pi_fuel(reserve=1h)": round(p_fuel_exhaustion(1.0), 4),
                      "pi_fuel(reserve=2h)": round(p_fuel_exhaustion(2.0), 4),
                      "calibrated": False},
        "icing_scenario": {"description": "serious icing weather, T 13 C, Td 12 C, descent power",
                           **ice_only, "counterfactuals": ice_cfs},
        "scenario": {"description": "serious icing weather with a 0.5 h fuel reserve on a 3 h flight",
                     **scenario, "counterfactuals": cfs},
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
