"""Uncertainty on the combined failure-mode risk for both failure studies.

Revised 2026-09-14. The combination is the leaky noisy-OR fitted by maximum
likelihood (leaky_noisy_or.py). Uncertainty in the corpus terms is obtained
by a nonparametric bootstrap over records: each resample refits the leak and
the per-parent strengths p_i and recomputes the base rates, so the intervals
cover the sampling variability of every corpus-estimated quantity at once.

The physics-tier occurrence probabilities used in the scenarios are held at
their calibrated values, so the intervals cover the corpus terms only. The
physics terms carry model-form uncertainty that resampling cannot express;
sensitivity_carb_icing.py addresses it by perturbing the constants.

Reads the scenario definitions from loc_risk.json and ef_risk.json so the
point estimates and the intervals come from one specification.
Output: physics/out/risk_uncertainty.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.leaky_noisy_or import (bootstrap_fit, combine,  # noqa: E402
                                    fit_leaky_noisy_or, load_presence)

N_BOOT = 500
SEED = 0


def ci(v: np.ndarray) -> list[float]:
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


def run(spec: dict, df) -> dict:
    target = spec["target"]
    names = [r["factor"] for r in spec["factors"]]
    X = df[names].to_numpy(dtype=np.int8)
    y = df[target].to_numpy(dtype=np.int8)
    fit0 = fit_leaky_noisy_or(X, y)
    leaks, ps, pis = bootstrap_fit(X, y, N_BOOT, SEED, theta0=fit0["theta"])

    base = combine(pis, ps, leaks)
    contrib = np.empty((N_BOOT, len(names)))
    for j in range(len(names)):
        pj = pis.copy()
        pj[:, j] = 0.0
        contrib[:, j] = base - combine(pj, ps, leaks)

    def draws(overrides: dict[str, float]) -> np.ndarray:
        pi = pis.copy()
        for k, v in overrides.items():
            pi[:, names.index(k)] = v
        return combine(pi, ps, leaks)

    def block(entry: dict) -> dict:
        v = draws(entry["pi_overrides"])
        return {"mean": round(float(v.mean()), 4), "ci95": ci(v),
                "ratio_vs_baseline_ci95": ci(v / base)}

    out = {
        "target": target, "n_records": int(len(df)), "n_boot": N_BOOT,
        "empirical_prevalence": round(float(y.mean()), 4),
        "leak": {"mean": round(float(leaks.mean()), 4), "ci95": ci(leaks)},
        "combined_baseline": {"mean": round(float(base.mean()), 4), "ci95": ci(base)},
        "factors": [{"factor": names[j], "tier": spec["factors"][j]["tier"],
                     "pi_base": round(float(pis[:, j].mean()), 4), "pi_ci95": ci(pis[:, j]),
                     "p": round(float(ps[:, j].mean()), 4), "p_ci95": ci(ps[:, j]),
                     "contribution": round(float(contrib[:, j].mean()), 4),
                     "contribution_ci95": ci(contrib[:, j])}
                    for j in range(len(names))],
        "scenario": {"combined": block(spec["scenario"]),
                     "counterfactuals": {k: block(v) for k, v in
                                         spec["scenario"]["counterfactuals"].items()}},
    }
    if "icing_scenario" in spec:
        out["icing_scenario"] = {"combined": block(spec["icing_scenario"]),
                                 "counterfactuals": {k: block(v) for k, v in
                                                     spec["icing_scenario"]["counterfactuals"].items()}}
    if "variants" in spec:
        out["variants"] = {k: block(v) for k, v in spec["variants"].items()}
    return out


def main() -> None:
    df = load_presence()
    out = {"loss_of_control": run(json.load(open("physics/out/loc_risk.json")), df),
           "engine_failure": run(json.load(open("physics/out/ef_risk.json")), df)}
    with open("physics/out/risk_uncertainty.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, r in out.items():
        print(f"\n=== {k}: empirical {r['empirical_prevalence']}, combined "
              f"{r['combined_baseline']['mean']} {r['combined_baseline']['ci95']}, "
              f"leak {r['leak']['mean']} {r['leak']['ci95']}")
        for f_ in r["factors"]:
            print(f"  {f_['factor']:38s} pi={f_['pi_base']:.3f} p={f_['p']:.3f} "
                  f"{f_['p_ci95']} Delta={f_['contribution']:.4f} {f_['contribution_ci95']}")
        s = r["scenario"]
        print(f"  scenario {s['combined']}")
        for lab, v in s["counterfactuals"].items():
            print(f"    {lab:40s} {v}")


if __name__ == "__main__":
    main()
