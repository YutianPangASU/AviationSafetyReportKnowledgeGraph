"""Execute the calibration of paper Eq. (calib) for each physics-tier factor.

Eq. (calib) asks that the occurrence model, averaged over the observed
operating-point distribution D of the accident population, reproduce the
corpus marginal P(F). The physics supplies the shape of the occurrence
probability across operating points and the corpus its level. This script
executes the calibration wherever the drivers of a factor are recorded, so
the manuscript can state for which factors Eq. (calib) was run, over which
D, and with what result. The reference class of every calibrated value is
"given a reportable accident", the population the corpus observes.

Executions:

  isotonic link    pi_cal(O) = g(score(O)), g monotone, fitted out of fold on
                   the physics score (the condensable water, or the exceedance
                   availability) against the extracted factor label. Its mean
                   over D equals the prevalence by construction. Used where a
                   per-record physics score exists: carburetor icing (score =
                   condensable water w in hPa from temperature and dewpoint;
                   revised 2026-09-30 from the g/m3 guard margin, which at
                   w = 0 still varies with the throttle-plate temperature and
                   so gave the warmed operating points different floors),
                   turbulence (score = Dryden gust-stall exceedance from the
                   recorded surface wind) and wind shear (score = shear
                   margin from the recorded surface wind).
  stratified       pi_cal(phase, n) = P(F | phase) * Lambda(phase, n) /
  rescale          Lambda(phase), where Lambda(phase) is the availability
                   marginalized over the phase prior of jsbsim_stall.py.
                   Used for stall, whose drivers (airspeed, weight, load
                   factor) are unrecorded, so only the phase stratum is
                   observed and the physics supplies the within-phase shape.
  scalar rescale   pi_cal(O) = P(F) * Lambda(O) / mean_D Lambda(O). Reported
                   as a diagnostic only: for the exceedance-type models the
                   mean of Lambda over D is dominated by a few extreme records
                   and the rescale saturates at 1 for moderate scenarios.
  not executed     fuel exhaustion: fuel on board and flight time are not
                   coded fields, so D does not exist. The fuel term enters
                   the engine-failure study at its physics level and the
                   manuscript labels it uncalibrated.

Output: physics/out/calibration.json, read by loc_risk_model.py and
ef_risk_model.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing  # noqa: E402
from physics.factor_models import C172, p_gust_stall_dryden  # noqa: E402
from physics.violation_rate import build_driver_index, CHAINS  # noqa: E402

KG = "event_extraction/out/causation_kg"
OUT = "physics/out/calibration.json"
STALL_CACHE = "physics/out/stall_prior_values.json"
ICE, TURB, SHEAR, STALL = ("CARBURETOR_OR_INDUCTION_ICING", "TURBULENCE_ENCOUNTER",
                           "WIND_SHEAR_OR_GUST", "STALL")
V_APPROACH = 1.25 * C172.vs1g_kcas
V_APPROACH_PLUS = 1.44 * C172.vs1g_kcas   # a 10 kt higher approach speed
N_FOLDS = 5
SEED = 0
PHASE_MAP = {"initial_climb": "climb", "go_around": "takeoff",
             "final_approach": "approach", "descent": "cruise",
             "en_route": "cruise", "emergency_descent": "approach"}


def isotonic_link(score: np.ndarray, y: np.ndarray) -> dict:
    """Out-of-fold isotonic fit of P(y | score) and the full-data link."""
    pred = np.zeros(len(y))
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(score, y):
        iso = IsotonicRegression(out_of_bounds="clip").fit(score[tr], y[tr])
        pred[te] = iso.predict(score[te])
    full = IsotonicRegression(out_of_bounds="clip").fit(score, y)
    return {"model": full,
            "auc_oof": float(roc_auc_score(y, pred)),
            "brier_oof": float(np.mean((pred - y) ** 2)),
            "brier_base_rate": float(y.mean() * (1 - y.mean())),
            "mean_oof": float(pred.mean()), "prevalence_in_D": float(y.mean()),
            "n_D": int(len(y))}


def curve(model, lo, hi, n=300):
    xs = np.linspace(lo, hi, n)
    return {"score": [round(float(v), 6) for v in xs],
            "p": [round(float(v), 5) for v in model.predict(xs)]}


def shear_margin(wind_kts: float, v_kcas: float) -> float:
    """Shear guard margin, Table tbl:guards: the shear loss the recorded wind
    can supply, wind, against the loss required, V - V_s1g (knots)."""
    return wind_kts - (v_kcas - C172.vs1g_kcas)


def main() -> None:
    os.makedirs("physics/out", exist_ok=True)
    df = pd.read_parquet(os.path.join(KG, "factor_vectors.parquet"))
    df["ev_id"] = df["record_id"].str.rsplit("_", n=1).str[0]
    prevalence = {c: float(df[c].mean()) for c in
                  (STALL, TURB, SHEAR, ICE, "FUEL_EXHAUSTION_OR_STARVATION")}
    drv = build_driver_index()
    out: dict = {"reference_class": "given a reportable accident",
                 "prevalence": prevalence, "factors": {}}

    # --- carburetor icing: isotonic link on the condensable water w (hPa) -----
    # The score is w itself, so every operating point with no condensable
    # subfreezing water ties at w = 0 and the link returns one floor value
    # there (the icing share of the accidents with w = 0). The link is fitted
    # on D, whose icing prevalence differs from the corpus base rate used in
    # the combination of Eq. (combine); "calibrated_corpus" rescales the link
    # to the corpus denominator, pi = P_corpus(ice) g(w) / P_D(ice), i.e. the
    # link supplies the relative risk and the corpus the level.
    wx = df[df["ev_id"].map(lambda e: e in drv and drv[e].temp_c is not None)]
    T = wx["ev_id"].map(lambda e: drv[e].temp_c).to_numpy(float)
    Td = wx["ev_id"].map(lambda e: drv[e].dew_c).to_numpy(float)
    y = wx[ICE].to_numpy(int)
    res = [p_carb_icing(t, d, "descent") for t, d in zip(T, Td)]
    pi_chart = np.array([r.p_ice for r in res])
    w = np.array([r.ice_index_hpa for r in res])
    link = isotonic_link(w, y)
    g = link.pop("model")
    to_corpus = prevalence[ICE] / float(y.mean())

    def point(temp_c: float, dew_c: float) -> dict:
        r = p_carb_icing(temp_c, dew_c, "descent")
        cal = float(g.predict([r.ice_index_hpa])[0])
        return {"chart": r.p_ice, "w_hpa": r.ice_index_hpa, "margin": r.margin_gm3,
                "calibrated": cal, "calibrated_corpus": cal * to_corpus}

    pts = {"worked_T13_Td12_descent": point(13.0, 12.0),
           "heat_T43_Td12_descent": point(43.0, 12.0),
           "drier_T13_Tdm5_descent": point(13.0, -5.0)}
    pts["worked_T13_Td12_descent"]["scalar_rescaled"] = float(
        min(1.0, pts["worked_T13_Td12_descent"]["chart"] * y.mean() / pi_chart.mean()))
    zero = w == 0
    out["factors"][ICE] = {
        "method": "isotonic link on the condensable water w (hPa)",
        "D": "NTSB accidents with usable temperature and dewpoint",
        **link,
        "corpus_base_rate": prevalence[ICE],
        "to_corpus_factor": to_corpus,
        "floor": {"n_w_zero": int(zero.sum()), "share_of_D": float(zero.mean()),
                  "icing_share": float(y[zero].mean()),
                  "link_value": float(g.predict([0.0])[0]),
                  "link_value_corpus": float(g.predict([0.0])[0]) * to_corpus},
        "chart_mean_over_D": float(pi_chart.mean()),
        "scalar_rescale_factor": float(y.mean() / pi_chart.mean()),
        "points": pts,
        "knots": {"w_hpa": [round(float(v), 6) for v in g.X_thresholds_],
                  "p": [round(float(v), 6) for v in g.y_thresholds_]},
        "curve": curve(g, 0.0, float(w.max())),
    }

    # --- turbulence: isotonic link on the Dryden gust-stall exceedance ---------
    ww = df[df["ev_id"].map(lambda e: e in drv and drv[e].wind_kts is not None)]
    wind = ww["ev_id"].map(lambda e: drv[e].wind_kts).to_numpy(float)
    lam = np.array([p_gust_stall_dryden(C172, V_APPROACH, w) for w in wind])
    yt = ww[TURB].to_numpy(int)
    link_t = isotonic_link(lam, yt)
    gt = link_t.pop("model")
    scen = {}
    for w in (10, 20, 25, 30, 35, 40):
        l1 = p_gust_stall_dryden(C172, V_APPROACH, w)
        l2 = p_gust_stall_dryden(C172, V_APPROACH_PLUS, w)
        scen[str(w)] = {"lambda": l1, "calibrated": float(gt.predict([l1])[0]),
                        "lambda_plus10kt": l2, "calibrated_plus10kt": float(gt.predict([l2])[0]),
                        "scalar_rescaled": float(min(1.0, prevalence[TURB] * l1 / lam.mean()))}
    out["factors"][TURB] = {
        "method": "isotonic link on the Dryden gust-stall exceedance at 1.25 Vs",
        "D": "NTSB accidents with a recorded surface wind", **link_t,
        "wind_kts_quantiles": {str(q): float(np.percentile(wind, q)) for q in (10, 50, 90)},
        "lambda_mean_over_D": float(lam.mean()),
        "scalar_rescale_factor": float(prevalence[TURB] / lam.mean()),
        "scenario_by_wind_kts": scen,
        "curve": curve(gt, 0.0, float(lam.max())),
    }

    # --- wind shear: isotonic link on the shear margin -------------------------
    sm = np.array([shear_margin(w, V_APPROACH) for w in wind])
    ys = ww[SHEAR].to_numpy(int)
    link_s = isotonic_link(sm, ys)
    gs = link_s.pop("model")
    scen_s = {}
    for w in (10, 20, 25, 30, 35, 40):
        m1, m2 = shear_margin(w, V_APPROACH), shear_margin(w, V_APPROACH_PLUS)
        scen_s[str(w)] = {"margin": m1, "calibrated": float(gs.predict([m1])[0]),
                          "margin_plus10kt": m2, "calibrated_plus10kt": float(gs.predict([m2])[0])}
    out["factors"][SHEAR] = {
        "method": "isotonic link on the shear margin wind - (V - Vs) at 1.25 Vs",
        "D": "NTSB accidents with a recorded surface wind", **link_s,
        "scenario_by_wind_kts": scen_s,
        "curve": curve(gs, float(sm.min()), float(sm.max())),
    }

    # --- stall: stratified rescale over the phase of the first event node ----
    stall = json.load(open(STALL_CACHE))
    lam_ph = {k: v["p_stall"] for k, v in stall["by_phase"].items()}
    ph = {}
    with open(CHAINS) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            ev = [n for n in rec.get("chain", []) if n.get("class") == "event"]
            p = (ev[0].get("phase_of_flight") if ev else None) or "unknown"
            ph[rec["record_id"]] = PHASE_MAP.get(p, p)
    d = df.set_index("record_id").join(pd.Series(ph, name="phase"), how="inner")
    rate = d.groupby("phase")[STALL].mean().to_dict()
    size = d.groupby("phase")[STALL].size().to_dict()
    man = stall["maneuvering_load_factor"]
    lam_man = man["default"]["p_stall"]
    out["factors"][STALL] = {
        "method": "phase-stratified rescale: P(stall | phase) x Lambda(phase, n) / Lambda(phase)",
        "D": "phase of flight of the first event node of every chain, with the "
             "phase prior of jsbsim_stall.py inside each stratum",
        "n_D": int(len(d)),
        "phase_rate": {k: round(float(v), 4) for k, v in rate.items()},
        "phase_size": {k: int(v) for k, v in size.items()},
        "lambda_by_phase": lam_ph,
        "scalar_rescale_factor": float(prevalence[STALL] / sum(
            size[p] / len(d) * lam_ph.get(p, 0.0) for p in size)),
        "scenario_maneuvering": {
            lab: {"n_mean": v["n_mean"], "n_sd": v["n_sd"], "lambda": v["p_stall"],
                  "calibrated": float(min(1.0, rate["maneuvering"] * v["p_stall"] / lam_man))}
            for lab, v in man.items()},
    }
    out["factors"]["FUEL_EXHAUSTION_OR_STARVATION"] = {
        "method": None, "executed": False,
        "reason": "fuel on board and flight time are not coded fields, so no D exists"}

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out["factors"].items():
        keys = ("method", "n_D", "prevalence_in_D", "auc_oof", "brier_oof",
                "brier_base_rate", "mean_oof", "scalar_rescale_factor", "executed")
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in v.items() if kk in keys})
    print("icing points:", json.dumps(out["factors"][ICE]["points"], indent=1))
    print("turbulence scenario:", json.dumps(out["factors"][TURB]["scenario_by_wind_kts"], indent=1))
    print("shear scenario:", json.dumps(out["factors"][SHEAR]["scenario_by_wind_kts"], indent=1))
    print("stall:", out["factors"][STALL]["phase_rate"], out["factors"][STALL]["scenario_maneuvering"])


if __name__ == "__main__":
    main()
