"""Feasibility check: NTSB injury/damage fields as risk ground truth.

Joins the extracted factor vectors (55,940 records with resolved severity)
to the NTSB events/aircraft tables (weather drivers, highest injury, damage,
light, phase, aircraft class) and asks, per layer of the hybrid framework,
what the recorded outcome can and cannot validate:

  occurrence layer   physics pi_ice vs the icing-factor label (AUC, reliability
                     by decile, pairwise weather-bin ordering), and a calibrated
                     link (isotonic on the guard margin, out of fold) that fixes
                     the level inside the accident population;
  consequence layer  extracted factors vs realized serious/fatal, out of fold
                     and time-blocked, against a pre-outcome covariate model
                     (weather, light, phase, aircraft, FAR part) to size the
                     narrative leakage;
  fused depth-2 risk P_cal(ice|wx) * P(EF|ice) vs the realized joint (ice AND
                     engine failure);
  physics vs severity  whether pi_ice carries any severity signal at all.

Usage (qwen-vllm env has sklearn + pandas):
  ~/miniconda3/envs/qwen-vllm/bin/python physics/severity_groundtruth_check.py
Output: physics/out/severity_groundtruth_check.json  (about 4 minutes)
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict  # noqa: E402

DBS = ["data/NTSB_ASRS/avall.mdb", "data/NTSB_ASRS/Pre2008.mdb"]
FV = "event_extraction/out/causation_kg/factor_vectors.parquet"
OUT = "physics/out/severity_groundtruth_check.json"
OUTCOMES = {"GROUND_IMPACT", "WATER_IMPACT", "MIDAIR_COLLISION", "GROUND_COLLISION",
            "RUNWAY_EXCURSION_OR_OVERRUN", "INFLIGHT_BREAKUP", "EMERGENCY_LANDING",
            "SUCCESSFUL_RECOVERY", "INJURY_OR_FATALITY"}


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def export(db: str, table: str):
    p = subprocess.Popen(["mdb-export", db, table], stdout=subprocess.PIPE,
                         text=True, stderr=subprocess.DEVNULL)
    yield from csv.DictReader(p.stdout)
    p.wait()


def fnum(v: str):
    v = (v or "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


def build_index():
    ev, ac, occ = {}, {}, {}
    for db in DBS:
        if not os.path.exists(db):
            continue
        for r in export(db, "events"):
            t, dp = fnum(r.get("wx_temp")), fnum(r.get("wx_dew_pt"))
            tc = dc = None
            if (t is not None and dp is not None and not (t == 0 and dp == 0)
                    and -60 < t < 130 and -60 < dp < 130 and dp <= t + 2):
                tc, dc = f_to_c(t), f_to_c(min(dp, t))
            w, wd = fnum(r.get("wind_vel_kts")), fnum(r.get("wind_dir_deg"))
            if not (w is not None and 0 <= w < 200 and not (w == 0 and not wd)):
                w = None
            ev[r["ev_id"]] = dict(
                temp_c=tc, dew_c=dc, wind_kts=w, year=fnum(r.get("ev_year")),
                ev_type=(r.get("ev_type") or "").strip(),
                hi_inj=(r.get("ev_highest_injury") or "").strip(),
                light=(r.get("light_cond") or "").strip(),
                wxb=(r.get("wx_cond_basic") or "").strip(),
                ceil=fnum(r.get("sky_ceil_ht")), vis=fnum(r.get("vis_sm")))
        for r in export(db, "aircraft"):
            ac[(r["ev_id"], r["Aircraft_Key"].strip())] = dict(
                damage=(r.get("damage") or "").strip(),
                cat=(r.get("acft_category") or "").strip(),
                neng=(r.get("num_eng") or "").strip(),
                far=(r.get("far_part") or "").strip(),
                home=(r.get("homebuilt") or "").strip(),
                typefly=(r.get("type_fly") or "").strip())
        for r in export(db, "Occurrences"):
            occ.setdefault((r["ev_id"], r["Aircraft_Key"].strip()),
                           (r.get("Phase_of_Flight") or "").strip())
    return ev, ac, occ


def main() -> None:
    ev, ac, occ = build_index()
    fv = pd.read_parquet(FV)
    nt = fv[fv.source.str.startswith("NTSB_ASRS")].copy()
    nt["ev_id"] = nt.record_id.str.rsplit("_", n=1).str[0]
    nt["ak"] = nt.record_id.str.rsplit("_", n=1).str[1]
    nt = nt.join(pd.DataFrame.from_dict(ev, orient="index"), on="ev_id")
    for c in ("damage", "cat", "neng", "far", "home", "typefly"):
        nt[c] = [ac.get((e, a), {}).get(c, "") for e, a in zip(nt.ev_id, nt.ak)]
    nt["phase"] = [occ.get((e, a), "") for e, a in zip(nt.ev_id, nt.ak)]
    out = {"n_ntsb_records": int(len(nt)),
           "coverage": {"temp_dewpoint": int(nt.temp_c.notna().sum()),
                        "wind": int(nt.wind_kts.notna().sum()),
                        "ev_type": nt.ev_type.value_counts().to_dict(),
                        "highest_injury": nt.hi_inj.value_counts().to_dict(),
                        "damage": nt.damage.value_counts().to_dict()},
           "damage_x_injury": {f"{d}|{i}": int(v) for (d, i), v in
                               pd.crosstab(nt.damage, nt.hi_inj).stack().items() if v}}

    nt["Y_sev"] = (nt.sev_ordinal >= 3).astype(int)
    nt["Y_fatal"] = (nt.sev_ordinal >= 4).astype(int)
    wx = nt[nt.temp_c.notna()].copy()
    ice = [p_carb_icing(t, d, "descent") for t, d in zip(wx.temp_c, wx.dew_c)]
    wx["pi_ice"] = [r.p_ice for r in ice]
    wx["m_ice"] = [r.margin_gm3 for r in ice]
    yi = wx.CARBURETOR_OR_INDUCTION_ICING.values
    ysev, yfat = wx.Y_sev.values, wx.Y_fatal.values
    out["weather_subset"] = {"n": int(len(wx)), "p_serious_fatal": round(float(ysev.mean()), 4),
                             "n_icing": int(yi.sum())}

    # --- occurrence layer: physics vs the factor label it models
    out["physics_pi_ice_auc"] = {
        "vs_icing_factor": round(roc_auc_score(yi, wx.pi_ice), 4),
        "vs_engine_failure": round(roc_auc_score(wx.ENGINE_FAILURE, wx.pi_ice), 4),
        "vs_serious_fatal": round(roc_auc_score(ysev, wx.pi_ice), 4),
        "vs_fatal": round(roc_auc_score(yfat, wx.pi_ice), 4)}
    sub = wx[yi == 1]
    out["physics_pi_ice_auc"]["vs_serious_fatal_within_icing_accidents"] = round(
        roc_auc_score(sub.Y_sev, sub.pi_ice), 4)
    wx["dec"] = pd.qcut(wx.pi_ice.rank(method="first"), 10, labels=False)
    rel = wx.groupby("dec").agg(pi_mean=("pi_ice", "mean"),
                                icing_rate=("CARBURETOR_OR_INDUCTION_ICING", "mean"))
    out["reliability_pi_ice_by_decile"] = rel.round(4).to_dict(orient="index")

    # calibrated link: isotonic on the guard margin, out of fold
    cal = np.zeros(len(wx))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(wx):
        iso = IsotonicRegression(out_of_bounds="clip").fit(wx.m_ice.values[tr], yi[tr])
        cal[te] = iso.predict(wx.m_ice.values[te])
    ef = wx.ENGINE_FAILURE.values
    q = float((yi * ef).sum() / yi.sum())
    joint = yi * ef
    out["calibrated_icing_link"] = {
        "auc": round(roc_auc_score(yi, cal), 4),
        "brier": round(brier_score_loss(yi, cal), 5),
        "brier_raw_pi_ice": round(brier_score_loss(yi, wx.pi_ice), 5),
        "brier_base_rate": round(brier_score_loss(yi, np.full(len(yi), yi.mean())), 5),
        "mean_cal": round(float(cal.mean()), 4), "prevalence": round(float(yi.mean()), 4),
        "fused_depth2": {"q_ef_given_ice": round(q, 4),
                         "mean_R": round(float((cal * q).mean()), 4),
                         "realized_joint": round(float(joint.mean()), 4),
                         "auc_vs_joint": round(roc_auc_score(joint, cal * q), 4),
                         "brier": round(brier_score_loss(joint, cal * q), 5),
                         "brier_base": round(brier_score_loss(
                             joint, np.full(len(joint), joint.mean())), 5)}}

    # pairwise weather-bin ordering (relative-risk check, level-free)
    wx["Tbin"] = pd.cut(wx.temp_c, [-40, 0, 5, 10, 15, 20, 25, 45])
    wx["Sbin"] = pd.cut(wx.dew_c - wx.temp_c, [-60, -15, -8, -4, -2, 1])
    tab = (wx.groupby(["Tbin", "Sbin"], observed=True)
             .agg(n=("pi_ice", "size"), pi=("pi_ice", "mean"),
                  rate=("CARBURETOR_OR_INDUCTION_ICING", "mean")).reset_index())
    tab = tab[tab.n >= 200]
    agree = tot = 0
    for i, j in combinations(range(len(tab)), 2):
        a, b = tab.iloc[i], tab.iloc[j]
        if abs(a.pi - b.pi) < 0.02 or abs(a.rate - b.rate) < 0.002:
            continue
        tot += 1
        agree += int((a.pi > b.pi) == (a.rate > b.rate))
    out["pairwise_bin_ordering"] = {"agree": agree, "pairs": tot,
                                    "frac": round(agree / max(tot, 1), 3)}

    # --- consequence layer: what predicts realized severity, and from what information
    causal = [c for c in fv.columns if c.isupper() and c not in OUTCOMES]
    X_f = wx[causal].values.astype(float)
    X_p = np.c_[wx.pi_ice.values, np.log1p(wx.wind_kts.fillna(0).values),
                wx.wind_kts.isna().values.astype(float)]
    pre = pd.get_dummies(wx[["light", "wxb", "cat", "neng", "far", "home", "typefly",
                             "phase"]].astype(str))
    pre = pre.loc[:, pre.sum() >= 30]
    pre["temp"], pre["dew"] = wx.temp_c.values, wx.dew_c.values
    pre["spread"] = wx.temp_c.values - wx.dew_c.values
    pre["wind"] = wx.wind_kts.fillna(0).values
    pre["wind_na"] = wx.wind_kts.isna().values.astype(float)
    pre["ceil"] = wx.ceil.fillna(-1).values
    pre["vis"] = wx.vis.fillna(-1).values
    pre["pi_ice"], pre["year"] = wx.pi_ice.values, wx.year.values
    X_pre = pre.values.astype(float)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    def oof_lr(X, y):
        return cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=skf,
                                 method="predict_proba")[:, 1]

    def oof_gbm(X, y):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        return cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]

    sev = {}
    for tgt, y in (("serious_fatal", ysev), ("fatal", yfat)):
        sev[tgt] = {"prevalence": round(float(y.mean()), 4)}
        for lab, X in (("lr_factors", X_f), ("lr_physics_drivers", X_p),
                       ("lr_factors_plus_physics", np.c_[X_f, X_p])):
            p = oof_lr(X, y)
            sev[tgt][lab] = {"auc": round(roc_auc_score(y, p), 4),
                             "pr_auc": round(average_precision_score(y, p), 4),
                             "brier": round(brier_score_loss(y, p), 5)}
    for lab, X in (("gbm_pre_outcome_covariates", X_pre), ("gbm_extracted_factors", X_f),
                   ("gbm_pre_outcome_plus_factors", np.c_[X_pre, X_f])):
        p = oof_gbm(X, ysev)
        sev["serious_fatal"][lab] = {"auc": round(roc_auc_score(ysev, p), 4),
                                     "brier": round(brier_score_loss(ysev, p), 5)}
    tr, te = (wx.year < 2010).values, (wx.year >= 2010).values
    m = LogisticRegression(max_iter=2000).fit(X_f[tr], ysev[tr])
    sev["serious_fatal"]["time_blocked_lr_factors"] = {
        "n_test": int(te.sum()), "auc": round(roc_auc_score(ysev[te], m.predict_proba(X_f[te])[:, 1]), 4)}
    out["severity_models"] = sev

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
