"""Held-out comparison of the hybrid against either layer alone on an
occurrence task (review 2026-09-14, comment 9).

Task: predict, per NTSB accident record, whether the chain carries the
carburetor-icing factor (and, second target, icing together with engine
failure), from information available before the flight: the recorded
weather and the pre-flight covariates the events table carries. Records
before 2015 train the fitted arms; records from 2015 onward score them.

Arms
  chart physics       pi_ice from the chart-calibrated model, no fit.
  physics calibrated  isotonic link on the condensable-water margin, fitted
                      on the training years (Eq. calib executed).
  corpus only         logistic regression on month, hour, light condition,
                      basic weather condition and state, no physics.
  corpus + raw wx     the same plus temperature and dewpoint as raw features
                      (a fitted reference that sees the same inputs as the
                      physics).
  hybrid              corpus covariates plus the calibrated physics
                      probability as a feature (physics shape, corpus level,
                      corpus covariates).
  hybrid, corrupted   the same with the sign-inverted margin, and with the
                      margin permuted across records.

Metrics on the test years: AUC with a DeLong interval, average precision,
Brier score, and the reliability of the hybrid by decile. Severity is not
a target here: the physics term carries no severity signal
(severity_groundtruth_check.py), and the manuscript says so.

Output: physics/out/heldout_occurrence.json
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing  # noqa: E402

KG = "event_extraction/out/causation_kg/factor_vectors.parquet"
DBS = ["data/NTSB_ASRS/avall.mdb", "data/NTSB_ASRS/Pre2008.mdb"]
OUT = "physics/out/heldout_occurrence.json"
SPLIT_YEAR = 2015
SEED = 0
COLS = ["ev_id", "ev_date", "ev_time", "light_cond", "wx_cond_basic", "ev_state",
        "wx_temp", "wx_dew_pt"]


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def export_events() -> pd.DataFrame:
    rows = []
    for db in DBS:
        if not os.path.exists(db):
            continue
        p = subprocess.Popen(["mdb-export", db, "events"], stdout=subprocess.PIPE,
                             text=True, stderr=subprocess.DEVNULL)
        r = csv.reader(p.stdout)
        hdr = next(r)
        ix = {c: i for i, c in enumerate(hdr)}
        for row in r:
            rows.append({c: (row[ix[c]] if c in ix else "") for c in COLS})
        p.wait()
    df = pd.DataFrame(rows)
    df["year"] = pd.to_datetime(df["ev_date"], errors="coerce").dt.year
    df["month"] = pd.to_datetime(df["ev_date"], errors="coerce").dt.month
    t = pd.to_numeric(df["ev_time"], errors="coerce")
    df["hour"] = (t // 100).where(t.between(0, 2359))
    for c in ("wx_temp", "wx_dew_pt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = (df["wx_temp"].notna() & df["wx_dew_pt"].notna()
          & ~((df["wx_temp"] == 0) & (df["wx_dew_pt"] == 0))
          & df["wx_temp"].between(-59, 129) & df["wx_dew_pt"].between(-59, 129)
          & (df["wx_dew_pt"] <= df["wx_temp"] + 2))
    df = df[ok].copy()
    df["T"] = df["wx_temp"].map(f_to_c)
    df["Td"] = np.minimum(df["wx_dew_pt"].map(f_to_c), df["T"])
    return df.drop_duplicates("ev_id")


def delong_ci(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    pos, neg = y == 1, y == 0
    sp, sn = s[pos], s[neg]
    sn_s = np.sort(sn)
    v10 = (np.searchsorted(sn_s, sp, "left") + np.searchsorted(sn_s, sp, "right")) / (2 * len(sn))
    sp_s = np.sort(sp)
    v01 = 1 - (np.searchsorted(sp_s, sn, "left") + np.searchsorted(sp_s, sn, "right")) / (2 * len(sp))
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / len(sp) + v01.var(ddof=1) / len(sn))
    return float(auc), float(auc - 1.96 * se), float(auc + 1.96 * se)


def score(y: np.ndarray, s: np.ndarray, prob: bool = True) -> dict:
    auc, lo, hi = delong_ci(y, s)
    out = {"auc": round(auc, 4), "auc_ci95": [round(lo, 4), round(hi, 4)],
           "average_precision": round(float(average_precision_score(y, s)), 4)}
    if prob:
        out["brier"] = round(float(brier_score_loss(y, np.clip(s, 0, 1))), 5)
    return out


def main() -> None:
    os.makedirs("physics/out", exist_ok=True)
    ev = export_events()
    fv = pd.read_parquet(KG, columns=["record_id", "CARBURETOR_OR_INDUCTION_ICING",
                                      "ENGINE_FAILURE"])
    fv["ev_id"] = fv["record_id"].str.rsplit("_", n=1).str[0]
    df = fv.merge(ev, on="ev_id", how="inner").dropna(subset=["year"])
    df["y_ice"] = df["CARBURETOR_OR_INDUCTION_ICING"].astype(int)
    df["y_ice_ef"] = (df["CARBURETOR_OR_INDUCTION_ICING"].astype(int)
                      & df["ENGINE_FAILURE"].astype(int)).astype(int)
    res = [p_carb_icing(t, d, "descent") for t, d in zip(df["T"], df["Td"])]
    df["chart"] = [r.p_ice for r in res]
    df["margin"] = [r.margin_gm3 for r in res]
    for c in ("light_cond", "wx_cond_basic", "ev_state"):
        df[c] = df[c].fillna("").replace("", "UNK")
    df["month"] = df["month"].fillna(0).astype(int).astype(str)
    df["hour"] = df["hour"].fillna(-1).astype(int).astype(str)

    tr, te = df[df["year"] < SPLIT_YEAR], df[df["year"] >= SPLIT_YEAR]
    out = {"split_year": SPLIT_YEAR, "n_train": int(len(tr)), "n_test": int(len(te)),
           "targets": {}}
    rng = np.random.default_rng(SEED)
    cat_cols = ["month", "hour", "light_cond", "wx_cond_basic", "ev_state"]

    def logistic(cols_cat, cols_num):
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat)]
            + ([("num", StandardScaler(), cols_num)] if cols_num else []))
        return make_pipeline(ct, LogisticRegression(max_iter=3000, C=1.0))

    for target in ("y_ice", "y_ice_ef"):
        ytr, yte = tr[target].to_numpy(int), te[target].to_numpy(int)
        arms = {}
        # chart physics, no fit
        arms["chart physics (no fit)"] = score(yte, te["chart"].to_numpy(), prob=True)
        # calibrated physics
        iso = IsotonicRegression(out_of_bounds="clip").fit(tr["margin"], ytr)
        cal_te = iso.predict(te["margin"])
        arms["physics, calibrated on train years"] = score(yte, cal_te)
        # corpus only
        m = logistic(cat_cols, []).fit(tr, ytr)
        arms["corpus only (covariates)"] = score(yte, m.predict_proba(te)[:, 1])
        # corpus + raw weather
        m = logistic(cat_cols, ["T", "Td"]).fit(tr, ytr)
        arms["corpus + raw temperature and dewpoint"] = score(yte, m.predict_proba(te)[:, 1])
        # hybrid: covariates + calibrated physics (as log-odds feature)
        tr2, te2 = tr.copy(), te.copy()
        eps = 1e-4
        tr2["phys"] = np.log((iso.predict(tr["margin"]) + eps) / (1 - iso.predict(tr["margin"]) + eps))
        te2["phys"] = np.log((cal_te + eps) / (1 - cal_te + eps))
        m = logistic(cat_cols, ["phys"]).fit(tr2, ytr)
        hyb = m.predict_proba(te2)[:, 1]
        arms["hybrid (covariates + calibrated physics)"] = score(yte, hyb)
        # corrupted: sign-inverted margin
        iso_neg = IsotonicRegression(out_of_bounds="clip").fit(-tr["margin"], ytr)
        tr2["phys"] = np.log((iso_neg.predict(-tr["margin"]) + eps) / (1 - iso_neg.predict(-tr["margin"]) + eps))
        te2["phys"] = np.log((iso_neg.predict(-te["margin"]) + eps) / (1 - iso_neg.predict(-te["margin"]) + eps))
        m = logistic(cat_cols, ["phys"]).fit(tr2, ytr)
        arms["hybrid, sign-inverted margin"] = score(yte, m.predict_proba(te2)[:, 1])
        # corrupted: permuted margin
        ptr = rng.permutation(tr["margin"].to_numpy())
        pte = rng.permutation(te["margin"].to_numpy())
        iso_p = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
        tr2["phys"] = np.log((iso_p.predict(ptr) + eps) / (1 - iso_p.predict(ptr) + eps))
        te2["phys"] = np.log((iso_p.predict(pte) + eps) / (1 - iso_p.predict(pte) + eps))
        m = logistic(cat_cols, ["phys"]).fit(tr2, ytr)
        arms["hybrid, permuted margin"] = score(yte, m.predict_proba(te2)[:, 1])
        # reliability of the hybrid by decile
        dec = pd.qcut(hyb, 10, labels=False, duplicates="drop")
        rel = pd.DataFrame({"p": hyb, "y": yte, "d": dec}).groupby("d").agg(
            p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size"))
        out["targets"][target] = {
            "n_pos_train": int(ytr.sum()), "n_pos_test": int(yte.sum()),
            "prevalence_test": round(float(yte.mean()), 4),
            "arms": arms,
            "hybrid_reliability_by_decile": [
                {"decile": int(i), "p_mean": round(float(r.p_mean), 4),
                 "realized": round(float(r.y_rate), 4), "n": int(r.n)}
                for i, r in rel.iterrows()],
        }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "targets"}))
    for t, v in out["targets"].items():
        print(f"\n== {t}: train pos {v['n_pos_train']}, test pos {v['n_pos_test']} "
              f"(prevalence {v['prevalence_test']})")
        for arm, s in v["arms"].items():
            print(f"  {arm:44s} AUC {s['auc']:.3f} {s['auc_ci95']} AP {s['average_precision']:.4f} "
                  f"Brier {s.get('brier')}")


if __name__ == "__main__":
    main()
