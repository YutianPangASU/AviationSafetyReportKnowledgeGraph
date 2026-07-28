"""Information-ceiling analysis for the carburetor-icing occurrence model.

The physics model scores AUC 0.61 separating narrative-attributed icing
accidents from all other weather-reported accidents, using only recorded
temperature and dewpoint. Read alone that number invites the reading that the
model is weak. The question it cannot answer is how much separation those two
covariates *can* support at all.

This script answers it by fitting supervised models to the same label from the
same two inputs and reading off their cross-validated AUC:

  physics      p_carb_icing(T, Td, power)         zero parameters fitted to label
  logistic     (T, Td)                            linear in the raw drivers
  logistic+    (T, Td, T-Td, RH, quadratic terms) physically-motivated features
  gbdt         (T, Td)                            nonparametric upper bound

If the fitted models cluster near the physics AUC, the physics model sits at the
information ceiling of the recorded weather and the residual is unrecorded state
(carburetor heat use, engine susceptibility, exposure time) rather than model
error. If they clear it substantially, the physics model is leaving recoverable
signal on the table.

Reuses the join in validate_carb_icing.py.

Output: physics/out/carb_icing_ceiling.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing, relative_humidity  # noqa: E402
from physics.validate_carb_icing import KG, FACTOR, build_weather_index  # noqa: E402

OUT = "physics/out"
N_SPLITS = 5
SEED = 0


def load_joined() -> pd.DataFrame:
    """Same join as validate_carb_icing: factor vectors -> NTSB recorded weather."""
    wx = build_weather_index()
    print(f"weather index: {len(wx)} NTSB events with usable temp/dewpoint")

    df = pd.read_parquet(KG, columns=["record_id", "category", FACTOR,
                                      "ENGINE_FAILURE", "sev_ordinal"]).copy()
    df["ev_id"] = df["record_id"].str.rsplit("_", n=1).str[0]
    df["wx"] = df["ev_id"].map(wx)
    df = df[df["wx"].notna()].copy()
    df["temp_c"] = df["wx"].str[0]
    df["dew_c"] = df["wx"].str[1]
    return df


def cv_auc(model, X: np.ndarray, y: np.ndarray) -> float:
    """Out-of-fold AUC, so the fitted models get no in-sample advantage."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def boot_ci(score: np.ndarray, y: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """Percentile bootstrap CI on AUC, resampling cases and controls separately."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
        aucs[b] = roc_auc_score(y[i], score[i])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_joined()
    y = df[FACTOR].to_numpy(dtype=int)
    T = df["temp_c"].to_numpy(dtype=float)
    Td = df["dew_c"].to_numpy(dtype=float)

    out: dict = {
        "n_total": int(len(df)),
        "n_icing": int(y.sum()),
        "n_other": int((1 - y).sum()),
        "n_splits": N_SPLITS,
        "note": "fitted AUCs are out-of-fold; physics AUC fits nothing to the label",
    }

    # --- physics, zero label-fitted parameters -------------------------------
    for power in ("cruise", "descent"):
        s = np.array([p_carb_icing(t, d, power).p_ice for t, d in zip(T, Td)])
        lo, hi = boot_ci(s, y)
        out[f"physics_{power}"] = {
            "auc": round(float(roc_auc_score(y, s)), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "fitted_params": 0,
        }

    # --- fitted reference models on the SAME two inputs ----------------------
    RH = np.array([relative_humidity(t, d) for t, d in zip(T, Td)])
    spread = T - Td

    feature_sets = {
        "logistic_T_Td": (make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000)),
                          np.column_stack([T, Td])),
        "logistic_physical": (make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=2000)),
                              np.column_stack([T, Td, spread, RH, T ** 2,
                                               Td ** 2, T * Td])),
        "gbdt_T_Td": (HistGradientBoostingClassifier(random_state=SEED),
                      np.column_stack([T, Td])),
    }
    for name, (model, X) in feature_sets.items():
        auc = cv_auc(model, X, y)
        out[name] = {"auc": round(auc, 4), "n_features": int(X.shape[1])}
        print(f"{name:22s} AUC {auc:.4f}")

    best_fitted = max(out[k]["auc"] for k in feature_sets)
    phys = out["physics_descent"]["auc"]
    out["ceiling_summary"] = {
        "physics_descent_auc": phys,
        "best_fitted_auc": round(best_fitted, 4),
        "gap": round(best_fitted - phys, 4),
        "physics_share_of_ceiling": round(
            (phys - 0.5) / (best_fitted - 0.5), 4) if best_fitted > 0.5 else None,
    }

    with open(os.path.join(OUT, "carb_icing_ceiling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["ceiling_summary"], indent=2))


if __name__ == "__main__":
    main()
