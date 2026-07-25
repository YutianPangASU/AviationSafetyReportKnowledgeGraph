"""Empirical validation of the carburetor-icing physics model against the corpus.

Tests the hybrid claim: do the accidents the narratives attribute to carburetor
icing actually fall in the weather where the physics model predicts high icing
risk? If the physics discriminates icing accidents from the rest by their
recorded temperature/dewpoint, the model earns its place as the node's Tier-1
occurrence probability.

Join: factor_vectors.parquet (record_id, CARBURETOR flag) -> NTSB ev_id (strip
the aircraft suffix) -> events.wx_temp / wx_dew_pt (Fahrenheit -> Celsius;
(0,0) pairs dropped as missing).

Outputs:
  physics/out/carb_icing_validation.json   summary stats
  physics/out/carb_icing_accidents.csv     (temp_c, dew_c, p_ice) for icing
                                           accidents -> paper figure input
"""
from __future__ import annotations
import csv, json, os, subprocess, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing

KG = "event_extraction/out/causation_kg/factor_vectors.parquet"
DBS = ["data/NTSB_ASRS/avall.mdb", "data/NTSB_ASRS/Pre2008.mdb"]
OUT = "physics/out"
FACTOR = "CARBURETOR_OR_INDUCTION_ICING"
POWER = "descent"  # icing power loss is typically discovered at low/descent power


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def build_weather_index() -> dict[str, tuple[float, float]]:
    """ev_id -> (temp_c, dew_c), Fahrenheit converted, (0,0)/implausible dropped."""
    wx: dict[str, tuple[float, float]] = {}
    for db in DBS:
        if not os.path.exists(db):
            continue
        p = subprocess.Popen(["mdb-export", db, "events"], stdout=subprocess.PIPE,
                             text=True, stderr=subprocess.DEVNULL)
        r = csv.reader(p.stdout)
        hdr = next(r)
        ix = {c: i for i, c in enumerate(hdr)}
        i_id, i_t, i_d = ix["ev_id"], ix["wx_temp"], ix["wx_dew_pt"]
        for row in r:
            t, d = row[i_t].strip(), row[i_d].strip()
            if not (t and d) or (t == "0" and d == "0"):
                continue
            try:
                tf, df = float(t), float(d)
            except ValueError:
                continue
            if not (-60 < tf < 130 and -60 < df < 130) or df > tf + 2:
                continue
            wx[row[i_id]] = (f_to_c(tf), f_to_c(df))
        p.wait()
    return wx


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    wx = build_weather_index()
    print(f"weather index: {len(wx)} NTSB events with usable temp/dewpoint")

    df = pd.read_parquet(KG, columns=["record_id", "category", FACTOR,
                                      "ENGINE_FAILURE", "sev_ordinal"])
    # record_id = "<ev_id>_<aircraft#>", ev_id like 20001213X25852; strip suffix.
    # Non-NTSB (FAA_AIDS) ids simply won't be in the weather index and drop out.
    df = df.copy()
    df["ev_id"] = df["record_id"].str.rsplit("_", n=1).str[0]
    df["wx"] = df["ev_id"].map(wx)
    df = df[df["wx"].notna()].copy()
    df["temp_c"] = df["wx"].str[0]
    df["dew_c"] = df["wx"].str[1]

    def auc(pos, neg):
        """P(score_pos > score_neg): rank-based, ties counted as 0.5."""
        import numpy as np
        allv = np.concatenate([pos, neg])
        order = allv.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(allv) + 1)
        # average ranks for ties
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        avg = {i: (csum[i] - cnt[i] + 1 + csum[i]) / 2.0 for i in range(len(cnt))}
        ar = np.array([avg[i] for i in inv])
        r_pos = ar[:len(pos)].sum()
        return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

    def summ(g, col):
        return {"n": int(len(g)),
                "mean_p_ice": round(float(g[col].mean()), 4),
                "median_p_ice": round(float(g[col].median()), 4),
                "frac_serious_or_moderate": round(float((g[col] >= 0.40).mean()), 4),
                "frac_serious": round(float((g[col] >= 0.80).mean()), 4)}

    out = {"n_joined_total": int(len(df))}
    for power in ("cruise", "descent"):
        col = f"p_ice_{power}"
        df[col] = df.apply(lambda r: p_carb_icing(r["temp_c"], r["dew_c"], power).p_ice, axis=1)
        icing = df[df[FACTOR] == 1]
        other = df[df[FACTOR] == 0]
        out[power] = {
            "icing_accidents": summ(icing, col),
            "non_icing_accidents": summ(other, col),
            "auc_icing_vs_other": round(auc(icing[col].values, other[col].values), 4),
            "icing_and_engine_failure": summ(icing[icing["ENGINE_FAILURE"] == 1], col),
        }
    df["p_ice"] = df["p_ice_descent"]
    icing = df[df[FACTOR] == 1]
    with open(os.path.join(OUT, "carb_icing_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    icing[["temp_c", "dew_c", "p_ice"]].to_csv(
        os.path.join(OUT, "carb_icing_accidents.csv"), index=False)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
