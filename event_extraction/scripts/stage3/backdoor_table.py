"""Paper table: graph-level counterfactuals with and without backdoor adjustment.

bn_do_smoke_test.py reports the backdoor-adjusted interventional effect only.
For the manuscript the informative comparison is adjusted *against* crude,
because the crude contrast P(Y|X=1) - P(Y|X=0) is what a study that skipped
structure learning would report. The difference between the two columns is the
confounding that the learned per-category DAG removes, and it is the only
evidence that learning the DAG was worth doing.

For each factor X in a category, with Y = serious or fatal outcome:

    crude      P(Y | X=1) - P(Y | X=0)
    adjusted   P(Y | do(X=1)) - P(Y | do(X=0))   via backdoor on Z = pa_G(X)
    shift      crude - adjusted

Nonparametric bootstrap over records gives a 95 % interval on the adjusted
effect, so the manuscript can report interventional effects with uncertainty.

Run:
    python event_extraction/scripts/stage3/backdoor_table.py \
        --categories LOC-I SCF-PP
Output: event_extraction/out/causation_kg/backdoor_table_{cat}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bn_do_smoke_test import (OUTCOME_FACTORS, SEV_SERIOUS_ORDINAL,
                              load_dag_parents, p_y_do_x)

N_BOOT = 400
SEED = 0


def crude_effect(sub: pd.DataFrame, x: str, alpha: float = 1.0) -> float:
    """Unadjusted contrast — what a frequency-only analysis would report."""
    g1, g0 = sub[sub[x] == 1], sub[sub[x] == 0]
    if len(g1) == 0 or len(g0) == 0:
        return float("nan")
    p1 = (g1["Y"].sum() + alpha) / (len(g1) + 2 * alpha)
    p0 = (g0["Y"].sum() + alpha) / (len(g0) + 2 * alpha)
    return float(p1 - p0)


def run_category(cat: str, df: pd.DataFrame, dag_dir: Path, min_stability: float,
                 min_support: int, max_parents: int, top_k: int) -> dict:
    sub = df[df["category"] == cat].copy()
    sub["Y"] = (sub["sev_ordinal"] >= SEV_SERIOUS_ORDINAL).astype(int)
    baseline = float(sub["Y"].mean())

    dag_parents = load_dag_parents(dag_dir / f"{cat}.csv", min_stability)
    factor_cols = [c for c in df.columns
                   if c.isupper() and c != "Y" and c not in OUTCOME_FACTORS]
    support = {c: int(sub[c].sum()) for c in factor_cols}
    candidates = [c for c in factor_cols if support[c] >= min_support]

    rows = []
    for x in candidates:
        parents = [p for p in dag_parents.get(x, [])
                   if p in sub.columns and support.get(p, 0) >= min_support]
        parents = sorted(parents, key=lambda p: -support[p])[:max_parents]
        p0, p1 = p_y_do_x(sub, x, parents)
        rows.append({
            "factor": x,
            "n_present": support[x],
            "adjustment_set": parents,
            "crude_effect": round(crude_effect(sub, x), 4),
            "adjusted_effect": round(p1 - p0, 4),
            "p_serious_do_absent": round(p0, 4),
            "p_serious_do_present": round(p1, 4),
        })
        rows[-1]["confounding_shift"] = round(
            rows[-1]["crude_effect"] - rows[-1]["adjusted_effect"], 4)

    rows.sort(key=lambda r: -r["adjusted_effect"])
    keep = rows[:top_k]

    # bootstrap CI on the adjusted effect, for the reported factors only
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(sub))
    boot: dict[str, list[float]] = {r["factor"]: [] for r in keep}
    for _ in range(N_BOOT):
        bs = sub.iloc[rng.choice(idx, len(idx), replace=True)]
        for r in keep:
            try:
                b0, b1 = p_y_do_x(bs, r["factor"], r["adjustment_set"])
                boot[r["factor"]].append(b1 - b0)
            except Exception:
                pass
    for r in keep:
        v = np.array(boot[r["factor"]], dtype=float)
        v = v[np.isfinite(v)]
        r["ci95"] = [round(float(np.percentile(v, 2.5)), 4),
                     round(float(np.percentile(v, 97.5)), 4)] if len(v) else None

    n_adjusted = sum(1 for r in rows if r["adjustment_set"])
    shifts = [abs(r["confounding_shift"]) for r in rows if r["adjustment_set"]]
    return {
        "category": cat,
        "n_records": int(len(sub)),
        "baseline_p_serious": round(baseline, 4),
        "n_candidate_factors": len(rows),
        "n_with_nonempty_adjustment_set": n_adjusted,
        "mean_abs_confounding_shift_adjusted_factors": (
            round(float(np.mean(shifts)), 4) if shifts else None),
        "max_abs_confounding_shift": (
            round(float(np.max(shifts)), 4) if shifts else None),
        "n_boot": N_BOOT,
        "interventions": keep,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-vectors",
                    default="event_extraction/out/causation_kg/factor_vectors.parquet",
                    type=Path)
    ap.add_argument("--dag-dir",
                    default="event_extraction/out/aggregate_kg/per_category_dag",
                    type=Path)
    ap.add_argument("--categories", nargs="+", default=["LOC-I", "SCF-PP"])
    ap.add_argument("--min-stability", default=0.5, type=float)
    ap.add_argument("--min-support", default=30, type=int)
    ap.add_argument("--max-parents", default=3, type=int)
    ap.add_argument("--top-k", default=8, type=int)
    args = ap.parse_args()

    df = pd.read_parquet(args.factor_vectors)
    for cat in args.categories:
        res = run_category(cat, df, args.dag_dir, args.min_stability,
                           args.min_support, args.max_parents, args.top_k)
        out = Path("event_extraction/out/causation_kg") / f"backdoor_table_{cat}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"\n=== {cat}: n={res['n_records']:,}, "
              f"baseline={res['baseline_p_serious']:.3f}, "
              f"mean |shift| on adjusted factors="
              f"{res['mean_abs_confounding_shift_adjusted_factors']} ===")
        print(f"{'factor':40s} {'n':>6} {'crude':>7} {'adj':>7} {'shift':>7}  Z")
        for r in res["interventions"]:
            z = ",".join(a[:12] for a in r["adjustment_set"]) or "-"
            print(f"{r['factor']:40s} {r['n_present']:>6,} "
                  f"{r['crude_effect']:>7.3f} {r['adjusted_effect']:>7.3f} "
                  f"{r['confounding_shift']:>7.3f}  {z}")
        print(f"written -> {out}")


if __name__ == "__main__":
    main()
