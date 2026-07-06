"""Phase-5 readiness gate: do()-intervention ranking via backdoor adjustment.

For one category (default LOC-I), estimate for every causal factor X

    P(Y = serious/fatal | do(X = 0))   and   P(Y | do(X = 1))

using Pearl's backdoor adjustment with the Stage-3 PC DAG supplying the
adjustment set: Z = parents(X) in the reoriented, bootstrap-stable DAG.

    P(Y | do(X=x)) = sum_z  P^(Y | X=x, Z=z) * P^(Z=z)

computed directly from the per-accident factor vectors (Laplace-smoothed,
with pooled fallback for empty strata). Root factors reduce to
P(Y | X=x). Factors are ranked by effect = P(Y|do(X=1)) - P(Y|do(X=0)) —
the achievable risk change at that intervention point.

Note: a naive discrete BN with all factors as parents of the outcome needs a
2 x 2^44 CPT (256 TiB) — backdoor adjustment on the learned structure is the
correct, tractable computation of the same interventional query.

Run (after build_kg_v4.py and the Stage-3 pipeline):
    /home/yp6443/miniconda3/envs/qwen-vllm/bin/python \\
        event_extraction/scripts/stage3/bn_do_smoke_test.py --category LOC-I
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SEV_SERIOUS_ORDINAL = 3  # serious_injury; fatal = 4

# Outcome factor_types are consequences, not intervention points — "do(no
# injury)" is tautological. They stay out of the candidate list.
OUTCOME_FACTORS = {
    "GROUND_IMPACT", "WATER_IMPACT", "MIDAIR_COLLISION", "GROUND_COLLISION",
    "RUNWAY_EXCURSION_OR_OVERRUN", "INFLIGHT_BREAKUP", "EMERGENCY_LANDING",
    "SUCCESSFUL_RECOVERY", "INJURY_OR_FATALITY",
}


def load_dag_parents(dag_csv: Path, min_stability: float) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {}
    with dag_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                stab = float(r.get("bootstrap_stability") or 0)
            except ValueError:
                stab = 0.0
            if stab >= min_stability:
                parents.setdefault(r["dst"], []).append(r["src"])
    return parents


def p_y_do_x(sub: pd.DataFrame, x: str, parents: list[str], alpha: float = 1.0
             ) -> tuple[float, float]:
    """Backdoor-adjusted (P(Y|do(X=0)), P(Y|do(X=1))) with Laplace smoothing.
    Empty (x, z) strata fall back to the stratum's pooled P(Y|Z=z)."""
    n = len(sub)
    out = [0.0, 0.0]
    if parents:
        grouped = sub.groupby(parents, observed=True)
        for _, g in grouped:
            w = len(g) / n
            p_pool = (g["Y"].sum() + alpha) / (len(g) + 2 * alpha)
            for xv in (0, 1):
                gx = g[g[x] == xv]
                if len(gx) == 0:
                    out[xv] += w * p_pool
                else:
                    out[xv] += w * (gx["Y"].sum() + alpha) / (len(gx) + 2 * alpha)
    else:
        for xv in (0, 1):
            gx = sub[sub[x] == xv]
            out[xv] = ((gx["Y"].sum() + alpha) / (len(gx) + 2 * alpha)
                       if len(gx) else float(sub["Y"].mean()))
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-vectors",
                    default="event_extraction/out/causation_kg/factor_vectors.parquet",
                    type=Path)
    ap.add_argument("--dag-dir",
                    default="event_extraction/out/aggregate_kg/per_category_dag",
                    type=Path)
    ap.add_argument("--category", default="LOC-I")
    ap.add_argument("--min-stability", default=0.5, type=float,
                    help="bootstrap_stability floor for DAG edges used as adjustment sets")
    ap.add_argument("--min-support", default=30, type=int,
                    help="min records with factor present to rank it")
    ap.add_argument("--max-parents", default=3, type=int,
                    help="cap adjustment-set size (largest-support parents kept)")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.factor_vectors)
    sub = df[df["category"] == args.category].copy()
    if len(sub) < 100:
        raise SystemExit(f"only {len(sub)} records in {args.category} — too few")
    sub["Y"] = (sub["sev_ordinal"] >= SEV_SERIOUS_ORDINAL).astype(int)
    baseline = float(sub["Y"].mean())
    print(f"{args.category}: {len(sub):,} records, "
          f"baseline P(serious/fatal) = {baseline:.3f}")

    dag_parents = load_dag_parents(args.dag_dir / f"{args.category}.csv",
                                   args.min_stability)
    factor_cols = [c for c in df.columns
                   if c.isupper() and c != "Y" and c not in OUTCOME_FACTORS]
    support = {c: int(sub[c].sum()) for c in factor_cols}
    candidates = [c for c in factor_cols if support[c] >= args.min_support]
    print(f"DAG adjustment sets from {args.dag_dir.name} "
          f"(stability >= {args.min_stability}); {len(candidates)} candidate factors")

    rows = []
    for x in candidates:
        parents = [p for p in dag_parents.get(x, [])
                   if p in sub.columns and support.get(p, 0) >= args.min_support]
        parents = sorted(parents, key=lambda p: -support[p])[: args.max_parents]
        p0, p1 = p_y_do_x(sub, x, parents)
        rows.append({
            "factor": x,
            "n_present": support[x],
            "adjustment_set": parents,
            "p_serious_do_absent": round(p0, 4),
            "p_serious_do_present": round(p1, 4),
            "effect_do_present_vs_absent": round(p1 - p0, 4),
            "risk_reduction_vs_baseline": round(baseline - p0, 4),
        })

    rows.sort(key=lambda r: -r["effect_do_present_vs_absent"])
    print(f"\n=== do(X) intervention ranking — {args.category} "
          f"(backdoor-adjusted) ===")
    print(f"{'factor':42s} {'n':>6} {'P(do=0)':>8} {'P(do=1)':>8} "
          f"{'effect':>7} {'adjust on':>30}")
    for r in rows[:20]:
        adj = ",".join(a[:14] for a in r["adjustment_set"]) or "-"
        print(f"{r['factor']:42s} {r['n_present']:>6,} "
              f"{r['p_serious_do_absent']:>8.3f} {r['p_serious_do_present']:>8.3f} "
              f"{r['effect_do_present_vs_absent']:>7.3f} {adj:>30s}")

    out = args.out or Path(f"event_extraction/out/causation_kg/"
                           f"bn_do_smoke_{args.category.replace(':', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "category": args.category, "n_records": int(len(sub)),
        "baseline_p_serious": round(baseline, 4),
        "method": "backdoor adjustment; Z = stable PC-DAG parents "
                  f"(stability >= {args.min_stability}, max {args.max_parents})",
        "interventions": rows}, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
