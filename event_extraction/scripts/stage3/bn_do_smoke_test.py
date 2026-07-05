"""Phase-5 readiness gate: parameterized do()-intervention smoke test.

Fits a discrete Bayesian network on the v4 factor-vector table for one
category (default LOC-I), using the Stage-3 reoriented DAG as structure plus
a severity target node, then computes

    P(outcome_severity >= serious | do(X = absent))    for candidate factors X

and reports the achievable risk reduction per intervention point, ranked.
This is the artifact the whole causation KG exists to support: if this
produces sane, graded numbers, the KG is ready for counterfactual work.

Structure: the per-category Stage-3 DAG edges (bootstrap-stable, causal
factors only) + an edge from every factor with support >= --min-support to
the binary target SERIOUS (severity >= serious_injury). CPDs are fit with
Bayesian (BDeu) smoothing. do(X=0) is computed by graph surgery (pgmpy's
do-operator) followed by exact inference.

Run (after build_kg_v4.py has produced factor_vectors.parquet):
    /home/yp6443/miniconda3/envs/qwen-vllm/bin/python \\
        event_extraction/scripts/stage3/bn_do_smoke_test.py --category LOC-I
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

TARGET = "SERIOUS"


def load_dag_edges(dag_csv: Path, min_stability: float) -> list[tuple[str, str]]:
    edges = []
    with dag_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                stab = float(r.get("bootstrap_stability") or 0)
            except ValueError:
                stab = 0.0
            if stab >= min_stability:
                edges.append((r["src"], r["dst"]))
    return edges


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
                    help="bootstrap_stability floor for structure edges")
    ap.add_argument("--min-support", default=30, type=int,
                    help="min records with factor present to consider intervening on it")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.factor_vectors)
    sub = df[df["category"] == args.category].copy()
    if len(sub) < 100:
        raise SystemExit(f"only {len(sub)} records in {args.category} — too few")
    sub[TARGET] = (sub["sev_ordinal"] >= 3).astype(int)  # serious_injury or fatal
    baseline = sub[TARGET].mean()
    print(f"{args.category}: {len(sub):,} records, "
          f"baseline P(serious/fatal) = {baseline:.3f}")

    dag_csv = args.dag_dir / f"{args.category}.csv"
    struct_edges = load_dag_edges(dag_csv, args.min_stability)
    factor_cols = [c for c in df.columns
                   if c.isupper() and c not in (TARGET,) and sub[c].sum() >= args.min_support]
    struct_edges = [(s, d) for s, d in struct_edges
                    if s in factor_cols and d in factor_cols]
    edges = struct_edges + [(f, TARGET) for f in factor_cols]
    nodes = sorted({n for e in edges for n in e})
    data = sub[[c for c in nodes if c != TARGET] + [TARGET]].astype(int)

    model = DiscreteBayesianNetwork(edges)
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu",
              equivalent_sample_size=10)
    print(f"BN: {len(nodes)} nodes, {len(edges)} edges "
          f"({len(struct_edges)} inter-factor from Stage-3 DAG)")

    rows = []
    for x in factor_cols:
        n_present = int(sub[x].sum())
        try:
            intervened = model.do([x])
            inf = VariableElimination(intervened)
            q = inf.query([TARGET], evidence={x: 0}, show_progress=False)
            p_do_absent = float(q.values[1])
            q1 = inf.query([TARGET], evidence={x: 1}, show_progress=False)
            p_do_present = float(q1.values[1])
        except Exception as e:
            print(f"  [skip {x}: {type(e).__name__}: {e}]")
            continue
        rows.append({
            "factor": x,
            "n_present": n_present,
            "p_serious_do_absent": round(p_do_absent, 4),
            "p_serious_do_present": round(p_do_present, 4),
            "risk_reduction_vs_baseline": round(baseline - p_do_absent, 4),
            "effect_present_vs_absent": round(p_do_present - p_do_absent, 4),
        })

    rows.sort(key=lambda r: -r["effect_present_vs_absent"])
    print(f"\n=== do(X = absent) intervention ranking — {args.category} ===")
    print(f"{'factor':42s} {'n':>6} {'P(do=0)':>8} {'P(do=1)':>8} {'effect':>8}")
    for r in rows[:20]:
        print(f"{r['factor']:42s} {r['n_present']:>6} "
              f"{r['p_serious_do_absent']:>8.3f} {r['p_serious_do_present']:>8.3f} "
              f"{r['effect_present_vs_absent']:>8.3f}")

    out = args.out or Path(f"event_extraction/out/causation_kg/"
                           f"bn_do_smoke_{args.category.replace(':','_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "category": args.category, "n_records": len(sub),
        "baseline_p_serious": round(baseline, 4),
        "n_structure_edges": len(struct_edges),
        "interventions": rows}, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
