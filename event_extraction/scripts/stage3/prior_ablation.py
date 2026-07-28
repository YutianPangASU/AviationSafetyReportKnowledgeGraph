"""Ablation: how much do the language-model priors change the learned DAG?

The same model family performs the extraction and supplies the ordering priors
that orient the PC skeleton, so a reader may reasonably suspect that the causal
structure reflects the model rather than the corpus. The pipeline already
produced a no-prior baseline (per_category_dag_noprior), where PC runs on the
factor-presence data alone with no language-model input.

This script compares the two on three levels, from the least to the most
consequential for the manuscript:

  skeleton      undirected edge sets: Jaccard, edges unique to each run
  orientation   agreement in direction on the shared skeleton, and SHD
  downstream    the backdoor-adjusted interventional effects computed with
                each run's parent sets

Only the third level matters for the claims. If the reported effects move very
little when the priors are removed, the priors affect presentation of the graph
rather than the numbers the paper reports.

Run:
    python event_extraction/scripts/stage3/prior_ablation.py \
        --categories LOC-I SCF-PP
Output: event_extraction/out/causation_kg/prior_ablation.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bn_do_smoke_test import OUTCOME_FACTORS, SEV_SERIOUS_ORDINAL, p_y_do_x

MIN_SUPPORT = 30
MAX_PARENTS = 3


def read_edges(path: Path, min_stability: float | None) -> list[tuple[str, str]]:
    edges = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if min_stability is not None:
                try:
                    if float(r.get("bootstrap_stability") or 0) < min_stability:
                        continue
                except ValueError:
                    continue
            edges.append((r["src"], r["dst"]))
    return edges


def parents_from(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s, d in edges:
        out.setdefault(d, []).append(s)
    return out


def compare_structure(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> dict:
    sa, sb = set(a), set(b)
    ua = {frozenset(e) for e in sa}
    ub = {frozenset(e) for e in sb}
    shared = ua & ub
    # orientation agreement on the shared skeleton
    agree = sum(1 for e in shared
                if (tuple(sorted(e)) in sa) == (tuple(sorted(e)) in sb)
                and (next(iter([x for x in sa if frozenset(x) == e]), None)
                     == next(iter([x for x in sb if frozenset(x) == e]), None)))
    # structural Hamming distance: missing + extra + reversed
    reversed_edges = sum(1 for e in shared
                         if next(x for x in sa if frozenset(x) == e)
                         != next(x for x in sb if frozenset(x) == e))
    return {
        "n_edges_with_prior": len(sa),
        "n_edges_no_prior": len(sb),
        "skeleton_shared": len(shared),
        "skeleton_only_with_prior": len(ua - ub),
        "skeleton_only_no_prior": len(ub - ua),
        "skeleton_jaccard": round(len(shared) / len(ua | ub), 4) if (ua | ub) else None,
        "orientation_agree_on_shared": agree,
        "orientation_reversed_on_shared": reversed_edges,
        "shd": len(ua - ub) + len(ub - ua) + reversed_edges,
    }


def downstream(cat: str, df: pd.DataFrame, pa_prior: dict, pa_noprior: dict,
               top_k: int) -> dict:
    sub = df[df["category"] == cat].copy()
    sub["Y"] = (sub["sev_ordinal"] >= SEV_SERIOUS_ORDINAL).astype(int)
    factor_cols = [c for c in df.columns
                   if c.isupper() and c != "Y" and c not in OUTCOME_FACTORS]
    support = {c: int(sub[c].sum()) for c in factor_cols}
    candidates = [c for c in factor_cols if support[c] >= MIN_SUPPORT]

    def eff(pa_map, x):
        ps = [p for p in pa_map.get(x, [])
              if p in sub.columns and support.get(p, 0) >= MIN_SUPPORT]
        ps = sorted(ps, key=lambda p: -support[p])[:MAX_PARENTS]
        p0, p1 = p_y_do_x(sub, x, ps)
        return p1 - p0, ps

    rows = []
    for x in candidates:
        e_p, z_p = eff(pa_prior, x)
        e_n, z_n = eff(pa_noprior, x)
        rows.append({"factor": x, "n_present": support[x],
                     "effect_with_prior": round(e_p, 4),
                     "effect_no_prior": round(e_n, 4),
                     "delta": round(e_p - e_n, 4),
                     "Z_with_prior": z_p, "Z_no_prior": z_n})
    rows.sort(key=lambda r: -r["effect_with_prior"])
    d = np.array([abs(r["delta"]) for r in rows])
    # do the two runs rank the intervention points the same way?
    top_p = [r["factor"] for r in sorted(rows, key=lambda r: -r["effect_with_prior"])[:top_k]]
    top_n = [r["factor"] for r in sorted(rows, key=lambda r: -r["effect_no_prior"])[:top_k]]
    return {
        "n_factors_compared": len(rows),
        "mean_abs_delta_effect": round(float(d.mean()), 4),
        "max_abs_delta_effect": round(float(d.max()), 4),
        "n_effects_changed_gt_0.02": int((d > 0.02).sum()),
        f"top{top_k}_overlap": len(set(top_p) & set(top_n)),
        f"top{top_k}_with_prior": top_p,
        f"top{top_k}_no_prior": top_n,
        "per_factor": rows[:top_k],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-vectors",
                    default="event_extraction/out/causation_kg/factor_vectors.parquet",
                    type=Path)
    ap.add_argument("--dag-dir",
                    default="event_extraction/out/aggregate_kg/per_category_dag",
                    type=Path)
    ap.add_argument("--noprior-dir",
                    default="event_extraction/out/aggregate_kg/per_category_dag_noprior",
                    type=Path)
    ap.add_argument("--categories", nargs="+", default=["LOC-I", "SCF-PP"])
    ap.add_argument("--min-stability", default=0.5, type=float)
    ap.add_argument("--top-k", default=8, type=int)
    args = ap.parse_args()

    df = pd.read_parquet(args.factor_vectors)
    out = {"min_stability_with_prior": args.min_stability,
           "note": "no-prior run carries no bootstrap column, so all its edges are used",
           "categories": {}}

    for cat in args.categories:
        e_prior = read_edges(args.dag_dir / f"{cat}.csv", args.min_stability)
        e_noprior = read_edges(args.noprior_dir / f"{cat}.csv", None)
        res = compare_structure(e_prior, e_noprior)
        res["downstream"] = downstream(cat, df, parents_from(e_prior),
                                       parents_from(e_noprior), args.top_k)
        out["categories"][cat] = res
        print(f"\n=== {cat} ===")
        print(f"edges: prior {res['n_edges_with_prior']}, "
              f"no-prior {res['n_edges_no_prior']}, "
              f"skeleton Jaccard {res['skeleton_jaccard']}, SHD {res['shd']}")
        d = res["downstream"]
        print(f"downstream: mean |delta effect| {d['mean_abs_delta_effect']}, "
              f"max {d['max_abs_delta_effect']}, "
              f"changed>0.02 {d['n_effects_changed_gt_0.02']}/{d['n_factors_compared']}, "
              f"top-{args.top_k} overlap {d[f'top{args.top_k}_overlap']}/{args.top_k}")

    p = Path("event_extraction/out/causation_kg/prior_ablation.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {p}")


if __name__ == "__main__":
    main()
