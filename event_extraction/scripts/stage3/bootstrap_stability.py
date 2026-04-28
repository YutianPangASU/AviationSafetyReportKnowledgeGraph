"""Piece 5 — bootstrap stability scoring for the per-category PC DAGs.

For each category:
  * load the per-accident presence matrix (recomputed from extraction)
  * resample 80 % of accidents B times, rerun PC each time, count edge stability
  * merge stability scores into the previously written DAG

LLM and Laplacian priors are FIXED across bootstraps (per the Stage-3 plan)
because re-running LLM judgments per bootstrap would be both expensive and
methodologically wrong (they're a prior, not a sample).

Output: appends `bootstrap_stability` column to per_category_dag/{cat}.csv
        and updates per_category_dag/{cat}.gpickle with edge attribute.

Run:
    python event_extraction/scripts/stage3/bootstrap_stability.py \\
        --categories LOC-I --bootstraps 30
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx

from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.utils.cit import chisq

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import EVENT_VOCAB, VOCAB_INDEX, stream_records_by_category, event_indices_in_record

def build_X(extraction: Path, enriched: Path, target_cat: str) -> np.ndarray:
    rows = []
    for cat, rec in stream_records_by_category(extraction, enriched):
        if cat != target_cat:
            continue
        local = event_indices_in_record(rec)
        present = sorted(set(local.values()))
        if not present:
            continue
        row = np.zeros(len(EVENT_VOCAB), dtype=np.int32)
        for idx in present:
            row[idx] = 1
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(EVENT_VOCAB)), dtype=np.int32)

def pc_edges_set(Xa: np.ndarray, alpha: float, active_idx: list[int]) -> set[tuple[int, int]]:
    """Run PC, return set of (src_vocab_idx, dst_vocab_idx) directed edges (skip undirected)."""
    cg = cl_pc(Xa, alpha=alpha, indep_test=chisq, verbose=False, show_progress=False)
    A = cg.G.graph
    n = A.shape[0]
    out: set[tuple[int, int]] = set()
    for ai in range(n):
        for aj in range(n):
            if ai == aj:
                continue
            if A[ai, aj] == -1 and A[aj, ai] == 1:
                out.add((active_idx[ai], active_idx[aj]))
            elif A[ai, aj] == -1 and A[aj, ai] == -1 and ai < aj:
                # CPDAG undirected — count both directions for stability of presence
                out.add((active_idx[ai], active_idx[aj]))
                out.add((active_idx[aj], active_idx[ai]))
    return out

def run_one(cat: str, args):
    print(f"\n=== {cat} ===")
    stem = cat.replace(":", "_").replace("/", "_")
    pkl = args.dag_dir / f"{stem}.gpickle"
    csv = args.dag_dir / f"{stem}.csv"
    if not pkl.exists():
        print("  no DAG; skip"); return
    with pkl.open("rb") as f:
        G: nx.DiGraph = pickle.load(f)

    print(f"  building presence matrix…")
    X = build_X(args.extraction, args.enriched, cat)
    n_acc, N = X.shape
    print(f"  X: {n_acc} accidents")
    if n_acc < 50:
        print("  skip: too few accidents"); return
    col_sums = X.sum(axis=0)
    active_idx = [i for i, c in enumerate(col_sums) if args.min_event_count <= c <= n_acc - args.min_event_count]
    if len(active_idx) < 4:
        print("  skip: too few active events"); return
    Xa = X[:, active_idx]

    rng = random.Random(args.seed)
    counts: Counter = Counter()
    for b in range(args.bootstraps):
        idx = [rng.randrange(0, n_acc) for _ in range(int(0.8 * n_acc))]
        Xb = Xa[idx, :]
        try:
            edges = pc_edges_set(Xb, args.alpha, active_idx)
        except Exception as e:
            print(f"    [b={b}] PC failed: {type(e).__name__}: {e}"); continue
        for e in edges:
            counts[e] += 1
        if (b + 1) % 5 == 0:
            print(f"    bootstrap {b+1}/{args.bootstraps}")

    # write back to graph
    for u, v, d in G.edges(data=True):
        ui = VOCAB_INDEX.get(u); vi = VOCAB_INDEX.get(v)
        if ui is None or vi is None:
            continue
        cnt = counts.get((ui, vi), 0)
        d["bootstrap_stability"] = cnt / args.bootstraps if args.bootstraps else 0.0

    with pkl.open("wb") as f:
        pickle.dump(G, f)
    # rewrite CSV with the new column
    with csv.open("w") as f:
        f.write("src,dst,direction_score,support_count,llm_prior,laplacian_prior,bootstrap_stability,pc_directed,pc_oriented_by_llm,from_cpdag_undirected\n")
        for u, v, d in G.edges(data=True):
            f.write(",".join(str(x if x is not None else "") for x in (
                u, v, d.get("direction_score"), d.get("support_count"),
                d.get("llm_prior"), d.get("laplacian_prior"), d.get("bootstrap_stability"),
                int(bool(d.get("pc_directed"))), int(bool(d.get("pc_oriented_by_llm"))),
                int(bool(d.get("from_cpdag_undirected"))),
            )) + "\n")
    print(f"  edges with stability >= 0.5: {sum(1 for _, _, d in G.edges(data=True) if (d.get('bootstrap_stability') or 0) >= 0.5)}")
    print(f"  -> {pkl.name} + {csv.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", default="event_extraction/out/full_corpus_v3.jsonl", type=Path)
    ap.add_argument("--enriched", default="data/corpus/corpus_enriched.jsonl", type=Path)
    ap.add_argument("--dag-dir", default="event_extraction/out/aggregate_kg/per_category_dag", type=Path)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--bootstraps", default=30, type=int)
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--min-event-count", default=5, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    targets = args.categories or [
        p.stem for p in args.dag_dir.glob("*.gpickle")
    ]
    print(f"Bootstrapping {len(targets)} categories × {args.bootstraps} samples each")
    for cat in targets:
        run_one(cat, args)

if __name__ == "__main__":
    main()
