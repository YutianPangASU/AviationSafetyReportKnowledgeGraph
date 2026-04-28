"""Piece 4-alt — denser association DAG built directly from LLM order +
precedence, skipping PC's conditional-independence pruning.

PC produces the *minimum* causal skeleton needed to explain the data — it
removes edges that are explained by transitive paths (A → C dropped if
A → B → C). That's correct for causal identifiability but produces a graph
much sparser than published HFACS-style accident chains.

This script produces a complementary view: every directed pair where either
(a) the LLM judged it directionally with confidence >= conf_threshold, or
(b) the precedence asymmetry is strong (direction_score >= dir_threshold),
plus a co-occurrence support floor. No CI pruning. The result is denser by
design — a "strong-association DAG" that complements the PC DAG.

Cycles are broken by removing the lowest-confidence edge in each cycle.

Run:
    python event_extraction/scripts/stage3/build_dense_dag.py \\
        --precedence-dir event_extraction/out/aggregate_kg/per_category_precedence \\
        --llm-order-dir event_extraction/out/aggregate_kg/per_category_llm_order \\
        --out-dir event_extraction/out/aggregate_kg/per_category_dag_dense \\
        --categories LOC-I \\
        --min-cooccur 5 --conf-threshold 0.5 --dir-threshold 0.65
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import EVENT_VOCAB, VOCAB_INDEX

def llm_lookup(p: Path) -> dict[tuple[str, str], dict]:
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()).get("pair_results", []):
        out[(r["a"], r["b"])] = r
        rel = r["relation"]
        flipped = {"a_before_b": "b_before_a", "b_before_a": "a_before_b"}.get(rel, rel)
        out[(r["b"], r["a"])] = {**r, "relation": flipped, "a": r["b"], "b": r["a"]}
    return out

def break_cycles(G: nx.DiGraph) -> int:
    """Remove the lowest-weight edge in each cycle until acyclic. Returns
    number of edges removed."""
    removed = 0
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        # cycle is a list of (u, v, key) tuples
        edges_in_cycle = [(u, v) for u, v, *_ in cycle]
        # weight: prefer higher direction_score; fall back to llm_prior; fall back to support
        def edge_weight(uv):
            u, v = uv
            d = G[u][v]
            return (d.get("direction_score", 0.5), d.get("llm_prior") or 0.0, d.get("support_count") or 0)
        weakest = min(edges_in_cycle, key=edge_weight)
        G.remove_edge(*weakest)
        removed += 1
    return removed

def run_one(cat: str, args) -> nx.DiGraph:
    print(f"\n=== {cat} ===")
    stem = cat.replace(":", "_").replace("/", "_")
    pre_npz = args.precedence_dir / f"{stem}.npz"
    if not pre_npz.exists():
        print("  no precedence; skip"); return nx.DiGraph()
    arrs = np.load(pre_npz)
    cooccur = arrs["cooccur"]
    precede = arrs["precede"]
    presence = arrs["presence"]

    laplacian = None
    lap_p = args.laplacian_dir / f"{stem}.npz" if args.laplacian_dir else None
    if lap_p and lap_p.exists():
        laplacian = np.load(lap_p)["prior"]

    llm = llm_lookup(args.llm_order_dir / f"{stem}.json") if args.llm_order_dir else {}

    G = nx.DiGraph()
    for i, ev in enumerate(EVENT_VOCAB):
        G.add_node(ev, presence=int(presence[i]))

    n_emitted = n_skipped_support = n_skipped_lap = n_skipped_unrelated = 0
    N = len(EVENT_VOCAB)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            sup = int(cooccur[i, j])
            if sup < args.min_cooccur:
                n_skipped_support += 1; continue
            lap = float(laplacian[i, j]) if laplacian is not None else None
            if lap is not None and lap < args.laplacian_threshold:
                n_skipped_lap += 1; continue

            llm_rec = llm.get((EVENT_VOCAB[i], EVENT_VOCAB[j]))
            llm_relation = llm_rec["relation"] if llm_rec else None
            llm_conf = float(llm_rec.get("confidence", 0.0)) if llm_rec else 0.0
            # Precedence-based direction score
            pij = int(precede[i, j]); pji = int(precede[j, i])
            ds = (pij + 1) / (pij + pji + 2)

            # Decision rules to emit i -> j
            emit = False
            evidence = None
            if llm_relation == "a_before_b" and llm_conf >= args.conf_threshold:
                emit = True; evidence = "llm"
            elif llm_relation == "b_before_a" and llm_conf >= args.conf_threshold:
                emit = False  # the j->i edge will be added when (j,i) is iterated
            elif llm_relation in ("concurrent", "unrelated"):
                if llm_relation == "unrelated":
                    n_skipped_unrelated += 1
                emit = False
            else:  # no LLM judgment
                if ds >= args.dir_threshold:
                    emit = True; evidence = "precedence"

            if emit:
                G.add_edge(EVENT_VOCAB[i], EVENT_VOCAB[j],
                           direction_score=ds,
                           support_count=sup,
                           llm_prior=llm_conf if evidence == "llm" else None,
                           laplacian_prior=lap,
                           bootstrap_stability=0.0,  # filled later if bootstrap is run
                           evidence=evidence,
                           pc_directed=0, pc_oriented_by_llm=0, from_cpdag_undirected=0,
                           reoriented=0, llm_evidence=llm_relation)
                n_emitted += 1
    print(f"  candidates: {n_emitted} emitted, {n_skipped_support} dropped by support, "
          f"{n_skipped_lap} by laplacian, {n_skipped_unrelated} unrelated")
    n_removed = break_cycles(G)
    if n_removed:
        print(f"  cycle-break: removed {n_removed} weakest-confidence edges to enforce DAG")
    return G

def write_outputs(G: nx.DiGraph, cat: str, out_dir: Path):
    stem = cat.replace(":", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / f"{stem}.gpickle"
    csv = out_dir / f"{stem}.csv"
    with pkl.open("wb") as f:
        pickle.dump(G, f)
    with csv.open("w") as f:
        f.write("src,dst,direction_score,support_count,llm_prior,laplacian_prior,bootstrap_stability,evidence,pc_directed,pc_oriented_by_llm,from_cpdag_undirected,reoriented,llm_evidence\n")
        for u, v, d in G.edges(data=True):
            f.write(",".join(str(x) if x is not None else "" for x in (
                u, v, d.get("direction_score"), d.get("support_count"),
                d.get("llm_prior"), d.get("laplacian_prior"), d.get("bootstrap_stability"),
                d.get("evidence"),
                int(bool(d.get("pc_directed"))), int(bool(d.get("pc_oriented_by_llm"))),
                int(bool(d.get("from_cpdag_undirected"))),
                int(bool(d.get("reoriented"))), d.get("llm_evidence"),
            )) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precedence-dir", default="event_extraction/out/aggregate_kg/per_category_precedence", type=Path)
    ap.add_argument("--llm-order-dir", default="event_extraction/out/aggregate_kg/per_category_llm_order", type=Path)
    ap.add_argument("--laplacian-dir", default="event_extraction/out/aggregate_kg/per_category_laplacian", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/per_category_dag_dense", type=Path)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--min-cooccur", default=5, type=int)
    ap.add_argument("--conf-threshold", default=0.5, type=float, help="LLM confidence threshold to emit")
    ap.add_argument("--dir-threshold", default=0.65, type=float, help="precedence direction score threshold (used when no LLM judgment)")
    ap.add_argument("--laplacian-threshold", default=0.2, type=float)
    args = ap.parse_args()
    targets = args.categories or [
        json.loads(p.read_text())["category"]
        for p in args.precedence_dir.glob("*.json")
    ]
    print(f"Building dense association DAGs for {len(targets)} categories")
    for cat in targets:
        G = run_one(cat, args)
        if G.number_of_edges() > 0:
            write_outputs(G, cat, args.out_dir)
            print(f"  -> {cat}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if __name__ == "__main__":
    main()
