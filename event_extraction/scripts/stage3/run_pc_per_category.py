"""Piece 4 — PC algorithm with optional LLM-order + Laplacian priors.

For each CICTT category, build a per-accident binary presence matrix
X ∈ {0,1}^{n_accidents × 42} (event_type i appeared in accident k), then run
the PC algorithm (causal-learn) to recover a CPDAG. Optionally seed the
skeleton with a prior:

  * support filter   - drop pairs with cooccur < min_cooccur (cuts noise)
  * Laplacian prior  - require prior >= laplacian_threshold (cuts noise further)
  * LLM order        - resolve undirected edges by majority of LLM judgments

Output (per category):
  per_category_dag/{cat}.gpickle  - networkx DiGraph
  per_category_dag/{cat}.csv      - src,dst,direction_score,support_count,llm_prior,laplacian_prior

Run baseline (no priors):
    python event_extraction/scripts/stage3/run_pc_per_category.py \\
        --extraction event_extraction/out/full_corpus_v3.jsonl \\
        --enriched data/corpus/corpus_enriched.jsonl \\
        --precedence-dir event_extraction/out/aggregate_kg/per_category_precedence \\
        --out-dir event_extraction/out/aggregate_kg/per_category_dag \\
        --min-cooccur 10 --no-laplacian --no-llm-order \\
        --categories LOC-I

Run with priors:
    python event_extraction/scripts/stage3/run_pc_per_category.py \\
        --llm-order-dir event_extraction/out/aggregate_kg/per_category_llm_order \\
        --laplacian-dir event_extraction/out/aggregate_kg/per_category_laplacian \\
        --laplacian-threshold 0.4 \\
        --categories LOC-I
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import EVENT_VOCAB, VOCAB_INDEX, stream_records_by_category, event_indices_in_record

# causal-learn PC
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.utils.cit import chisq

def build_presence_X(extraction: Path, enriched: Path, target_cat: str) -> np.ndarray:
    """Binary presence matrix for one category."""
    rows: list[np.ndarray] = []
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
    if not rows:
        return np.zeros((0, len(EVENT_VOCAB)), dtype=np.int32)
    return np.vstack(rows)

def llm_order_lookup(llm_path: Path) -> dict[tuple[str, str], dict]:
    """{(a, b): {relation, confidence}} — keyed both directions for convenience."""
    if not llm_path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    for r in json.loads(llm_path.read_text()).get("pair_results", []):
        a, b = r["a"], r["b"]
        out[(a, b)] = r
        # mirror with relation flipped
        rel = r["relation"]
        if rel == "a_before_b": mirrored = "b_before_a"
        elif rel == "b_before_a": mirrored = "a_before_b"
        else: mirrored = rel
        out[(b, a)] = {**r, "relation": mirrored, "a": b, "b": a}
    return out

def run_pc_one(cat: str, args) -> nx.DiGraph:
    print(f"\n=== {cat} ===")
    stem = cat.replace(":", "_").replace("/", "_")

    # --- inputs ---
    pre_npz = args.precedence_dir / f"{stem}.npz"
    if not pre_npz.exists():
        print(f"  no precedence file; skip"); return nx.DiGraph()
    pre_arrs = np.load(pre_npz)
    cooccur = pre_arrs["cooccur"]
    precede = pre_arrs["precede"]
    presence_counts = pre_arrs["presence"]

    laplacian_prior = None
    if not args.no_laplacian:
        lap_path = args.laplacian_dir / f"{stem}.npz"
        if lap_path.exists():
            laplacian_prior = np.load(lap_path)["prior"]
            print(f"  laplacian prior loaded")

    llm_lookup: dict[tuple[str, str], dict] = {}
    if not args.no_llm_order:
        llm_lookup = llm_order_lookup(args.llm_order_dir / f"{stem}.json")
        print(f"  llm order entries: {len(llm_lookup) // 2 * 2}")

    # --- presence matrix X ---
    print(f"  building presence matrix…")
    X = build_presence_X(args.extraction, args.enriched, cat)
    n_acc, N = X.shape
    print(f"  X shape: {n_acc} accidents × {N} event types")
    if n_acc < args.min_accidents:
        print(f"  skip: only {n_acc} accidents (< min {args.min_accidents})")
        return nx.DiGraph()

    # Determine which event types have enough variation to run PC on
    col_sums = X.sum(axis=0)
    active_idx = [i for i, c in enumerate(col_sums) if args.min_event_count <= c <= n_acc - args.min_event_count]
    print(f"  active event types (var-filtered): {len(active_idx)} / {N}")
    if len(active_idx) < 4:
        print("  skip: too few active event types")
        return nx.DiGraph()
    Xa = X[:, active_idx]

    # --- run PC ---
    # alpha = significance for CI tests
    print(f"  running PC (alpha={args.alpha}, indep_test=chisq) …")
    cg = cl_pc(Xa, alpha=args.alpha, indep_test=chisq, verbose=False, show_progress=False)
    # cg.G.graph is N×N int matrix; entry (i, j) = 1 means i->j, 0 = none, -1 = j<-i, etc.
    # Per causal-learn docs: G[i,j] = 1 and G[j,i] = -1 means i -> j.
    # Undirected (in CPDAG): G[i,j] = G[j,i] = -1
    A = cg.G.graph
    n = A.shape[0]

    G_out = nx.DiGraph()
    for i in range(N):
        G_out.add_node(EVENT_VOCAB[i],
                       presence=int(presence_counts[i]),
                       active=(i in active_idx))

    # build active->vocab index map
    a2v = {a: v for a, v in enumerate(active_idx)}

    skipped_lap = skipped_support = 0
    for ai in range(n):
        for aj in range(n):
            if ai == aj:
                continue
            i = a2v[ai]; j = a2v[aj]
            # PC edge directions:
            #   A[i,j] == -1 and A[j,i] == 1     -> directed i -> j
            #   A[i,j] == -1 and A[j,i] == -1    -> undirected (CPDAG)
            v_ij = A[ai, aj]; v_ji = A[aj, ai]
            directed = (v_ij == -1 and v_ji == 1)
            undirected = (v_ij == -1 and v_ji == -1) and i < j  # only emit once
            if not (directed or undirected):
                continue
            sup = int(cooccur[i, j])
            if sup < args.min_cooccur:
                skipped_support += 1; continue
            lap = float(laplacian_prior[i, j]) if laplacian_prior is not None else None
            if lap is not None and lap < args.laplacian_threshold:
                skipped_lap += 1; continue
            # Direction resolution
            llm_score = None
            if undirected and llm_lookup:
                rec = llm_lookup.get((EVENT_VOCAB[i], EVENT_VOCAB[j]))
                if rec is not None:
                    rel = rec["relation"]; conf = float(rec.get("confidence", 0.0))
                    if rel == "a_before_b":
                        directed = True; llm_score = conf
                    elif rel == "b_before_a":
                        i, j = j, i
                        directed = True; llm_score = conf
                    elif rel == "concurrent":
                        # leave as undirected; both directions get a soft edge
                        pass
            # Direction score: PC-directed -> 1.0; CPDAG-undirected with LLM -> conf; else fallback to precedence asymmetry
            if directed and llm_score is not None:
                direction_score = llm_score
            elif directed:
                # use precedence asymmetry as the PC's preferred direction
                pij = int(precede[i, j]); pji = int(precede[j, i])
                direction_score = (pij + 1) / (pij + pji + 2)
            else:
                # undirected — emit both with low confidence; rely on bootstrap stability later
                direction_score = 0.5
            attrs = {
                "direction_score": float(direction_score),
                "support_count": sup,
                "llm_prior": llm_score,
                "laplacian_prior": lap,
                "pc_directed": bool(directed and llm_score is None),
                "pc_oriented_by_llm": bool(directed and llm_score is not None),
                "from_cpdag_undirected": bool(undirected),
            }
            if directed:
                G_out.add_edge(EVENT_VOCAB[i], EVENT_VOCAB[j], **attrs)
            else:
                G_out.add_edge(EVENT_VOCAB[i], EVENT_VOCAB[j], **attrs)
                G_out.add_edge(EVENT_VOCAB[j], EVENT_VOCAB[i], **attrs)
    print(f"  edges: {G_out.number_of_edges()} kept, {skipped_support} dropped by support, {skipped_lap} by laplacian")
    return G_out

def write_outputs(G: nx.DiGraph, cat: str, out_dir: Path):
    stem = cat.replace(":", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / f"{stem}.gpickle"
    csv = out_dir / f"{stem}.csv"
    with pkl.open("wb") as f:
        pickle.dump(G, f)
    with csv.open("w") as f:
        f.write("src,dst,direction_score,support_count,llm_prior,laplacian_prior,pc_directed,pc_oriented_by_llm,from_cpdag_undirected\n")
        for u, v, d in G.edges(data=True):
            f.write(",".join(str(x if x is not None else "") for x in (
                u, v, d.get("direction_score"), d.get("support_count"),
                d.get("llm_prior"), d.get("laplacian_prior"),
                int(bool(d.get("pc_directed"))), int(bool(d.get("pc_oriented_by_llm"))),
                int(bool(d.get("from_cpdag_undirected"))),
            )) + "\n")
    print(f"  -> {pkl.name} + {csv.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", default="event_extraction/out/full_corpus_v3.jsonl", type=Path)
    ap.add_argument("--enriched", default="data/corpus/corpus_enriched.jsonl", type=Path)
    ap.add_argument("--precedence-dir", default="event_extraction/out/aggregate_kg/per_category_precedence", type=Path)
    ap.add_argument("--llm-order-dir", default="event_extraction/out/aggregate_kg/per_category_llm_order", type=Path)
    ap.add_argument("--laplacian-dir", default="event_extraction/out/aggregate_kg/per_category_laplacian", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/per_category_dag", type=Path)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--min-accidents", default=50, type=int)
    ap.add_argument("--min-event-count", default=5, type=int,
                    help="event type must appear in at least this many accidents to enter PC")
    ap.add_argument("--min-cooccur", default=10, type=int)
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--laplacian-threshold", default=0.4, type=float)
    ap.add_argument("--no-laplacian", action="store_true")
    ap.add_argument("--no-llm-order", action="store_true")
    args = ap.parse_args()

    targets: list[str] = []
    if args.categories:
        targets = args.categories
    else:
        for p in args.precedence_dir.glob("*.json"):
            cat = json.loads(p.read_text())["category"]
            targets.append(cat)
    print(f"PC over {len(targets)} categories")
    for cat in targets:
        try:
            G = run_pc_one(cat, args)
            if G.number_of_edges() > 0:
                write_outputs(G, cat, args.out_dir)
        except Exception as e:
            print(f"  [error] {cat}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
