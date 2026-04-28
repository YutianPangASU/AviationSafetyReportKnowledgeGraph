"""Piece 4.5 — re-orient PC-output edges using LLM order + precedence asymmetry.

PC alone cannot reliably orient edges from binary co-occurrence data when no
v-structure exists. This pass walks every directed edge in the PC output and
flips it if the combined evidence (precedence asymmetry + LLM order) clearly
points the other way. Skeleton (which pairs are connected) is preserved; only
direction is potentially swapped.

Decision rule per edge u -> v with attrs:
  - direction_score ∈ [0, 1] (precedence asymmetry; 0.5 = tie, > 0.5 favours u -> v)
  - llm_prior      (None or float; populated only when the edge was originally
                    CPDAG-undirected and resolved by LLM)

For PC-directed edges we look up the LLM judgment from the side-loaded JSON
(if available) since llm_prior in the CSV was None for those.

Action:
  * If direction_score >= 0.5 AND (no LLM judgment or LLM agrees) -> keep
  * If direction_score < 0.5 AND LLM says the opposite -> flip
  * If direction_score < 0.3 AND no LLM signal -> flip on precedence alone
  * Else -> keep (low confidence, leave the user to filter via direction_score)

Adds two new edge attrs: `reoriented` (bool) and `llm_evidence` (relation or None).

Run:
    python event_extraction/scripts/stage3/reorient_dag.py \\
        --in-dir event_extraction/out/aggregate_kg/per_category_dag \\
        --llm-order-dir event_extraction/out/aggregate_kg/per_category_llm_order
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx

def load_llm_order(p: Path) -> dict[tuple[str, str], dict]:
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()).get("pair_results", []):
        out[(r["a"], r["b"])] = r
        rel = r["relation"]
        flipped = {"a_before_b": "b_before_a", "b_before_a": "a_before_b"}.get(rel, rel)
        out[(r["b"], r["a"])] = {**r, "relation": flipped, "a": r["b"], "b": r["a"]}
    return out

def reorient_one(cat: str, args) -> int:
    stem = cat
    pkl = args.in_dir / f"{stem}.gpickle"
    csv = args.in_dir / f"{stem}.csv"
    if not pkl.exists():
        return 0
    with pkl.open("rb") as f:
        G: nx.DiGraph = pickle.load(f)
    llm = load_llm_order(args.llm_order_dir / f"{stem}.json") if args.llm_order_dir else {}

    # Decide flips first; only mutate after iteration is done
    flips: list[tuple[str, str, dict]] = []
    keeps: list[tuple[str, str, dict]] = []
    for u, v, d in list(G.edges(data=True)):
        ds = float(d.get("direction_score") or 0.5)
        rec = llm.get((u, v))
        rel = rec["relation"] if rec else None
        flip = False
        # LLM says opposite direction
        if rel == "b_before_a" and ds < 0.5:
            flip = True
        # No LLM but precedence is strongly opposite
        elif rel is None and ds < 0.3:
            flip = True
        d["llm_evidence"] = rel
        if flip:
            d["reoriented"] = True
            flips.append((u, v, d))
        else:
            d["reoriented"] = False
            keeps.append((u, v, d))

    # Apply flips: remove old, add reversed
    for u, v, d in flips:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        # invert direction_score for the new edge
        d2 = dict(d)
        d2["direction_score"] = 1.0 - float(d.get("direction_score") or 0.5)
        G.add_edge(v, u, **d2)

    # Save
    with pkl.open("wb") as f:
        pickle.dump(G, f)
    with csv.open("w") as f:
        f.write("src,dst,direction_score,support_count,llm_prior,laplacian_prior,bootstrap_stability,pc_directed,pc_oriented_by_llm,from_cpdag_undirected,reoriented,llm_evidence\n")
        for u, v, d in G.edges(data=True):
            row = [u, v,
                   d.get("direction_score"), d.get("support_count"),
                   d.get("llm_prior"), d.get("laplacian_prior"), d.get("bootstrap_stability"),
                   int(bool(d.get("pc_directed"))), int(bool(d.get("pc_oriented_by_llm"))),
                   int(bool(d.get("from_cpdag_undirected"))),
                   int(bool(d.get("reoriented"))),
                   d.get("llm_evidence")]
            f.write(",".join(str(x) if x is not None else "" for x in row) + "\n")
    print(f"  {cat:20s}  flipped {len(flips):3d}  kept {len(keeps):3d}")
    return len(flips)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="event_extraction/out/aggregate_kg/per_category_dag", type=Path)
    ap.add_argument("--llm-order-dir", default="event_extraction/out/aggregate_kg/per_category_llm_order", type=Path)
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()
    targets = args.categories or [p.stem for p in args.in_dir.glob("*.gpickle")]
    print(f"Reorienting {len(targets)} categories")
    total = 0
    for cat in targets:
        total += reorient_one(cat, args)
    print(f"\nTotal flips: {total}")

if __name__ == "__main__":
    main()
