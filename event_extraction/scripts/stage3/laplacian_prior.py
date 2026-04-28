"""Piece 3 — Laplacian similarity prior over the 42-event vocabulary.

Adapts Zhang et al. (NeurIPS CaLM 2024) to our setting. We build a similarity
matrix S in R^{42×42} from:
  * pairwise LLM ordering judgments (a_before_b / b_before_a / concurrent / unrelated)
    -> turned into a directed similarity proxy: 'concurrent' contributes most,
       a directional ordering contributes some, 'unrelated' contributes 0.
  * cooccurrence support (PMI-normalized) as a secondary signal.

Compute the unnormalized Laplacian L = D - S, take the bottom-k eigenvectors as
a low-dim event embedding, and use squared distances in that embedding as the
soft prior over edge presence: closer pairs are more likely to share a true
causal edge in the ground-truth DAG.

Output per category:
  per_category_laplacian/{cat}.npz  - keyed: similarity, laplacian_emb, prior
  per_category_laplacian/{cat}.json - meta

The prior matrix is in [0, 1]; PC's skeleton seeding uses prior >= threshold to
keep the edge in the candidate set.

Run:
    python event_extraction/scripts/stage3/laplacian_prior.py \\
        --llm-order-dir event_extraction/out/aggregate_kg/per_category_llm_order \\
        --precedence-dir event_extraction/out/aggregate_kg/per_category_precedence \\
        --out-dir event_extraction/out/aggregate_kg/per_category_laplacian \\
        --categories LOC-I
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import EVENT_VOCAB, VOCAB_INDEX

# similarity contribution per relation type
RELATION_SIM = {
    "concurrent":  1.0,
    "a_before_b":  0.6,
    "b_before_a":  0.6,
    "unrelated":   0.0,
}

def build_similarity(pair_results: list[dict], cooccur: np.ndarray) -> np.ndarray:
    N = len(EVENT_VOCAB)
    S = np.zeros((N, N), dtype=np.float64)
    for r in pair_results:
        i = VOCAB_INDEX.get(r["a"]); j = VOCAB_INDEX.get(r["b"])
        if i is None or j is None:
            continue
        rel = r.get("relation", "unrelated")
        conf = float(r.get("confidence", 0.0))
        s = RELATION_SIM.get(rel, 0.0) * conf
        S[i, j] += s; S[j, i] += s
    # Add a small co-occurrence-PMI term so unjudged but co-occurring pairs are not at zero
    total = max(1.0, cooccur.sum())
    pres_marg = cooccur.sum(axis=0) + 1e-9
    for i in range(N):
        for j in range(N):
            if i == j or cooccur[i, j] == 0:
                continue
            p_ij = cooccur[i, j] / total
            p_i = pres_marg[i] / total
            p_j = pres_marg[j] / total
            pmi = math.log(p_ij / (p_i * p_j + 1e-12) + 1e-9)
            S[i, j] += max(0.0, pmi) * 0.05  # small weight; LLM judgements dominate
    np.fill_diagonal(S, 0.0)
    # symmetrise (already symmetric in expectation; enforce numerically)
    S = 0.5 * (S + S.T)
    return S

def laplacian_embedding(S: np.ndarray, k: int = 8) -> np.ndarray:
    """Bottom-k eigenvectors of L = D - S (excluding the trivial zero-eigenvector)."""
    D = np.diag(S.sum(axis=1))
    L = D - S
    # symmetric => use eigh
    w, V = np.linalg.eigh(L)
    # ascending eigenvalues; skip the first if it is ~0 (constant vector)
    keep = V[:, 1:1 + k]
    return keep

def embedding_to_prior(emb: np.ndarray) -> np.ndarray:
    """Squared-distance kernel in [0, 1] from the embedding. Closer pairs => higher prior."""
    N = emb.shape[0]
    d2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d2[i, j] = float(np.sum((emb[i] - emb[j]) ** 2))
    # sigma = median distance for a sane scale
    nz = d2[d2 > 0]
    sigma = float(np.median(nz)) if nz.size else 1.0
    K = np.exp(-d2 / max(1e-9, sigma))
    np.fill_diagonal(K, 0.0)
    return K

def run_one(cat: str, args):
    stem = cat.replace(":", "_").replace("/", "_")
    llm_path = args.llm_order_dir / f"{stem}.json"
    pre_npz = args.precedence_dir / f"{stem}.npz"
    if not llm_path.exists() or not pre_npz.exists():
        print(f"  [skip] {cat}: missing inputs")
        return
    pair_results = json.loads(llm_path.read_text())["pair_results"]
    cooccur = np.load(pre_npz)["cooccur"]
    S = build_similarity(pair_results, cooccur)
    emb = laplacian_embedding(S, k=args.k)
    prior = embedding_to_prior(emb)

    out_npz = args.out_dir / f"{stem}.npz"
    out_json = args.out_dir / f"{stem}.json"
    np.savez(out_npz, similarity=S, laplacian_emb=emb, prior=prior)
    out_json.write_text(json.dumps({
        "category": cat, "vocab": EVENT_VOCAB, "k_eigenvectors": args.k,
    }, indent=2))
    # quick sanity: top-3 most-similar to LOSS_OF_CONTROL_INFLIGHT
    if "LOSS_OF_CONTROL_INFLIGHT" in VOCAB_INDEX:
        i = VOCAB_INDEX["LOSS_OF_CONTROL_INFLIGHT"]
        ranked = sorted(enumerate(prior[i]), key=lambda kv: -kv[1])[:6]
        print(f"  [{cat}] LOC-I-similar (Laplacian prior): " + ", ".join(
            f"{EVENT_VOCAB[j]}({p:.2f})" for j, p in ranked if j != i))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-order-dir", default="event_extraction/out/aggregate_kg/per_category_llm_order", type=Path)
    ap.add_argument("--precedence-dir", default="event_extraction/out/aggregate_kg/per_category_precedence", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/per_category_laplacian", type=Path)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--k", default=8, type=int, help="eigenvectors to keep")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.categories or [
        json.loads(p.read_text())["category"] for p in args.llm_order_dir.glob("*.json")
    ]
    print(f"Targets: {targets}")
    for cat in targets:
        run_one(cat, args)

if __name__ == "__main__":
    main()
