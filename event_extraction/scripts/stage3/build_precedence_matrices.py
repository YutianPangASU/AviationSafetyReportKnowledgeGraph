"""Piece 1 — per-category co-occurrence + temporal precedence matrices.

For each CICTT category C, build:
  * presence  P_C ∈ R^N: number of accidents where event-type i appeared (N=42)
  * cooccur   C_C ∈ R^{42×42}: accidents where both event-types i, j appear
  * precede   M_C ∈ R^{42×42}: accidents where event-type i temporally / causally
              preceded event-type j, derived from extracted edges of types
              {CAUSES, CONTRIBUTES_TO, TRIGGERS, ENABLES, PRECEDES}
              interpreted as i->j ordering

Output files (per category) under
  event_extraction/out/aggregate_kg/per_category_precedence/
    {cat}.npz                - keyed arrays: presence, cooccur, precede
    {cat}.json               - {category, n_accidents, vocab, ...}

Run:
    python event_extraction/scripts/stage3/build_precedence_matrices.py \\
        --extraction event_extraction/out/full_corpus_v4.jsonl \\
        --enriched data/corpus/corpus_enriched.jsonl \\
        --out-dir event_extraction/out/aggregate_kg/per_category_precedence
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# allow running as a script from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import (EVENT_VOCAB, VOCAB_INDEX, stream_records_by_category,
                     event_indices_in_record, precedence_pairs_in_record)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", default="event_extraction/out/full_corpus_v4.jsonl", type=Path)
    ap.add_argument("--enriched", default="data/corpus/corpus_enriched.jsonl", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/per_category_precedence", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    N = len(EVENT_VOCAB)
    presence: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N, dtype=np.int32))
    cooccur:  dict[str, np.ndarray] = defaultdict(lambda: np.zeros((N, N), dtype=np.int32))
    precede:  dict[str, np.ndarray] = defaultdict(lambda: np.zeros((N, N), dtype=np.int32))
    n_accidents: dict[str, int] = defaultdict(int)

    print(f"Streaming {args.extraction.name}…")
    for cat, rec in stream_records_by_category(args.extraction, args.enriched):
        n_accidents[cat] += 1
        local_to_idx = event_indices_in_record(rec)
        present_indices = sorted(set(local_to_idx.values()))
        # presence
        for i in present_indices:
            presence[cat][i] += 1
        # cooccur (symmetric, only count each unordered pair once per record)
        for i in present_indices:
            for j in present_indices:
                if i != j:
                    cooccur[cat][i, j] += 1
        # precede (v4: chain caused_by back-references; v3 legacy: typed edges)
        for si, di in precedence_pairs_in_record(rec, local_to_idx):
            precede[cat][si, di] += 1

    print(f"\nWriting per-category matrices -> {args.out_dir}")
    for cat in sorted(n_accidents):
        npz_path = args.out_dir / f"{cat.replace(':', '_').replace('/', '_')}.npz"
        json_path = npz_path.with_suffix(".json")
        np.savez(npz_path,
                 presence=presence[cat], cooccur=cooccur[cat], precede=precede[cat])
        json_path.write_text(json.dumps({
            "category": cat,
            "n_accidents": n_accidents[cat],
            "vocab": EVENT_VOCAB,
            "precedence_source": "v4 chain caused_by back-references",
        }, indent=2))
        # quick sanity row
        top_present = sorted(enumerate(presence[cat]), key=lambda kv: -kv[1])[:3]
        print(f"  {cat:30s} accidents={n_accidents[cat]:>5}  top: " +
              ", ".join(f"{EVENT_VOCAB[i]}({c})" for i, c in top_present))
    print(f"\nDone. {len(n_accidents)} categories written.")

if __name__ == "__main__":
    main()
