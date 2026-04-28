"""Piece 2 — Vashishtha-style LLM causal-order prompting (pairwise variant).

For each CICTT category, ask the LLM about ordered pairs of event types that
co-occur frequently in that category. The LLM returns one of
{"a_before_b", "b_before_a", "concurrent", "unrelated"} along with a
confidence in [0, 1]. guided_json constrains the response.

Pairwise (rather than full triplets) is faster and the Laplacian prior built
on top of it (piece 3) is well-defined for a directed graph induced by the
pairwise judgments. Triplet-based cycle resolution can be layered on later.

Selection: only pairs with cooccur >= MIN_PAIR_COOCCUR are queried, capped at
TOP_K_PAIRS_PER_CATEGORY. This keeps the total call count tractable while
covering the high-support pairs that PC will care about.

Run:
    python event_extraction/scripts/stage3/llm_causal_order.py \\
        --in-dir event_extraction/out/aggregate_kg/per_category_precedence \\
        --out-dir event_extraction/out/aggregate_kg/per_category_llm_order \\
        --categories LOC-I \\
        --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _common import EVENT_VOCAB

PAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relation", "confidence"],
    "properties": {
        "relation": {"enum": ["a_before_b", "b_before_a", "concurrent", "unrelated"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
    },
}

SYSTEM = """You are an aviation safety domain expert. You will be given a CICTT accident category and a pair of event types from the ACE-Graph 42-event vocabulary. Decide the typical temporal/causal order between the two event types in accidents of this category.

Return a single JSON object: {"relation": "<a_before_b|b_before_a|concurrent|unrelated>", "confidence": <0..1>, "rationale": "<one sentence>"}.

Definitions:
  * a_before_b   - In a typical accident of this category, A occurs (or starts to develop) before B; A is causally upstream of B.
  * b_before_a   - The reverse direction.
  * concurrent   - A and B typically occur in the same temporal window or are aspects of the same state; neither is upstream.
  * unrelated    - These two event types do not have a typical causal/temporal ordering in accidents of this category.

Be strict about ordering: only choose `a_before_b` or `b_before_a` if there is a clear typical ordering. Otherwise prefer `concurrent` or `unrelated`."""

def user_prompt(cat: str, a: str, b: str) -> str:
    return (
        f"CICTT category: {cat}\n"
        f"Event A: {a}\n"
        f"Event B: {b}\n\n"
        "What is the typical temporal/causal ordering of A and B in accidents of this category?"
    )

async def query_pair(client, endpoint, model, cat, a, b, sem):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(cat, a, b)},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": PAIR_SCHEMA,
    }
    async with sem:
        resp = await client.post(f"{endpoint}/chat/completions", json=body, timeout=180)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)

async def run_category(cat: str, args, client, sem, cooccur, vocab) -> dict:
    """Query pairs for a single category. Returns dict with pair_results list and meta."""
    out_path = args.out_dir / f"{cat.replace(':', '_').replace('/', '_')}.json"
    cached: dict[str, dict] = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        for r in prev.get("pair_results", []):
            cached[f"{r['a']}__{r['b']}"] = r

    # Build candidate pairs (a < b lexicographically, asymmetric coverage of (a,b) and (b,a) is implicit)
    N = len(vocab)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            c = int(cooccur[i, j])
            if c >= args.min_cooccur:
                pairs.append((c, i, j))
    # sort by co-occurrence desc, take top K
    pairs.sort(key=lambda r: -r[0])
    pairs = pairs[: args.top_k]
    print(f"  [{cat}] candidate pairs: {len(pairs)} (cooccur >= {args.min_cooccur}, top {args.top_k})")

    # Filter out cached
    todo = [(c, i, j) for c, i, j in pairs if f"{vocab[i]}__{vocab[j]}" not in cached]
    print(f"  [{cat}] {len(todo)} new pairs to query, {len(cached)} cached")

    n_done = 0
    t0 = time.time()
    async def worker(c, i, j):
        nonlocal n_done
        try:
            obj = await query_pair(client, args.endpoint, args.model, cat, vocab[i], vocab[j], sem)
        except Exception as e:
            obj = {"relation": "unrelated", "confidence": 0.0, "rationale": f"err:{type(e).__name__}"}
        n_done += 1
        if n_done % 50 == 0:
            rate = n_done / (time.time() - t0 + 1e-9)
            print(f"    [{cat}] {n_done}/{len(todo)} ({rate:.1f}/s)")
        return {
            "a": vocab[i], "b": vocab[j], "cooccur": c,
            "relation": obj["relation"], "confidence": float(obj["confidence"]),
            "rationale": obj.get("rationale", ""),
        }
    new_results = await asyncio.gather(*[worker(c, i, j) for c, i, j in todo])
    pair_results = list(cached.values()) + new_results
    return {
        "category": cat,
        "n_pairs": len(pair_results),
        "min_cooccur": args.min_cooccur,
        "top_k": args.top_k,
        "pair_results": pair_results,
    }

async def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets: list[str] = []
    if args.categories:
        targets = args.categories
    else:
        # default: every category with a precedence file
        for p in args.in_dir.glob("*.json"):
            cat = json.loads(p.read_text())["category"]
            targets.append(cat)
    print(f"Targets: {targets}")

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        for cat in targets:
            stem = cat.replace(":", "_").replace("/", "_")
            jpath = args.in_dir / f"{stem}.json"
            npz = args.in_dir / f"{stem}.npz"
            if not jpath.exists() or not npz.exists():
                print(f"  [skip] missing precedence for {cat}")
                continue
            meta = json.loads(jpath.read_text())
            arrs = np.load(npz)
            cooccur = arrs["cooccur"]
            result = await run_category(cat, args, client, sem, cooccur, EVENT_VOCAB)
            out_path = args.out_dir / f"{stem}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"  [{cat}] -> {out_path}  ({result['n_pairs']} pair results)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="event_extraction/out/aggregate_kg/per_category_precedence", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/per_category_llm_order", type=Path)
    ap.add_argument("--categories", nargs="*", default=None,
                    help="categories to run; default = all categories with precedence files")
    ap.add_argument("--min-cooccur", default=10, type=int,
                    help="only query pairs with cooccur >= this many accidents")
    ap.add_argument("--top-k", default=300, type=int,
                    help="cap on pairs per category (sorted by cooccur desc)")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--concurrency", default=4, type=int)
    args = ap.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
