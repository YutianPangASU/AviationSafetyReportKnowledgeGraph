"""Chain-ordering eval against the TSB Canada EventPhaseSequence.

Companion to eval_chain_order.py, which scores against NTSB `seq_of_events`.
The TSB sequence is coded by a different agency and its EventPhaseSequence
number is temporal order, so this eval tests the paper's claim that the low
NTSB all-pairs tau (0.31) reflects a coding convention rather than a model
failure: on a gold standard that is chronological throughout, the all-pairs
tau is the headline number, with no cross-occurrence restriction needed.

Method per record (>= 2 gold events, >= 2 chain nodes):
  1. The same LLM judge as semantic_eval.py matches each TSB event entry
     ("EVENT NAME (phase: PHASE)") to chain-node indices.
  2. For every gold pair (i < j) where both matched, concordant if the
     earliest matched chain position of i precedes that of j.
  3. tau = (concordant - discordant) / (concordant + discordant), pooled and
     per record.

Run (vLLM server up, extraction done):
    python event_extraction/scripts/eval_tsb_order.py \
        --extraction event_extraction/out/tsb_order_extraction.jsonl \
        --out event_extraction/out/tsb_order_judged.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_eval import JUDGE_SCHEMA, JUDGE_SYSTEM, event_payload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD = ROOT / "event_extraction/out/tsb_order_gold.jsonl"
INPUT = ROOT / "event_extraction/out/tsb_order_input.jsonl"


def gold_strings(seq: list[dict]) -> list[str]:
    out = []
    for e in seq:
        s = e["event"]
        if e.get("phase"):
            s += f" (phase: {e['phase']})"
        out.append(s)
    return out


async def judge_one(client, endpoint, model, factors, events, narrative, sem):
    user = (
        f"NARRATIVE EXCERPT (truncated):\n{narrative[:1200]}\n\n"
        f"TSB CODED EVENT SEQUENCE (in investigator order):\n"
        + "\n".join(f"  [{i}] {s}" for i, s in enumerate(factors))
        + "\n\nEXTRACTED CHAIN NODES (in extracted order):\n"
        + "\n".join(f"  ({i}) {s}" for i, s in enumerate(events))
        + f"\n\nReturn JSON {{\"matches\": {{\"0\": [...], ..., \"{len(factors)-1}\": [...]}}}} — for each sequence entry, the chain-node indices with the same meaning. Use [] for no-match."
    )
    body = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": 1024,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "judge", "schema": JUDGE_SCHEMA,
                                            "strict": True}},
    }
    async with sem:
        resp = await client.post(f"{endpoint}/chat/completions", json=body, timeout=300)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    raw = json.loads(content).get("matches", {})
    return {str(i): list(raw.get(str(i), [])) for i in range(len(factors))}


def pair_counts(matches: dict[str, list], n_seq: int, n_chain: int) -> tuple[int, int, int]:
    """(concordant, discordant, n_matched_entries) for one record."""
    pos: list[int | None] = []
    for i in range(n_seq):
        idxs = [int(e) for e in matches.get(str(i), [])
                if isinstance(e, (int, str)) and str(e).lstrip("-").isdigit()
                and 0 <= int(e) < n_chain]
        pos.append(min(idxs) if idxs else None)
    conc = disc = 0
    for i in range(n_seq):
        for j in range(i + 1, n_seq):
            pi, pj = pos[i], pos[j]
            if pi is None or pj is None or pi == pj:
                continue
            if pi < pj:
                conc += 1
            else:
                disc += 1
    return conc, disc, sum(1 for p in pos if p is not None)


async def run(args) -> int:
    gold = {}
    with GOLD.open() as f:
        for line in f:
            j = json.loads(line)
            gold[j["record_id"]] = j["seq"]
    text = {}
    with INPUT.open() as f:
        for line in f:
            j = json.loads(line)
            text[j["record_id"]] = j["text"]

    work = []
    with Path(args.extraction).open() as f:
        for line in f:
            r = json.loads(line)
            rid = r.get("record_id")
            if not r.get("ok") or rid not in gold:
                continue
            seq = gold_strings(gold[rid])
            events = event_payload(r)
            if len(seq) >= 2 and len(events) >= 2:
                work.append((rid, seq, events))
    print(f"{len(work):,} records eligible for TSB ordering eval")

    out_path = Path(args.out)
    cache = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    cache.add(json.loads(line)["record_id"])
                except Exception:
                    pass
    todo = [w for w in work if w[0] not in cache]
    print(f"{len(todo):,} pending after cache")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    n_done = 0
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        with out_path.open("a") as outf:
            async def worker(w):
                nonlocal n_done
                rid, seq, events = w
                try:
                    matches = await judge_one(client, args.endpoint, args.model,
                                              seq, events, text.get(rid, ""), sem)
                except Exception as e:
                    print(f"  [judge-err] {rid}: {type(e).__name__}: {e}")
                    return
                async with lock:
                    outf.write(json.dumps({"record_id": rid, "matches": matches,
                                           "n_seq": len(seq),
                                           "n_chain": len(events)}) + "\n")
                    outf.flush()
                    n_done += 1
                    if n_done % 50 == 0:
                        print(f"  judged {n_done}/{len(todo)} "
                              f"({n_done/(time.time()-t0+1e-9):.1f}/s)")
            await asyncio.gather(*[worker(w) for w in todo])

    conc = disc = 0
    taus = []
    n_scored = matched_entries = total_entries = 0
    with out_path.open() as f:
        for line in f:
            j = json.loads(line)
            c, d, m = pair_counts(j["matches"], j["n_seq"], j["n_chain"])
            matched_entries += m
            total_entries += j["n_seq"]
            if c + d == 0:
                continue
            n_scored += 1
            conc += c
            disc += d
            taus.append((c - d) / (c + d))
    pooled = (conc - disc) / (conc + disc) if conc + disc else None
    summary = {
        "tag": args.tag,
        "extraction_file": str(args.extraction),
        "n_records_scored": n_scored,
        "n_pairs": conc + disc,
        "gold_entry_match_rate": round(matched_entries / total_entries, 4)
        if total_entries else None,
        "pooled_kendall_tau": round(pooled, 4) if pooled is not None else None,
        "mean_per_record_tau": round(sum(taus) / len(taus), 4) if taus else None,
        "frac_records_tau_ge_0.75": round(
            sum(1 for t in taus if t >= 0.75) / len(taus), 4) if taus else None,
    }
    sp = Path(str(args.out).replace(".jsonl", ".summary.json"))
    sp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="tsb-order")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
