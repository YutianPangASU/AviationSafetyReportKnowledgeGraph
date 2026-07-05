"""Chain-ordering eval — Kendall's tau against NTSB `seq_of_events`.

Measures whether the v4 extracted chain places factors in the same order as
the NTSB investigator's coded sequence (the Phase-2 gate is tau >= 0.75 on
the narrative-only track; the grounded track is trivially high since the
prompt contains the order and serves only as a ceiling check).

Method, per record with >= 2 ordered seq entries:
  1. LLM-judge matches each seq entry (in `order`) to chain-node indices —
     same judge machinery as semantic_eval.py.
  2. For every pair of seq entries (i < j) where both matched, compare the
     chain positions of their (earliest) matched nodes: concordant if
     chain_i < chain_j, discordant if chain_i > chain_j, tie skipped.
  3. tau = (concordant - discordant) / (concordant + discordant), pooled
     across records; per-record tau distribution also reported.

IMPORTANT (2026-07-05 calibration finding): WITHIN one occurrence, the
seq_of_events `order` field is an NTSB coding convention (proximate subject
first, contributing details after), NOT chronology — scoring against it
yields tau ~0.31 even when the investigator ordering is shown to the model.
BETWEEN occurrences the ordering IS temporal (occurrence 1 precedes
occurrence 2). The headline metric is therefore `cross_occurrence_tau`,
computed on pairs from different occurrences only; the raw all-pairs tau is
kept as a secondary diagnostic.

Only NTSB Pre2008 records carry `seq_of_events`, so coverage is that subset
of the calibration sample.

Use:
    python event_extraction/scripts/eval_chain_order.py \\
        --extraction event_extraction/out/calibration_2k_v4_narrative.jsonl \\
        --tag v4-narrative \\
        --out event_extraction/out/chain_order_v4_narrative.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_eval import ENRICHED, JUDGE_SCHEMA, JUDGE_SYSTEM, event_payload  # noqa: E402


def ordered_seq_entries(structured: dict) -> list[tuple[str, int]]:
    """All seq_of_events entries with usable subject text, in investigator
    order, as (text, occurrence_no). Unlike semantic_eval.cause_strings,
    non-cause entries are kept — ordering is the target here, not
    attribution."""
    rows = []
    for v in (structured.get("seq_of_events") or []):
        subj = (v.get("subj_text") or "").strip()
        if not subj or subj == "None":
            continue
        mod = (v.get("modifier_text") or "").strip()
        txt = f"{subj} ({mod})" if mod and mod != "None" else subj
        rows.append((v.get("order", 0), txt, v.get("occurrence_no", 0)))
    rows.sort(key=lambda r: r[0])
    return [(t, o) for _, t, o in rows]


async def judge_one(client, endpoint, model, factors, events, narrative, sem):
    user = (
        f"NARRATIVE EXCERPT (truncated):\n{narrative[:1200]}\n\n"
        f"NTSB SEQUENCE-OF-EVENTS ENTRIES (in investigator order):\n"
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


def tau_from_matches(matches: dict[str, list[int]], n_seq: int, n_chain: int,
                     occ_nos: list[int] | None = None,
                     cross_occurrence_only: bool = False) -> tuple[int, int]:
    """Return (concordant, discordant) pair counts for one record. With
    cross_occurrence_only, pairs within the same occurrence are skipped and
    concordance is judged on occurrence order (the temporal signal)."""
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
            if cross_occurrence_only:
                if not occ_nos or occ_nos[i] == occ_nos[j]:
                    continue
                if (occ_nos[i] < occ_nos[j]) == (pi < pj):
                    conc += 1
                else:
                    disc += 1
            elif pi < pj:
                conc += 1
            else:
                disc += 1
    return conc, disc


async def run(args) -> int:
    structured_by_id: dict[str, dict] = {}
    text_by_id: dict[str, str] = {}
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("structured") and (r["structured"].get("seq_of_events")):
                structured_by_id[r["record_id"]] = r["structured"]
                text_by_id[r["record_id"]] = r.get("text", "")
    print(f"{len(structured_by_id):,} records with seq_of_events")

    work = []
    with Path(args.extraction).open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok") or r["record_id"] not in structured_by_id:
                continue
            entries = ordered_seq_entries(structured_by_id[r["record_id"]])
            seq = [t for t, _ in entries]
            occ_nos = [o for _, o in entries]
            events = event_payload(r)
            if len(seq) >= 2 and len(events) >= 2:
                work.append((r["record_id"], seq, occ_nos, events))
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work):,} records eligible for ordering eval")

    out_path = Path(args.out)
    cache: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                    cache[j["record_id"]] = j
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
                rid, seq, occ_nos, events = w
                try:
                    matches = await judge_one(
                        client, args.endpoint, args.model, seq, events,
                        text_by_id.get(rid, ""), sem)
                except Exception as e:
                    print(f"  [judge-err] {rid}: {type(e).__name__}: {e}")
                    return
                async with lock:
                    outf.write(json.dumps({
                        "record_id": rid, "matches": matches,
                        "n_seq": len(seq), "n_chain": len(events),
                        "occ_nos": occ_nos}) + "\n")
                    outf.flush()
                    n_done += 1
                    if n_done % 50 == 0:
                        print(f"  judged {n_done}/{len(todo)} "
                              f"({n_done/(time.time()-t0+1e-9):.1f}/s)")
            await asyncio.gather(*[worker(w) for w in todo])

    occ_by_id = {rid: occ for rid, _, occ, _ in work}
    stats = {"all_pairs": [0, 0, []], "cross_occurrence": [0, 0, []]}
    with out_path.open() as f:
        for line in f:
            j = json.loads(line)
            occ_nos = j.get("occ_nos") or occ_by_id.get(j["record_id"])
            for key, cross in (("all_pairs", False), ("cross_occurrence", True)):
                if cross and (not occ_nos or len(occ_nos) != j["n_seq"]):
                    continue
                c, d = tau_from_matches(j["matches"], j["n_seq"], j["n_chain"],
                                        occ_nos=occ_nos,
                                        cross_occurrence_only=cross)
                if c + d == 0:
                    continue
                stats[key][0] += c
                stats[key][1] += d
                stats[key][2].append((c - d) / (c + d))

    summary = {"tag": args.tag, "extraction_file": str(args.extraction)}
    for key, (c, d, taus) in stats.items():
        summary[key] = {
            "n_records_scored": len(taus),
            "n_pairs": c + d,
            "pooled_kendall_tau": (c - d) / (c + d) if c + d else 0.0,
            "mean_per_record_tau": sum(taus) / len(taus) if taus else 0.0,
            "frac_records_tau_ge_0.75": (sum(1 for t in taus if t >= 0.75) / len(taus)
                                         if taus else 0.0),
        }
    # headline metric: cross-occurrence (temporal); all-pairs is diagnostic only
    summary["headline_tau"] = summary["cross_occurrence"]["pooled_kendall_tau"]
    print(f"\n=== Chain-order eval ({args.tag}) ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--concurrency", default=12, type=int)
    ap.add_argument("--limit", default=0, type=int)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
