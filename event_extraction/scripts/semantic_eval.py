"""LLM-as-judge semantic recall matcher.

For each record in a calibration set:
  1. Pull NTSB cause-flagged factors from the joined structured payload.
  2. Pull extracted events from a v1 or v3 extraction file.
  3. Ask the local Qwen (via vLLM) which extracted events match each factor.
     Output is forced to JSON via `guided_json`: dict mapping factor index ->
     list of event indices that semantically match.
  4. Aggregate recall (factors with >=1 match) and precision (events that
     match >=1 factor).

Use:
    python event_extraction/scripts/semantic_eval.py \\
        --extraction event_extraction/out/calibration_2k_v3.jsonl \\
        --tag v3 \\
        --out event_extraction/out/semantic_eval_v3.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            },
        }
    },
}

JUDGE_SYSTEM = """You are an aviation-safety domain matcher. You will be given:
  * a NTSB-narrative summary
  * a list of NTSB-investigator cause-flagged factors (numbered)
  * a list of LLM-extracted causal events (numbered, with event_type and trigger)

For each NTSB factor, list the event indices whose meaning is **semantically equivalent or directly implied**. A match is real if a domain expert reading the extracted event would recognize it as describing the same factor (allow paraphrase, narrative-vs-structured-vocab differences, and HFACS-vs-natural-language differences).

Return a JSON object with shape `{"matches": {"<factor_idx>": [event_idxs...]}}`. Empty lists are valid for factors with no match. Do not include factors with no match? — DO include them with []. Only the listed factor indices may appear as keys.

Be strict: a generic event ("ground impact") does NOT match a specific NTSB factor ("PILOT'S FAILURE TO MAINTAIN AIRSPEED") just because both are about the accident. Match only when the EVENT meaning equals or implies the FACTOR meaning."""

def cause_strings(structured: dict) -> list[str]:
    out = []
    for f in (structured.get("findings") or []):
        if (f.get("cause_factor") or "").upper() in ("C", "F") and f.get("description"):
            out.append(f["description"])
    for v in (structured.get("seq_of_events") or []):
        if (v.get("cause_factor") or "").upper() == "C" and v.get("subj_text"):
            txt = v["subj_text"]
            if v.get("modifier_text") and v["modifier_text"] != "None":
                txt = f"{txt} ({v['modifier_text']})"
            out.append(txt)
    return out

def event_payload(rec: dict) -> list[str]:
    out = []
    for n in (rec.get("nodes") or []):
        if not isinstance(n, dict) or n.get("kind") != "event":
            continue
        et = n.get("event_type") or n.get("subtype") or "event"
        cr = n.get("cause_role")
        tag = f"[{et}{f' role={cr}' if cr else ''}]"
        trig = n.get("trigger") or ""
        out.append(f"{tag} {trig}".strip())
    return out

async def judge_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    record_id: str,
    factors: list[str],
    events: list[str],
    narrative_excerpt: str,
    sem: asyncio.Semaphore,
) -> dict[str, list[int]]:
    if not factors or not events:
        return {str(i): [] for i in range(len(factors))}
    user = (
        f"NARRATIVE EXCERPT (truncated):\n{narrative_excerpt[:1200]}\n\n"
        f"NTSB CAUSE-FLAGGED FACTORS:\n"
        + "\n".join(f"  [{i}] {s}" for i, s in enumerate(factors))
        + f"\n\nEXTRACTED EVENTS:\n"
        + "\n".join(f"  ({i}) {s}" for i, s in enumerate(events))
        + f"\n\nReturn JSON {{\"matches\": {{\"0\": [...], \"1\": [...], ...}}}} with one key per factor (0..{len(factors)-1}). Use [] for no-match."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": JUDGE_SCHEMA,
    }
    async with sem:
        resp = await client.post(f"{endpoint}/chat/completions", json=body, timeout=300)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    obj = json.loads(content)
    raw = obj.get("matches", {})
    return {str(i): list(raw.get(str(i), [])) for i in range(len(factors))}

async def run(args: argparse.Namespace) -> int:
    print(f"Loading enriched corpus index…")
    structured_by_id: dict[str, dict] = {}
    text_by_id: dict[str, str] = {}
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("structured"):
                structured_by_id[r["record_id"]] = r["structured"]
                text_by_id[r["record_id"]] = r.get("text", "")
    print(f"  {len(structured_by_id):,} records with supervision")

    print(f"Loading {args.extraction}…")
    extraction_by_id: dict[str, dict] = {}
    with Path(args.extraction).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok"):
                extraction_by_id[r["record_id"]] = r
    print(f"  {len(extraction_by_id):,} ok extractions")

    # Build the work list
    work: list[tuple[str, list[str], list[str]]] = []
    for rid, ex in extraction_by_id.items():
        if rid not in structured_by_id:
            continue
        factors = cause_strings(structured_by_id[rid])
        events = event_payload(ex)
        if not factors:
            continue
        work.append((rid, factors, events))
    if args.limit:
        work = work[: args.limit]
    print(f"  {len(work):,} records to evaluate (factors x events)")

    sem = asyncio.Semaphore(args.concurrency)
    cache: dict[str, dict[str, list[int]]] = {}
    out_path = Path(args.out)

    # Resume support: skip records already in the output file
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                    cache[j["record_id"]] = j["matches"]
                except Exception:
                    pass
    todo = [w for w in work if w[0] not in cache]
    print(f"  {len(todo):,} pending after cache (cached: {len(cache):,})")

    n_done = 0
    t0 = time.time()
    write_lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with out_path.open("a") as outf:
            async def worker(w):
                nonlocal n_done
                rid, factors, events = w
                try:
                    matches = await judge_one(
                        client, args.endpoint, args.model, rid, factors, events,
                        text_by_id.get(rid, ""), sem,
                    )
                except Exception as e:
                    matches = {str(i): [] for i in range(len(factors))}
                    print(f"  [judge-err] {rid}: {type(e).__name__}: {e}")
                async with write_lock:
                    outf.write(json.dumps({
                        "record_id": rid, "matches": matches,
                        "n_factors": len(factors), "n_events": len(events),
                    }) + "\n")
                    outf.flush()
                    n_done += 1
                    if n_done % 50 == 0:
                        rate = n_done / (time.time() - t0 + 1e-9)
                        print(f"  judged {n_done}/{len(todo)} ({rate:.1f}/s)")
            await asyncio.gather(*[worker(w) for w in todo])

    # Aggregate from the full output file (cache + new)
    agg = {}
    with out_path.open() as f:
        for line in f:
            j = json.loads(line)
            agg[j["record_id"]] = j

    n_factors = 0
    n_factors_matched = 0
    n_events = 0
    n_events_matched = 0
    per_record_recalls: list[float] = []
    for rid, ex in extraction_by_id.items():
        if rid not in agg:
            continue
        m = agg[rid]
        matches = m["matches"]
        nf = m["n_factors"]
        ne = m["n_events"]
        n_factors += nf
        n_events += ne
        matched_events_set: set[int] = set()
        rec_match = 0
        for fi in range(nf):
            evs = matches.get(str(fi), [])
            if evs:
                rec_match += 1
                for e in evs:
                    try:
                        ei = int(e)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= ei < ne:
                        matched_events_set.add(ei)
        n_factors_matched += rec_match
        n_events_matched += len(matched_events_set)
        if nf:
            per_record_recalls.append(rec_match / nf)

    def pct(a, b): return f"{100*a/b:.2f}%" if b else "n/a"
    summary = {
        "tag": args.tag,
        "extraction_file": str(args.extraction),
        "n_records_evaluated": len(per_record_recalls),
        "n_factors_total": n_factors,
        "n_factors_matched": n_factors_matched,
        "recall": n_factors_matched / n_factors if n_factors else 0,
        "n_events_total": n_events,
        "n_events_matched_to_a_factor": n_events_matched,
        "precision": n_events_matched / n_events if n_events else 0,
        "mean_per_record_recall": sum(per_record_recalls) / len(per_record_recalls) if per_record_recalls else 0,
    }
    print(f"\n=== Semantic eval ({args.tag}) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", required=True, help="JSONL of extractions to evaluate")
    ap.add_argument("--tag", required=True, help="label for this eval run (e.g. v1, v3)")
    ap.add_argument("--out", required=True, help="JSONL judge-output (resumes if exists)")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--concurrency", default=12, type=int)
    ap.add_argument("--limit", default=0, type=int)
    args = ap.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
