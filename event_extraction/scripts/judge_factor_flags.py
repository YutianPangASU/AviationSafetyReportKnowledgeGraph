"""Judge the extracted chains against the NTSB factor (F) flags (review 2026-10-10).

The main evaluation (semantic_eval.py) scores the chains against the cause
flags: avall findings flagged C or F and pre-2008 sequence entries flagged C.
Pre-2008 entries flagged F, which NTSB uses for contributing factors, are not
in that reference, so a node the extractor marks as contributing has little
to match. This script judges those F entries with the same prompt and model
call as semantic_eval.py, and re-judges the original reference on the same
server as a reproduction check, since the server here runs in FP8 rather than
the bfloat16 of the original run.

Precision by node role is then reported against the cause flags (original
reference), the factor flags, and their union.

Use:
    python event_extraction/scripts/judge_factor_flags.py \\
        --endpoints http://localhost:8000/v1 http://localhost:8001/v1

Output: event_extraction/out/factor_flags/judge_F.jsonl, judge_C_repro.jsonl,
        event_extraction/out/factor_flags/factor_flag_precision.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_eval import cause_strings, event_payload, judge_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"
EXTRACTION = ROOT / "event_extraction/out/calibration_2k_v4_narrative.jsonl"
JUDGE = ROOT / "event_extraction/out/semantic_eval_v4_narrative.jsonl"
OUT_DIR = ROOT / "event_extraction/out/factor_flags"


def factor_strings(structured: dict) -> list[str]:
    """Pre-2008 sequence entries flagged F, formatted as in cause_strings."""
    out = []
    for v in (structured.get("seq_of_events") or []):
        if (v.get("cause_factor") or "").upper() == "F" and v.get("subj_text"):
            txt = v["subj_text"]
            if v.get("modifier_text") and v["modifier_text"] != "None":
                txt = f"{txt} ({v['modifier_text']})"
            out.append(txt)
    return out


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                out[r["record_id"]] = r
    return out


async def judge_all(work, out_path: Path, endpoints, model, concurrency) -> None:
    done = load_jsonl(out_path)
    todo = [w for w in work if w[0] not in done]
    print(f"{out_path.name}: {len(todo)} pending, {len(done)} cached")
    sems = [asyncio.Semaphore(concurrency) for _ in endpoints]
    lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with out_path.open("a") as outf:
            async def worker(k, w):
                rid, factors, events, text = w
                j = k % len(endpoints)
                try:
                    m = await judge_one(client, endpoints[j], model, rid, factors, events,
                                        text, sems[j])
                except Exception as e:  # recorded as no match, as in semantic_eval.py
                    print(f"  [judge-err] {rid}: {type(e).__name__}: {e}")
                    m = {str(i): [] for i in range(len(factors))}
                async with lock:
                    outf.write(json.dumps({"record_id": rid, "matches": m,
                                           "n_factors": len(factors),
                                           "n_events": len(events)}) + "\n")
                    outf.flush()
            await asyncio.gather(*[worker(k, w) for k, w in enumerate(todo)])


def precision_by_role(chains, ids, *judges) -> dict:
    """Nodes matched by any of the given judge outputs, by cause role, over the
    records in ids (a record absent from a judge output has no match there)."""
    tot, hit = defaultdict(int), defaultdict(int)
    for rid in ids:
        rec = chains[rid]
        matched = set()
        for jd in judges:
            for lst in (jd.get(rid) or {}).get("matches", {}).values():
                matched.update(int(e) for e in lst)
        for k, n in enumerate(rec["chain"]):
            role = n.get("cause_role") or "none"
            tot[role] += 1
            hit[role] += k in matched
    out = {r: {"nodes": tot[r], "matched": hit[r], "precision": round(hit[r] / tot[r], 4)}
           for r in sorted(tot)}
    n = tot["primary"] + tot["contributing"]
    h = hit["primary"] + hit["contributing"]
    out["primary_or_contributing"] = {"nodes": n, "matched": h, "precision": round(h / n, 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", default=["http://localhost:8000/v1"])
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--concurrency", default=12, type=int)
    ap.add_argument("--quantization", default="fp8", help="recorded in the output only")
    args = ap.parse_args()

    original = load_jsonl(JUDGE)
    chains = {r["record_id"]: r for r in load_jsonl(EXTRACTION).values() if r.get("ok")}
    structured, text = {}, {}
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in original and r.get("structured"):
                structured[r["record_id"]] = r["structured"]
                text[r["record_id"]] = r.get("text", "")

    work_f, work_c = [], []
    for rid in sorted(original):
        if rid not in chains or rid not in structured:
            continue
        events = event_payload(chains[rid])
        fs = factor_strings(structured[rid])
        if fs:
            work_f.append((rid, fs, events, text[rid]))
        work_c.append((rid, cause_strings(structured[rid]), events, text[rid]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f_path, c_path = OUT_DIR / "judge_F.jsonl", OUT_DIR / "judge_C_repro.jsonl"
    asyncio.run(judge_all(work_f, f_path, args.endpoints, args.model, args.concurrency))
    asyncio.run(judge_all(work_c, c_path, args.endpoints, args.model, args.concurrency))
    judge_f, judge_c = load_jsonl(f_path), load_jsonl(c_path)
    ids = [w[0] for w in work_c]

    # reproduction: per-finding agreement of the re-run with the original judge
    agree = n = recalled_orig = recalled_new = 0
    for rid, r in judge_c.items():
        o = original.get(rid)
        if not o or o["n_factors"] != r["n_factors"]:
            continue
        for i in range(r["n_factors"]):
            a, b = bool(o["matches"].get(str(i))), bool(r["matches"].get(str(i)))
            n += 1
            agree += a == b
            recalled_orig += a
            recalled_new += b

    n_f = sum(r["n_factors"] for r in judge_f.values())
    rec_f = sum(sum(bool(v) for v in r["matches"].values()) for r in judge_f.values())
    out = {
        "server": {"model": args.model, "quantization": args.quantization,
                   "endpoints": len(args.endpoints)},
        "reproduction_of_cause_flag_judge": {
            "findings": n, "agreement": round(agree / n, 4),
            "recall_original": round(recalled_orig / n, 4),
            "recall_rerun": round(recalled_new / n, 4)},
        "factor_flag_entries": {"records": len(judge_f), "entries": n_f,
                                "recalled": rec_f, "recall": round(rec_f / n_f, 4)},
        "precision_by_role": {
            "cause_flags_original": precision_by_role(chains, ids, original),
            "cause_flags_rerun": precision_by_role(chains, ids, judge_c),
            "factor_flags": precision_by_role(chains, ids, judge_f),
            "cause_or_factor_flags": precision_by_role(chains, ids, judge_c, judge_f),
        },
    }
    (OUT_DIR / "factor_flag_precision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
