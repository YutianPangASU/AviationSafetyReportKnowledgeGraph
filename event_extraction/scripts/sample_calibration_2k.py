"""Sample 2,000 NTSB records for the v3 calibration run.

Selection rules:
  * record exists in the v1 extraction (so we can compare extractions on the same input)
  * record has structured supervision (Findings or seq_of_events)
  * has at least one cause-flagged supervision item
  * narrative length 200..2000 words
  * stratified ~80/20 between Pre2008 and avall (to test both supervision regimes)

Output: event_extraction/out/calibration_2k_input.jsonl with fields the runner expects.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"
V1 = ROOT / "event_extraction/out/full_corpus_events.jsonl"
LOCI_INPUT = ROOT / "event_extraction/out/loci_input_records.jsonl"
OUT = ROOT / "event_extraction/out/calibration_2k_input.jsonl"

TARGET = 2000
SEED = 42
SPLIT_PRE2008 = 0.80   # majority Pre2008 since that's what v1 covered most of

def has_cause(structured) -> bool:
    if not structured:
        return False
    if any((f.get("cause_factor") or "").upper() in ("C", "F")
           for f in (structured.get("findings") or [])):
        return True
    if any((v.get("cause_factor") or "").upper() == "C"
           for v in (structured.get("seq_of_events") or [])):
        return True
    return False

def main():
    print("Indexing v1 extraction record_ids…")
    v1_ids: set[str] = set()
    with V1.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok"):
                v1_ids.add(r["record_id"])
    print(f"  v1 ok records: {len(v1_ids):,}")

    # Already-evaluated LOC-I cases — exclude so we don't duplicate
    loci_ids: set[str] = set()
    if LOCI_INPUT.exists():
        with LOCI_INPUT.open() as f:
            for line in f:
                loci_ids.add(json.loads(line)["record_id"])
    print(f"  loci cases to exclude: {len(loci_ids)}")

    print("Scanning enriched corpus…")
    pre2008_pool: list[dict] = []
    avall_pool: list[dict] = []
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in loci_ids:
                continue
            if r["record_id"] not in v1_ids:
                continue
            if not has_cause(r.get("structured")):
                continue
            wc = len((r.get("text") or "").split())
            if wc < 200 or wc > 2000:
                continue
            if r["source"] == "NTSB_ASRS:Pre2008":
                pre2008_pool.append(r)
            elif r["source"] == "NTSB_ASRS:avall":
                avall_pool.append(r)
    print(f"  pre2008 pool: {len(pre2008_pool):,}; avall pool: {len(avall_pool):,}")

    rng = random.Random(SEED)
    rng.shuffle(pre2008_pool)
    rng.shuffle(avall_pool)
    n_pre2008 = int(TARGET * SPLIT_PRE2008)
    n_avall = TARGET - n_pre2008
    n_avall = min(n_avall, len(avall_pool))
    n_pre2008 = TARGET - n_avall
    selected = pre2008_pool[:n_pre2008] + avall_pool[:n_avall]
    rng.shuffle(selected)
    print(f"  selected: {n_pre2008} Pre2008 + {n_avall} avall = {len(selected)} total")

    with OUT.open("w") as g:
        for r in selected:
            g.write(json.dumps({
                "record_id": r["record_id"],
                "source": r["source"],
                "text": r["text"],
            }) + "\n")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
