"""Select 10 LOC-I (Loss of Control - In Flight) cases for the manual
leave-one-out feasibility check (README §7.1).

Criteria:
  * NTSB Pre2008 record with Occurrence_Code == 250 (LOC-I)
  * Narrative text >= 1000 chars (rich enough for hand-graphing)
  * >= 3 cause-flagged seq_of_events entries (so leave-one-out has multiple
    candidate factors to remove)
  * Already present in the v1 extraction (so we can compare hand-built vs.
    LLM-built graphs on the same case)
  * Mix of phases of flight where possible

Output:
  event_extraction/out/loci_feasibility_cases.jsonl - 10 records, each with:
    record_id, narrative, aircraft, injury, structured_seq_of_events,
    structured_occurrences, primary_cause_text, llm_extraction (v1 graph)
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"
EXTRACTED = ROOT / "event_extraction/out/full_corpus_events.jsonl"
OUT = ROOT / "event_extraction/out/loci_feasibility_cases.jsonl"

LOCI_CODE = 250  # NTSB Occurrence_Code for "LOSS OF CONTROL - IN FLIGHT"
TARGET_N = 10
MIN_TEXT_CHARS = 1000
MIN_CAUSE_FACTORS = 3
RNG_SEED = 42

def main():
    print("Indexing v1 extraction…")
    extraction_by_id: dict[str, dict] = {}
    with EXTRACTED.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok"):
                extraction_by_id[r["record_id"]] = r
    print(f"  {len(extraction_by_id):,} extracted records")

    print("Scanning enriched corpus for LOC-I candidates…")
    candidates = []
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "NTSB_ASRS:Pre2008":
                continue
            s = r.get("structured")
            if not s:
                continue
            occs = s.get("occurrences") or []
            if not any(o.get("occurrence_code") == LOCI_CODE for o in occs):
                continue
            seq = s.get("seq_of_events") or []
            cf = [v for v in seq if (v.get("cause_factor") or "").upper() == "C"]
            if len(cf) < MIN_CAUSE_FACTORS:
                continue
            text = r.get("text") or ""
            if len(text) < MIN_TEXT_CHARS:
                continue
            if r["record_id"] not in extraction_by_id:
                continue
            candidates.append(r)

    print(f"  {len(candidates):,} LOC-I candidates meeting all criteria")

    rng = random.Random(RNG_SEED)
    rng.shuffle(candidates)

    # Stratify by phase to get a varied sample
    by_phase: dict[str, list[dict]] = {}
    for r in candidates:
        ph = (r["structured"].get("phase_of_flight") or "UNKNOWN").upper()
        by_phase.setdefault(ph, []).append(r)

    print(f"  phases: {[(k, len(v)) for k, v in sorted(by_phase.items(), key=lambda kv: -len(kv[1]))][:8]}")

    selected: list[dict] = []
    seen_phases: set[str] = set()
    # First pass: one per phase to maximize variety
    for ph, lst in sorted(by_phase.items(), key=lambda kv: -len(kv[1])):
        if len(selected) >= TARGET_N:
            break
        selected.append(lst[0])
        seen_phases.add(ph)
    # Second pass: fill remaining slots from the largest phase buckets
    if len(selected) < TARGET_N:
        for ph, lst in sorted(by_phase.items(), key=lambda kv: -len(kv[1])):
            for r in lst[1:]:
                if len(selected) >= TARGET_N:
                    break
                selected.append(r)
            if len(selected) >= TARGET_N:
                break
    selected = selected[:TARGET_N]

    print(f"\nSelected {len(selected)} cases:")
    with OUT.open("w") as outf:
        for i, r in enumerate(selected, 1):
            ext = extraction_by_id[r["record_id"]]
            payload = {
                "case_no": i,
                "record_id": r["record_id"],
                "ntsb_no": (r["structured"].get("aircraft") or {}).get("ntsb_no"),
                "phase_of_flight": r["structured"].get("phase_of_flight"),
                "primary_cause_text": r["structured"].get("primary_cause_text"),
                "narrative": r["text"],
                "aircraft": r["structured"].get("aircraft"),
                "injury": r["structured"].get("injury"),
                "occurrences": r["structured"].get("occurrences"),
                "seq_of_events": r["structured"].get("seq_of_events"),
                "v1_extraction": {
                    "nodes": ext.get("nodes"),
                    "edges": ext.get("edges"),
                },
            }
            outf.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"  [{i:2d}] {r['record_id']}  phase={r['structured'].get('phase_of_flight')}  "
                  f"cause={(r['structured'].get('primary_cause_text') or '')[:60]!r}  "
                  f"seq_events={len(r['structured'].get('seq_of_events') or [])}  "
                  f"v1_nodes={len(ext.get('nodes') or [])}  v1_edges={len(ext.get('edges') or [])}")

    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
