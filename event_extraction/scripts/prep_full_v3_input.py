"""Prepare the v3 full re-extraction input list.

Filter corpus_enriched.jsonl to:
  * records that have `structured != null` (i.e. NTSB Pre2008 / avall / REPORT
    or FAA AIDS — sources with structured supervision)
  * narrative length in [200, 2000] words (matches v1 filter)
  * v3-extracted records that ALREADY succeeded in calibration_2k_v3.jsonl are
    pre-emptied to the output via copy, so resume picks up where calibration
    left off.

Outputs:
  * event_extraction/out/full_corpus_v3_input.jsonl  - record_id/source/text
  * event_extraction/out/full_corpus_v3.jsonl        - seeded with the 1,984
    successful v3 calibration extractions (resume skips them)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"
CAL_V3 = ROOT / "event_extraction/out/calibration_2k_v3.jsonl"
INP = ROOT / "event_extraction/out/full_corpus_v3_input.jsonl"
OUT = ROOT / "event_extraction/out/full_corpus_v3.jsonl"

MIN_WORDS = 200
MAX_WORDS = 2000

def main():
    by_source = Counter()
    skipped_by_reason = Counter()
    with ENRICHED.open() as f, INP.open("w") as g:
        for line in f:
            r = json.loads(line)
            if not r.get("structured"):
                skipped_by_reason["no_structured_supervision"] += 1
                continue
            text = r.get("text") or ""
            wc = len(text.split())
            if wc < MIN_WORDS:
                skipped_by_reason["too_short"] += 1
                continue
            if wc > MAX_WORDS:
                skipped_by_reason["too_long"] += 1
                continue
            g.write(json.dumps({
                "record_id": r["record_id"],
                "source": r["source"],
                "text": text,
            }) + "\n")
            by_source[r["source"]] += 1
    total = sum(by_source.values())
    print(f"Selected {total:,} records for v3 re-extraction")
    print(f"\nBy source:")
    for s, c in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {s:30s} {c:>10,}")
    print(f"\nSkipped:")
    for k, c in skipped_by_reason.most_common():
        print(f"  {k:30s} {c:>10,}")

    # Seed output with successful calibration extractions (resume will skip them)
    seeded = 0
    with CAL_V3.open() as f, OUT.open("w") as g:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok"):
                g.write(line)
                seeded += 1
    print(f"\nSeeded {OUT.name} with {seeded:,} successful calibration extractions")
    print(f"  -> resume mode will skip these and process {total - seeded:,} new records")

if __name__ == "__main__":
    main()
