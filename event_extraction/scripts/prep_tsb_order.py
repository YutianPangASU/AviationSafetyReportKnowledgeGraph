"""Build the TSB Canada ordering-validation set.

Why this exists: the v4 ordering evaluation against NTSB `seq_of_events`
scores tau = 0.31 on all pairs and 0.82 on cross-occurrence pairs, and the
paper argues the gap is an NTSB coding convention (within one occurrence the
proximate subject is listed first, not the chronologically first event). The
TSB of Canada codes an event-and-phase sequence for every occurrence, written
by a different agency under a different convention, in which the
EventPhaseSequence number is the temporal order. Scoring the extracted chains
against that sequence tests the chronology claim on an independent gold
standard: if the model orders chronologically, the TSB all-pairs tau should
land near the NTSB cross-occurrence value, not near 0.31.

Selection: single-aircraft occurrences only (multi-aircraft occurrences carry
one sequence per aircraft, which makes the gold ambiguous), at least two coded
events, and an English summary of at least MIN_WORDS words so the narrative
has enough content to extract a chain from.

Outputs:
  event_extraction/out/tsb_order_input.jsonl   {record_id, source, text}
                                               for extract_vllm --input-records
  event_extraction/out/tsb_order_gold.jsonl    {record_id, seq: [{order, event,
                                               phase, desc}]}

Run:  python event_extraction/scripts/prep_tsb_order.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EVENTS = ROOT / "data/TSB_CANADA/ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv"
OCCUR = ROOT / "data/TSB_CANADA/ASISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv"
OUT_INPUT = ROOT / "event_extraction/out/tsb_order_input.jsonl"
OUT_GOLD = ROOT / "event_extraction/out/tsb_order_gold.jsonl"

MIN_WORDS = 150
MAX_WORDS = 2000
SAMPLE = 1500
SEED = 0


def read_events() -> dict[str, dict[str, list[dict]]]:
    """OccNo -> AcID -> [{order, event, phase, desc}] sorted by sequence."""
    per: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with EVENTS.open(encoding="utf-8-sig", errors="replace") as f:
        r = csv.reader(f)
        hdr = next(r)
        ix = {c: i for i, c in enumerate(hdr)}
        for row in r:
            try:
                order = int(row[ix["EventPhaseSequence"]] or 0)
            except ValueError:
                continue
            per[row[ix["OccNo"]]][row[ix["AcID"]]].append({
                "order": order,
                "event": row[ix["EventID_DisplayEng"]].strip(),
                "phase": row[ix["PhaseID_DisplayEng"]].strip(),
                "desc": row[ix["FullEventDescEng"]].strip(),
            })
    for acs in per.values():
        for seq in acs.values():
            seq.sort(key=lambda e: e["order"])
    return per


def main() -> None:
    per = read_events()
    pool: list[tuple[str, str, list[dict]]] = []
    with OCCUR.open(encoding="utf-8-sig", errors="replace") as f:
        r = csv.reader(f)
        hdr = next(r)
        ix = {c: i for i, c in enumerate(hdr)}
        first = hdr[0]
        for row in r:
            occ = row[ix["OccNo"]]
            summary = (row[ix["Summary"]] or "").strip()
            wc = len(summary.split())
            if not (MIN_WORDS <= wc <= MAX_WORDS):
                continue
            acs = per.get(occ)
            if not acs or len(acs) != 1:      # single-aircraft only
                continue
            seq = next(iter(acs.values()))
            if len(seq) < 2:
                continue
            pool.append((occ, summary, seq))

    print(f"pool: {len(pool)} single-aircraft occurrences with "
          f">={MIN_WORDS}-word summaries and >=2 coded events")
    rng = random.Random(SEED)
    sample = rng.sample(pool, min(SAMPLE, len(pool)))
    sample.sort(key=lambda t: t[0])

    with OUT_INPUT.open("w") as fi, OUT_GOLD.open("w") as fg:
        for occ, summary, seq in sample:
            rid = f"TSB_{occ}"
            fi.write(json.dumps({"record_id": rid, "source": "TSB_CANADA",
                                 "text": summary}) + "\n")
            fg.write(json.dumps({"record_id": rid, "seq": seq}) + "\n")
    n_ev = sum(len(s) for _, _, s in sample)
    print(f"wrote {len(sample)} records ({n_ev} gold events, "
          f"{n_ev/len(sample):.2f}/record) -> {OUT_INPUT.name}, {OUT_GOLD.name}")


if __name__ == "__main__":
    main()
