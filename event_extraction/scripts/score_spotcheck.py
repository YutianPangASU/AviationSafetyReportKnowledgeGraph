"""Score the reduced human spot check of the judge (review 2026-09-30).

One domain-literate annotator adjudicates the 50 reports of
event_extraction/out/adjudication/spotcheck_50.csv (the first 50 records of
the seed-0 sample_100 drawn by eval_extraction_extras.py). For every
(finding, node) pair the annotator writes 1 in annotator_match when the
extracted node states the NTSB cause-flagged finding and 0 otherwise; the
narrative, findings and nodes of each record are in spotcheck_50.jsonl.
Blank cells count as 0.

Reported, as percentage agreement with the judge rather than as a new gold
standard: agreement over all pairs, agreement over the findings (a finding
is recalled when at least one node matches), the finding-level recall under
the judge and under the annotator, and the disagreeing findings for review.

Usage:
  python3 event_extraction/scripts/score_spotcheck.py \
      event_extraction/out/adjudication/spotcheck_50_annotated.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    judge = [int(r["judge_match"] or 0) for r in rows]
    ann = [int((r["annotator_match"] or "0").strip() or 0) for r in rows]
    blank = sum(1 for r in rows if not (r["annotator_match"] or "").strip())
    pair_agree = sum(a == b for a, b in zip(judge, ann)) / len(rows)

    fj, fa = defaultdict(int), defaultdict(int)
    text = {}
    for r, j, a in zip(rows, judge, ann):
        k = (r["record_id"], r["finding_idx"])
        fj[k] |= j
        fa[k] |= a
        text[k] = r["finding_text"]
    keys = sorted(fj)
    find_agree = sum(fj[k] == fa[k] for k in keys) / len(keys)
    out = {
        "records": len({r["record_id"] for r in rows}),
        "pairs": len(rows), "blank_pairs_counted_as_0": blank,
        "findings": len(keys),
        "pair_agreement": round(pair_agree, 4),
        "finding_agreement": round(find_agree, 4),
        "finding_recall_judge": round(sum(fj[k] for k in keys) / len(keys), 4),
        "finding_recall_annotator": round(sum(fa[k] for k in keys) / len(keys), 4),
        "disagreements": [{"record_id": k[0], "finding_idx": k[1], "finding": text[k],
                           "judge": fj[k], "annotator": fa[k]}
                          for k in keys if fj[k] != fa[k]],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
