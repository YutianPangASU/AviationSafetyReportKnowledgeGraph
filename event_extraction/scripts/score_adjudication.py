"""Score the human adjudication of the judge (review 2026-09-14, comment 5).

Input: two copies of event_extraction/out/adjudication/sample_100.csv, one
per annotator, with the annotator_match column filled with 1 (the extracted
node states the NTSB finding) or 0 for every (finding, node) pair. Rows
left blank count as 0.

Reports Cohen's kappa between the two annotators, between each annotator and
the judge, the finding-level recall under each rater (a finding is recalled
when at least one node matches), and the pairs where the judge and both
annotators disagree, for inspection.

Usage:
  python3 event_extraction/scripts/score_adjudication.py \
      event_extraction/out/adjudication/sample_100_annotatorA.csv \
      event_extraction/out/adjudication/sample_100_annotatorB.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load(path: str):
    rows = list(csv.DictReader(open(path)))
    key = [(r["record_id"], r["finding_idx"], r["node_idx"]) for r in rows]
    judge = np.array([int(r["judge_match"] or 0) for r in rows])
    ann = np.array([int((r["annotator_match"] or "0").strip() or 0) for r in rows])
    return rows, key, judge, ann


def finding_recall(rows, marks) -> float:
    hit = defaultdict(int)
    for r, m in zip(rows, marks):
        hit[(r["record_id"], r["finding_idx"])] |= int(m)
    return float(np.mean(list(hit.values())))


def main() -> None:
    pa, pb = sys.argv[1], sys.argv[2]
    rows_a, key_a, judge, a = load(pa)
    rows_b, key_b, judge_b, b = load(pb)
    assert key_a == key_b and (judge == judge_b).all(), "the two files must be copies of the same sample"
    out = {
        "n_records": len({r["record_id"] for r in rows_a}),
        "n_pairs": len(rows_a),
        "kappa_annotators": round(kappa(a, b), 3),
        "kappa_judge_vs_A": round(kappa(judge, a), 3),
        "kappa_judge_vs_B": round(kappa(judge, b), 3),
        "pair_agreement_judge_vs_A": round(float((judge == a).mean()), 4),
        "pair_agreement_judge_vs_B": round(float((judge == b).mean()), 4),
        "finding_recall": {"judge": round(finding_recall(rows_a, judge), 4),
                           "annotator_A": round(finding_recall(rows_a, a), 4),
                           "annotator_B": round(finding_recall(rows_b, b), 4)},
        "disagreements_judge_vs_both": [
            {"record_id": r["record_id"], "finding": r["finding_text"], "node": r["factor_type"],
             "trigger": r["trigger"], "judge": int(j), "A": int(x), "B": int(y)}
            for r, j, x, y in zip(rows_a, judge, a, b) if x == y and j != x][:200],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements_judge_vs_both"}, indent=2))
    print("disagreements (judge vs both annotators):", len(out["disagreements_judge_vs_both"]))
    with open("event_extraction/out/adjudication/adjudication_scores.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
