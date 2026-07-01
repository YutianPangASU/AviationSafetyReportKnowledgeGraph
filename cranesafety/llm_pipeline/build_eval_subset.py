"""Build a common evaluation subset + ready-to-send prompts for any LLM.

We score every model on the SAME sentences so agreement rates are comparable.
The subset keeps all "informative" records (any IN/MAYBE from a human) plus a
seeded random sample of pure-OUT records to anchor the negative / false-positive
rate. For each (record, cohort) it emits the exact system + user prompt, so the
identical instructions can be sent to a local model, an API model, or pasted by
hand.

Outputs (in --outdir):
  eval_subset_ids.json  - list of record_ids in the subset
  eval_tasks.jsonl      - one row per (record, cohort): prompts + gold + indices
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from guidelines import COHORTS, system_prompt
from label_crane import build_user_prompt, target_cohorts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../labeled_crane_corpus_sample.jsonl")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--n-negatives", type=int, default=190)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.corpus).open()]
    informative, negative = [], []
    for r in records:
        has_pos = any(
            lb["value"] in ("IN", "MAYBE")
            for s in r["sentences"] for lb in s["labels"]
        )
        (informative if has_pos else negative).append(r)

    rng = random.Random(args.seed)
    rng.shuffle(negative)
    subset = informative + negative[: args.n_negatives]
    subset_ids = [r["record_id"] for r in subset]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval_subset_ids.json").write_text(json.dumps(subset_ids))

    tasks = []
    for r in subset:
        for cohort in target_cohorts(r):
            tasks.append({
                "record_id": r["record_id"],
                "cohort": cohort,
                "indices": [s["index"] for s in r["sentences"]],
                "system": system_prompt(cohort),
                "user": build_user_prompt(r),
            })
    with (outdir / "eval_tasks.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"subset: {len(subset)} records "
          f"({len(informative)} informative + {len(subset)-len(informative)} negatives)")
    print(f"tasks (record x cohort): {len(tasks)}")
    print(f"wrote {outdir/'eval_subset_ids.json'} and {outdir/'eval_tasks.jsonl'}")


if __name__ == "__main__":
    main()
