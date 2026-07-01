"""Cross-model comparison: human gold vs Qwen / Opus / Fable on the eval subset.

Every model is scored on the SAME (record, cohort, sentence) keys -- the common
evaluation subset -- so agreement rates are directly comparable. Reports, per
model: binary agreement (IN+MAYBE vs OUT), Cohen's kappa, and IN precision /
recall / F1; broken out per cohort. Also emits a human inter-annotator agreement
(IAA) reference and a per-sentence side-by-side table for qualitative review.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

PRIORITY = {"IN": 2, "MAYBE": 1, "OUT": 0}
COHORTS = ["Caused by Wind", "Mobile Crane Accident", "Static Crane Accident"]


def bin_(v: str) -> str:
    return "OUT" if v == "OUT" else "IN"


def human_gold(corpus: Path):
    """Returns (gold, iaa_pairs). gold[(rid,cohort,idx)] = {gold,text}."""
    gold: dict[tuple, dict] = {}
    iaa: list[tuple[str, str]] = []   # (annotator_a, annotator_b) binary values on doubly-labeled sentences
    for line in corpus.open():
        r = json.loads(line)
        for s in r["sentences"]:
            per: dict[str, list[str]] = defaultdict(list)
            for lb in s["labels"]:
                per[lb["label"]].append(lb["value"])
            for cohort, vals in per.items():
                cnt = Counter(vals)
                best = max(cnt, key=lambda v: (cnt[v], PRIORITY[v]))
                gold[(r["record_id"], cohort, s["index"])] = {"gold": best, "text": s["text"]}
                if len(vals) >= 2:   # first two annotators as an IAA pair
                    iaa.append((bin_(vals[0]), bin_(vals[1])))
    return gold, iaa


def load_model(paths: list[Path]) -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    for p in paths:
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok") is False:
                continue
            for idx, val in r["labels"].items():
                out[(r["record_id"], r["cohort"], int(idx))] = str(val).upper()
    return out


def metrics(gold_vals: list[str], pred_vals: list[str]) -> dict[str, float]:
    g = [bin_(x) for x in gold_vals]
    p = [bin_(x) for x in pred_vals]
    n = len(g)
    acc = sum(a == b for a, b in zip(g, p)) / n
    kappa = cohen_kappa_score(g, p) if len(set(g)) > 1 and len(set(p)) > 1 else float("nan")
    pr, rc, f1, _ = precision_recall_fscore_support(
        g, p, labels=["IN"], average=None, zero_division=0
    )
    return {"n": n, "acc": acc, "kappa": float(kappa),
            "in_prec": float(pr[0]), "in_rec": float(rc[0]), "in_f1": float(f1[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../labeled_crane_corpus_sample.jsonl")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    out = Path(args.outdir)

    gold, iaa = human_gold(Path(args.corpus))
    subset_ids = set(json.loads((out / "eval_subset_ids.json").read_text()))

    models = {
        "Qwen3.6-35B": load_model([out / "llm_labels_qwen.jsonl"]),
        "Opus-4.8": load_model([Path(p) for p in sorted(glob.glob(str(out / "opus" / "*.jsonl")))]),
        "Fable-5": load_model([Path(p) for p in sorted(glob.glob(str(out / "fable" / "*.jsonl")))]),
    }

    # Common keys: gold keys within the subset that ALL models predicted.
    keys = [k for k in gold if k[0] in subset_ids]
    for m, d in models.items():
        keys = [k for k in keys if k in d]
    keys.sort()
    print(f"common eval keys (all models predicted): {len(keys)} "
          f"sentence-cohort pairs across {len(subset_ids)} records\n")

    # Human IAA reference (binary) on doubly-annotated sentences.
    if iaa:
        a = [x[0] for x in iaa]
        b = [x[1] for x in iaa]
        iaa_acc = sum(x == y for x, y in zip(a, b)) / len(iaa)
        iaa_k = cohen_kappa_score(a, b)
        print(f"Human inter-annotator agreement (n={len(iaa)} doubly-labeled): "
              f"acc={iaa_acc:.4f}  kappa={iaa_k:.4f}\n")

    # ---- Overall table ----
    hdr = f"{'model':<14}{'n':>6}{'agree':>8}{'kappa':>8}{'IN_prec':>9}{'IN_rec':>8}{'IN_f1':>8}"
    print("OVERALL (binary: IN+MAYBE vs OUT)")
    print(hdr)
    summary: dict[str, Any] = {"n_keys": len(keys), "overall": {}, "by_cohort": {}}
    for m, d in models.items():
        r = metrics([gold[k]["gold"] for k in keys], [d[k] for k in keys])
        summary["overall"][m] = r
        print(f"{m:<14}{r['n']:>6}{r['acc']:>8.3f}{r['kappa']:>8.3f}"
              f"{r['in_prec']:>9.3f}{r['in_rec']:>8.3f}{r['in_f1']:>8.3f}")

    # ---- Per-cohort ----
    for cohort in COHORTS:
        ck = [k for k in keys if k[1] == cohort]
        if not ck:
            continue
        print(f"\n{cohort} (n={len(ck)})")
        print(hdr)
        summary["by_cohort"][cohort] = {}
        for m, d in models.items():
            r = metrics([gold[k]["gold"] for k in ck], [d[k] for k in ck])
            summary["by_cohort"][cohort][m] = r
            print(f"{m:<14}{r['n']:>6}{r['acc']:>8.3f}{r['kappa']:>8.3f}"
                  f"{r['in_prec']:>9.3f}{r['in_rec']:>8.3f}{r['in_f1']:>8.3f}")

    # ---- Side-by-side per-sentence table (for qualitative review) ----
    rows = []
    for k in keys:
        rid, cohort, idx = k
        row = {"record_id": rid, "cohort": cohort, "index": idx,
               "human": gold[k]["gold"]}
        for m, d in models.items():
            row[m] = d[k]
        # disagreement flag (any model differs from human, binary)
        h = bin_(row["human"])
        row["all_agree"] = all(bin_(row[m]) == h for m in models)
        row["text"] = gold[k]["text"]
        rows.append(row)
    with (out / "side_by_side.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    n_dis = sum(not r["all_agree"] for r in rows)
    print(f"\nwrote {out/'side_by_side.jsonl'} ({n_dis} sentences where >=1 model "
          f"disagrees with human)")
    print(f"wrote {out/'compare_summary.json'}")


if __name__ == "__main__":
    main()
