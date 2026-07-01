"""Compare LLM sentence labels against the human gold labels.

Reports (1) agreement rates -- overall and per cohort, under both a 3-class
(IN/OUT/MAYBE) and a binary (IN+MAYBE vs OUT) view -- with Cohen's kappa and
per-class precision/recall/F1 for the rare positive class, and (2) a dump of
every disagreement with sentence text for qualitative review.

Usage:
    python analyze.py --corpus ../labeled_crane_corpus_sample.jsonl \
        --llm out/llm_labels.jsonl --outdir out
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

PRIORITY = {"IN": 2, "MAYBE": 1, "OUT": 0}


def human_gold(corpus: Path) -> dict[tuple[Any, str, int], dict[str, Any]]:
    """{(record_id, cohort, index): {gold, text, authors}} via majority vote.

    Ties break toward the stronger label (IN > MAYBE > OUT), matching the
    guide's "lean towards yes". Same-cohort multi-annotator conflicts are rare.
    """
    gold: dict[tuple[Any, str, int], dict[str, Any]] = {}
    for line in corpus.open():
        r = json.loads(line)
        for s in r["sentences"]:
            per_cohort: dict[str, list[str]] = defaultdict(list)
            for lb in s["labels"]:
                per_cohort[lb["label"]].append(lb["value"])
            for cohort, vals in per_cohort.items():
                cnt = Counter(vals)
                best = max(cnt, key=lambda v: (cnt[v], PRIORITY[v]))
                gold[(r["record_id"], cohort, s["index"])] = {
                    "gold": best, "text": s["text"], "n_annot": len(vals),
                }
    return gold


def load_llm(path: Path) -> dict[tuple[Any, str, int], str]:
    out: dict[tuple[Any, str, int], str] = {}
    n_fail = 0
    for line in path.open():
        r = json.loads(line)
        if not r.get("ok"):
            n_fail += 1
            continue
        for idx, val in r["labels"].items():
            out[(r["record_id"], r["cohort"], int(idx))] = val
    if n_fail:
        print(f"  (warning: {n_fail} failed LLM records skipped)")
    return out


def block(title: str, gold: list[str], pred: list[str]) -> dict[str, Any]:
    n = len(gold)
    agree = sum(g == p for g, p in zip(gold, pred))
    kappa = cohen_kappa_score(gold, pred) if len(set(gold)) > 1 else float("nan")
    print(f"\n### {title}  (n={n})")
    print(f"  agreement (accuracy): {agree/n:.4f}   Cohen's kappa: {kappa:.4f}")
    # Per-class P/R/F for positive-ish classes present.
    labels = sorted(set(gold) | set(pred), key=lambda v: -PRIORITY.get(v, 0))
    p, r, f, sup = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    print(f"  {'class':<6} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for i, lb in enumerate(labels):
        print(f"  {lb:<6} {p[i]:>6.3f} {r[i]:>7.3f} {f[i]:>6.3f} {int(sup[i]):>8}")
    # Confusion (gold rows -> pred cols)
    cats = sorted(set(gold) | set(pred), key=lambda v: -PRIORITY.get(v, 0))
    conf = Counter(zip(gold, pred))
    print("  confusion (gold\\pred): " + " ".join(f"{c:>6}" for c in cats))
    for g in cats:
        print(f"    {g:<20} " + " ".join(f"{conf[(g,c)]:>6}" for c in cats))
    return {"n": n, "accuracy": agree / n, "kappa": float(kappa)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../labeled_crane_corpus_sample.jsonl")
    ap.add_argument("--llm", default="out/llm_labels.jsonl")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    gold = human_gold(Path(args.corpus))
    llm = load_llm(Path(args.llm))

    keys = [k for k in gold if k in llm]
    missing = len(gold) - len(keys)
    print(f"joined {len(keys)} sentence-cohort pairs "
          f"({missing} human pairs had no LLM prediction)")

    g3 = [gold[k]["gold"] for k in keys]
    p3 = [llm[k] for k in keys]

    def bin_(v: str) -> str:
        return "OUT" if v == "OUT" else "IN"   # collapse MAYBE -> IN

    summary: dict[str, Any] = {"overall": {}, "by_cohort": {}}

    print("\n" + "=" * 70 + "\nOVERALL\n" + "=" * 70)
    summary["overall"]["three_class"] = block("3-class (IN/OUT/MAYBE)", g3, p3)
    summary["overall"]["binary"] = block(
        "Binary (IN+MAYBE vs OUT)", [bin_(x) for x in g3], [bin_(x) for x in p3]
    )

    for cohort in ["Caused by Wind", "Mobile Crane Accident", "Static Crane Accident"]:
        ck = [k for k in keys if k[1] == cohort]
        if not ck:
            continue
        g = [gold[k]["gold"] for k in ck]
        p = [llm[k] for k in ck]
        print("\n" + "=" * 70 + f"\nCOHORT: {cohort}\n" + "=" * 70)
        summary["by_cohort"][cohort] = {
            "three_class": block("3-class", g, p),
            "binary": block("Binary (IN+MAYBE vs OUT)",
                            [bin_(x) for x in g], [bin_(x) for x in p]),
        }

    # Disagreement dump for qualitative review (binary view = what matters most).
    disagreements = []
    for k in keys:
        g, p = gold[k]["gold"], llm[k]
        if bin_(g) != bin_(p):
            rid, cohort, idx = k
            disagreements.append({
                "record_id": rid, "cohort": cohort, "index": idx,
                "human": g, "llm": p,
                "type": "LLM_false_positive" if bin_(p) == "IN" else "LLM_false_negative",
                "text": gold[k]["text"], "n_annot": gold[k]["n_annot"],
            })
    disagreements.sort(key=lambda d: (d["cohort"], d["type"], d["record_id"], d["index"]))

    outdir = Path(args.outdir)
    dpath = outdir / "disagreements.jsonl"
    with dpath.open("w", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70 + "\nDISAGREEMENT BREAKDOWN\n" + "=" * 70)
    by = Counter((d["cohort"], d["type"]) for d in disagreements)
    print(f"total binary disagreements: {len(disagreements)} of {len(keys)} "
          f"({len(disagreements)/len(keys)*100:.1f}%)")
    for (cohort, typ), c in sorted(by.items()):
        print(f"  {cohort:<24} {typ:<20} {c}")
    print(f"\nwrote {dpath}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
