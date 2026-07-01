"""Document-level comparison: human gold vs Qwen / Opus / Fable, full dataset.

The fair task: ONE label per (report, cohort). Human report-gold is derived from
the (uniform) sentence labels. Every model made the same single decision per
report, so agreement rates are directly comparable. Reports overall + per-cohort
agreement, Cohen's kappa, and IN precision/recall/F1 under the binary
(IN+MAYBE vs OUT) view; dumps disagreements (with model evidence) for review.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

COHORTS = ["Caused by Wind", "Mobile Crane Accident", "Static Crane Accident"]


def bin_(v: str) -> str:
    return "OUT" if v == "OUT" else "IN"


def human_report_gold(corpus: Path):
    """{(record_id, cohort): {"gold":..., "text":full_report}}. Report label =
    IN if any sentence IN; elif any MAYBE; else OUT. Also returns per-report IAA."""
    gold: dict[tuple, dict] = {}
    iaa_a, iaa_b = [], []
    for line in corpus.open():
        r = json.loads(line)
        text = " ".join(s["text"] for s in r["sentences"])
        by_cohort_vals: dict[str, list[str]] = defaultdict(list)
        by_cohort_annot: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for s in r["sentences"]:
            for lb in s["labels"]:
                by_cohort_vals[lb["label"]].append(lb["value"])
                by_cohort_annot[lb["label"]][lb["author"]].append(lb["value"])
        for cohort, vals in by_cohort_vals.items():
            if "IN" in vals:
                g = "IN"
            elif "MAYBE" in vals:
                g = "MAYBE"
            else:
                g = "OUT"
            gold[(r["record_id"], cohort)] = {"gold": g, "text": text}
            # report-level IAA: annotators who labeled this report/cohort
            annots = by_cohort_annot[cohort]
            if len(annots) >= 2:
                def report_val(v):
                    return "IN" if "IN" in v else ("MAYBE" if "MAYBE" in v else "OUT")
                a_list = [report_val(v) for v in annots.values()]
                iaa_a.append(bin_(a_list[0]))
                iaa_b.append(bin_(a_list[1]))
    return gold, (iaa_a, iaa_b)


def load_doc(paths) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for p in paths:
        for line in Path(p).open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") is False:
                continue
            lab = str(r.get("label", "OUT")).upper()
            if lab not in {"IN", "OUT", "MAYBE"}:
                lab = "OUT"
            out[(r["record_id"], r["cohort"])] = {"label": lab, "evidence": r.get("evidence", "")}
    return out


def metrics(gold_vals, pred_vals):
    g = [bin_(x) for x in gold_vals]
    p = [bin_(x) for x in pred_vals]
    n = len(g)
    acc = sum(a == b for a, b in zip(g, p)) / n
    kappa = (cohen_kappa_score(g, p)
             if len(set(g)) > 1 and len(set(p)) > 1 else float("nan"))
    pr, rc, f1, _ = precision_recall_fscore_support(
        g, p, labels=["IN"], average=None, zero_division=0)
    return {"n": n, "acc": acc, "kappa": float(kappa),
            "in_prec": float(pr[0]), "in_rec": float(rc[0]), "in_f1": float(f1[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../labeled_crane_corpus_sample.jsonl")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    out = Path(args.outdir)

    gold, (iaa_a, iaa_b) = human_report_gold(Path(args.corpus))
    models = {
        "Qwen3.6-35B": load_doc([out / "doc_qwen.jsonl"]),
        "Opus-4.8": load_doc(sorted(glob.glob(str(out / "doc_opus" / "*.jsonl")))),
        "Fable-5": load_doc(sorted(glob.glob(str(out / "doc_fable" / "*.jsonl")))),
    }
    keys = sorted(k for k in gold if all(k in d for d in models.values()))
    print(f"scored on {len(keys)} (report, cohort) pairs (full dataset)\n")

    if iaa_a:
        acc = sum(a == b for a, b in zip(iaa_a, iaa_b)) / len(iaa_a)
        k = cohen_kappa_score(iaa_a, iaa_b)
        print(f"Human inter-annotator agreement (report-level, n={len(iaa_a)} "
              f"doubly-labeled reports): acc={acc:.4f} kappa={k:.4f}\n")

    hdr = (f"{'labeler':<14}{'n':>6}{'agree':>8}{'kappa':>8}"
           f"{'IN_prec':>9}{'IN_rec':>8}{'IN_f1':>7}")
    summary = {"n": len(keys), "overall": {}, "by_cohort": {}}
    print("OVERALL (binary IN+MAYBE vs OUT)")
    print(hdr)
    for m, d in models.items():
        r = metrics([gold[k]["gold"] for k in keys], [d[k]["label"] for k in keys])
        summary["overall"][m] = r
        print(f"{m:<14}{r['n']:>6}{r['acc']:>8.3f}{r['kappa']:>8.3f}"
              f"{r['in_prec']:>9.3f}{r['in_rec']:>8.3f}{r['in_f1']:>7.3f}")

    for cohort in COHORTS:
        ck = [k for k in keys if k[1] == cohort]
        npos = sum(bin_(gold[k]["gold"]) == "IN" for k in ck)
        print(f"\n{cohort}  (n={len(ck)}, human-IN reports={npos})")
        print(hdr)
        summary["by_cohort"][cohort] = {}
        for m, d in models.items():
            r = metrics([gold[k]["gold"] for k in ck], [d[k]["label"] for k in ck])
            summary["by_cohort"][cohort][m] = r
            print(f"{m:<14}{r['n']:>6}{r['acc']:>8.3f}{r['kappa']:>8.3f}"
                  f"{r['in_prec']:>9.3f}{r['in_rec']:>8.3f}{r['in_f1']:>7.3f}")

    # Confusion vs human (binary) per model + disagreement dump.
    print("\nConfusion vs human (binary), gold\\pred  IN / OUT:")
    dis = []
    for m, d in models.items():
        c = Counter((bin_(gold[k]["gold"]), bin_(d[k]["label"])) for k in keys)
        print(f"  {m:<12} IN->[IN={c[('IN','IN')]}, OUT={c[('IN','OUT')]}]  "
              f"OUT->[IN={c[('OUT','IN')]}, OUT={c[('OUT','OUT')]}]")
    for k in keys:
        h = gold[k]["gold"]
        row = {"record_id": k[0], "cohort": k[1], "human": h}
        for m, d in models.items():
            row[m] = d[k]["label"]
            row[m + "_evidence"] = d[k]["evidence"]
        if not all(bin_(row[m]) == bin_(h) for m in models):
            row["text"] = gold[k]["text"]
            dis.append(row)
    with (out / "doc_disagreements.jsonl").open("w", encoding="utf-8") as f:
        for r in dis:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "doc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(dis)} reports with >=1 model disagreeing with human "
          f"-> {out/'doc_disagreements.jsonl'}")
    print(f"wrote {out/'doc_summary.json'}")


if __name__ == "__main__":
    main()
