"""Final comparison table: human vs Qwen / Opus / Fable at BOTH granularities.

- Document level: ONE label per (report, cohort), full dataset (1193 pairs).
- Sentence level: one label per (report, cohort, sentence), on the 300-record
  evaluation subset (1809 pairs; all informative reports + a negative sample).

For each labeler we report binary (IN+MAYBE vs OUT) agreement with the human
gold, Cohen's kappa, and IN precision/recall/F1. A human inter-annotator
agreement (IAA) row anchors the ceiling at each level. Emits Markdown + JSON.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

OUT = Path("out")
CORPUS = Path("../labeled_crane_corpus_sample.jsonl")
COHORTS = ["Caused by Wind", "Mobile Crane Accident", "Static Crane Accident"]


def b(v):
    return "OUT" if v == "OUT" else "IN"


def m(gold, pred):
    g = [b(x) for x in gold]
    p = [b(x) for x in pred]
    n = len(g)
    acc = sum(a == c for a, c in zip(g, p)) / n
    k = cohen_kappa_score(g, p) if len(set(g)) > 1 and len(set(p)) > 1 else float("nan")
    pr, rc, f1, _ = precision_recall_fscore_support(g, p, labels=["IN"], zero_division=0)
    return n, acc, float(k), float(pr[0]), float(rc[0]), float(f1[0])


# ---------- load corpus once ----------
records = [json.loads(l) for l in CORPUS.open()]

# ---------- SENTENCE-LEVEL gold + models (subset) ----------
sent_gold = {}   # (rid,cohort,idx) -> value (majority, tie->IN)
sent_iaa = []
PRIO = {"IN": 2, "MAYBE": 1, "OUT": 0}
for r in records:
    for s in r["sentences"]:
        per = defaultdict(list)
        for lb in s["labels"]:
            per[lb["label"]].append(lb["value"])
        for coh, vals in per.items():
            from collections import Counter
            cnt = Counter(vals)
            best = max(cnt, key=lambda v: (cnt[v], PRIO[v]))
            sent_gold[(r["record_id"], coh, s["index"])] = best
            if len(vals) >= 2:
                sent_iaa.append((b(vals[0]), b(vals[1])))


def load_sent(paths):
    d = {}
    for p in paths:
        for line in Path(p).open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok") is False:
                continue
            for idx, val in r["labels"].items():
                d[(r["record_id"], r["cohort"], int(idx))] = str(val).upper()
    return d


sent_models = {
    "Qwen3.6-35B": load_sent([OUT / "llm_labels_qwen.jsonl"]),
    "Opus-4.8": load_sent(sorted(glob.glob(str(OUT / "opus" / "*.jsonl")))),
    "Fable-5": load_sent(sorted(glob.glob(str(OUT / "fable" / "*.jsonl")))),
}
subset_ids = set(json.loads((OUT / "eval_subset_ids.json").read_text()))
sent_keys = [k for k in sent_gold if k[0] in subset_ids]
for d in sent_models.values():
    sent_keys = [k for k in sent_keys if k in d]

# ---------- DOCUMENT-LEVEL gold + models (full) ----------
doc_gold = {}
doc_iaa = []
for r in records:
    per_vals = defaultdict(list)
    per_annot = defaultdict(lambda: defaultdict(list))
    for s in r["sentences"]:
        for lb in s["labels"]:
            per_vals[lb["label"]].append(lb["value"])
            per_annot[lb["label"]][lb["author"]].append(lb["value"])
    for coh, vals in per_vals.items():
        g = "IN" if "IN" in vals else ("MAYBE" if "MAYBE" in vals else "OUT")
        doc_gold[(r["record_id"], coh)] = g
        annots = per_annot[coh]
        if len(annots) >= 2:
            rv = lambda v: "IN" if "IN" in v else ("MAYBE" if "MAYBE" in v else "OUT")
            a = [rv(v) for v in annots.values()]
            doc_iaa.append((b(a[0]), b(a[1])))


def load_doc(paths):
    d = {}
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
            d[(r["record_id"], r["cohort"])] = lab if lab in {"IN", "OUT", "MAYBE"} else "OUT"
    return d


doc_models = {
    "Qwen3.6-35B": load_doc([OUT / "doc_qwen.jsonl"]),
    "Opus-4.8": load_doc(sorted(glob.glob(str(OUT / "doc_opus" / "*.jsonl")))),
    "Fable-5": load_doc(sorted(glob.glob(str(OUT / "doc_fable" / "*.jsonl")))),
}
doc_keys = [k for k in doc_gold if all(k in d for d in doc_models.values())]

# ---------- render ----------
def rows(keys, gold, models, iaa):
    out = []
    ia = cohen_kappa_score([x[0] for x in iaa], [x[1] for x in iaa])
    iacc = sum(x == y for x, y in iaa) / len(iaa)
    out.append(("Human IAA", len(iaa), iacc, ia, None, None, None))
    for name, d in models.items():
        out.append((name, *m([gold[k] for k in keys], [d[k] for k in keys])))
    return out


def md_table(title, rws):
    lines = [f"### {title}", "",
             "| Labeler | n | Agreement | Cohen κ | IN prec | IN recall | IN F1 |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for name, n, acc, k, p, rc, f1 in rws:
        pp = "–" if p is None else f"{p:.3f}"
        rr = "–" if rc is None else f"{rc:.3f}"
        ff = "–" if f1 is None else f"{f1:.3f}"
        kk = "–" if k != k else f"{k:.3f}"  # nan guard
        lines.append(f"| {name} | {n} | {acc:.3f} | {kk} | {pp} | {rr} | {ff} |")
    return "\n".join(lines)


doc_rows = rows(doc_keys, doc_gold, doc_models, doc_iaa)
sent_rows = rows(sent_keys, sent_gold, sent_models, sent_iaa)

report = []
report.append("# Human vs LLM agreement — crane cohort labeling\n")
report.append("Binary view (IN+MAYBE vs OUT). Positive class = IN (report/sentence "
              "belongs to the cohort).\n")
report.append(md_table(
    "Document level — one label per (report, cohort), full dataset", doc_rows))
report.append("")
report.append(md_table(
    "Sentence level — one label per sentence, 300-report eval subset", sent_rows))
report.append("")

# Per-cohort document-level agreement (headline task)
report.append("### Document level, per cohort (agreement / κ)\n")
report.append("| Cohort | n | " + " | ".join(doc_models) + " |")
report.append("|---|--:|" + "|".join(["--:"] * len(doc_models)) + "|")
for coh in COHORTS:
    ck = [k for k in doc_keys if k[1] == coh]
    npos = sum(b(doc_gold[k]) == "IN" for k in ck)
    cells = []
    for name, d in doc_models.items():
        _, acc, k, *_ = m([doc_gold[x] for x in ck], [d[x] for x in ck])
        cells.append(f"{acc:.3f} / {k:.2f}")
    report.append(f"| {coh} (IN={npos}) | {len(ck)} | " + " | ".join(cells) + " |")

text = "\n".join(report) + "\n"
(OUT / "RESULTS_TABLE.md").write_text(text)
print(text)
print(f"\nwrote {OUT/'RESULTS_TABLE.md'}")
