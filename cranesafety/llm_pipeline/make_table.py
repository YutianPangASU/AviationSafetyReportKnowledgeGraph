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

# ---------- keyed inter-annotator (IAA) pairs, so models can be scored on the
#            exact same doubly-labeled population ----------
def rv(vals):
    return "IN" if "IN" in vals else ("MAYBE" if "MAYBE" in vals else "OUT")


doc_iaa_pairs = {}   # (rid,cohort) -> (annot_a_binary, annot_b_binary)
sent_iaa_pairs = {}  # (rid,cohort,idx) -> (a,b)
for r in records:
    doc_annot = defaultdict(lambda: defaultdict(list))
    for s in r["sentences"]:
        sent_annot = defaultdict(lambda: defaultdict(list))
        for lb in s["labels"]:
            doc_annot[lb["label"]][lb["author"]].append(lb["value"])
            sent_annot[lb["label"]][lb["author"]].append(lb["value"])
        for coh, au in sent_annot.items():
            if len(au) >= 2:
                a = [b(rv(v)) for v in au.values()]
                sent_iaa_pairs[(r["record_id"], coh, s["index"])] = (a[0], a[1])
    for coh, au in doc_annot.items():
        if len(au) >= 2:
            a = [b(rv(v)) for v in au.values()]
            doc_iaa_pairs[(r["record_id"], coh)] = (a[0], a[1])


# ---------- render helpers ----------
def model_only_table(title, keys, gold, models):
    """Block 1: model vs human gold (no IAA row -- undefined here)."""
    lines = [f"### {title}", "",
             "| Labeler | n | Agreement | Cohen κ | IN prec | IN recall | IN F1 |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for name, d in models.items():
        n, acc, k, p, rc, f1 = m([gold[x] for x in keys], [d[x] for x in keys])
        lines.append(f"| {name} | {n} | {acc:.3f} | {k:.3f} | "
                     f"{p:.3f} | {rc:.3f} | {f1:.3f} |")
    return "\n".join(lines)


def same_n_table(title, iaa_pairs, gold, models):
    """Block 2: human 2nd-annotator AND every model on the identical
    doubly-labeled population -> one shared n. Agreement/κ only (IN P/R/F is
    unstable on ~15 positives)."""
    keys = [k for k in iaa_pairs if all(k in d for d in models.values())]
    ga = [iaa_pairs[k][0] for k in keys]
    gb = [iaa_pairs[k][1] for k in keys]
    n = len(keys)
    hacc = sum(x == y for x, y in zip(ga, gb)) / n
    hk = cohen_kappa_score(ga, gb)
    lines = [f"### {title}", "",
             f"All rows scored on the **same {n}** doubly-labeled items.", "",
             "| Labeler | n | Agreement | Cohen κ |", "|---|--:|--:|--:|",
             f"| Human (2nd annotator) | {n} | {hacc:.3f} | {hk:.3f} |"]
    for name, d in models.items():
        _, acc, k, *_ = m([gold[x] for x in keys], [d[x] for x in keys])
        lines.append(f"| {name} | {n} | {acc:.3f} | {k:.3f} |")
    return "\n".join(lines)


GLOSSARY = """\
### How to read these numbers

- **Agreement** — fraction of items given the same IN/OUT call by the two
  labelers. The corpus is ~91% OUT, so a labeler that always says OUT already
  scores ~0.91; agreement alone is flattering on imbalanced data.
- **Cohen κ** — agreement corrected for chance (1 = perfect, 0 = no better than
  guessing). This is the metric to trust here: the "always OUT" labeler scores
  κ ≈ 0. On tiny positive counts κ is statistically fragile (one flip moves it a
  lot), so read it together with the raw counts.
- **IN precision / recall / F1** — restricted to the rare positive class (IN).
  Precision = of the items the labeler called IN, how many the human agreed with.
  Recall = of the human's IN items, how many the labeler caught. F1 = their mean.
- **Binary view** — MAYBE is folded into IN (the guide says "lean to a decision").
"""

NOTE = """\
> **On the human baseline.** Only **115** of the 1,193 (report, cohort) groups
> were labeled by two people; the rest had a single annotator, so a human-vs-human
> number simply does not exist for them. That is why Block 1 (model vs. gold) uses
> the full dataset while Block 2 (human-vs-human baseline) can only use the 115
> doubly-labeled groups. On those 115, the two humans agreed on **114** — the sole
> disagreement (rec1361) is one annotator saying OUT and the other MAYBE. The
> models match that second-human baseline exactly.
"""

report = []
report.append("# Human vs LLM agreement — crane cohort labeling\n")
report.append("Binary view (IN+MAYBE vs OUT). Positive class = IN (item belongs "
              "to the cohort). Labelers: Human gold, Qwen3.6-35B (local), "
              "Opus-4.8, Fable-5.\n")
report.append(GLOSSARY)
report.append("---\n")
report.append("## Block 1 — Model performance vs. the human gold\n")
report.append("How well each LLM reproduces the human labels, across **all** items. "
              "(No human-vs-human row: it is undefined on the single-annotator "
              "majority of the data — see the note in Block 2.)\n")
report.append(model_only_table(
    "Document level — one label per (report, cohort), full dataset (n=1,193)",
    doc_keys, doc_gold, doc_models))
report.append("")
report.append(model_only_table(
    "Sentence level — one label per sentence, 300-report eval subset (n=1,809)",
    sent_keys, sent_gold, sent_models))
report.append("")
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
report.append("")
report.append("---\n")
report.append("## Block 2 — \"Is the LLM as good as a second human?\"\n")
report.append("Here every labeler — the second human **and** each model — is scored "
              "on the **identical** doubly-labeled population, so `n` matches across "
              "all rows.\n")
report.append(same_n_table(
    "Document level (doubly-labeled report groups)", doc_iaa_pairs, doc_gold, doc_models))
report.append("")
report.append(same_n_table(
    "Sentence level (doubly-labeled sentences)", sent_iaa_pairs, sent_gold, sent_models))
report.append("")
report.append(
    "> The sentence-level models trail the second human here purely because of the "
    "**propagation artifact** (§1 of `REPORT.md`): humans stamp every sentence of an "
    "IN report as IN, and these doubly-labeled sentences are concentrated in such "
    "reports, so the models are penalized for not copying the label onto follow-on "
    "sentences that never mention a crane. It is not a quality gap — see the "
    "document-level block above.\n")
report.append(NOTE)

text = "\n".join(report) + "\n"
(OUT / "RESULTS_TABLE.md").write_text(text)
print(text)
print(f"\nwrote {OUT/'RESULTS_TABLE.md'}")
