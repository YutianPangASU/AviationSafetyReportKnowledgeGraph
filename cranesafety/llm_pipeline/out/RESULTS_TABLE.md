# Human vs LLM agreement — crane cohort labeling

Binary view (IN+MAYBE vs OUT). Positive class = IN (item belongs to the cohort). Labelers: Human gold, Qwen3.6-35B (local), Opus-4.8, Fable-5.

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

---

## Block 1 — Model performance vs. the human gold

How well each LLM reproduces the human labels, across **all** items. (No human-vs-human row: it is undefined on the single-annotator majority of the data — see the note in Block 2.)

### Document level — one label per (report, cohort), full dataset (n=1,193)

| Labeler | n | Agreement | Cohen κ | IN prec | IN recall | IN F1 |
|---|--:|--:|--:|--:|--:|--:|
| Qwen3.6-35B | 1193 | 0.959 | 0.747 | 0.796 | 0.745 | 0.770 |
| Opus-4.8 | 1193 | 0.946 | 0.695 | 0.689 | 0.764 | 0.724 |
| Fable-5 | 1193 | 0.956 | 0.748 | 0.732 | 0.818 | 0.773 |

### Sentence level — one label per sentence, 300-report eval subset (n=1,809)

| Labeler | n | Agreement | Cohen κ | IN prec | IN recall | IN F1 |
|---|--:|--:|--:|--:|--:|--:|
| Qwen3.6-35B | 1809 | 0.725 | 0.376 | 0.974 | 0.348 | 0.512 |
| Opus-4.8 | 1809 | 0.648 | 0.177 | 0.946 | 0.162 | 0.277 |
| Fable-5 | 1809 | 0.738 | 0.410 | 0.951 | 0.389 | 0.552 |

### Document level, per cohort (agreement / κ)

| Cohort | n | Qwen3.6-35B | Opus-4.8 | Fable-5 |
|---|--:|--:|--:|--:|
| Caused by Wind (IN=16) | 871 | 0.994 / 0.84 | 0.993 / 0.81 | 0.993 / 0.81 |
| Mobile Crane Accident (IN=57) | 170 | 0.859 / 0.68 | 0.800 / 0.58 | 0.853 / 0.68 |
| Static Crane Accident (IN=37) | 152 | 0.868 / 0.63 | 0.842 / 0.56 | 0.855 / 0.63 |

---

## Block 2 — "Is the LLM as good as a second human?"

Here every labeler — the second human **and** each model — is scored on the **identical** doubly-labeled population, so `n` matches across all rows.

### Document level (doubly-labeled report groups)

All rows scored on the **same 115** doubly-labeled items.

| Labeler | n | Agreement | Cohen κ |
|---|--:|--:|--:|
| Human (2nd annotator) | 115 | 0.991 | 0.796 |
| Qwen3.6-35B | 115 | 0.991 | 0.796 |
| Opus-4.8 | 115 | 0.991 | 0.796 |
| Fable-5 | 115 | 0.991 | 0.796 |

### Sentence level (doubly-labeled sentences)

All rows scored on the **same 140** doubly-labeled items.

| Labeler | n | Agreement | Cohen κ |
|---|--:|--:|--:|
| Human (2nd annotator) | 140 | 0.964 | 0.719 |
| Qwen3.6-35B | 140 | 0.929 | 0.268 |
| Opus-4.8 | 140 | 0.929 | 0.268 |
| Fable-5 | 140 | 0.929 | 0.268 |

> The sentence-level models trail the second human here purely because of the **propagation artifact** (§1 of `REPORT.md`): humans stamp every sentence of an IN report as IN, and these doubly-labeled sentences are concentrated in such reports, so the models are penalized for not copying the label onto follow-on sentences that never mention a crane. It is not a quality gap — see the document-level block above.

> **On the human baseline.** Only **115** of the 1,193 (report, cohort) groups
> were labeled by two people; the rest had a single annotator, so a human-vs-human
> number simply does not exist for them. That is why Block 1 (model vs. gold) uses
> the full dataset while Block 2 (human-vs-human baseline) can only use the 115
> doubly-labeled groups. On those 115, the two humans agreed on **114** — the sole
> disagreement (rec1361) is one annotator saying OUT and the other MAYBE. The
> models match that second-human baseline exactly.

