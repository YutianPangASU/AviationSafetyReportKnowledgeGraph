# LLM vs human labeling — crane safety cohorts

**Task.** Reproduce the human cohort annotations with LLMs and measure agreement.
Three cohorts (`Caused by Wind`, `Mobile Crane Accident`, `Static Crane Accident`),
value ∈ {IN, OUT, MAYBE}. Corpus: 1,190 OSHA-style reports, 6,930 sentences,
1,193 (report, cohort) annotation groups.

**Labelers compared.** Human (gold) · **Qwen3.6-35B-A3B** (local vLLM) ·
**Opus-4.8** · **Fable-5**. All received the *identical* guideline text the humans
used (`../Labelling Guide.pdf`, transcribed in `guidelines.py`), temperature 0.

---

## 1. Granularity: the human labels are effectively document-level

The file is stored per sentence (each sentence has an index + label), but the
**content** is uniform per report. Splitting each (report, cohort) group by
annotator:

| pattern (within one annotator, one report) | count |
|---|--:|
| every sentence OUT | 1,083 |
| every sentence IN | 109 |
| mixed within a report | **0** |

The one apparent "mixed" group (record 1361) is two annotators disagreeing
(OUT vs MAYBE) on the *same* sentences, not one annotator varying across the
report. **So each annotator made one decision per report and stamped every
sentence with it.** We therefore evaluate at both levels:

- **Document level** (fair task): one label per (report, cohort), full dataset.
- **Sentence level** (raw format): one label per sentence, on a 300-report subset
  enriched with all 110 informative reports + a negative sample.

Sentence-level scoring *penalizes the LLMs for not propagating* a report-wide
judgment onto continuation sentences (injury, hospital transport, aftermath) that
don't themselves name the crane — which is why sentence-level recall looks poor
even though the models identify the right reports. The document-level numbers are
the meaningful comparison.

---

## 2. Agreement (binary: IN+MAYBE vs OUT)

### Document level — full dataset (n = 1,193)

| Labeler | Agreement | Cohen κ | IN prec | IN recall | IN F1 |
|---|--:|--:|--:|--:|--:|
| **Human IAA (ceiling)** | 0.991 | 0.796 | – | – | – |
| Qwen3.6-35B | 0.959 | 0.747 | 0.796 | 0.745 | 0.770 |
| Opus-4.8 | 0.946 | 0.695 | 0.689 | 0.764 | 0.724 |
| Fable-5 | 0.956 | 0.748 | 0.732 | 0.818 | 0.773 |

All three land at **95–96% agreement, κ ≈ 0.70–0.75**, approaching the
human-to-human ceiling (κ = 0.80). Qwen (local) is on par with the frontier models.

### Document level, per cohort (agreement / κ)

| Cohort | n | Qwen | Opus | Fable |
|---|--:|--:|--:|--:|
| Caused by Wind (IN=16) | 871 | 0.994 / 0.84 | 0.993 / 0.81 | 0.993 / 0.81 |
| Mobile Crane (IN=57) | 170 | 0.859 / 0.68 | 0.800 / 0.58 | 0.853 / 0.68 |
| Static Crane (IN=37) | 152 | 0.868 / 0.63 | 0.842 / 0.56 | 0.855 / 0.63 |

Wind is nearly solved. The crane-type cohorts are harder — and that is where the
human labels themselves are least consistent (see §3).

### Sentence level — 300-report subset (n = 1,809)

| Labeler | Agreement | Cohen κ | IN prec | IN recall | IN F1 |
|---|--:|--:|--:|--:|--:|
| Human IAA | 0.992 | 0.733 | – | – | – |
| Qwen3.6-35B | 0.725 | 0.376 | 0.974 | 0.348 | 0.512 |
| Opus-4.8 | 0.648 | 0.177 | 0.946 | 0.162 | 0.277 |
| Fable-5 | 0.738 | 0.410 | 0.951 | 0.389 | 0.552 |

High IN precision, low IN recall: the models only mark the sentence that names the
crane/wind, not the whole report. This is the propagation artifact, not a labeling
error — see §1.

---

## 3. Qualitative: what the disagreements are

Only **80 of 1,193 reports (6.7%)** have any model disagreeing with the human.
Inspecting them (with each model's cited evidence, `out/doc_disagreements.jsonl`)
shows most are **human inconsistencies, not model errors**:

**A. Mobile ↔ Static cohort confusion (the biggest theme).** Humans repeatedly
put a crane in the wrong type-cohort; the models classify the type correctly:
- Truck/crawler/wheel cranes labeled **Static** by humans — e.g. rec2033 *"truck
  crane"*, rec2034 *"Link Belt crawler crane"* — are **mobile**; models say OUT for Static.
- Overhead/bridge/gantry cranes labeled **Mobile** by humans — e.g. rec3721 *"5-ton
  overhead crane"*, rec3820 *"double overhead bridge crane"* — are **static**; models say OUT for Mobile.
Models even resolve model numbers (rec3774: *Link Belt LS 218* → crawler → mobile).

**B. Human false negatives (15 reports, unanimous LLM=IN vs human=OUT).** The
report clearly names a crane type or a wind effect and the human still marked OUT:
- rec1226 / rec1510 (Wind): *"pushed into motion by wind shear"*, *"the wind blew the
  boom and load line into the power line"* — plainly wind-caused; human OUT.
- rec3388 *"truck-mounted crane"*, rec3774 (named crawler) — plainly mobile; human OUT.

**C. MAYBE hedges (12 reports).** Humans reserve MAYBE for genuine ambiguity; the
models (following *"lean to a decision"*) commit to OUT:
- Wind-farm / windmill / wind-tower reports where the word *"wind"* is a red herring
  (rec349, rec1361, rec1666) — no wind actually acts on the crane; models say OUT.
- Reports naming *"a crane"* with no type (rec3761, rec4094) — the guideline itself
  says *"if the type is not mentioned, classify OUT,"* so the models follow the rubric
  more literally than the hedging human.

**Direction of error.** Models are slightly liberal on crane-type cohorts
(false-positive IN 21–38 reports) and rarely miss a clearly-typed report. Net: the
residual gap to the human ceiling is dominated by **human labeling noise on the
Mobile/Static distinction**, not model capability.

---

## 4. Practical takeaways

1. **A local 35B model (Qwen3.6) matches frontier models here** (κ 0.75 vs 0.70–0.75)
   and runs the full corpus in ~1 min on 2 GPUs — viable as the production labeler.
2. **Label at the report level**, not the sentence level — it matches how humans
   actually annotated and avoids the propagation artifact.
3. **The LLMs can be used to audit the human labels.** The 33 unanimous
   LLM-vs-human disagreements are a high-value re-review queue; a majority are human
   Mobile/Static mix-ups or missed cranes.
4. **Tighten the guideline** on two points that cause most noise: (a) an explicit
   crane-type table (crawler/truck/wheel/RT = mobile; tower/gantry/overhead/bridge/
   jib = static), and (b) MAYBE usage for "wind tower/windmill" reports.

## Files
- `guidelines.py` — cohort rubric (from the PDF) + prompt builders (sentence & doc).
- `PROMPT.md` — portable prompt to send any LLM.
- `doc_label.py` / `label_crane.py` — Qwen labelers (doc / sentence).
- `compare_doc.py` / `compare_models.py` / `make_table.py` — scoring.
- `out/RESULTS_TABLE.md`, `out/doc_summary.json`, `out/doc_disagreements.jsonl`.
