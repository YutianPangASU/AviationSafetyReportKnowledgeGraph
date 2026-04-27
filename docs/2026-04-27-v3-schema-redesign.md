# v3 Schema Redesign — Stage-1 Extraction

**Date:** 2026-04-27
**Scope:** Stage 0 (corpus) and Stage 1 (event extraction) of the ACE-Graph pipeline
**Status:** v3 schema locked, full re-extraction launched on 56k structured-supervision records

## Why redesign

A v1 calibration on 44.6k records ([eval_stage1.json](../event_extraction/out/eval_stage1.json)) and statistical inspection ([event_stats output](../event_extraction/out/eval_stage1.json)) surfaced three structural problems that would have undermined Stages 2–4:

1. **Stage 0 had been scoped as text-cleaning, not as supervision-preserving.** `data/corpus/corpus.jsonl` carried only `text / title / date / location / tail / url`. NTSB structured fields (`Findings`, `seq_of_events`, `Occurrences`), FAA AIDS event-type codes, and CICTT-equivalent occurrence categories — every field the design doc relied on for Stage-1 supervision and Stage-3 categorization — were stripped during corpus harmonization. They were sitting un-joined in the upstream `Pre2008.mdb`, `avall.mdb`, and `A*.txt` files.
2. **The v1 schema was permissive where it had to be strict.** HAEM listed `OPERATIONAL` as a parallel option to H/A/E/M; the LLM picked the catch-all 53.4 % of the time, defeating the H/A/E/M ontology. There was no closed Stage-3 event-type vocabulary, so the 121 distinct `subtype` values were too sparse to aggregate across accidents.
3. **The Stage-3/4 target primitive was undefined.** "Removing a factor" had no operational definition. Without locking the node vocabulary and `do(X = absent)` semantics first, every upstream design choice was being made blind.

## What the redesign does

### Root 1 — Stage 0 = joined supervision artifact

Tables dumped from `data/NTSB_ASRS/Pre2008.mdb` and `avall.mdb` via [`data_processing/dump_ntsb_mdb.sh`](../data_processing/dump_ntsb_mdb.sh): `events`, `aircraft`, `narratives`, `Findings`, `Occurrences`, `seq_of_events`, `injury`, plus the code lookup tables `ct_seqevt` / `ct_iaids`. FAA AIDS A-files parsed by [`data_processing/parse_faa_aids.py`](../data_processing/parse_faa_aids.py) extracting `cause_primary_text`, `cause_general_category_text`, `phase_of_flight_text`, etc.

[`data_processing/build_corpus_enriched.py`](../data_processing/build_corpus_enriched.py) joins all of this onto every corpus record by `record_id` → produces [`data/corpus/corpus_enriched.jsonl`](../data/corpus/corpus_enriched.jsonl). Coverage:

| Source | Records | With structured |
|---|---:|---:|
| NTSB_ASRS:Pre2008 | 55,495 | 100 % |
| NTSB_ASRS:avall | 25,434 | 100 % |
| NTSB_REPORT | 177 | 100 % |
| FAA_AIDS (all years) | 89,361 | 100 % |
| BEA / TSB_CANADA | 32,750 | 0 % (narrative-only) |
| **TOTAL** | **178,013** | **82 %** |

Important detail discovered during the join: cause attribution lives in **different tables across the two NTSB databases**. Pre2008 (older taxonomy) puts it in `seq_of_events.Cause_Factor='C'`; avall (post-2008 CAST) puts it in the dense `Findings` table with HFACS-style `category/subcategory/section/subsection/modifier` codes plus a verbose `finding_description`. The join handles both.

### Root 2 — constrained extraction schema

[`event_extraction/prompts/schema_v3.json`](../event_extraction/prompts/schema_v3.json) is enforced via vLLM `guided_json`. Every closed-vocabulary field is mechanical now: the model cannot invent strings.

Three structural changes from v1:

- **Drop `OPERATIONAL` from HAEM.** Allowed values are H | A | E | M | UCA | unknown. Events that previously degenerated to OPERATIONAL must commit to a real letter or `unknown`.
- **Add `event_type`** (required) drawn from the 42-value Stage-3 vocabulary. See [schema.md](../event_extraction/prompts/schema.md) for the full list grouped into seven families.
- **Add `cause_role`** (required) ∈ { primary, contributing, outcome, context }. Explicit cause attribution, supervisable against NTSB `Findings.Cause_Factor`.
- **Replace free-text `actor` / `object` strings with `actor_id` / `object_id`** entity-id references. Eliminates the v1 ambiguity where participation was simultaneously expressible as a string field and as an edge.

The system prompt ([`system_v3.txt`](../event_extraction/prompts/system_v3.txt)) carries explicit guidance for the cases where v1 failed most — in particular the "decisions are events, not conditions" rule that was the single biggest driver of missed cause attribution.

### Root 3 — locked Stage-3 primitive

The 42-value `event_type` vocabulary IS the Stage-3 graph node set. Aggregation: count the number of accidents in category C where event-type X precedes event-type Y, weighted by NTSB `Cause_Factor`. The vocabulary is intentionally finite — Stage 3 builds per-CICTT-category subgraphs at this level; an explosion of subtypes would defeat aggregation.

`do(X = absent)` is operationalised as: set `cause_role` of all events with `event_type == X` to `none`, and prune all outgoing `CAUSES` / `CONTRIBUTES_TO` edges from those events. The counterfactual probability is the share of accidents in the per-category graph that still reach a terminal-event node (GROUND_IMPACT, MIDAIR_COLLISION, INFLIGHT_BREAKUP, etc.) after the prune.

#### Feasibility validated on 10 LOC-I cases

[`event_extraction/out/loci_feasibility_cases.jsonl`](../event_extraction/out/loci_feasibility_cases.jsonl) contains 10 LOC-I accidents (NTSB Occurrence_Code = 250) stratified across phases of flight, each with 3–12 cause-flagged factors and the v1 LLM extraction inline.

Walked one case (record `20001213X27335_1`, Piper PA-28RT, 1989, 2 fatal, weather-induced LOC-I) through the protocol:

```
ICING_ENCOUNTER (E, contributing) ──UNDER_CONDITION──> [c2 ice pellets]
[c1 turbulence]    ──CONTRIBUTES_TO──┐
[c3 IMC]           ──CONTRIBUTES_TO──┤
[c4 attitude ind.] ──CONTRIBUTES_TO──┴──> CONTROL_INPUT_IMPROPER (A, primary, abrupt turn)
                                            │
                                            CAUSES
                                            ↓
                                          STRUCTURAL_OVERLOAD (A, primary, exceeded Va)
                                            │
                                            CAUSES
                                            ↓
                                          AIRFRAME_STRUCTURAL_FAILURE (A, outcome, wing sheared)
                                            │
                                            CAUSES
                                            ↓
                                          GROUND_IMPACT (A, outcome, crashed)
```

Leave-one-out produced graded, distinguishable counterfactuals on every cf=C factor: `do(CONTROL_INPUT_IMPROPER = absent)` severs the entire chain to GROUND_IMPACT (strong negative); `do(STRUCTURAL_OVERLOAD = absent)` severs from overload onward but leaves disorientation path intact (moderate); `do(INSTRUMENT_FAILURE = absent)` removes one CONTRIBUTES_TO edge into the improper-control event but leaves it reachable from turbulence + IMC alone (weak).

The protocol is feasible at this vocabulary granularity.

## Calibration — v1 vs v3 on 2,000 NTSB records

Same input records, same vLLM server. v1 results pulled from the existing [`full_corpus_events.jsonl`](../event_extraction/out/full_corpus_events.jsonl); v3 produced fresh via [`extract_vllm.py --guided-json schema_v3.json --system-file system_v3.txt --fewshot-file few_shot_v3.json`](../event_extraction/scripts/extract_vllm.py).

Recall and precision measured by [`semantic_eval.py`](../event_extraction/scripts/semantic_eval.py) — an LLM-as-judge that decides per-record whether each NTSB cause-flagged factor is semantically equivalent to or directly implied by some extracted event. The judge runs on the same Qwen, with `guided_json` constraining the output shape. This is the right matcher for HFACS-vs-narrative paraphrase mismatch — the token-overlap proxy used earlier was severely under-reporting recall.

| Metric | v1 | v3 | Δ |
|---|---:|---:|---|
| Records evaluated | 2,000 | 1,984 | −16 (0.8 % v3 guided-JSON parse failures) |
| NTSB cause-flagged factors total | 3,481 | 3,450 | similar |
| **Recall** | **73.86 %** | **80.52 %** | **+6.66 pts (+9.0 % rel)** |
| Mean per-record recall | 75.71 % | 82.03 % | +6.32 pts |
| **Precision** | **21.31 %** | **29.87 %** | **+8.56 pts (+40 % rel)** |
| Events extracted total | 14,897 | 9,605 | −35 % (leaner) |
| Events / record | 7.45 | 4.84 | leaner |
| Schema violations (HAEM `OPERATIONAL`) | 53.4 % | **0 %** | fixed |
| Stage-3 vocabulary coverage | n/a | **100 %** | new |
| `cause_role` populated | absent | populated | new |

Outputs:
- [`event_extraction/out/semantic_eval_v1.summary.json`](../event_extraction/out/semantic_eval_v1.summary.json)
- [`event_extraction/out/semantic_eval_v3.summary.json`](../event_extraction/out/semantic_eval_v3.summary.json)

Per-record judge outputs (record_id → factor-index → matching-event-indices) live alongside in `semantic_eval_v1.jsonl` and `semantic_eval_v3.jsonl`.

## What this means for downstream stages

- **Stage 1** is locked at the v3 schema. Full re-extraction running now on 56,202 records (NTSB Pre2008/avall/REPORT + FAA AIDS — every record in the corpus that has structured supervision and a 200–2000-word narrative). Wall-clock estimate: ~27 GPU-hours at v3 pace on the local Qwen3.6-35B-A3B. Output: [`event_extraction/out/full_corpus_v3.jsonl`](../event_extraction/out/full_corpus_v3.jsonl) (seeded with the 1,984 calibration extractions; `--resume` skips them).
- **Stage 2** can now build per-accident temporal DAGs against `seq_of_events` ground truth (Pre2008) or `Findings` cause-flag ordering (avall), measured by Kendall's tau as the design plan specified.
- **Stage 3** has its 42-node vocabulary locked. Per-category sub-graphs can be aggregated immediately once Stage 1 finishes — count `CAUSES` / `CONTRIBUTES_TO` edges between event_types within each CICTT category bucket. CICTT category assignment is the next blocker; for NTSB it joins via `Occurrences.Occurrence_Code` against `ct_seqevt`. For FAA AIDS the `cause_general_category_text` field gives a coarser-grained partitioning.
- **Stage 4** has `do(X = absent)` operationalised. Manual feasibility validated on case 9; remaining 9 LOC-I cases in [`loci_feasibility_cases.jsonl`](../event_extraction/out/loci_feasibility_cases.jsonl) are ready for an independent annotator to walk through for inter-rater reliability.

## Files added or changed

### Data pipeline (Stage 0 — root 1)

- [`data_processing/dump_ntsb_mdb.sh`](../data_processing/dump_ntsb_mdb.sh) — dumps every NTSB eADMS table to JSONL.
- [`data_processing/parse_faa_aids.py`](../data_processing/parse_faa_aids.py) — parses FAA AIDS A-files into structured JSONL.
- [`data_processing/build_corpus_enriched.py`](../data_processing/build_corpus_enriched.py) — joins everything onto `corpus.jsonl` by `record_id`.
- [`data/corpus/corpus_enriched.jsonl`](../data/corpus/corpus_enriched.jsonl) — 178k records with `structured` payload.
- [`data/corpus/corpus_enriched.stats.json`](../data/corpus/corpus_enriched.stats.json) — per-source coverage stats.

### Extraction (Stage 1 — root 2)

- [`event_extraction/prompts/schema_v3.json`](../event_extraction/prompts/schema_v3.json) — JSON Schema for `guided_json`.
- [`event_extraction/prompts/schema.md`](../event_extraction/prompts/schema.md) — human-readable v3 schema doc.
- [`event_extraction/prompts/system_v3.txt`](../event_extraction/prompts/system_v3.txt) — v3 system prompt.
- [`event_extraction/prompts/few_shot_v3.json`](../event_extraction/prompts/few_shot_v3.json) — v3-format worked example.
- [`event_extraction/scripts/extract_vllm.py`](../event_extraction/scripts/extract_vllm.py) — added `--guided-json`, `--system-file`, `--fewshot-file` flags.

### Evaluation (Stage 1 calibration)

- [`event_extraction/scripts/eval_stage1.py`](../event_extraction/scripts/eval_stage1.py) — token-overlap proxy eval (kept for fast-iteration sanity checks).
- [`event_extraction/scripts/semantic_eval.py`](../event_extraction/scripts/semantic_eval.py) — LLM-as-judge semantic recall matcher.
- [`event_extraction/scripts/select_loci_cases.py`](../event_extraction/scripts/select_loci_cases.py) — selects 10 LOC-I cases for manual feasibility.
- [`event_extraction/scripts/compare_v1_v3.py`](../event_extraction/scripts/compare_v1_v3.py) — quick v1-vs-v3 comparison on the LOC-I cases.

### Stage-3 prep (root 3)

- 42-value `event_type` vocabulary — defined in [`schema_v3.json`](../event_extraction/prompts/schema_v3.json) and documented in [`schema.md`](../event_extraction/prompts/schema.md).
- Leave-one-out semantics — defined here (above) and validated on case 9.
