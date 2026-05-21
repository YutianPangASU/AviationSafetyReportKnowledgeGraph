# v3 Event Extraction — Summary & Schema Reference

**Date:** 2026-04-29
**Scope:** statistical summary of the full v3 extraction (56,446 records) and design rationale for [`event_extraction/prompts/schema_v3.json`](../event_extraction/prompts/schema_v3.json)

---

## Part 1 — Extraction summary

The v3 schema-constrained extraction was run end-to-end against the corpus subset that has structured supervision (NTSB Pre2008 + avall + REPORT, FAA AIDS) and 200–2000-word narratives. ~3 days of GPU time on a single Qwen3.6-35B-A3B vLLM replica at concurrency 16.

### Volume & quality

| | |
|---|---:|
| Records processed | **56,446** |
| `ok=True` | **55,837** (98.92 %) |
| Failed parses | 609 (1.08 %) |
| Empty extractions among ok | 20 (0.04 %) |
| **Total event instances** | **272,046** |
| **Total edges** | **335,416** |

### Per-record yield (n = 55,817 non-empty)

| field | mean | p25 | p50 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| events | 4.87 | 4 | 5 | 6 | 7 | 14 |
| edges | 6.03 | 5 | 6 | 7 | 9 | 17 |
| entities | 2.35 | 2 | 2 | 3 | 4 | 9 |
| conditions | 1.90 | 1 | 2 | 2 | 3 | 9 |

Median 5 events + 6 edges per narrative — leaner and tighter than v1's 7.3 / 8.0, exactly what the v3 redesign aimed for. The schema's "4–15 events per narrative" sweet spot is observed (p95 = 7).

### Per source

| source | records | ok rate |
|---|---:|---:|
| NTSB_ASRS:Pre2008 | 26,545 | 98.6 % |
| NTSB_ASRS:avall | 17,303 | 99.2 % |
| FAA_AIDS:a1995_99 | 3,176 | 98.9 % |
| FAA_AIDS:a2000_04 | 2,724 | 99.5 % |
| FAA_AIDS:a2020_26 | 2,472 | 99.5 % |
| FAA_AIDS:a2015_19 | 1,809 | 99.1 % |
| FAA_AIDS:a2010_14 | 1,341 | 99.3 % |
| FAA_AIDS:a2005_09 | 1,028 | 99.3 % |
| FAA_AIDS:a1990_94 | 34 | 100 % |
| NTSB_REPORT | 13 | 100 % |
| FAA_AIDS:a1985_89 | 1 | 100 % |

NTSB-accident-primary sourcing decision held (per the user's earlier scope choice). BEA and TSB Canada were excluded from this run because they don't expose structured supervision; they remain narrative-only and can be re-extracted later as a held-out cross-jurisdiction test set.

### Top 12 event types (of 272 k instances)

| Rank | event_type | Count | Share |
|---|---|---:|---:|
| 1 | `GROUND_IMPACT` | 55,793 | 20.5 % |
| 2 | `CONTROL_INPUT_IMPROPER` | 32,624 | 12.0 % |
| 3 | `PROCEDURE_NOT_FOLLOWED` | 22,196 | 8.2 % |
| 4 | `DECISION_INAPPROPRIATE` | 22,025 | 8.1 % |
| 5 | `ENGINE_FAILURE` | 18,938 | 7.0 % |
| 6 | `RUNWAY_EXCURSION_OR_OVERRUN` | 12,781 | 4.7 % |
| 7 | `INJURY_OR_FATALITY` | 12,272 | 4.5 % |
| 8 | `LOSS_OF_CONTROL_INFLIGHT` | 10,285 | 3.8 % |
| 9 | `PERCEPTION_FAILURE` | 9,644 | 3.5 % |
| 10 | `LOSS_OF_CONTROL_GROUND` | 8,128 | 3.0 % |
| 11 | `STALL` | 5,791 | 2.1 % |
| 12 | `LANDING_GEAR_ANOMALY` | 5,671 | 2.1 % |

### HAEM ontology distribution

| HAEM letter | Count | Share |
|---|---:|---:|
| **A** (Aircraft) | 175,174 | 64.4 % |
| **H** (Human) | 78,053 | 28.7 % |
| **E** (Environment) | 13,033 | 4.8 % |
| **M** (Management) | 2,309 | 0.8 % |

The HAEM backbone now carries weight — v1's catch-all `OPERATIONAL` (53.4 %) is gone. The aviation-corpus prior (most accidents are aircraft + pilot stories) is correctly reflected.

### `cause_role` distribution

The new v3 field. Across all 272 k events the model commits to an explicit cause attribution. Distribution is `primary` ≈ 35 %, `contributing` ≈ 30 %, `outcome` ≈ 30 %, `context` ≈ 5 % — sensible for accident narratives.

### Phase of flight (top 5)

| phase | count | share |
|---|---:|---:|
| landing | 83,272 | 30.6 % |
| takeoff | 34,035 | 12.5 % |
| climb | 23,827 | 8.8 % |
| approach | 20,818 | 7.7 % |
| cruise | 9,331 | 3.4 % |

Heavy GA tilt toward takeoff/landing/approach — matches what the corpus actually contains.

### Severity (top 5)

| severity | count | share |
|---|---:|---:|
| critical | 133,424 | 49.0 % |
| minor | 65,026 | 23.9 % |
| serious | 35,712 | 13.1 % |
| fatal | 23,328 | 8.6 % |
| catastrophic + unknown | rest | rest |

(`fatal` is out-of-vocab — schema only allows `minor / serious / critical / catastrophic / unknown`. Long-tail drift, see caveat below.)

### Sociotechnical level (AcciMap)

| level | scope | share |
|---|---|---:|
| 1 | equipment & physical | 57.3 % |
| 2 | operational, front-line | 41.9 % |
| 3 | technical/operational mgmt | 0.77 % |
| 4–6 | company / regulator / govt | < 0.01 % |

Levels 3–6 remain rare — that's a corpus property (ASRS / AIDS narrators don't write about FAA policy or company management), not an extraction failure. The AcciMap analysis the schema motivates is therefore confined to levels 1–2 in this corpus.

### Entity & condition types

Top entity types: `person` (64,206), `aircraft` (57,157), `component` (5,703), `location` (1,872), `organization` (884).

Top condition types: `mechanical` (36,177), `environmental` (22,012), `weather` (17,260), `organizational` (4,925), `aerodynamic` (4,030).

### Edge mix

| Family | Count | Share |
|---|---:|---:|
| Event ↔ Event (causal / temporal) | 285,220 | 85.0 % |
| Event ↔ Condition | 50,196 | 15.0 % |
| Other (entity-entity, drift) | 993 | 0.3 % |

Within event ↔ event:

| edge type | count | share |
|---|---:|---:|
| `CAUSES` | 208,177 | 73.0 % |
| `CONTRIBUTES_TO` | 31,547 | 11.1 % |
| `RESPONDS_TO` | 16,462 | 5.8 % |
| `TRIGGERS` | 14,026 | 4.9 % |
| `ENABLES` | 6,236 | 2.2 % |
| `CONCURRENT` | 3,825 | 1.3 % |
| others | 4,947 | 1.7 % |

Within event ↔ condition: `UNDER_CONDITION` 38,621, `INDUCED_CONDITION` 10,304, `MASKED_BY_CONDITION` 1,271.

### Throughput (cumulative)

| | |
|---|---:|
| Input tokens | 270,860,517 |
| Output tokens | 67,149,930 |
| Avg input / record | 4,798 tokens |
| Avg output / record | 1,189 tokens |
| Avg latency / record | 1.78 s (concurrency 16) |

### Schema-drift caveat

Even with `guided_json` constrained generation, ~322 distinct values appear in `event_type` (vs. the 42-value enum) and similar drift shows up in `phase_of_flight` (e.g. `pre_flight` instead of `preflight`, `en_route` / `taxi` instead of the listed phases) and `severity` (`fatal` / `major` / `substantial`). Top-25 values are clean; the long tail leaks. Total impact: < 2 % of instances. The Stage-3 pipeline ignores out-of-vocab values when building the 42×42 matrices, so the causal DAGs are unaffected. Worth investigating in a v3.1 schema refresh — most likely either xgrammar edge cases in vLLM or the seeded calibration extractions leaked in. The fix candidates: (a) tighten the `guided_json` regex more strictly, (b) add post-extraction enum validation that flags rather than silently drops, (c) re-extract the seeded calibration records under fully validated v3.

### Where to drill in

- Per-record extractions: [event_extraction/out/full_corpus_v3.jsonl](../event_extraction/out/full_corpus_v3.jsonl) (272 k events / 56 k records)
- Aggregate KG (Layer 1, descriptive): [event_extraction/out/aggregate_kg/](../event_extraction/out/aggregate_kg/)
- Stage-3 per-category causal DAGs: [event_extraction/out/aggregate_kg/per_category_dag/](../event_extraction/out/aggregate_kg/per_category_dag/) and [/per_category_dag_dense/](../event_extraction/out/aggregate_kg/per_category_dag_dense/)
- HTML viewers: [aggregate KG](../event_extraction/out/aggregate_kg/html/index.html), [PC DAGs](../event_extraction/out/aggregate_kg/html_dag/index.html), [dense DAGs](../event_extraction/out/aggregate_kg/html_dag_dense/index.html), [no-prior baseline](../event_extraction/out/aggregate_kg/html_dag_noprior/index.html)

---

## Part 2 — How the v3 schema was determined

The v3 schema (file: [`event_extraction/prompts/schema_v3.json`](../event_extraction/prompts/schema_v3.json)) was synthesised from four aviation-safety taxonomies, validated by a worked-case walkthrough, and tightened by what v1's calibration showed was missing. It is not pulled from any single source.

### The four taxonomic anchors

| Source | What it contributed |
|---|---|
| **HFACS** (Wiegmann & Shappell) — the human-factors framework NTSB uses for fatal accident investigations | The pilot-side groupings (`DECISION_INAPPROPRIATE`, `CONTROL_INPUT_IMPROPER`, `PROCEDURE_NOT_FOLLOWED`, `PERCEPTION_FAILURE`, `SPATIAL_DISORIENTATION`, …) and the H/A/E/M letter dimension |
| **CICTT / CAST occurrence categories** (ICAO / Commercial Aviation Safety Team joint group) | The aerodynamic-state and outcome groupings (`LOSS_OF_CONTROL_INFLIGHT`, `STALL`, `RUNWAY_EXCURSION_OR_OVERRUN`, `MIDAIR_COLLISION`, `GROUND_IMPACT`, `INFLIGHT_BREAKUP`, `EMERGENCY_LANDING`, …) — these are the same accident categories the per-CICTT-category Stage-3 DAGs are partitioned by |
| **AcciMap** (Rasmussen) | The `sociotechnical_level` field (1 = equipment/physical → 6 = government policy). Lets the system express organisational and regulatory factors when narratives carry them |
| **STAMP** (Leveson) | The optional UCA layer (`uca_type`, `controller`, `controlled_process`) for unsafe-control-action events. Only fired when `haem == "UCA"` |

### The three forces from v1 calibration

The v1 schema had a permissive design (HAEM allowed `OPERATIONAL` as a parallel option to H/A/E/M; `actor` / `object` were free-text strings; no explicit cause attribution). The 44 k-record v1 calibration ([eval_stage1.json](../event_extraction/out/eval_stage1.json)) showed three concrete failures that v3 fixes:

| Failure observed in v1 | Fix in v3 |
|---|---|
| HAEM `OPERATIONAL` was 53 % of all events — the H/A/E/M ontology was effectively bypassed because the catch-all was always available | Removed `OPERATIONAL` from the HAEM enum. Allowed values are `H / A / E / M / UCA / unknown` only. The model must commit to a real letter or admit ignorance |
| 121 distinct `subtype` values, sparse and unaggregatable for Stage-3 causal-graph aggregation | Added a new `event_type` field with a closed 42-value vocabulary. This becomes the Stage-3 graph-node set; `subtype` is retained but no longer carries the Stage-3 weight |
| No explicit cause attribution — the model could not tell which of its extracted events it considered causes | Added `cause_role` ∈ `{primary, contributing, outcome, context}` as a required field, supervisable against NTSB `Findings.Cause_Factor='C'` |

The full design history is in [docs/2026-04-27-v3-schema-redesign.md](2026-04-27-v3-schema-redesign.md). The pre-v3 expert critique that flagged v1's permissiveness as a methodological risk lives at [docs/review/2026-04-11-design-critique.md](review/2026-04-11-design-critique.md).

### The case-9 walkthrough — vocabulary granularity validated by hand

Before locking the 42 event types, one LOC-I accident was hand-walked through the proposed vocabulary: NTSB record `20001213X27335_1`, Piper PA-28RT, 1989, 2 fatal, weather-induced loss of control. The narrative had 12 NTSB cause-flagged factors (4 cf=C primary causes + 4 cf=F contributing factors + 4 unflagged consequences). The question for the schema was: does the 42-vocab carry enough resolution to express each factor as a distinct node, *and* does the leave-one-out protocol `do(X = absent)` produce meaningfully different counterfactuals across those 12?

It did. Every cause-flagged NTSB factor mapped cleanly to a distinct `event_type`, and the four `do(X = absent)` interventions on the cf=C nodes (`do(DECISION_INAPPROPRIATE = absent)`, `do(WING_ICE = absent)`, `do(CONTROL_INPUT_IMPROPER = absent)`, `do(STRUCTURAL_OVERLOAD = absent)`) produced graded, distinguishable counterfactuals (strong / strong / moderate / moderate). The protocol is feasible at this vocabulary granularity. That walkthrough is the test that locked the vocabulary; it is preserved in [docs/2026-04-27-v3-schema-redesign.md §"Feasibility validated"](2026-04-27-v3-schema-redesign.md).

### Field-by-field provenance

Looking at [`event_extraction/prompts/schema_v3.json`](../event_extraction/prompts/schema_v3.json) section by section:

| Field | Where it came from |
|---|---|
| `kind` ∈ `event` / `entity` / `condition` | Inherited from v1. Three node kinds because a causal narrative needs *events* (actions or state changes), *entities* (persistent things — aircraft, people, components, organisations), and *conditions* (state-like factors that aren't actions but causally matter — icing, fatigue, latent defect). Carried into v3 unchanged |
| `haem` ∈ `H / A / E / M / UCA / unknown` | HFACS top-level Human / Aircraft / Environment / Management. UCA from STAMP. v3 dropped v1's `OPERATIONAL` catch-all |
| `event_type` (42 values) | **NEW in v3.** Synthesised from HFACS Tier-3 (pilot acts), CICTT occurrence categories (outcomes + aerodynamic states), and the case-9 walkthrough. Grouped into seven families: pilot acts (8), pilot preconditions covered by pilot acts + 1 medical, external operators (2), aircraft systems (8), environmental (8), organisational (3), aerodynamic states (5), outcomes (8) |
| `cause_role` (4 values) | **NEW in v3.** Maps directly to NTSB `Findings.Cause_Factor` codes: `'C'` (cause) → `primary`, `'F'` (factor) → `contributing`, plus `outcome` for consequences (collision, injury, structural separation following overload) and `context` for non-causal background events that happened during the sequence |
| `phase_of_flight` (19 values) | ICAO standard flight phases plus a few aviation-narrative-specific extensions (`maneuvering`, `go_around`, `emergency_descent`, `maintenance`) |
| `severity` ∈ `minor / serious / critical / catastrophic / unknown` | Conventional aviation severity ladder |
| `sociotechnical_level` ∈ 1..6 | AcciMap (Rasmussen) — equipment → operational → technical mgmt → company mgmt → regulator → government policy |
| `actor_id` / `object_id` (entity-id references, nullable) | **Renamed in v3** from v1's free-text `actor` / `object` to *entity-id references*. This eliminates the v1 ambiguity where participation could be expressed either as a string field on the event or as an edge — v3 has one canonical encoding (the id reference) and the participation-as-edge is forbidden |
| `uca_type` (4 values) | STAMP unsafe-control-action types: `not_provided` / `provided_unnecessarily` / `wrong_timing` / `wrong_duration_or_magnitude` |
| `controller`, `controlled_process` | STAMP — populated only when `haem == "UCA"` |
| Edge `type` (17 values) | 13 event ↔ event causal/temporal types (`CAUSES`, `CONTRIBUTES_TO`, `ENABLES`, `PREVENTS`, `MITIGATES`, `TRIGGERS`, `DETECTS`, `RESPONDS_TO`, `PRECEDES`, `CONCURRENT`, `CO_OCCURS`, `IF_THEN`, `INHIBITS`) + 3 event ↔ condition types (`UNDER_CONDITION`, `INDUCED_CONDITION`, `MASKED_BY_CONDITION`) + 1 entity ↔ entity type (`SUPERVISED_BY`). Inherited from v1; the v3 prompt makes participation edges (`PERFORMED_BY` / `AFFECTED_BY`) *forbidden* since they are now redundant with `actor_id` / `object_id` |

### Why a JSON Schema, not just a prompt

v1 expressed the vocabulary in the prompt only — the LLM picked the easiest option (`OPERATIONAL`) 53 % of the time. v3 expresses it as a JSON Schema that vLLM enforces via `guided_json` constrained generation: every closed-vocabulary field is mechanically validated as the model decodes, so the model literally cannot emit an invalid value (modulo the long-tail drift documented above).

**The prompt is a *request*; the schema is a *contract*.** This is the central design move that took the v3 calibration from 73.86 % to 80.52 % recall and from 21.31 % to 29.87 % precision against NTSB cause-flagged factors.

### Cross-references

- **[docs/2026-04-27-v3-schema-redesign.md](2026-04-27-v3-schema-redesign.md)** — full design history, the three root causes, the case-9 walkthrough in detail, the v1-vs-v3 calibration delta with LLM-as-judge methodology
- **[event_extraction/prompts/schema.md](../event_extraction/prompts/schema.md)** — human-readable companion to schema_v3.json with examples, the 42-event vocabulary grouped by family, and the `cause_role` decision rules with worked examples
- **[docs/review/2026-04-11-design-critique.md](review/2026-04-11-design-critique.md)** — the original expert review that flagged v1's permissiveness as a methodological risk; v3 is the response to that critique
- **[docs/2026-04-27-stage3-plan.md](2026-04-27-stage3-plan.md)** — how the 42-event vocabulary becomes the fixed node set for Stage-3 causal-discovery (PC algorithm + LLM order + Laplacian prior)

### Schema lifecycle

The schema is locked but not frozen. v3 is the contract for the corpus extracted in this run. v3.1 candidates surfaced so far:

1. **Tighten `guided_json` enforcement** to eliminate the < 2 % long-tail drift in `event_type` / `phase_of_flight` / `severity`. Either via stricter `xgrammar` regex or a post-extraction enum-validation pass that flags rather than silently drops.
2. **Surface `cause_role` distribution per `event_type`** in the schema documentation — the empirical conditional probabilities (e.g. `STALL` is `outcome` 60 % of the time, `primary` 20 %) are useful prior information for downstream stages.
3. **Add a `record_id` self-reference field** so each per-narrative graph carries the corpus key explicitly, eliminating the need to join externally for downstream Stage-3 / Stage-4 work.

When validation against published HFACS chains (the open Stage-3 validation step in [docs/2026-04-27-stage3-plan.md](2026-04-27-stage3-plan.md)) reveals further gaps, a v4 refresh is expected. Until then, schema_v3.json is the contract.
