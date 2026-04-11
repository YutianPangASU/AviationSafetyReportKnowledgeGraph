# ACE-Graph: Aviation Causal Event Graph — Design Document

**Status:** Draft — Sections 1-2 approved in brainstorming; Sections 3-6 pending.
**Created:** 2026-04-10
**Working title:** Aviation Causal Event Graph (ACE-Graph)
**One-line summary:** A unified cross-jurisdiction aviation safety corpus, hierarchical event extraction pipeline, per-category causal graphs, and counterfactual reasoning layer evaluated via leave-one-out probable-cause on NTSB accidents.

---

## 1. Motivation and Scope

### 1.1 Goal

Build an end-to-end, reproducible pipeline that transforms unstructured aviation accident reports into a causal reasoning system. The system should be able to answer, for a real accident, questions of the form: *"If contributing factor X had been absent, would the accident still be likely to occur?"* — i.e., Rung 3 counterfactual reasoning in Pearl's ladder.

### 1.2 Primary research contribution

A **unified pipeline paper** that integrates four recent methodological ingredients into an aviation-safety-specific system:

1. HABERT-style hierarchical event extraction (Zhao et al., 2024, 2025)
2. LLM-generated causal order as expert prior (Vashishtha et al., ICLR 2025)
3. LLM-curated Laplacian similarity prior for concurrent-cause effect estimation (Zhang et al., NeurIPS CaLM 2024)
4. CausalCoT + Pearl's three-rung evaluation on real accidents (Jin et al., CLadder, NeurIPS 2023)

No prior work combines these into an aviation-safety pipeline. The integration itself — plus the cross-jurisdiction corpus construction and the leave-one-out probable-cause benchmark — is the contribution.

### 1.3 Data sources

All four public aviation safety databases are treated as a **single unified corpus**, not as primary-plus-validation silos:

- **NTSB** (US) — `avall.mdb`, rich narratives + curated `Events_Sequence`, `seq_of_events`, `Findings` tables
- **TSB Canada** — public CSVs including `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv`
- **BEA** (France) — scraped notified events from `bea.aero`; optional full-report scrape pass
- **FAA AIDS** (US) — tab-delimited accident/incident records + edited remarks

Cross-jurisdiction harmonization is a first-class problem, not a side task.

### 1.4 Scope decisions (approved in brainstorming)

| Decision | Choice |
|---|---|
| Research contribution | Unified pipeline paper (4-stage) |
| Data sources | All four, merged into one corpus |
| Event labeling | Hybrid: canonical NTSB Phase/Occurrence taxonomy (coarse) + LLM free-form subevents (fine) |
| Causal graph scope | Per-accident-category sub-graphs (one per ICAO CICTT category) |
| Evaluation | Probable-cause leave-one-out, grounded in NTSB `Findings` + `probable_cause` |
| First milestone | Build one category (Loss of Control In-flight, LOC-I) end-to-end before expanding |
| Benchmark release | Yes — curate leave-one-out eval set as a reusable artifact |

### 1.5 Out of scope

- Real-time or operational decision support
- Runway/airspace simulation
- Fleet-level risk dashboards
- Any intervention with safety-of-life implications

---

## 2. High-Level Architecture

Five stages. Stage 0 is the corpus-construction prerequisite; Stages 1-4 mirror Pearl's ladder from raw signal to counterfactual reasoning.

```
Stage 0  Corpus Harmonization
   four raw sources
     -> schema alignment
     -> richness filtering
     -> deduplication
     -> taxonomy harmonization (ICAO CICTT anchor)
   => Unified Aviation Safety Corpus (UASC) + UASC-Rich subset

Stage 1  Hybrid Event Extraction
   UASC-Rich records
     -> HABERT-style coarse labels (Phase / Occurrence)
     -> LLM free-form subevent role extraction (actor, action, object, time)
     -> reconciled against structured Findings / seq_of_events where available
   => Event table (record_id, event_id, event_type, roles, time_marker, confidence)

Stage 2  Per-Accident Temporal Graph
   Event table + narrative time cues + structured sequences
     -> topological ordering within each accident
     -> confidence-weighted temporal edges
   => Per-accident temporal DAGs

Stage 3  Per-Category Causal Graph
   per-accident DAGs, grouped by CICTT category
     -> LLM triplet-prompted causal order over event-type vocabulary (Paper 1)
     -> Laplacian similarity prior over event types (Paper 3)
     -> constraint-based / score-based discovery with the prior
   => One causal DAG per category + backdoor adjustment sets

Stage 4  Counterfactual Reasoning + Evaluation
   per-category causal DAG + factor set of an accident
     -> CausalCoT adapted for aviation (Paper 2)
     -> Answer Rung-2 (intervention) and Rung-3 (counterfactual) queries
   => Leave-one-out probable-cause benchmark
   => Released curated benchmark as standalone dataset artifact
```

### 2.1 Why per-category sub-graphs

A single global DAG over thousands of event types becomes dense and uninterpretable; rare factors get swamped. Per-category sub-graphs (LOC-I, CFIT, RE, MAC, SCF-PP, ...) give the eventual paper a natural per-section structure, keep each graph small enough to validate against domain knowledge, and let rare factors stand out within their category.

### 2.2 Why build LOC-I first

Loss of Control In-flight is:
- The largest single category by fatality count
- Well-documented in both NTSB and ICAO safety literature
- Has clear, often short causal chains that are tractable to validate
- Sample size is large enough to support the LLM-prior-assisted discovery

Success on LOC-I gives an early failure signal. If the pipeline fails on LOC-I, it will fail on rarer categories too.

---

## 3. Stage 0 — Corpus Harmonization and Richness Filtering

### 3.1 Source-by-source reality check

| Source | Narrative depth | Structured events | Cause/findings | Expected pass rate |
|---|---|---|---|---|
| **NTSB** `avall.mdb` | 3 narrative fields (`narr_accp`, `narr_accf`, `narr_cause`), often hundreds of tokens | `Events_Sequence`, `seq_of_events`, `Occurrences` | `Findings` table + `narr_cause` | ~70-80% pass |
| **TSB Canada** CSVs | Structured summary fields, shorter free text | `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv` | Limited structured causes | ~40% pass |
| **BEA** scraped | Title + `summary`; full reports require separate scrape of `detail_url` | None natively | Buried in full report PDFs | ~20-30% after full-report fetch |
| **FAA AIDS** txt | E-files short, redacted remarks | A-file coded fields only | Coded cause fields only | ~10% pass |

**Implication:** NTSB will dominate the reasoning-ready subset. This is acceptable as long as it is documented and the cross-source test splits include non-NTSB records.

### 3.2 Canonical record schema (UASC)

One schema for all four sources. Field set chosen so downstream stages can pull what they need without reading source-specific formats.

```
UASC record
├── ids
│   ├── uasc_id              # stable unified ID
│   ├── source               # {NTSB, BEA, TSB, FAA_AIDS}
│   ├── source_id            # original record key
│   └── dup_cluster_id       # shared across cross-source duplicates
├── context
│   ├── occurrence_date
│   ├── country, location, airport_icao, lat_lon
│   ├── aircraft_make, model, category, engines
│   └── operation_type, flight_phase_raw
├── outcome
│   ├── severity             # {fatal, serious, minor, none, damage_only}
│   ├── fatalities, injuries
│   └── accident_category    # harmonized to ICAO CICTT
├── text
│   ├── narrative_factual    # "what happened" (always required)
│   ├── narrative_cause      # "why it happened" (required for reasoning-ready)
│   └── narrative_analysis   # investigator analysis (optional)
├── structured_events        # list[{order, phase, event_type, time_marker?}]
├── findings                 # list[{finding_type, subject_code, human_readable}]
└── quality
    ├── richness_score       # 0.0 - 1.0
    ├── richness_tier        # {reasoning_ready, partial, metadata_only}
    └── harmonization_flags  # which fields were inferred vs. original
```

Every field earns its place:
- Stage 1 consumes `narrative_*` and `structured_events`
- Stage 2 consumes `structured_events` plus time markers in narratives
- Stage 3 buckets by `accident_category` and uses `findings` for supervision/prior
- Stage 4 evaluates against `findings` + `narrative_cause` in the leave-one-out benchmark

### 3.3 Richness filter — definition of "reasoning-ready"

A record is **reasoning-ready** if and only if it passes all five gates:

1. **Factual narrative present and substantial** — `narrative_factual` ≥ 150 tokens. Screens out FAA AIDS one-liners. Cutoff will be tuned on the NTSB distribution during Stage 0 build.
2. **Cause statement present** — either `narrative_cause` ≥ 50 tokens, *or* ≥ 2 `findings` rows with a subject code. Ensures the "why" exists.
3. **Temporal structure recoverable** — either `structured_events` has ≥ 2 ordered entries, *or* `narrative_factual` contains ≥ 2 explicit temporal markers ("then", "shortly after", "at 14:23", "during cruise"). Ensures we can order sub-events.
4. **Context minimum** — `occurrence_date`, `flight_phase_raw`, `aircraft_category`, and `severity` all non-null.
5. **Harmonizable category** — `accident_category` maps to a CICTT code (original or inferred via rule + LLM classifier).

### 3.4 Richness score

Continuous score in [0, 1]:

```
richness = 0.25 * norm(len(narrative_factual), 150, 800)
         + 0.20 * norm(len(narrative_cause),  50, 400)
         + 0.20 * min(1, n_structured_events / 5)
         + 0.15 * min(1, n_findings / 4)
         + 0.10 * n_temporal_markers_capped
         + 0.10 * (all_context_present ? 1 : 0)
```

Tiers:
- `reasoning_ready` — passes all 5 gates, score ≥ 0.5 → enters the core corpus for Stages 1-4
- `partial` — passes ≥ 3 gates, score ≥ 0.3 → used only for cross-source validation / OOD test splits
- `metadata_only` — excluded from modeling, kept for provenance and citation

### 3.5 Taxonomy harmonization

Anchor on **ICAO CICTT** (Common Taxonomy Team):

- **Accident category:** NTSB already uses CICTT in `Occurrences`; TSB uses ICAO codes; BEA uses French-translated ICAO terminology; FAA AIDS requires rule-based + LLM-assisted mapping from older cause codes.
- **Flight phase:** ICAO flight phase taxonomy (Standing, Taxi, Takeoff, Initial Climb, Climb, Cruise, Descent, Approach, Landing, Emergency). Lookup tables per source.
- **Sub-event types (fine grain):** retain Zhao et al.'s NTSB **Phase / Occurrence / Subject** hierarchy as the canonical fine ontology; no cross-source equivalent exists at that granularity. Non-NTSB records get events projected into NTSB Subject codes via a small LLM-assisted classifier trained on NTSB data.

All mappings are explicit crosswalk tables committed to the repo. Every projection is recorded in `harmonization_flags` so original-vs-inferred is always traceable.

### 3.6 Cross-source deduplication

Cross-jurisdiction duplicates occur (a US accident may appear in both NTSB and FAA AIDS).

Strategy: blocking + pairwise match.
- **Blocking key:** `(occurrence_date ± 1 day, country, aircraft_category)`
- **Match features:** tail-number exact match, aircraft make/model match, location TF-IDF similarity ≥ 0.7, severity agreement
- **Decision rule:** weighted score ≥ threshold → same cluster; borderline cases logged for manual review
- **Cluster resolution:** richest record becomes canonical; other members linked via `dup_cluster_id` and contribute any unique fields

### 3.7 Deliverables

1. **UASC v1.0** — unified parquet file with the canonical schema
2. **UASC-Rich** — the `reasoning_ready` subset (primary input to Stages 1-4)
3. **Harmonization tables** — committed crosswalks for category, phase, subject codes
4. **Deduplication audit log** — cluster IDs + decisions
5. **Corpus statistics report** — per-source counts, richness distribution, category distribution, duplicate rate (also a paper artifact)
6. **Data card** — provenance, licenses, known limitations

### 3.8 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Over-filtering leaves an NTSB-only corpus | `partial` tier preserves cross-source test sets even if non-NTSB records don't enter training |
| BEA full-report scraping is fragile | Treat as optional sub-project; if it fails, BEA drops to `partial` tier with explicit limitation note |
| Fine Subject code projection on non-NTSB records introduces noise | `harmonization_flags` records every inference; Stage 3 uses coarse (Occurrence) labels as primary bucketing, fine Subject only where origin-NTSB |

---

## 4. Stage 1 — Hybrid Event Extraction

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- HABERT model reuse vs. retraining on UASC-Rich
- LLM choice and prompt design for free-form subevent role extraction
- Reconciliation rules between HABERT coarse labels and LLM free-form spans
- How to use NTSB `Findings` / `seq_of_events` as supervision signal
- Output event table schema
- Per-record confidence scoring

---

## 5. Stage 2 — Per-Accident Temporal Graph

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- Temporal edge construction rules
- Handling of partial orderings and simultaneous events
- Time-marker normalization ("2 minutes later" vs. timestamps)
- Merging structured `seq_of_events` with narrative-derived ordering
- Per-accident DAG representation and storage

---

## 6. Stage 3 — Per-Category Causal Graph

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- CICTT category list and sub-graph granularity
- LLM triplet-prompting protocol for causal order (adapting Vashishtha et al. to event-type vocabulary)
- Laplacian similarity prior construction (adapting Zhang et al. to NTSB Subject codes)
- Choice of discovery algorithm (PC vs. CaMML vs. score-based) given the prior
- Edge-weight interpretation and stability analysis
- Per-category validation against published HFACS / ICAO chains

---

## 7. Stage 4 — Counterfactual Reasoning and Evaluation

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- CausalCoT adaptation: prompt templates, estimand selection, backdoor-set construction
- Leave-one-out protocol: exactly how a factor is "removed" and how the counterfactual probability is measured
- Ground-truth construction from NTSB `Findings` + `probable_cause`
- Baselines: retrieval, raw LLM QA, random, graph-without-prior
- Benchmark release: dataset card, splits, metrics, versioning
- Metrics: accuracy per rung, cross-jurisdiction generalization, stratification by category

---

## 8. Evaluation Plan (outline)

- **Primary metric:** leave-one-out probable-cause counterfactual accuracy on NTSB
- **Secondary metrics:** per-category structural agreement with HFACS / ICAO chains; Rung-1 associational accuracy; Rung-2 interventional accuracy
- **Cross-jurisdiction generalization:** held-out test splits from BEA / TSB Canada / FAA AIDS `partial`-tier records
- **Ablations:** (a) remove LLM causal-order prior, (b) remove Laplacian similarity prior, (c) remove free-form fine extraction, (d) global graph vs. per-category
- **Qualitative:** per-category DAG visualizations + case studies on high-profile accidents

---

## 9. References

1. Vashishtha, A. et al. *Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference.* ICLR 2025.
2. Jin, Z. et al. *CLadder: Assessing Causal Reasoning in Language Models.* NeurIPS 2023.
3. Zhang, Q. et al. *Leveraging LLM-Generated Structural Prior for Causal Inference with Concurrent Causes.* NeurIPS CaLM Workshop 2024.
4. Zhao, X. et al. *Hierarchical Multilabel Classification for Fine-Level Event Extraction from Aviation Accident Reports.* INFORMS Journal on Data Science, 2025.
5. Zhao, X. *Hierarchical Sequential Event Prediction and Translation from Aviation Accident Report Data.* ASU Dissertation, 2022.

---

## Appendix A — Open Questions from Brainstorming

These are questions from the brainstorming session that were left open for the next round:

1. Should the richness filter gates (150 token floor, 50 token cause floor) be tuned empirically on NTSB distributions before being fixed?
2. Is BEA full-report scraping worth the extra sub-project, or drop BEA to summaries-only?
3. Build a fresh unified fine ontology, or retain NTSB Phase/Occurrence/Subject as canonical with projection for non-NTSB records (current choice)?
4. Stage 1 topics (Section 4) — pending
5. Stage 2 topics (Section 5) — pending
6. Stage 3 topics (Section 6) — pending
7. Stage 4 topics (Section 7) — pending
