# ACE-Graph: Aviation Causal Event Graph — Design Document (v2)

**Status:** Draft v2 — Section 3 revised per review; Sections 4-7 remain TBD stubs with expanded planned topics; new sections added for evaluation, related work, limitations, reproducibility, and ethics.
**Created:** 2026-04-10 | **Revised:** 2026-04-11
**Working title:** Aviation Causal Event Graph (ACE-Graph)
**One-line summary:** A unified cross-jurisdiction aviation safety corpus, hierarchical event extraction pipeline, per-category causal graphs, and LLM-assisted counterfactual question answering evaluated via leave-one-out agreement with NTSB investigator judgments.

---

## 1. Motivation and Scope

### 1.1 Goal

Build an end-to-end, reproducible pipeline that transforms unstructured aviation accident reports into a causal reasoning system. The system should be able to answer, for a real accident, questions of the form: *"If contributing factor X had been absent, would the accident still be likely to occur?"*

This is LLM-assisted counterfactual question answering grounded in a learned causal graph structure. While the question targets Pearl's Rung 3 (counterfactual reasoning), the system does not perform formal counterfactual inference via abduction-action-prediction on a fully parameterized structural causal model (SCM). Instead, it combines a learned causal graph with LLM-generated counterfactual judgments, evaluated against NTSB investigator judgments as a proxy for causal ground truth. This distinction is discussed further in Section 10 (Limitations).

### 1.2 Primary research contribution

A **unified pipeline paper** that integrates four recent methodological ingredients into an aviation-safety-specific system:

1. HABERT-style hierarchical event extraction (Zhao et al., 2024, 2025)
2. LLM-generated causal order as expert prior (Vashishtha et al., ICLR 2025)
3. LLM-curated Laplacian similarity prior for concurrent-cause effect estimation (Zhang et al., NeurIPS CaLM 2024)
4. CLadder-inspired evaluation framework for causal QA on real accidents (Jin et al., NeurIPS 2023)

No prior work combines these into an aviation-safety pipeline. The integration itself — plus the cross-jurisdiction corpus construction and the leave-one-out probable-cause benchmark — is the contribution.

**Ingredient compatibility as a contribution.** These four methods were developed for different settings: HABERT for NTSB-specific event ontologies, Vashishtha et al. for general causal inference benchmarks, Zhang et al. under specific SCM assumptions (e.g., additive noise), and CLadder as an evaluation framework (not a reasoning method). Demonstrating that they can be composed into a coherent aviation-safety pipeline — and characterizing where their assumptions hold or must be relaxed — is itself a methodological contribution. Section 9 (Related Work) discusses each ingredient's assumptions and the adaptations required.

### 1.3 Data sources

All four public aviation safety databases are treated as a **single unified corpus**, not as primary-plus-validation silos:

- **NTSB** (US) — `avall.mdb`, rich narratives + curated `Events_Sequence`, `seq_of_events`, `Findings` tables
- **TSB Canada** — public CSVs including `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv`
- **BEA** (France) — scraped notified events from `bea.aero`; optional full-report scrape pass
- **FAA AIDS** (US) — tab-delimited accident/incident records + edited remarks

Cross-jurisdiction harmonization is a first-class problem, not a side task.

**Honest framing on source balance.** While all four sources contribute to the unified schema, NTSB records are expected to dominate the reasoning-ready subset (~70-80% pass rate vs. 10-40% for other sources). The cross-jurisdiction contribution is the harmonization *methodology* and unified *schema*, not a balanced multi-source corpus. Cross-jurisdiction evaluation uses partial-tier non-NTSB records as out-of-distribution test sets to measure pipeline transfer, not as equal-weight training data.

### 1.4 Scope decisions (approved in brainstorming)

| Decision | Choice |
|---|---|
| Research contribution | Unified pipeline paper (4-stage) |
| Data sources | All four, merged into one corpus |
| Event labeling | Hybrid: canonical NTSB Phase/Occurrence taxonomy (coarse) + LLM free-form subevents (fine) |
| Causal graph scope | Per-accident-category sub-graphs (one per ICAO CICTT category) |
| Evaluation | Leave-one-out agreement with NTSB investigator judgment (`Findings` + `probable_cause`) |
| First milestone | Build one category (Loss of Control In-flight, LOC-I) end-to-end before expanding |
| Benchmark release | Yes — curate leave-one-out eval set as a reusable artifact |

### 1.5 Out of scope

- Real-time or operational decision support
- Runway/airspace simulation
- Fleet-level risk dashboards
- Any intervention with safety-of-life implications
- Formal SCM parameterization (functional form estimation)

---

## 2. High-Level Architecture

Five stages. Stage 0 is the corpus-construction prerequisite; Stages 1-4 progress from raw data toward counterfactual question answering, loosely corresponding to increasing levels of causal reasoning.

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
     -> LLM triplet-prompted causal order over event-type vocabulary
     -> Laplacian similarity prior over event types
     -> constraint-based / score-based discovery with the prior
   => One causal DAG per category + backdoor adjustment sets

Stage 4  Counterfactual QA + Evaluation
   per-category causal DAG + factor set of an accident
     -> CausalCoT adapted for aviation
     -> Answer interventional and counterfactual queries
        (evaluated as agreement with investigator judgment,
         not formal SCM computation)
   => Leave-one-out probable-cause benchmark
   => Released curated benchmark as standalone dataset artifact
```

### 2.1 Why per-category sub-graphs

A single global DAG over thousands of event types becomes dense and uninterpretable; rare factors get swamped. Per-category sub-graphs (LOC-I, CFIT, RE, MAC, SCF-PP, ...) give the eventual paper a natural per-section structure, keep each graph small enough to validate against domain knowledge, and let rare factors stand out within their category.

**Multi-label handling.** Some accidents involve multiple CICTT categories simultaneously (e.g., LOC-I triggered by SCF-PP). We assign each accident a *primary* category and record *secondary* categories as metadata. Stage 3 uses primary assignment for sub-graph construction but also constructs a small set of cross-category bridge edges representing known multi-category causal pathways.

**Minimum sample size.** Per-category causal discovery requires sufficient sample size for stable graph estimation. Categories below a minimum threshold (to be determined empirically during Stage 0, expected ~50-100 reasoning-ready records) will be excluded from per-category analysis or merged into an "Other" bucket. The paper will report results only for categories meeting this threshold.

### 2.2 Why build LOC-I first

Loss of Control In-flight is:
- The largest single category by fatality count (CICTT, 2024)
- Well-documented in both NTSB and ICAO safety literature
- Has clear, often short causal chains that are tractable to validate
- Sample size is large enough to support the LLM-prior-assisted discovery

Success on LOC-I gives an early failure signal. If the pipeline fails on LOC-I, it will fail on rarer categories too.

### 2.3 Cross-stage consistency checks

The feed-forward pipeline has no inherent error-correction mechanism. We define three explicit consistency checks to detect error propagation:

1. **Stage 1 -> Stage 2 (event vocabulary alignment):** After Stage 2 produces temporal DAGs, verify that the event vocabulary in the DAGs is a subset of Stage 1's extraction output. Flag any orphaned event types that appear in structured data but were missed by extraction.

2. **Stage 2 -> Stage 3 (temporal support for causal edges):** After Stage 3 produces causal graphs, verify that high-confidence causal edges (top quartile by weight) are supported by temporal ordering in Stage 2. A causal edge A -> B where B precedes A in >50% of per-accident temporal DAGs is flagged for manual review.

3. **Stage 3 -> Stage 4 (counterfactual graph consistency):** After Stage 4 generates counterfactual answers, verify that removing a non-ancestor node in the causal graph does not change the predicted outcome. Violations indicate the LLM counterfactual generator is not faithful to the graph structure.

### 2.4 Error propagation awareness

The four-stage pipeline means errors in Stage N propagate to all downstream stages. This is a known limitation of feed-forward architectures. We mitigate this by:

- Reporting per-stage evaluation metrics (Section 8), not just end-to-end results
- Committing to an error-propagation sensitivity analysis: artificially injecting noise at each stage boundary and measuring downstream impact
- Considering a joint/iterative Stage 1-2 approach (where temporal structure informs event extraction and vice versa) as a future extension if the feed-forward results are unsatisfactory

### 2.5 LLM usage policy and circularity mitigation

LLMs are used in multiple stages, creating a risk of correlated biases propagating through the pipeline. The table below catalogs every LLM usage point and the mitigation strategy.

| Stage | LLM Role | Model Family | Structured-Data Fallback | Conflict Resolution |
|---|---|---|---|---|
| 0 | CICTT category classification for FAA AIDS records | Family A (e.g., GPT-4 class) | Rule-based mapping from FAA cause codes | Rules primary; LLM supplements ambiguous cases |
| 1 | Free-form subevent role extraction | Family A | NTSB `Findings` + `seq_of_events` | **Structured data is primary.** LLM output supplements when structured data is absent or ambiguous. When both exist and disagree, structured data wins; disagreement is logged. |
| 3 | Triplet-prompted causal order prior | Family B (e.g., Claude class) | None (this is the LLM-prior contribution) | Ablation baseline: Stage 3 without LLM prior (PC algorithm on event co-occurrence) |
| 3 | Laplacian similarity prior curation | Family B | None | Same ablation baseline |
| 4 | CausalCoT counterfactual QA | Family B | N/A — this is the inference target | Evaluated against investigator judgment; single-stage LLM baseline (see Section 8) |

**Key policies:**

1. **Model family diversification.** Extraction stages (0, 1) and reasoning stages (3, 4) must use different model families to reduce correlated bias. If the same LLM's misconceptions about aviation causality (e.g., over-weighting pilot error, under-weighting maintenance failures — a known bias in public discourse that LLMs absorb) shape both extraction *and* causal reasoning, the evaluation measures self-consistency rather than causal reasoning quality (Sheth et al., 2025).

2. **Structured data primacy.** Where structured data exists (NTSB `Findings`, `seq_of_events`, `Occurrences`), it is the primary signal. LLM output is supplementary. This is enforced in Stage 1 reconciliation and Stage 3 prior construction.

3. **Mandatory ablations.** The evaluation (Section 8) includes a no-LLM-prior baseline (pure constraint-based discovery from structured data) to isolate the LLM contribution from the data contribution.

---

## 3. Stage 0 — Corpus Harmonization and Richness Filtering

### 3.1 Source-by-source reality check

| Source | Narrative depth | Structured events | Cause/findings | Expected pass rate |
|---|---|---|---|---|
| **NTSB** `avall.mdb` | 3 narrative fields (`narr_accp`, `narr_accf`, `narr_cause`), often hundreds of words | `Events_Sequence`, `seq_of_events`, `Occurrences` | `Findings` table + `narr_cause` | ~70-80% pass |
| **TSB Canada** CSVs | Structured summary fields, shorter free text | `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv` | Limited structured causes | ~40% pass |
| **BEA** scraped | Title + `summary`; full reports require separate scrape of `detail_url` | None natively | Buried in full report PDFs | ~20-30% with full-report fetch; **~5% summaries-only** |
| **FAA AIDS** txt | E-files short, redacted remarks | A-file coded fields only | Coded cause fields only | ~10% pass |

**Implication.** NTSB will dominate the reasoning-ready subset (likely 80-90%). This is explicitly documented and the paper will frame the cross-jurisdiction contribution as the harmonization *methodology*, not a balanced multi-source evaluation. Cross-source test splits will include non-NTSB `partial`-tier records to measure pipeline transfer.

**BEA realistic projection.** The 20-30% pass rate is conditional on successful full-report scraping from `detail_url`. Since scraping is marked as optional (Section 3.8), the realistic BEA pass rate without it is ~5%. The paper will report both scenarios.

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
│   ├── accident_category_primary   # harmonized to ICAO CICTT
│   └── accident_category_secondary # list of additional CICTT codes (may be empty)
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

Changes from v1: `accident_category` split into `accident_category_primary` and `accident_category_secondary` to support multi-label category assignment (Section 2.1).

Every field earns its place:
- Stage 1 consumes `narrative_*` and `structured_events`
- Stage 2 consumes `structured_events` plus time markers in narratives
- Stage 3 buckets by `accident_category_primary` and uses `findings` for supervision/prior; `accident_category_secondary` informs cross-category bridge edges
- Stage 4 evaluates against `findings` + `narrative_cause` in the leave-one-out benchmark

### 3.3 Richness filter — definition of "reasoning-ready"

A record is **reasoning-ready** if and only if it passes all five gates:

1. **Factual narrative present and substantial** — `narrative_factual` >= 120 words (tokenizer-independent). Screens out FAA AIDS one-liners. Cutoff will be tuned on the NTSB word-count distribution during Stage 0 build.
2. **Cause statement present** — either `narrative_cause` >= 40 words, *or* >= 2 `findings` rows with a subject code. Ensures the "why" exists.
3. **Temporal structure recoverable** — either `structured_events` has >= 2 ordered entries, *or* a temporal relation classifier identifies >= 2 temporal relations in `narrative_factual`. The initial implementation uses keyword matching ("then", "shortly after", "at [time]", "during [phase]") as a proxy; the production implementation will use a TimeML/TempEval-style temporal relation classifier (Pustejovsky et al., 2003; Verhagen et al., 2010) to reduce false positives from non-temporal uses of keywords like "then."
4. **Context minimum** — `occurrence_date`, `flight_phase_raw`, `aircraft_category`, and `severity` all non-null.
5. **Harmonizable category** — `accident_category_primary` maps to a CICTT code (original or inferred via rule + LLM classifier).

**Note on word counts vs. token counts.** v1 used token counts (150, 50), which are tokenizer-dependent. v2 uses word counts (120, 40) to ensure thresholds are stable across model choices. The approximate equivalence is 1 word ~ 1.3 tokens for English text.

### 3.4 Richness score

Continuous score in [0, 1]:

```
richness = w1 * lognorm(wordcount(narrative_factual), 120, 600)
         + w2 * lognorm(wordcount(narrative_cause),   40, 300)
         + w3 * min(1, n_structured_events / 5)
         + w4 * min(1, n_findings / 4)
         + w5 * min(1, n_temporal_relations / 3)
         + w6 * (all_context_present ? 1 : 0)
```

where:

```
lognorm(x, lo, hi) = clip((log(x) - log(lo)) / (log(hi) - log(lo)), 0, 1)
```

The log transform reflects diminishing marginal information gain from longer narratives: the difference between 120 and 200 words is more informative than the difference between 500 and 600 words.

**Default weights:** `w1=0.25, w2=0.20, w3=0.20, w4=0.15, w5=0.10, w6=0.10`. These are a **default configuration, not principled values**. They reflect a prior belief that narrative content (w1+w2=0.45) is more informative than structured metadata (w3+w4=0.35) or auxiliary signals (w5+w6=0.20). Appendix B describes the sensitivity analysis protocol that will be run during Stage 0 construction to validate or revise these weights.

**Limitations of the additive form.** The weighted sum assumes linear substitutability between components: a record with a long narrative and no structured events scores similarly to one with a short narrative and many structured events. In practice, some components are likely *complementary* (narrative + structured events together are worth more than the sum) while others may be *redundant* (narrative_cause and findings often encode the same information). As an alternative, we will also test a **k-of-n dominance filter**: a record is "reasoning-ready" if it exceeds component-specific thresholds on at least k of n criteria (e.g., 4 of 6), without collapsing to a single scalar. The final choice will be reported with justification.

**Tier thresholds.** Rather than fixing arbitrary thresholds (v1 used 0.5 and 0.3), we will set tier boundaries at natural break points in the richness score distribution, identified via Jenks natural breaks or KDE valley detection on the NTSB distribution. We report `score >= 0.5` and `score >= 0.3` as initial values pending this analysis.

Tiers:
- `reasoning_ready` — passes all 5 gates, score >= threshold_high -> enters the core corpus for Stages 1-4
- `partial` — passes >= 3 gates, score >= threshold_low -> used only for cross-source validation / OOD test splits
- `metadata_only` — excluded from modeling, kept for provenance and citation

### 3.5 Taxonomy harmonization

Anchor on **ICAO CICTT** (Common Taxonomy Team; CAST/ICAO, 2017):

- **Accident category:** NTSB already uses CICTT in `Occurrences`; TSB uses ICAO codes; BEA uses French-translated ICAO terminology; FAA AIDS requires rule-based + LLM-assisted mapping from older cause codes.
- **Flight phase:** ICAO flight phase taxonomy (Standing, Taxi, Takeoff, Initial Climb, Climb, Cruise, Descent, Approach, Landing, Emergency). Lookup tables per source.
- **Sub-event types (fine grain):** retain Zhao et al.'s NTSB **Phase / Occurrence / Subject** hierarchy as the canonical fine ontology; no cross-source equivalent exists at that granularity. Non-NTSB records get events projected into NTSB Subject codes via a small LLM-assisted classifier trained on NTSB data. Projected codes are treated as a **separate, lower-confidence tier** in Stage 3 to prevent noise from contaminating the fine-grained event vocabulary.

**FAA AIDS mapping validation.** The FAA AIDS -> CICTT mapping will be validated on dual-coded records: accidents that appear in both NTSB and FAA AIDS with known CICTT codes in NTSB. This provides an empirical accuracy estimate for the rule-based + LLM mapping.

**Multi-label assignment.** Accidents that span multiple CICTT categories (e.g., LOC-I + SCF-PP) are assigned a primary category based on the most proximate occurrence and secondary categories based on contributing factors. Both are recorded in the UASC schema (Section 3.2).

All mappings are explicit crosswalk tables committed to the repo. Every projection is recorded in `harmonization_flags` so original-vs-inferred is always traceable.

### 3.6 Cross-source deduplication

Cross-jurisdiction duplicates occur (a US accident may appear in both NTSB and FAA AIDS).

Strategy: blocking + pairwise match.
- **Blocking key:** `(occurrence_date +/- 1 day, country, aircraft_category)`. **Known limitation:** records where the date is recorded differently (occurrence date vs. report date) or where aircraft_category is coded differently across sources may be missed. As a complementary approach, we also apply LSH (locality-sensitive hashing) on concatenated text fields to catch blocking-key misses.
- **Match features:** tail-number exact match, aircraft make/model match, location text similarity, severity agreement
- **Decision rule:** weighted score >= threshold -> same cluster; threshold to be set via validation on a manually reviewed sample of known cross-source duplicates (not a fixed arbitrary value). Borderline cases logged for manual review.
- **Cluster resolution:** richest record becomes canonical; other members linked via `dup_cluster_id` and contribute any unique fields

### 3.7 Deliverables

1. **UASC v1.0** — unified parquet file with the canonical schema
2. **UASC-Rich** — the `reasoning_ready` subset (primary input to Stages 1-4)
3. **Harmonization tables** — committed crosswalks for category, phase, subject codes
4. **Deduplication audit log** — cluster IDs + decisions
5. **Corpus statistics report** — per-source counts, richness distribution, category distribution, duplicate rate (also a paper artifact)
6. **Data card** — provenance, licenses, known limitations
7. **Richness score analysis report** — sensitivity analysis results, final weight selection rationale, tier threshold justification (Appendix B)

### 3.8 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Over-filtering leaves an NTSB-only corpus | `partial` tier preserves cross-source test sets even if non-NTSB records don't enter training. Paper frames NTSB dominance honestly (Section 1.3). |
| BEA full-report scraping is fragile | Treat as optional sub-project; if it fails, BEA drops to `partial` tier with explicit limitation note. Paper reports both with-scraping and without-scraping pass rates. |
| Fine Subject code projection on non-NTSB records introduces noise | `harmonization_flags` records every inference; Stage 3 uses coarse (Occurrence) labels as primary bucketing, fine Subject only where origin-NTSB. Projected codes treated as lower-confidence tier. |
| Richness score weights are arbitrary | Sensitivity analysis (Appendix B) validates or revises weights before proceeding past Stage 0. Alternative k-of-n filter tested. |
| FAA AIDS category mapping is noisy | Validated on dual-coded NTSB/FAA AIDS records. Noisy records flagged in `harmonization_flags`. |

---

## 4. Stage 1 — Hybrid Event Extraction

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- HABERT model reuse vs. retraining on UASC-Rich
- LLM choice and prompt design for free-form subevent role extraction. **Constraint: must use a different model family from Stages 3-4 to mitigate LLM circularity (Section 2.5).**
- Reconciliation rules between HABERT coarse labels and LLM free-form spans. **Conflict resolution: when HABERT labels and LLM extraction disagree with NTSB structured `Findings`, `Findings` are primary.**
- How to use NTSB `Findings` / `seq_of_events` as supervision signal
- Output event table schema
- Per-record confidence scoring
- **Per-stage evaluation: extraction precision/recall/F1 measured against NTSB `Findings` and `seq_of_events` on a held-out evaluation set (Section 8).**

---

## 5. Stage 2 — Per-Accident Temporal Graph

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- Temporal edge construction rules
- Handling of partial orderings and simultaneous events
- Time-marker normalization ("2 minutes later" vs. timestamps)
- **Temporal relation extraction: evaluate TimeML/TempEval-based classifiers (Pustejovsky et al., 2003; Verhagen et al., 2010) in addition to keyword matching for temporal marker detection. Keywords are fragile — "then" appears in non-temporal contexts frequently.**
- Merging structured `seq_of_events` with narrative-derived ordering
- Per-accident DAG representation and storage
- **Per-stage evaluation: temporal ordering agreement with `seq_of_events` on NTSB records, measured via Kendall's tau (Section 8).**
- **Consistency check with Stage 1: verify event vocabulary alignment between extraction output and temporal DAG nodes (Section 2.3).**

---

## 6. Stage 3 — Per-Category Causal Graph

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- CICTT category list and sub-graph granularity
- **Minimum sample-size threshold per category:** categories below threshold excluded or merged into "Other." Threshold to be set empirically during Stage 0 corpus analysis.
- LLM triplet-prompting protocol for causal order (adapting Vashishtha et al. to event-type vocabulary)
- **Assumption audit for Vashishtha et al.:** What assumptions does the method make about expert quality? Do LLMs, as aviation-domain "experts," satisfy these assumptions? If not, what relaxations are needed?
- Laplacian similarity prior construction (adapting Zhang et al. to NTSB Subject codes)
- **Assumption audit for Zhang et al.:** What SCM assumptions (linearity, additive noise) does the Laplacian prior require? Are these compatible with aviation causal structure, where interactions are often nonlinear and multi-factorial?
- Choice of discovery algorithm (PC vs. CaMML vs. score-based) given the prior
- Edge-weight interpretation and stability analysis
- **Multi-label category handling: cross-category bridge edges for accidents assigned to multiple CICTT categories (Section 2.1).**
- Per-category validation against published HFACS classifications (Shappell & Wiegmann, 2000, 2003) and ICAO causal chains
- **No-LLM-prior ablation: PC algorithm on structured event co-occurrence data alone, as a mandatory baseline isolating the LLM contribution (Section 2.5).**
- **Per-stage evaluation: edge precision/recall against domain expert judgment on a manually annotated subset (Section 8).**

---

## 7. Stage 4 — Counterfactual Reasoning and Evaluation

**Status:** TBD — to be brainstormed in next session, but the leave-one-out protocol formalization below is a **must-complete prerequisite** before building Stages 1-3.

### 7.1 Leave-one-out protocol (formalized sketch)

**Definition.** "Removing a factor" is formalized as a do-calculus intervention `do(X = absent)` on the per-category causal DAG, where X is a node representing a contributing factor identified in the NTSB `Findings` table. This is *not* node deletion (which would alter graph structure) but a Pearl-style intervention that fixes the value of X while preserving the rest of the graph.

**Counterfactual estimand.** For each accident a with factor set F_a and outcome Y_a:

```
P(Y_a = accident | do(X_i = absent), F_a \ {X_i}, graph_category)
```

The system predicts whether the accident would still have occurred if factor X_i had been absent, given the remaining factors and the causal graph for the accident's category.

**Feasibility check.** Before building Stages 1-3, we will manually construct causal graphs for 10 LOC-I accidents from NTSB, apply the leave-one-out protocol by hand, and verify that:
- The protocol produces meaningful distinctions (not all factors are equally "necessary")
- The counterfactual question is answerable given the information available
- The manual results are stable across two independent annotators

If this feasibility check fails, the evaluation design must be revised before proceeding.

### 7.2 Planned topics to resolve
- CausalCoT adaptation: prompt templates, estimand selection, backdoor-set construction
- Practical implementation of `do(X = absent)`: how to approximate the counterfactual probability given a learned graph structure without a fully parameterized SCM
- **Expanded baselines (see Section 8.3)**
- **Ground-truth framing:** Evaluation is against NTSB investigator judgment (`Findings` + `probable_cause`), *not* objective causal truth. Three known biases in this proxy:
  1. **Investigator attribution bias:** NTSB probable cause reflects regulatory/legal judgment under time pressure and institutional norms, not purely scientific causal determination (see HFACS literature: Shappell & Wiegmann, 2003).
  2. **Narrative-findings non-independence:** The same investigators who wrote the narrative also wrote the findings. A system trained on narratives and evaluated against findings from the same document partly measures textual consistency.
  3. **Jurisdictional selection bias:** NTSB systematically attributes causes to factors that are *actionable* within FAA jurisdiction. Systemic, organizational, or regulatory causes are under-represented relative to their true causal role.
- **Secondary evaluation sources:** HFACS classifications by trained human factors analysts; published re-analyses of high-profile accidents where the causal story is well-established and independently verified. Commit to inter-rater reliability study on a subset of 50+ accidents.
- Benchmark release: dataset card, splits, metrics, versioning
- Metrics: accuracy per reasoning level, cross-jurisdiction generalization, stratification by category

---

## 8. Evaluation Plan

### 8.1 Per-stage evaluation metrics

Each pipeline stage is evaluated independently, not just end-to-end. This enables diagnosis of *where* the pipeline fails when final numbers are unsatisfactory.

| Stage | Metric | Gold Standard | Measurement |
|---|---|---|---|
| 0 — Harmonization | Category assignment accuracy | Dual-coded NTSB/FAA AIDS records with known CICTT codes | Precision/recall on CICTT code assignment |
| 1 — Event Extraction | Event extraction P/R/F1 | NTSB `Findings` + `seq_of_events` on held-out set | Token-level and span-level P/R/F1 |
| 2 — Temporal Graph | Temporal ordering accuracy | NTSB `seq_of_events` ordering | Kendall's tau rank correlation |
| 3 — Causal Graph | Causal edge quality | Domain expert judgment on manually annotated subset (n >= 50 edges per category) | Edge-level precision/recall vs. expert annotations |
| 4 — Counterfactual QA | Agreement with investigator judgment | NTSB `Findings` + `probable_cause` | Leave-one-out protocol accuracy (Section 7.1) |

### 8.2 Primary metric

**Leave-one-out agreement with NTSB investigator judgment** on the LOC-I category, then extended to additional categories.

This is *agreement with investigator judgment*, not causal accuracy. The distinction matters: NTSB findings encode regulatory perspective and institutional norms, not objective causal ground truth. The paper will frame the metric accordingly and discuss the implications in Section 10 (Limitations).

### 8.3 Baselines

| Baseline | What it tests |
|---|---|
| **Random** | Lower bound |
| **Retrieval-based** | Can similar-accident lookup answer counterfactual questions? |
| **Raw LLM QA** | Give the LLM the narrative directly and ask the counterfactual question. No pipeline. | 
| **Single-stage LLM** | Give GPT-4/Claude the full narrative + findings and ask for counterfactual directly. **Critical baseline:** if this matches the pipeline, the pipeline complexity is unjustified. |
| **NTSB structured data only** | Use `Findings` + `seq_of_events` directly, no NLP. How much does the pipeline add over what's already in the database? |
| **Graph-without-LLM-prior** | Stage 3 causal graph built via PC algorithm on event co-occurrence, no LLM priors. Isolates LLM contribution. |
| **Graph-without-Laplacian** | Remove Laplacian similarity prior only |
| **Global graph** | Single global DAG instead of per-category sub-graphs |
| **No fine extraction** | Remove LLM free-form extraction, use only HABERT coarse labels |

### 8.4 Cross-jurisdiction evaluation

Held-out test splits from BEA / TSB Canada / FAA AIDS `partial`-tier records. This tests whether the NTSB-trained pipeline transfers to non-NTSB text. It does *not* test independent causal ground truth from other jurisdictions (which is unavailable at the structured level).

### 8.5 Qualitative evaluation

Per-category DAG visualizations + case studies on high-profile accidents (e.g., Colgan Air 3407 for LOC-I, AF447 for pilot-automation interaction). Case studies include comparison with published NTSB probable cause and HFACS re-analyses.

---

## 9. Related Work

### 9.1 Aviation safety analysis frameworks

The dominant frameworks for systematic aviation accident analysis are manual, expert-driven taxonomies:

- **HFACS** (Human Factors Analysis and Classification System; Shappell & Wiegmann, 2000, 2003; Wiegmann & Shappell, 2003). Based on Reason's Swiss cheese model, HFACS classifies human causal factors at four levels: unsafe acts, preconditions for unsafe acts, unsafe supervision, and organizational influences. HFACS is the standard human-factors taxonomy used by NTSB and military aviation. ACE-Graph's Stage 3 per-category causal graphs can be validated against HFACS-classified accidents for structural agreement.

- **STAMP/STPA** (Systems-Theoretic Accident Model and Processes; Leveson, 2004, 2011). STAMP treats safety as a dynamic control problem rather than a failure prevention problem. STPA (Systems-Theoretic Process Analysis) identifies hazards from unsafe control actions. STAMP's systems-theoretic perspective is complementary to ACE-Graph's event-chain approach: STAMP captures systemic/organizational causes that event chains may miss.

- **Rasmussen's AcciMap** (Rasmussen, 1997). Models accident causation across multiple sociotechnical levels (regulatory, organizational, workplace). AcciMap's multi-level perspective motivates our consideration of systemic factors beyond the proximate event chain.

- **ICAO CICTT** (Common Taxonomy Team; CAST/ICAO, 1999-present). The standard occurrence category and flight phase taxonomy used across jurisdictions. ACE-Graph anchors on CICTT for cross-source harmonization (Section 3.5).

ACE-Graph differs from these frameworks by *automating* extraction and causal graph construction from unstructured text, rather than requiring manual expert classification. The evaluation (Section 8) validates against HFACS and CICTT as reference standards.

### 9.2 Aviation NLP and text mining

NLP has been applied to aviation safety reports with increasing sophistication:

- **Topic modeling and classification.** Early work applied LSA, LDA, and SVD to ASRS and NTSB narratives for topic discovery and report classification (Tanguy et al., 2016; Rose et al., various). Recent work uses transformer-based models: **SafeAeroBERT** (Kierszbaum & Lavedrine, 2023) pre-trains BERT on 400K+ ASRS and NTSB reports, achieving strong performance on report classification. **Aviation-BERT** (2023) takes a similar approach. A systematic review by Tikayat Ray et al. (2023) covers the field comprehensively.

- **HABERT** (Zhao et al., 2024, 2025). Hierarchical multilabel classification for fine-level event extraction from NTSB reports, using the NTSB Phase/Occurrence/Subject ontology. ACE-Graph's Stage 1 builds directly on HABERT for coarse-label extraction. Zhao's 2022 dissertation additionally develops sequential event prediction from NTSB data.

- **LLM-based analysis.** Preliminary work examines GPT-4 for ASRS analysis (Batuwita et al., 2023), finding potential for summarization and classification but noting hallucination risks — directly relevant to our LLM circularity concerns (Section 2.5).

ACE-Graph extends this literature by going beyond classification and topic modeling to causal graph construction and counterfactual reasoning.

### 9.3 Causal discovery and inference

**Foundational methods:**

- **Constraint-based discovery:** Spirtes, Glymour, & Scheines (2000) established the PC and FCI algorithms for learning causal graphs from observational data under faithfulness assumptions. ACE-Graph's Stage 3 uses constraint-based discovery as one algorithm option and as the no-LLM-prior ablation baseline.

- **SCM formalism and do-calculus:** Pearl (2009) provides the formal framework for causal reasoning, including the three-rung causal hierarchy (association, intervention, counterfactual) and do-calculus for computing interventional distributions. ACE-Graph's leave-one-out protocol (Section 7.1) uses do-calculus notation, though the actual computation relies on LLM-assisted approximation rather than formal SCM inference.

- **Causal inference and machine learning:** Peters, Janzing, & Scholkopf (2017) bridge causal inference and machine learning, covering identifiability, estimation, and the role of assumptions. Their treatment of assumptions is directly relevant to our assumption audits for Stage 3 ingredients.

**LLM-assisted causal inference (the four ingredients):**

- Vashishtha et al. (ICLR 2025) show that LLM-generated causal order — even when imperfect — substantially improves causal effect estimation when used as an expert prior. ACE-Graph adapts this to aviation event-type vocabularies. **Adaptation needed:** the method assumes experts have domain knowledge; LLMs' aviation-causality knowledge needs empirical validation.

- Zhang et al. (NeurIPS CaLM 2024) propose an LLM-curated Laplacian similarity prior for handling concurrent causes. **Adaptation needed:** the method assumes a specific SCM framework; compatibility with aviation's often nonlinear, multi-factorial causal structure requires investigation.

- Jin et al. (CLadder, NeurIPS 2023) develop a benchmark for assessing causal reasoning in LLMs across Pearl's three rungs. ACE-Graph adapts CLadder's evaluation framework (not its reasoning method) for aviation-specific counterfactual QA.

**LLM causal reasoning evaluation:** Sheth et al. (2025) evaluate LLMs for causal graph reasoning, finding systematic biases that vary across model families — motivating our cross-model diversification policy (Section 2.5). Yu et al. (2025) propose CausalEval for rigorous evaluation of causal claims, informing our per-stage evaluation design.

### 9.4 Temporal information extraction

Temporal ordering of events is critical for Stage 2.

- **TimeML** (Pustejovsky et al., 2003) defines a specification language for temporal and event expressions in text, including temporal relations (BEFORE, AFTER, SIMULTANEOUS, etc.). The TimeBank corpus provides training data for temporal relation classifiers.

- **TempEval** (Verhagen et al., 2007, 2010) provides shared tasks for temporal relation identification, establishing baselines and evaluation protocols. ACE-Graph's Stage 2 should leverage these methods rather than relying on fragile keyword matching for temporal structure recovery.

### 9.5 Event graphs and root cause analysis

Graph-based approaches to root cause analysis provide methodological parallels:

- **Groot** (Wang et al., 2021) uses event graphs for root cause analysis in industrial settings, constructing causal chains from structured event data. While developed for microservices, the event-graph construction methodology informs our Stage 2-3 pipeline.

- **RCAEval** (2024) proposes evaluation methodology for RCA systems with per-component metrics — the same principle motivating our per-stage evaluation design (Section 8.1).

### 9.6 LLM reasoning capabilities and limitations

- Bhaskar et al. (2025) analyze reasoning capabilities of language models, finding that LLMs can exhibit systematic reasoning failures that appear as consistent patterns rather than random errors — relevant to our circularity concern.

- Wei et al. (2026) survey agentic reasoning patterns in LLMs, informing the design of Stage 4's CausalCoT adaptation.

---

## 10. Limitations and Honest Framing

This section consolidates known limitations that must be clearly communicated in the paper.

### 10.1 NTSB dominance

The reasoning-ready corpus will be 80-90% NTSB records. The "unified cross-jurisdiction" contribution is the harmonization *methodology* and unified *schema*, not a balanced multi-source corpus. Claims about cross-jurisdiction generalization are limited to measuring pipeline transfer on partial-tier non-NTSB test sets.

### 10.2 Ground truth is investigator judgment

NTSB `probable_cause` and `Findings` reflect regulatory/legal judgment under time pressure and institutional norms, not scientific causal determination. Known biases include:
- Over-attribution to pilot error and under-attribution to systemic/organizational factors (Shappell & Wiegmann, 2003)
- Narrative-findings non-independence (same investigators wrote both)
- Jurisdictional selection bias toward FAA-actionable factors

The evaluation measures *agreement with investigator judgment*, which is a meaningful but imperfect proxy for causal reasoning quality.

### 10.3 LLM dependence and circularity

Despite mitigation measures (Section 2.5), the pipeline fundamentally depends on LLM capabilities. If LLMs have systematic blind spots in aviation causality (e.g., under-representing maintenance culture or organizational factors), the pipeline will inherit them. Cross-model diversification reduces but does not eliminate this risk.

### 10.4 Pearl's ladder position

The system performs LLM-assisted counterfactual question answering grounded in a learned causal graph. It does *not* perform formal Rung-3 counterfactual inference via the abduction-action-prediction procedure on a fully parameterized SCM. Formal counterfactual inference requires knowing the functional relationships in the SCM (not just the graph structure), which we do not estimate. The paper should frame the contribution as "counterfactual QA grounded in causal structure" rather than "Rung-3 counterfactual reasoning."

### 10.5 Rare category limitations

Per-category causal discovery requires sufficient sample sizes. Categories below the minimum threshold (Section 2.1) will be excluded. The paper will clearly report which categories are covered and which are excluded, with sample sizes.

### 10.6 Feed-forward pipeline errors

Errors in early stages propagate downstream without correction. Per-stage metrics (Section 8.1) enable diagnosis but not prevention. A joint/iterative approach is left as future work.

---

## 11. Computational Budget and Reproducibility

### 11.1 LLM call budget estimate

| Stage | Calls per record | Estimated corpus size | Total calls | Estimated cost (2026 API pricing) |
|---|---|---|---|---|
| 0 (category classification) | ~1 per FAA AIDS record | ~5,000 FAA AIDS | ~5,000 | ~$5 |
| 1 (subevent extraction) | ~2-3 per record | ~10,000 UASC-Rich | ~25,000 | ~$50-100 |
| 3 (causal order triplets) | ~50-100 per category (event-pair comparisons) | ~10 categories | ~500-1,000 | ~$10-20 |
| 3 (Laplacian prior) | ~20-50 per category | ~10 categories | ~200-500 | ~$5-10 |
| 4 (CausalCoT) | ~1 per leave-one-out query | ~2,000 eval queries | ~2,000 | ~$20-40 |
| **Total** | | | **~33,000-34,000** | **~$100-175** |

Budget is manageable. The dominant cost is Stage 1 extraction.

### 11.2 Reproducibility protocol

1. **LLM response caching.** All LLM calls are cached with full prompt, response, model ID, and timestamp. Cached responses are used for all downstream stages to ensure determinism.
2. **Temperature setting.** Temperature=0 (or minimum available) for all LLM calls where supported. For models without temperature control, we use the lowest-variance decoding option.
3. **Variance reporting.** For stochastic components (LLM extraction, causal discovery with random restarts), we report results across 3 independent runs with different random seeds. Mean and standard deviation are reported for all metrics.
4. **Version pinning.** All model versions, library versions, and data versions are pinned and recorded in a `requirements.txt` / `environment.yml` and a `DATA_VERSIONS.md` manifest.

### 11.3 Hardware requirements

Non-LLM components (HABERT fine-tuning, causal discovery algorithms) require a single GPU for training (estimated 4-8 hours on an A100 for HABERT) and CPU-only for inference and graph algorithms.

---

## 12. Ethical Considerations

1. **Fatality data.** Aviation accident data involves fatalities and serious injuries. All data used in this project is public record, released by government agencies for safety improvement purposes. We handle this data with appropriate gravity and do not sensationalize individual accidents.

2. **Named individuals.** Narratives may contain names of pilots, crew, or other individuals. We do not extract, store, or report personally identifiable information beyond what is already in the public record. Any case studies in the paper use official accident identifiers (e.g., NTSB number), not individual names.

3. **No safety-of-life decisions.** This system is a research tool for post-hoc analysis. It does not make real-time safety decisions, issue alerts, or recommend interventions. Any operational use would require extensive validation beyond the scope of this project (Section 1.5).

4. **Risk of misuse.** Automated causal attribution should not replace human investigation. The system's outputs are probabilistic assessments grounded in historical patterns, not definitive causal determinations. This limitation must be clearly communicated in any release.

5. **Data sourcing.** All data sources are publicly available government datasets. BEA data is scraped from a public-facing search engine; we comply with their robots.txt and rate-limit requests.

---

## 13. References

### Core pipeline ingredients
1. Vashishtha, A. et al. *Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference.* ICLR 2025.
2. Jin, Z. et al. *CLadder: Assessing Causal Reasoning in Language Models.* NeurIPS 2023.
3. Zhang, Q. et al. *Leveraging LLM-Generated Structural Prior for Causal Inference with Concurrent Causes.* NeurIPS CaLM Workshop 2024.
4. Zhao, X. et al. *Hierarchical Multilabel Classification for Fine-Level Event Extraction from Aviation Accident Reports.* INFORMS Journal on Data Science, 2025.
5. Zhao, X. *Hierarchical Sequential Event Prediction and Translation from Aviation Accident Report Data.* ASU Dissertation, 2022.

### Aviation safety frameworks
6. Shappell, S. A. & Wiegmann, D. A. *The Human Factors Analysis and Classification System — HFACS.* DOT/FAA/AM-00/7, Office of Aviation Medicine, 2000.
7. Wiegmann, D. A. & Shappell, S. A. *A Human Error Approach to Aviation Accident Analysis: The Human Factors Analysis and Classification System.* Ashgate, 2003.
8. Leveson, N. G. *A New Accident Model for Engineering Safer Systems.* Safety Science 42(4), 237-270, 2004.
9. Leveson, N. G. *Engineering a Safer World: Systems Thinking Applied to Safety.* MIT Press, 2011.
10. Rasmussen, J. *Risk Management in a Dynamic Society: A Modelling Problem.* Safety Science 27(2-3), 183-213, 1997.
11. CAST/ICAO Common Taxonomy Team. *Aviation Occurrence Categories: Definitions and Usage Notes.* Version 4.7, 2017.

### Causal discovery and inference
12. Pearl, J. *Causality: Models, Reasoning and Inference.* 2nd edition, Cambridge University Press, 2009.
13. Spirtes, P., Glymour, C. & Scheines, R. *Causation, Prediction, and Search.* 2nd edition, MIT Press, 2000.
14. Peters, J., Janzing, D. & Scholkopf, B. *Elements of Causal Inference: Foundations and Learning Algorithms.* MIT Press, 2017.

### LLM causal reasoning evaluation
15. Sheth, M. et al. *CausalGraph2LLM: Evaluating LLMs for Causal Graph Reasoning.* 2025.
16. Yu, H. et al. *CausalEval: Towards Better Causal Evaluation.* 2025.

### Aviation NLP
17. Tikayat Ray, A. et al. *Natural Language Processing (NLP) in Aviation Safety: Systematic Review of Research and Outlook into the Future.* Aerospace 10(7), 600, 2023.
18. Kierszbaum, S. & Lavedrine, J. *SafeAeroBERT: Towards a Safety-Informed Aerospace-Specific Language Model.* AIAA AVIATION Forum, 2023.
19. Batuwita, R. et al. *Examining the Potential of Generative Language Models for Aviation Safety Analysis: Case Study and Insights Using ASRS.* Aerospace 10(9), 770, 2023.

### Temporal information extraction
20. Pustejovsky, J. et al. *TimeML: Robust Specification of Event and Temporal Expressions in Text.* IWCS-5, 2003.
21. Verhagen, M. et al. *SemEval-2007 Task 15: TempEval Temporal Relation Identification.* SemEval 2007.
22. Verhagen, M. et al. *The TempEval Challenge: Identifying Temporal Relations in Text.* Language Resources and Evaluation 43(2), 161-179, 2010.

### Event graphs and RCA methodology
23. Wang, P. et al. *Groot: An Event-Graph-Based Approach for Root Cause Analysis in Industrial Settings.* ASE, 2021.

### LLM reasoning
24. Bhaskar, A. et al. *Language Models That Think.* 2025.
25. Wei, J. et al. *Agentic Reasoning for Large Language Models.* 2026.

---

## Appendix A — Open Questions from Brainstorming

These are questions from the brainstorming session that were left open for the next round:

1. ~~Should the richness filter gates (150 token floor, 50 token cause floor) be tuned empirically on NTSB distributions before being fixed?~~ **Resolved in v2:** Yes. Thresholds switched to word counts (120, 40) and will be tuned via sensitivity analysis (Appendix B).
2. Is BEA full-report scraping worth the extra sub-project, or drop BEA to summaries-only? **Updated:** Paper will report both scenarios; realistic pass rate without scraping is ~5%.
3. Build a fresh unified fine ontology, or retain NTSB Phase/Occurrence/Subject as canonical with projection for non-NTSB records (current choice)? **Unchanged,** but projected codes now treated as lower-confidence tier in Stage 3.
4. Stage 1 topics (Section 4) — pending, with expanded planned topics added
5. Stage 2 topics (Section 5) — pending, with expanded planned topics added
6. Stage 3 topics (Section 6) — pending, with expanded planned topics added
7. Stage 4 topics (Section 7) — **partially resolved:** leave-one-out protocol formalized (Section 7.1); remaining topics pending

---

## Appendix B — Richness Score Sensitivity Analysis Protocol

The richness score weights (Section 3.4) are a default configuration. This appendix defines the protocol for validating or revising them during Stage 0 construction.

### B.1 Objective

Determine whether the default weights produce a reasoning-ready subset that supports high-quality downstream event extraction (Stage 1), and whether alternative weight configurations or filtering methods perform better.

### B.2 Procedure

1. **Annotated calibration set.** Manually annotate 100 NTSB records (stratified by richness score quintile) for whether they yield a valid causal chain when processed through a simplified Stage 1 extraction (using the NTSB structured `Findings` + `seq_of_events` as supervision). Binary label: `valid_chain = {yes, no}`.

2. **Weight sensitivity grid.** Vary each weight in the default configuration by +/- 50% (capped to sum to 1.0), producing ~200 weight vectors on the 6-dimensional simplex. For each weight vector, compute the richness score for all records and the resulting reasoning-ready subset (using the Jenks-determined threshold).

3. **Downstream quality metric.** For each weight vector, measure the proportion of the reasoning-ready subset that has `valid_chain = yes` in the calibration set (precision), and the proportion of all `valid_chain = yes` records that fall in the reasoning-ready subset (recall).

4. **Pareto front.** Plot the Pareto front of precision vs. recall vs. corpus size. Select the weight vector that maximizes the F1 of valid-chain inclusion while maintaining a corpus size >= 5,000 records.

5. **Alternative: learned weights.** Fit a logistic regression predicting `valid_chain` from the six richness components. Compare the learned weights to the default configuration. If the learned model's F1 exceeds the grid-search best by >5%, adopt the learned weights.

6. **Alternative: k-of-n dominance filter.** Test the alternative filtering method (Section 3.4) where a record is reasoning-ready if it exceeds component-specific thresholds on >= k of 6 criteria. Compare against the weighted-sum approach on the same calibration set.

### B.3 Tier threshold selection

1. Compute the richness score distribution for all NTSB records using the selected weights.
2. Apply Jenks natural breaks (k=3) to identify cluster boundaries.
3. If the Jenks boundaries are close to the default values (0.5, 0.3 +/- 0.1), retain the defaults. Otherwise, adopt the Jenks boundaries with documentation.
4. Report the per-source distribution of records in each tier.

### B.4 Deliverable

A report (included as part of Deliverable 7, Section 3.7) containing:
- The sensitivity grid results and Pareto front visualization
- The selected weight vector with rationale
- Comparison with logistic regression and k-of-n alternatives
- The final tier thresholds with Jenks analysis
- Per-source record counts in each tier
