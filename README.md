# A Hybrid Framework for Explainable Aviation Risk

Companion repository for *A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction
from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis*
(Y. Pang, Y. Tian, J. Xie, X. Zhao, J. Hu, J.-P. Clarke and H.-A. Jacobsen).

The paper extracts one ordered causal chain from each of 56,202 NTSB and FAA accident reports with a
locally served language model, aggregates the chains into a causation graph, attaches physics
occurrence models to the factors whose drivers are physical, and combines the factors into a failure
mode probability that supports interventions and counterfactuals. Where the paper says that something
"is released with the code" or "is in the code repository", it is listed below.

Paper section, table and figure numbers refer to the IEEE build. Items marked "full build" appear
only in the longer manuscript.

## What the Paper Points To

| Paper item | Where it is |
|---|---|
| Full learned networks with bootstrap stabilities (Figs. of the loss of control and airframe failure networks; Sec. V-G) | `event_extraction/out/aggregate_kg/per_category_dag/LOC-I.csv` (60 edges) and `SCF-PP.csv` (42 edges). Columns: `src`, `dst`, `support_count`, `bootstrap_stability` (share of 30 bootstrap refits that keep the edge), `direction_score`, `llm_prior`, `laplacian_prior`, `pc_directed`, `pc_oriented_by_llm`, `from_cpdag_undirected`. The figures draw the edges with stability of at least 0.5. |
| Networks learned without the ordering priors (full build) | `event_extraction/out/aggregate_kg/per_category_dag_noprior/` |
| Complete graph-level intervention tables, all 26 factors with a nonempty adjustment set (Table IV) | `event_extraction/out/causation_kg/backdoor_table_LOC-I.json`, `backdoor_table_SCF-PP.json` (crude and adjusted effects, adjustment sets, 95% percentiles over 400 bootstrap resamples) |
| FCI marks and E-values (Table IV) | `event_extraction/out/causation_kg/fci_check_LOC-I.json`, `fci_check_SCF-PP.json` |
| Effect of the language model priors (full build) | `event_extraction/out/causation_kg/prior_ablation.json` |
| Judge outputs of the extraction evaluation (Sec. V-A) | recall: `event_extraction/out/semantic_eval_v4_narrative.jsonl`, `semantic_eval_v4_grounded.jsonl`; chain order: `chain_order_v4_narrative.jsonl`, `chain_order_v4_grounded.jsonl`; TSB order: `tsb_order_judged.jsonl`; repeat runs: `selfconsistency/` |
| Cross-family check on full investigation reports (Sec. V-A) | `event_extraction/gold496/`: inputs, Claude outputs in `outputs/`, judge cache in `judge_cache/`, gold findings `gold.jsonl`, scores `scores.json`, procedure in `RUNBOOK.md` and `COMPLETED.md` |
| Extraction prompt and schema | `event_extraction/prompts/system_v4.txt`, `schema_v4.json`, `few_shot_v4.json` (built by `event_extraction/scripts/build_fewshot_v4.py`); decoding settings in `event_extraction/scripts/extract_vllm.py` |
| Judge prompt | `JUDGE_SYSTEM` and `JUDGE_SCHEMA` in `event_extraction/scripts/semantic_eval.py`, shared by `eval_chain_order.py` and `eval_tsb_order.py`; the cross-family judge is in `event_extraction/gold496/score_gold.py` |
| Extracted chains | `event_extraction/out/full_corpus_v4.jsonl.gz` (raw run, 56,202 records, retries included) and `event_extraction/out/causation_kg/per_accident_chains.jsonl.gz` (the 55,940 valid chains) |
| Causation graph | `event_extraction/out/causation_kg/causation_edges.csv` (unfiltered edge list with support, conditional and lift), `causation_kg.graphml` (edges with support of at least 30), `factor_vectors.parquet` (binary presence table, one row per chain), `outcome_layer.csv` |
| Physics layer outputs | `physics/out/` (see the map below) |
| Constants and priors of the physics layer, with provenance | table below |

### Occurrence Categories

Records enter the category of their first recognized NTSB occurrence code (`categorize` in
`event_extraction/scripts/stage3/_common.py`). Two categories are used in the paper.

| File key | Name in the paper | NTSB occurrence codes | Notes |
|---|---|---|---|
| `LOC-I` | loss of control in flight | 250 loss of control in flight (94% of records), 100 abrupt maneuver, 110 uncontrolled altitude deviation, 380 roll over | |
| `SCF-PP` | airframe failure | 130 airframe, component or system failure (97%), 131 propeller failure, 140 decompression | The key is a legacy label. The nearest CICTT category is SCF-NP. Loss of engine power (350) is in neither category. |

Other keys of the same map are not used in the paper and some labels do not match their codes; see
the note in `_common.py` before using them.

## Reproduction Map

| Paper item | Script | Output |
|---|---|---|
| Corpus | `data_processing/build_corpus.py`, `build_corpus_enriched.py`, `event_extraction/scripts/prep_full_v3_input.py` | `data/corpus/` |
| Extraction | `event_extraction/scripts/extract_vllm.py` | `event_extraction/out/full_corpus_v4.jsonl` (released as `full_corpus_v4.jsonl.gz`) |
| Causation graph | `event_extraction/scripts/build_kg_v4.py` | `event_extraction/out/causation_kg/` |
| Per-category networks | `event_extraction/scripts/run_stage3_v4_pipeline.sh` (precedence, language model order, PC with and without priors, reorientation, 30 bootstrap refits) | `event_extraction/out/aggregate_kg/` |
| Fig. 1 framework | drawn in TikZ in the manuscript source | |
| Table I guards | `physics/guards.py` | |
| Table II and Fig. 3 engine failure | `physics/ef_risk_model.py`, `physics/risk_uncertainty.py`, `physics/make_ef_fig.py` | `physics/out/ef_risk.json`, `risk_uncertainty.json` |
| Table III probability of necessity | `physics/probability_of_necessity.py` | `physics/out/probability_of_necessity.json` |
| Table IV graph-level interventions | `event_extraction/scripts/stage3/backdoor_table.py`, `event_extraction/scripts/stage3/fci_check.py` | `event_extraction/out/causation_kg/backdoor_table_*.json`, `fci_check_*.json` |
| Fig. 2 carburetor icing surface | `physics/make_carb_icing_fig.py`, `physics/carb_icing_model.py` | reads `physics/out/carb_icing_accidents.csv` |
| Sec. V-A extraction fidelity | `event_extraction/scripts/semantic_eval.py`, `eval_chain_order.py`, `eval_extraction_extras.py`, `run_selfconsistency.sh`, `score_selfconsistency.py`, `validate_extraction.py` | `event_extraction/out/*_v4_*.summary.json`, `eval_extras_v4_*.json`, `selfconsistency/` |
| Sec. V-B admissibility monitor | `physics/violation_rate.py` | `physics/out/violation_rate.json` |
| Sec. V-C calibration | `physics/calibrate_physics.py` | `physics/out/calibration.json`, `stall_prior_values.json` |
| Sec. V-D separation ceiling | `physics/ceiling_carb_icing.py`, `physics/review_checks/icing_delong_bayes.py` | `physics/out/carb_icing_ceiling.json` |
| Sec. V-E routine weather | `physics/metar_control.py` (downloads NOAA ISD-Lite into `data/isd_lite/`) | `physics/out/metar_control.json` |
| Sec. V-F later years | `physics/heldout_occurrence.py`, `physics/severity_groundtruth_check.py` | `physics/out/heldout_occurrence.json`, `severity_groundtruth_check.json` |
| Carburetor icing validation (full build) | `physics/validate_carb_icing.py` | `physics/out/carb_icing_validation.json`, `carb_icing_accidents.csv` |
| Sensitivity to the constants (full build) | `physics/sensitivity_carb_icing.py` | `physics/out/carb_icing_sensitivity.json` |
| Loss of control study (full build) | `physics/loc_risk_model.py`, `make_loc_fig.py`, `make_loc_kg_fig.py`, `make_factor_physics_fig.py`, `make_locdag_top.py` | `physics/out/loc_risk.json` |

## Constants and Priors of the Physics Layer

Speeds are calibrated airspeeds for the Cessna 172 at gross weight and sea level. The constants live
in `physics/factor_models.py`, `jsbsim_stall.py`, `stall_model.py`, `carb_icing_model.py`,
`ef_risk_model.py` and `leaky_noisy_or.py`.

| Quantity | Value | Source |
|---|---|---|
| Aircraft constants | W 2,300 lbf, S 174 ft², lift slope 4.6 per rad, chord 4.9 ft, limit load factor 3.8 | Cessna 172 class values |
| Stall aerodynamics | C_L,max 1.47 clean at 16 degrees, 1.82 with flap, V_s 52 KCAS | JSBSim c172x lift tables |
| Design speeds | V_A 99 KCAS, V_C 120 KCAS | Cessna 172 class values |
| Maneuvering prior, weight | triangular on 0.70, 0.90 and 1.00 of gross | authors' judgment |
| Maneuvering prior, airspeed | Normal, mean 1.35 and spread 0.18 times V_s | authors' judgment |
| Maneuvering prior, load factor | Normal, mean 1.55 and spread 0.45, clipped to 1 to 6 | authors' judgment |
| Other phase priors | takeoff, climb, cruise, approach, landing (`physics/out/stall_prior_values.json`) | authors' judgment |
| Discrete gust scale | 5, 10 and 20 ft/s for light, moderate and severe or thunderstorm | single parameter fit (Hoblit 1988) |
| Shear scale | 10 kt, exponential | authors' judgment, microburst cores 20 to 40 kt |
| Dryden turbulence | sigma_w = 0.1 u_20, L_w 200 ft, encounter 120 s, approach at 1.25 V_s | low-altitude form (MIL-F-8785C) |
| Carburetor icing cooling | 12, 18, 25, 32 and 35 °C for takeoff, climb, cruise, descent and idle | set to the chart zones (FAA SAIB CE-09-35) |
| Carburetor icing exposure | kappa 0.60 per hPa, light zone onset w_thr 0.176 hPa | set to the chart zones (FAA SAIB CE-09-35) |
| Fuel margin | sigma = 0.25 t_flight | authors' judgment |
| Engine base hazard | 1e-4 per flight hour | order of magnitude, surrogate only |
| Leading factor rule | rho = 0.02 | Sec. III-B |

The engine failure study uses only the carburetor icing constants, the fuel margin and the leading
factor rule. The fuel term is not calibrated and enters the combination at its corpus base rate.

## Language Models

- Extractor, judge and ordering priors: Qwen3.6-35B-A3B, Hugging Face revision
  `53c43178507d69762986fbfa314f6e8d4d859409`, served by vLLM 0.19.1 in bfloat16, context 32,768,
  reasoning trace disabled. Extraction at temperature 0.2 with at most 6,144 output tokens and
  constrained JSON against `schema_v4.json`, one repair turn on violation. Judge at temperature 0
  with at most 1,024 output tokens.
- Cross-family extractors: `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`, run on 9 and
  10 August 2026 (`event_extraction/gold496/`).

## Source Data

All source records are public. They are not redistributed here; the steps below fetch them into
`data/`.

| Archive | How to obtain |
|---|---|
| NTSB aviation databases (`avall.mdb`, `Pre2008.mdb`) | download from https://app.ntsb.gov/avdata (see below), then `data_processing/dump_ntsb_mdb.sh` |
| NTSB published investigation reports | `data/scrape_ntsb_reports.py` |
| FAA Accident and Incident Data System | `data/download_faa_aids.py` |
| TSB Canada (order check) | https://www.tsb.gc.ca/eng/stats/aviation/data-5.html, files `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv` and `ASISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv` into `data/TSB_CANADA/` |
| BEA notified events | `data/scrape_bea.py` |
| NOAA Integrated Surface Database (routine weather) | fetched by `physics/metar_control.py` into `data/isd_lite/` |

---

# Development Notes: Data Sources and Environment

## Environment
Assume you have conda installed,
```
conda create -n ntsb python=3.10
conda activate ntsb
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia pandas
conda install -c conda-forge transformers
pip install wikipedia newspaper3k GoogleNews pyvis requests
```

## Data Accessibility

All datasets are stored under the `data/` directory:

```
data/
├── NTSB_ASRS/       # US NTSB Aviation Safety Reporting System
│   └── avall.mdb    # MS Access database with narratives, events, aircraft, etc.
├── NTSB_REPORTS/    # US NTSB published narrative investigation reports
│   ├── pdf/         # Source PDFs (AAR, AAB, SIR, AIR, ASR, SS, SRR)
│   ├── txt/         # pdfplumber-extracted text (primary downstream input)
│   ├── listing.json # Phase 1 enumeration cache
│   └── manifest.csv # Per-report manifest, joinable to avall.mdb by ntsb_accident_id
├── TSB_CANADA/       # Transport Safety Board of Canada
│   └── *.csv         # Public occurrence, aircraft, injuries, events data
├── BEA/              # French Bureau d'Enquetes et d'Analyses
│   └── bea_notified_events.csv   # All notified events scraped from bea.aero
└── FAA_AIDS/         # FAA Accident/Incident Data System (ASIAS)
    ├── a*.txt        # Accident/incident records by time period (TAB-delimited)
    ├── e*.txt        # Edited remarks/narratives by time period (TAB-delimited)
    ├── AcrftSer.txt  # Aircraft make/model/series lookup table
    ├── Airport.txt   # Airport and location lookup table
    ├── aidcodes.doc  # Code definitions and data dictionary
    └── Afilelayout.txt / Efilelayout.txt  # File layout descriptions
```

### NTSB data (US)
Source: https://app.ntsb.gov/avdata
```
wget https://app.ntsb.gov/avdata/Access/avall.zip
unzip DownloadFile\?fileID\=C\:\\avdata\\avall.zip -d data/NTSB_ASRS/
rm -rf DownloadFile\?fileID\=C\:\\avdata\\avall.zip
```

The `avall.mdb` Microsoft Access database format requires `mdbtools` on Linux:
```
sudo apt install mdbtools
```

### NTSB Published Reports (US, narratives)
Source: https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx

The full long-form NTSB published reports — Aircraft Accident Reports (AAR), Aircraft Accident Briefs (AAB), Special Investigation Reports (SIR), newer Aviation Investigation Reports (AIR), Aviation Safety Reports (ASR), Safety Studies (SS), and Safety Recommendation Reports (SRR). These are the highest-richness narratives in the NTSB universe and are joinable back to `avall.mdb` rows by `ntsb_accident_id`.

Requires Playwright (one-time install ~300 MB):
```
conda activate ntsb
pip install playwright pdfplumber
PLAYWRIGHT_BROWSERS_PATH=$PWD/data/NTSB_REPORTS/.playwright python -m playwright install chromium
```

To re-scrape:
```
conda activate ntsb
python data/scrape_ntsb_reports.py            # full run / resume
python data/scrape_ntsb_reports.py --limit 5  # smoke test (5 most recent reports)
python data/scrape_ntsb_reports.py --phase 4  # rebuild manifest only
```

Output: `data/NTSB_REPORTS/pdf/*.pdf`, `data/NTSB_REPORTS/txt/*.txt`, `data/NTSB_REPORTS/manifest.csv`. Manifest columns: `report_number`, `report_type`, `title`, `publication_date`, `accident_date`, `location`, `ntsb_accident_id`, `accident_id_source`, `pdf_url`, `pdf_path`, `txt_path`, `pdf_pages`, `txt_chars`, `extraction_status`, `scraped_at`.

### TSB data （Canada）
Source: https://www.tsb.gc.ca/eng/stats/aviation/data-5.html

CSV files covering occurrences, aircraft, injuries, events/phases, and survivability.

### BEA data (France)
Source: https://bea.aero/en/investigation-reports/notified-events/

~6500 notified aviation safety events scraped from the BEA search engine. To re-scrape:
```
conda activate ntsb
python scrape_bea.py
```

Fields: file_number, title, summary, date, location, state_of_occurrence, occurrence_class, human_consequences, aircraft_category, manufacturer_model, registration, state_of_registry, operator, operation_type, flight_phase, departure, destination, responsible_entity, detail_url.

### FAA AIDS data (US)
Source: https://www.asias.faa.gov/apex/f?p=100:189:::NO:::

FAA Accident/Incident Data System from the Aviation Safety Information Analysis and Sharing (ASIAS) system. Contains accident and incident records from pre-1975 to present, provided as TAB-delimited text files in zip archives. To re-download:
```
conda activate ntsb
python download_faa_aids.py
```

Data files:
- **A files** (`Apre1975.txt`, `a1975_79.txt`, ..., `a2020_26.txt`): Accident/incident records by time period. Layout described in `Afilelayout.txt`.
- **E files** (`e1975_79.txt`, ..., `e2020_26.txt`): Edited remarks/narratives (redacted per Privacy Act). Layout described in `Efilelayout.txt`.
- **AcrftSer.txt**: Aircraft make, model, and series lookup. Layout in `aircraftseries.doc`.
- **Airport.txt**: Airport and location lookup. Layout in `airport.doc`.
- **aidcodes.doc**: Associated table codes and definitions.
- **10-14-2010-Attention.doc**: Important changes to AIDS data format.


---

# ACE-Graph: Aviation Causal Event Graph — Design

**Status:** Draft v3 — Model-data hybrid framework with quantitative risk. LOC-I demo scope.
**Working title:** Aviation Causal Event Graph (ACE-Graph)
**One-line summary:** A causal knowledge graph extracted from aviation safety reports where each node embeds hierarchical failure models (fault trees, HRA, bow-tie) with tiered quantitative risk values, enabling counterfactual intervention analysis with measurable risk reduction.

The full versioned design document lives at [docs/superpowers/specs/2026-04-13-ace-graph-v3-design.md](docs/superpowers/specs/2026-04-13-ace-graph-v3-design.md). Prior versions: [v2](docs/superpowers/specs/2026-04-11-ace-graph-design-v2.md) | [v1](docs/superpowers/specs/2026-04-10-ace-graph-design.md) | [v1→v2 changelog](docs/superpowers/specs/2026-04-11-ace-graph-changelog.md) | [expert review](docs/review/2026-04-11-design-critique.md).

## 1. Motivation and Scope

### 1.1 Goal

Build an end-to-end, reproducible pipeline that transforms unstructured aviation accident reports into a causal reasoning system. The system should be able to answer, for a real accident, questions of the form: *"If contributing factor X had been absent, would the accident still be likely to occur?"*

This is LLM-assisted counterfactual question answering grounded in a learned causal graph structure. While the question targets Pearl's Rung 3 (counterfactual reasoning), the system does not perform formal counterfactual inference via abduction-action-prediction on a fully parameterized structural causal model (SCM). Instead, it combines a learned causal graph with LLM-generated counterfactual judgments, evaluated against NTSB investigator judgments as a proxy for causal ground truth.

### 1.2 Primary research contribution

A **unified pipeline paper** that integrates four recent methodological ingredients into an aviation-safety-specific system:

1. HABERT-style hierarchical event extraction (Zhao et al., 2024, 2025)
2. LLM-generated causal order as expert prior (Vashishtha et al., ICLR 2025)
3. LLM-curated Laplacian similarity prior for concurrent-cause effect estimation (Zhang et al., NeurIPS CaLM 2024)
4. CLadder-inspired evaluation framework for causal QA on real accidents (Jin et al., NeurIPS 2023)

No prior work combines these into an aviation-safety pipeline. The integration itself — plus the cross-jurisdiction corpus construction and the leave-one-out probable-cause benchmark — is the contribution.

**Ingredient compatibility as a contribution.** These four methods were developed for different settings. Demonstrating that they can be composed into a coherent aviation-safety pipeline — and characterizing where their assumptions hold or must be relaxed — is itself a methodological contribution.

### 1.3 Data sources

All four public aviation safety databases are treated as a **single unified corpus**, not as primary-plus-validation silos:

- **NTSB** (US) — `avall.mdb`, rich narratives + curated `Events_Sequence`, `seq_of_events`, `Findings` tables
- **TSB Canada** — public CSVs including `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv`
- **BEA** (France) — scraped notified events from `bea.aero`; optional full-report scrape pass
- **FAA AIDS** (US) — tab-delimited accident/incident records + edited remarks

Cross-jurisdiction harmonization is a first-class problem, not a side task.

**Honest framing on source balance.** While all four sources contribute to the unified schema, NTSB records are expected to dominate the reasoning-ready subset (~70-80% pass rate vs. 10-40% for other sources). The cross-jurisdiction contribution is the harmonization *methodology* and unified *schema*, not a balanced multi-source corpus.

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
     -> LLM triplet-prompted causal order over event-type vocabulary (Paper 1)
     -> Laplacian similarity prior over event types (Paper 3)
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

**Multi-label handling.** Some accidents involve multiple CICTT categories simultaneously (e.g., LOC-I triggered by SCF-PP). We assign each accident a *primary* category and record *secondary* categories as metadata. Stage 3 uses primary assignment for sub-graph construction but also constructs cross-category bridge edges.

**Minimum sample size.** Categories below a minimum threshold (expected ~50-100 reasoning-ready records) will be excluded from per-category analysis or merged into an "Other" bucket.

### 2.2 Why build LOC-I first

Loss of Control In-flight is:
- The largest single category by fatality count
- Well-documented in both NTSB and ICAO safety literature
- Has clear, often short causal chains that are tractable to validate
- Sample size is large enough to support the LLM-prior-assisted discovery

Success on LOC-I gives an early failure signal. If the pipeline fails on LOC-I, it will fail on rarer categories too.

### 2.3 Cross-stage consistency checks

Three explicit checks to detect error propagation:
1. **Stage 1 → 2 (event vocabulary alignment):** Verify event vocabulary in temporal DAGs is a subset of Stage 1 extraction output.
2. **Stage 2 → 3 (temporal support for causal edges):** Verify high-confidence causal edges are supported by temporal ordering. Flag A → B edges where B precedes A in >50% of per-accident DAGs.
3. **Stage 3 → 4 (counterfactual graph consistency):** Verify removing a non-ancestor node does not change predicted outcome.

### 2.4 LLM usage policy and circularity mitigation

LLMs are used in multiple stages, creating correlated bias risk. Key policies:

| Stage | LLM Role | Model Family | Conflict Resolution |
|---|---|---|---|
| 0 | CICTT classification (FAA AIDS) | Family A (GPT-4 class) | Rules primary; LLM supplements |
| 1 | Subevent role extraction | Family A | **Structured data primary.** LLM supplements. |
| 3 | Causal order + Laplacian prior | Family B (Claude class) | Ablation: no-LLM-prior baseline |
| 4 | CausalCoT counterfactual QA | Family B | Evaluated against investigator judgment |

1. **Model family diversification** — extraction (0, 1) and reasoning (3, 4) use different model families.
2. **Structured data primacy** — where NTSB `Findings`/`seq_of_events` exist, they are primary signal.
3. **Mandatory ablations** — no-LLM-prior baseline isolates the LLM contribution.

## 3. Stage 0 — Corpus Harmonization and Richness Filtering

### 3.1 Source-by-source reality check

| Source | Narrative depth | Structured events | Cause/findings | Expected pass rate |
|---|---|---|---|---|
| **NTSB** `avall.mdb` | 3 narrative fields (`narr_accp`, `narr_accf`, `narr_cause`), often hundreds of words | `Events_Sequence`, `seq_of_events`, `Occurrences` | `Findings` table + `narr_cause` | ~70-80% pass |
| **TSB Canada** CSVs | Structured summary fields, shorter free text | `ASISdb_MDOTW_VW_EVENTS_AND_PHASES_PUBLIC.csv` | Limited structured causes | ~40% pass |
| **BEA** scraped | Title + `summary`; full reports require separate scrape of `detail_url` | None natively | Buried in full report PDFs | ~20-30% with full-report fetch; **~5% summaries-only** |
| **FAA AIDS** txt | E-files short, redacted remarks | A-file coded fields only | Coded cause fields only | ~10% pass |

**Implication:** NTSB will dominate the reasoning-ready subset (likely 80-90%). This is explicitly documented and the paper will frame the cross-jurisdiction contribution as the harmonization *methodology*, not a balanced multi-source evaluation.

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

Every field earns its place:
- Stage 1 consumes `narrative_*` and `structured_events`
- Stage 2 consumes `structured_events` plus time markers in narratives
- Stage 3 buckets by `accident_category_primary` and uses `findings` for supervision/prior; `accident_category_secondary` informs cross-category bridge edges
- Stage 4 evaluates against `findings` + `narrative_cause` in the leave-one-out benchmark

### 3.3 Richness filter — definition of "reasoning-ready"

A record is **reasoning-ready** if and only if it passes all five gates:

1. **Factual narrative present and substantial** — `narrative_factual` >= 120 words (tokenizer-independent). Screens out FAA AIDS one-liners. Cutoff will be tuned on the NTSB word-count distribution during Stage 0 build.
2. **Cause statement present** — either `narrative_cause` >= 40 words, *or* >= 2 `findings` rows with a subject code. Ensures the "why" exists.
3. **Temporal structure recoverable** — either `structured_events` has >= 2 ordered entries, *or* a temporal relation classifier identifies >= 2 temporal relations in `narrative_factual`. Initial implementation uses keyword matching; production uses TimeML/TempEval-style classifiers.
4. **Context minimum** — `occurrence_date`, `flight_phase_raw`, `aircraft_category`, and `severity` all non-null.
5. **Harmonizable category** — `accident_category_primary` maps to a CICTT code (original or inferred via rule + LLM classifier).

### 3.4 Richness score

Continuous score in [0, 1]:

```
richness = w1 * lognorm(wordcount(narrative_factual), 120, 600)
         + w2 * lognorm(wordcount(narrative_cause),   40, 300)
         + w3 * min(1, n_structured_events / 5)
         + w4 * min(1, n_findings / 4)
         + w5 * min(1, n_temporal_relations / 3)
         + w6 * (all_context_present ? 1 : 0)

where: lognorm(x, lo, hi) = clip((log(x) - log(lo)) / (log(hi) - log(lo)), 0, 1)
```

**Default weights:** `w1=0.25, w2=0.20, w3=0.20, w4=0.15, w5=0.10, w6=0.10`. These are a **default configuration, not principled values**. Sensitivity analysis (Appendix B of v2 design doc) will validate or revise during Stage 0.

**Alternative filter:** Also testing a **k-of-n dominance filter** (reasoning-ready if >= k of 6 component thresholds are met) to address the additive form's substitutability assumption.

**Tier thresholds** set at natural break points via Jenks natural breaks or KDE valley detection (not arbitrary round numbers). Initial values pending analysis:
- `reasoning_ready` — passes all 5 gates, score >= threshold_high -> enters the core corpus for Stages 1-4
- `partial` — passes >= 3 gates, score >= threshold_low -> used only for cross-source validation / OOD test splits
- `metadata_only` — excluded from modeling, kept for provenance and citation

### 3.5 Taxonomy harmonization

Anchor on **ICAO CICTT** (Common Taxonomy Team):

- **Accident category:** NTSB already uses CICTT in `Occurrences`; TSB uses ICAO codes; BEA uses French-translated ICAO terminology; FAA AIDS requires rule-based + LLM-assisted mapping from older cause codes.
- **Flight phase:** ICAO flight phase taxonomy (Standing, Taxi, Takeoff, Initial Climb, Climb, Cruise, Descent, Approach, Landing, Emergency). Lookup tables per source.
- **Sub-event types (fine grain):** retain Zhao et al.'s NTSB **Phase / Occurrence / Subject** hierarchy as the canonical fine ontology; no cross-source equivalent exists at that granularity. Non-NTSB records get events projected into NTSB Subject codes via a small LLM-assisted classifier trained on NTSB data. Projected codes are treated as a **separate, lower-confidence tier** in Stage 3.

**FAA AIDS mapping validation.** The FAA AIDS -> CICTT mapping will be validated on dual-coded records (accidents in both NTSB and FAA AIDS with known CICTT codes).

**Multi-label assignment.** Accidents spanning multiple CICTT categories are assigned a primary category (most proximate occurrence) and secondary categories (contributing factors).

All mappings are explicit crosswalk tables committed to the repo. Every projection is recorded in `harmonization_flags` so original-vs-inferred is always traceable.

### 3.6 Cross-source deduplication

Cross-jurisdiction duplicates occur (a US accident may appear in both NTSB and FAA AIDS).

Strategy: blocking + pairwise match.
- **Blocking key:** `(occurrence_date +/- 1 day, country, aircraft_category)`. Complemented by LSH on concatenated text fields to catch blocking-key misses.
- **Match features:** tail-number exact match, aircraft make/model match, location text similarity, severity agreement
- **Decision rule:** weighted score >= threshold -> same cluster; threshold validated on a manually reviewed sample (not a fixed arbitrary value). Borderline cases logged for manual review.
- **Cluster resolution:** richest record becomes canonical; other members linked via `dup_cluster_id` and contribute any unique fields

### 3.7 Deliverables

1. **UASC v1.0** — unified parquet file with the canonical schema
2. **UASC-Rich** — the `reasoning_ready` subset (primary input to Stages 1-4)
3. **Harmonization tables** — committed crosswalks for category, phase, subject codes
4. **Deduplication audit log** — cluster IDs + decisions
5. **Corpus statistics report** — per-source counts, richness distribution, category distribution, duplicate rate (also a paper artifact)
6. **Data card** — provenance, licenses, known limitations
7. **Richness score analysis report** — sensitivity analysis results, final weight selection rationale, tier threshold justification

**Implemented (2026-04-27):** [`data/corpus/corpus.jsonl`](data/corpus/corpus.jsonl) carries narrative + minimal metadata for 178k records. [`data/corpus/corpus_enriched.jsonl`](data/corpus/corpus_enriched.jsonl) supplements it with structured supervision fields rejoined from `Pre2008.mdb`, `avall.mdb`, and FAA AIDS A-files — 145k records (82%) carry NTSB `Findings` / `seq_of_events` / `Occurrences` or FAA AIDS cause+phase. See [data_processing/README.md](data_processing/README.md). The full UASC parquet build per the design above is still pending; the JSONL form is what Stage 1 currently consumes.

### 3.8 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Over-filtering leaves an NTSB-only corpus | `partial` tier preserves cross-source test sets. Paper frames NTSB dominance honestly (Section 1.3). |
| BEA full-report scraping is fragile | Treat as optional sub-project; if it fails, BEA drops to `partial` tier. Paper reports both scenarios. |
| Fine Subject code projection on non-NTSB records introduces noise | `harmonization_flags` records every inference; projected codes treated as lower-confidence tier in Stage 3. |
| Richness score weights are arbitrary | Sensitivity analysis (Appendix B) validates or revises weights before proceeding past Stage 0. Alternative k-of-n filter tested. |
| FAA AIDS category mapping is noisy | Validated on dual-coded NTSB/FAA AIDS records. Noisy records flagged in `harmonization_flags`. |

## 4. Stage 1 — Hybrid Event Extraction

**Status: superseded by v4 (2026-07-05).** Extraction now uses the **v4 ordered-causal-chain schema**: per-narrative chains with `caused_by` back-references (acyclic by construction), a 57-value factor vocabulary (35 events + 12 first-class conditions + 9 outcomes), grounded prompting with NTSB `Findings`/`seq_of_events` in-context, and truly enforced constrained decoding (`response_format json_schema` — the legacy `guided_json` field is silently ignored by vLLM 0.19.x and was the root cause of all v3 enum drift). Full corpus: **56,154 records, zero schema violations** (`event_extraction/out/full_corpus_v4.jsonl`).

Calibration (2k records, LLM-as-judge): recall **90.8 %** narrative-only / **93.1 %** grounded (v3: 80.5 %), chain ordering Kendall τ **0.82/0.87** vs investigator occurrence order. See [docs/2026-07-05-v4-chain-schema-plan.md](docs/2026-07-05-v4-chain-schema-plan.md), [docs/2026-07-05-v4-phase2-calibration-results.md](docs/2026-07-05-v4-phase2-calibration-results.md), [docs/2026-07-05-v4-condition-vocab.md](docs/2026-07-05-v4-condition-vocab.md), and [event_extraction/README.md](event_extraction/README.md) for the current workflow. The causation KG + factor vectors + per-category `do()` intervention rankings live under `event_extraction/out/causation_kg/`.

Historical v3 notes (42-event vocabulary, v1-vs-v3 delta): [docs/2026-04-29-extraction-summary.md](docs/2026-04-29-extraction-summary.md), [docs/2026-04-27-v3-schema-redesign.md](docs/2026-04-27-v3-schema-redesign.md).

**Implementation:**

- LLM: Qwen3.6-35B-A3B served via vLLM, OpenAI-compatible endpoint.
- Schema: [`event_extraction/prompts/schema_v3.json`](event_extraction/prompts/schema_v3.json) enforced via vLLM `guided_json` constrained generation. Closed vocabularies (HAEM, `event_type`, `cause_role`, phase, severity, edge types, entity / condition types) are mechanically validated.
- Prompt: [`event_extraction/prompts/system_v3.txt`](event_extraction/prompts/system_v3.txt) + one worked example in [`few_shot_v3.json`](event_extraction/prompts/few_shot_v3.json).
- Runner: [`event_extraction/scripts/extract_vllm.py`](event_extraction/scripts/extract_vllm.py) with `--guided-json`, `--system-file`, `--fewshot-file` flags.
- Supervision rejoin (Stage 0 supplement): [`data_processing/build_corpus_enriched.py`](data_processing/build_corpus_enriched.py) joins NTSB `Findings` / `seq_of_events` / `Occurrences` and FAA AIDS cause/phase columns onto every corpus record. 145k of 178k records (82%) now carry structured supervision. See [data_processing/README.md](data_processing/README.md).

**Calibration on 2k NTSB records (LLM-as-judge against `Findings.Cause_Factor`):**

| Metric | v1 | v3 |
|---|---:|---:|
| Recall | 73.86% | **80.52%** |
| Precision | 21.31% | **29.87%** |
| Events / record | 7.45 | 4.84 (leaner) |
| Schema violations (HAEM `OPERATIONAL` catch-all) | 53% | **0%** |

**Resolved from the original "topics to resolve" list:**

- ~~HABERT model reuse vs. retraining~~ — superseded by single-model LLM extraction with constrained generation. HABERT remains a candidate for Stage-2 temporal-edge classification.
- ~~LLM choice and prompt design~~ — Qwen3.6-35B-A3B locally; v3 prompt + JSON Schema. Cross-model ablation (different family) is still planned for Stage-3 prior generation per the [design critique](docs/review/2026-04-11-design-critique.md) §2.
- ~~Reconciliation rules with `Findings`~~ — `cause_role: primary` is the supervised target; `Findings.Cause_Factor='C'` items are the gold standard.
- ~~How to use `Findings` / `seq_of_events` as supervision signal~~ — joined via [`build_corpus_enriched.py`](data_processing/build_corpus_enriched.py); evaluated semantically by [`semantic_eval.py`](event_extraction/scripts/semantic_eval.py).
- ~~Output event table schema~~ — [`schema_v3.json`](event_extraction/prompts/schema_v3.json), 42-value `event_type` vocabulary aligned to HFACS Tier-3 + CICTT.
- ~~Per-stage evaluation~~ — recall/precision against NTSB cause-flagged factors via LLM-as-judge; tracked in `out/semantic_eval_*.summary.json`.

**Still open:**

- Cross-model ablation for circularity mitigation.
- Per-record confidence scoring (currently `cause_role` carries the model's commitment level).
- Adversarial prompt iteration on the residual ~20% recall miss.

## 5. Stage 2 — Per-Accident Temporal Graph

**Status:** TBD — to be brainstormed in next session.

Planned topics to resolve:
- Temporal edge construction rules
- Handling of partial orderings and simultaneous events
- Time-marker normalization ("2 minutes later" vs. timestamps)
- **Temporal relation extraction: evaluate TimeML/TempEval-based classifiers in addition to keyword matching.**
- Merging structured `seq_of_events` with narrative-derived ordering
- Per-accident DAG representation and storage
- **Per-stage evaluation: temporal ordering agreement with `seq_of_events`, measured via Kendall's tau.**
- **Consistency check with Stage 1: verify event vocabulary alignment (Section 2.3).**

## 6. Stage 3 — Per-Category Causal Graph

**Status:** plan locked, execution starts 2026-04-28. See [docs/2026-04-27-stage3-plan.md](docs/2026-04-27-stage3-plan.md) for the seven-piece workplan with effort estimates, deliverables, validation tests, and concrete first-step commands. First milestone: validated LOC-I DAG by 2026-05-01; all-category DAGs by 2026-05-02.

Planned topics to resolve:
- CICTT category list and sub-graph granularity
- **Minimum sample-size threshold per category:** categories below threshold excluded or merged.
- LLM triplet-prompting protocol for causal order (adapting Vashishtha et al. to event-type vocabulary)
- **Assumption audit for Vashishtha et al.:** Do LLMs satisfy the expert-quality assumptions?
- Laplacian similarity prior construction (adapting Zhang et al. to NTSB Subject codes)
- **Assumption audit for Zhang et al.:** Are SCM assumptions (linearity, additive noise) compatible with aviation causal structure?
- Choice of discovery algorithm (PC vs. CaMML vs. score-based) given the prior
- Edge-weight interpretation and stability analysis
- **Multi-label: cross-category bridge edges for multi-CICTT accidents (Section 2.1).**
- Per-category validation against published HFACS classifications and ICAO chains
- **No-LLM-prior ablation: PC algorithm on structured event co-occurrence as mandatory baseline (Section 2.4).**
- **Per-stage evaluation: edge precision/recall against domain expert judgment on annotated subset.**

## 7. Stage 4 — Counterfactual Reasoning and Evaluation

**Status:** TBD — leave-one-out protocol formalized below; remaining topics pending.

### 7.1 Leave-one-out protocol (formalized)

"Removing a factor" is formalized as `do(X = absent)` on the per-category causal DAG — a Pearl-style intervention, not node deletion. The counterfactual estimand for accident *a* with factor set *F_a*:

```
P(Y_a = accident | do(X_i = absent), F_a \ {X_i}, graph_category)
```

**Feasibility check (must-complete before building Stages 1-3):** Manually construct causal graphs for 10 LOC-I accidents, apply leave-one-out by hand, verify the protocol produces meaningful distinctions and is stable across two annotators.

### 7.2 Remaining topics
- CausalCoT adaptation: prompt templates, estimand selection, backdoor-set construction
- **Ground-truth framing:** evaluation is against NTSB investigator judgment, *not* objective causal truth. Three known biases: investigator attribution bias, narrative-findings non-independence, jurisdictional selection bias.
- **Secondary evaluation:** HFACS classifications, published re-analyses of high-profile accidents, inter-rater reliability study (n >= 50).
- Baselines: retrieval, raw LLM QA, random, graph-without-prior, **single-stage LLM** (critical), **NTSB structured data only**
- Benchmark release: dataset card, splits, metrics, versioning
- Metrics: accuracy per reasoning level, cross-jurisdiction generalization, stratification by category

## 8. Evaluation Plan

### 8.1 Per-stage metrics

| Stage | Metric | Gold Standard |
|---|---|---|
| 0 — Harmonization | Category assignment accuracy | Dual-coded NTSB/FAA AIDS records |
| 1 — Event Extraction | Event extraction P/R/F1 | NTSB `Findings` + `seq_of_events` |
| 2 — Temporal Graph | Temporal ordering accuracy | NTSB `seq_of_events` (Kendall's tau) |
| 3 — Causal Graph | Causal edge quality | Domain expert judgment (n >= 50 edges/category) |
| 4 — Counterfactual QA | Investigator judgment agreement | NTSB `Findings` + `probable_cause` |

### 8.2 Primary metric
Leave-one-out agreement with NTSB investigator judgment (not "causal accuracy") on LOC-I, then extended.

### 8.3 Baselines (9 total)
Random, retrieval-based, raw LLM QA, **single-stage LLM** (critical), **NTSB structured data only**, graph-without-LLM-prior, graph-without-Laplacian, global graph, no fine extraction.

### 8.4 Cross-jurisdiction
Held-out test splits from BEA / TSB Canada / FAA AIDS `partial`-tier records — tests pipeline transfer, not independent causal ground truth.

### 8.5 Qualitative
Per-category DAG visualizations + case studies on high-profile accidents (e.g., Colgan Air 3407, AF447).

## 9. Related Work

See [v2 design doc Section 9](docs/superpowers/specs/2026-04-11-ace-graph-design-v2.md) for full treatment (25 references in 6 subsections: aviation safety frameworks, aviation NLP, causal discovery, temporal extraction, event graphs, LLM reasoning).

## 10. Limitations

1. **NTSB dominance** — reasoning-ready corpus will be 80-90% NTSB.
2. **Ground truth is investigator judgment** — not scientific causal determination. Known biases documented.
3. **LLM dependence** — cross-model diversification reduces but does not eliminate circularity risk.
4. **Pearl's ladder position** — LLM-assisted counterfactual QA, not formal Rung-3 SCM inference.
5. **Rare categories** — excluded below minimum sample-size threshold.
6. **Feed-forward errors** — per-stage metrics enable diagnosis but not prevention.

## 11. Computational Budget and Reproducibility

~33,000 LLM calls total (~$100-175). All calls cached, temperature=0, 3-run variance reported, all versions pinned. See [v2 design doc Section 11](docs/superpowers/specs/2026-04-11-ace-graph-design-v2.md).

## 12. Ethical Considerations

Public fatality data handled with appropriate gravity. No PII extraction beyond public record. No safety-of-life decisions. Automated causal attribution clearly framed as probabilistic, not definitive. See [v2 design doc Section 12](docs/superpowers/specs/2026-04-11-ace-graph-design-v2.md).

## 13. References

### Core pipeline ingredients
1. Vashishtha, A. et al. *Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference.* ICLR 2025.
2. Jin, Z. et al. *CLadder: Assessing Causal Reasoning in Language Models.* NeurIPS 2023.
3. Zhang, Q. et al. *Leveraging LLM-Generated Structural Prior for Causal Inference with Concurrent Causes.* NeurIPS CaLM Workshop 2024.
4. Zhao, X. et al. *Hierarchical Multilabel Classification for Fine-Level Event Extraction from Aviation Accident Reports.* INFORMS Journal on Data Science, 2025.
5. Zhao, X. *Hierarchical Sequential Event Prediction and Translation from Aviation Accident Report Data.* ASU Dissertation, 2022.

### Aviation safety frameworks
6. Shappell & Wiegmann. *HFACS.* DOT/FAA/AM-00/7, 2000.
7. Wiegmann & Shappell. *A Human Error Approach to Aviation Accident Analysis.* Ashgate, 2003.
8. Leveson. *A New Accident Model for Engineering Safer Systems.* Safety Science, 2004.
9. Rasmussen. *Risk Management in a Dynamic Society.* Safety Science, 1997.
10. CAST/ICAO. *Aviation Occurrence Categories.* v4.7, 2017.

### Causal discovery and inference
11. Pearl. *Causality.* 2nd ed., Cambridge, 2009.
12. Spirtes, Glymour & Scheines. *Causation, Prediction, and Search.* MIT Press, 2000.
13. Peters, Janzing & Scholkopf. *Elements of Causal Inference.* MIT Press, 2017.

Full reference list (25 papers) in [v2 design doc Section 13](docs/superpowers/specs/2026-04-11-ace-graph-design-v2.md).

## Appendix A — Open Questions from Brainstorming

1. ~~Should the richness filter gates be tuned empirically?~~ **Resolved in v2:** Yes. Switched to word counts (120, 40), validated via sensitivity analysis.
2. Is BEA full-report scraping worth the extra sub-project? **Updated:** Paper reports both scenarios; ~5% pass rate without scraping.
3. Build a fresh unified fine ontology, or retain NTSB as canonical? **Unchanged,** but projected codes now lower-confidence tier.
4. Stage 1 topics (Section 4) — pending, expanded planned topics added
5. Stage 2 topics (Section 5) — pending, expanded planned topics added
6. Stage 3 topics (Section 6) — pending, expanded planned topics added
7. Stage 4 topics (Section 7) — **partially resolved:** leave-one-out protocol formalized; remaining pending
