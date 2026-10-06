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

# Data Sources and Environment in Detail

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
