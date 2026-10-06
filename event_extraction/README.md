# Event extraction (Qwen3.6-35B-A3B via vLLM, schema v4 — ordered causal chains)

Extracts an **ordered causal chain** per narrative from the aviation safety
corpus (`data/corpus/corpus_enriched.jsonl`). The chain format encodes
causality as per-node `caused_by` back-references restricted to earlier
indices — acyclic by construction — and is the input to the causation KG and
the downstream counterfactual / risk analysis.

Design docs: [v4 plan](../docs/2026-07-05-v4-chain-schema-plan.md) ·
[condition vocabulary memo](../docs/2026-07-05-v4-condition-vocab.md).

## Layout

```
event_extraction/
    prompts/
        schema_v4.json         JSON Schema (enforced via response_format json_schema)
        system_v4.txt          v4 system prompt (chain rules + grounded mode)
        few_shot_v4.json       2 worked examples (1 grounded, 1 narrative-only)
        schema.md              v3 schema doc (historical)
        schema_v3.json / system_v3.txt / few_shot_v3.json
                               v3 contract — kept until the v4 corpus fully
                               replaces the v3 one, then removable
    scripts/
        serve_qwen.sh          launch one vLLM replica (GPUS=0,1 PORT=8000)
        extract_vllm.py        runner — v4 chain mode is the default
        build_fewshot_v4.py    regenerates few_shot_v4.json from hand annotations
        validate_extraction.py enum + structural validator (v3 and v4 formats)
        semantic_eval.py       LLM-as-judge recall/precision vs NTSB C/F factors
        eval_chain_order.py    Kendall's tau of chain order vs NTSB seq_of_events
        build_kg_v4.py         causation KG builder (chains -> factor vectors,
                               causal edges, outcome layer)
        build_kg_layer1.py     v3 aggregate-KG builder (until Phase 4 replaces it)
        sample_calibration_2k.py  provenance of the 2k calibration sample
        prep_full_v3_input.py  builds the 56k extraction input list
        verify_setup.py        sanity checks (no inference)
        stage3/                per-category causal-DAG pipeline (PC + priors)
            bn_do_smoke_test.py  Phase-5 gate: BN do()-intervention ranking
    out/                       extraction outputs and eval reports
    logs/                      vLLM server + runner logs
```

## Critical implementation note: constrained decoding

vLLM 0.19.x **silently ignores** the legacy `guided_json` request field — the
v3 corpus was extracted effectively unconstrained because of this, producing
enum drift in 82 % of records (see `out/validation_v3.summary.json`).
`extract_vllm.py` now sends the schema as OpenAI-standard
`response_format: {type: "json_schema", ...}`, which IS enforced. Verified
2026-07-05 by round-tripping an out-of-enum value.

What the schema cannot express (caused_by index monotonicity, outcome-as-cause)
is checked post-hoc by `validate_extraction.py` logic inside the runner, with
one repair round-trip; same-type self-links that survive the retry are
stripped (`sanitized: true`) rather than failing the record.

## Serve the model

```bash
GPUS=0,1 PORT=8000 bash event_extraction/scripts/serve_qwen.sh
# optional second replica:
GPUS=2,3 PORT=8001 bash event_extraction/scripts/serve_qwen.sh
```

## Run the v4 extraction

v4 files are the defaults — flags only needed to select records, output, and
supervision mode:

```bash
# grounded (production) — NTSB Findings / seq_of_events in the prompt
python3 event_extraction/scripts/extract_vllm.py \
    --input-records event_extraction/out/full_corpus_v3_input.jsonl \
    --out event_extraction/out/full_corpus_v4.jsonl \
    --supervision grounded \
    --endpoints http://localhost:8000/v1 --endpoints http://localhost:8001/v1 \
    --concurrency 16 --max-tokens 4096 --resume

# narrative-only (honest-eval track, calibration only)
python3 event_extraction/scripts/extract_vllm.py \
    --input-records event_extraction/out/calibration_2k_input.jsonl \
    --out event_extraction/out/calibration_2k_v4_narrative.jsonl \
    --supervision none --concurrency 16 --resume
```

Output is JSONL, one object per narrative:

```json
{"record_id": "...", "source": "NTSB_ASRS:Pre2008", "ok": true,
 "chain": [{"idx": 0, "class": "condition", "factor_type": "FUEL_EXHAUSTION_OR_STARVATION",
            "trigger": "...", "phase_of_flight": "climb", "cause_role": "primary",
            "caused_by": [{"src": 0, "strength": "direct", "evidence": "..."}]}, ...],
 "outcome_severity": "aircraft_damage_only",
 "retried": false, "sanitized": false, "latency_s": 4.1,
 "input_tokens": 5100, "output_tokens": 700}
```

## Validate & evaluate

```bash
# structural + enum validation (exit 2 on violations)
python3 event_extraction/scripts/validate_extraction.py \
    --extraction event_extraction/out/full_corpus_v4.jsonl

# recall / precision vs NTSB cause-flagged factors (LLM-as-judge)
python3 event_extraction/scripts/semantic_eval.py \
    --extraction event_extraction/out/calibration_2k_v4_narrative.jsonl \
    --tag v4-narrative --out event_extraction/out/semantic_eval_v4_narrative.jsonl

# chain ordering vs investigator seq_of_events (Kendall's tau)
python3 event_extraction/scripts/eval_chain_order.py \
    --extraction event_extraction/out/calibration_2k_v4_narrative.jsonl \
    --tag v4-narrative --out event_extraction/out/chain_order_v4_narrative.jsonl
```

Phase-2 gates (see plan doc): narrative-only recall >= 80 %, precision >= 45 %,
tau >= 0.75, zero post-retry structural violations. Current numbers live in
`out/semantic_eval_v4_*.summary.json` / `out/chain_order_v4_*.summary.json`.

## Build the causation KG

```bash
python3 event_extraction/scripts/build_kg_v4.py \
    --extraction event_extraction/out/full_corpus_v4.jsonl \
    --out-dir    event_extraction/out/causation_kg
```

Outputs under `out/causation_kg/`:

| File | Use |
|---|---|
| `per_accident_chains.jsonl` | provenance layer — per-accident chain + resolved severity |
| `factor_vectors.parquet` | records × 57 binary factors + outcome — causal-discovery / risk input |
| `causation_edges.csv` | causal-factor edges: support, P(dst\|src), lift, direct share, sample records |
| `outcome_layer.csv` | per-factor P(serious/fatal), risk ratio vs baseline |
| `causation_kg.graphml` | filtered causal view (min support) for Cytoscape/Gephi |
| `per_category/*.csv` | per-CICTT-category edge lists |

Outcome severity is resolved structured-first (NTSB injury/damage overrides the
model's record-level guess).

### Visualize

```bash
python3 event_extraction/scripts/viz_causation_kg.py
# -> out/causation_kg/html/causation_kg.html   (open in a browser)
```

Interactive Cytoscape view of the aggregate causation KG: nodes colored by
factor family and sized by record support, edge width by co-support, hover for
support / P(dst|src) / lift / direct share, click a node for its risk ratio and
top in/out edges, min-support + lift filters, plus a sortable edge table and
the per-category do() intervention rankings inline. The per-category Stage-3
causal DAGs have their own viewers under `out/aggregate_kg/html_dag/*.html`,
and `out/causation_kg/causation_kg.graphml` loads in Cytoscape desktop / Gephi.

**Publication-style figures** (layered causal diagrams, Graphviz, serif +
grayscale; tiers follow the AcciMap-motivated ordering organizational →
conditions → environment → crew actions → system failures → aerodynamic
states; solid edges = bootstrap stability ≥ 0.7, dashed ≥ 0.5; line weight ∝
record support; per-category node counts are within-category):

```bash
python3 event_extraction/scripts/viz_causation_academic.py \
    --category LOC-I CFIT SCF-PP ... --aggregate
# -> out/causation_kg/figures/dag_<CAT>.{svg,png,gv}
# -> out/causation_kg/figures/causation_kg_aggregate.{svg,png,gv}
```

SVGs drop straight into a paper; the `.gv` sources are committed so figure
styling is reproducible. Requires graphviz (`conda env ntsb`).

## Stage 3 / Phase 5

The per-category causal-DAG pipeline lives in `scripts/stage3/` (see its
README). After Stage 3 runs on v4 data, the readiness gate is:

```bash
/home/yp6443/miniconda3/envs/qwen-vllm/bin/python \
    event_extraction/scripts/stage3/bn_do_smoke_test.py --category LOC-I
```

which fits a discrete BN (pgmpy) on the factor vectors with the Stage-3 DAG as
structure and ranks `do(X = absent)` interventions by achievable risk
reduction — the counterfactual query the whole pipeline exists to answer.
