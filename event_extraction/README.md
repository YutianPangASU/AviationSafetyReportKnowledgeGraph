# Event extraction (Qwen3.6-35B-A3B via vLLM, schema v3)

Local OSS runner for pulling typed causal/temporal events out of the cleaned
aviation safety corpus (`data/corpus/corpus_enriched.jsonl`).

The pipeline is on **schema v3** — see [prompts/schema.md](prompts/schema.md)
and [docs/2026-04-27-v3-schema-redesign.md](../docs/2026-04-27-v3-schema-redesign.md)
for the design, the 42-event vocabulary, and the v1-vs-v3 calibration delta.

## Layout

```
event_extraction/
    prompts/
        schema.md              human-readable v3 schema documentation
        schema_v3.json         JSON Schema enforced via vLLM `guided_json`
        system_v3.txt          v3 system prompt
        few_shot_v3.json       one v3-format worked example
        system.txt             legacy v1 prompt (kept for reproducing v1 baseline)
        few_shot.json          legacy v1 examples
    scripts/
        serve_qwen.sh          launches one vLLM OpenAI-compatible replica
        extract_vllm.py        runner — supports v1 and v3 via flags
        eval_stage1.py         token-overlap calibration eval (fast proxy)
        semantic_eval.py       LLM-as-judge semantic recall matcher
        select_loci_cases.py   pull 10 LOC-I cases for manual feasibility check
        compare_v1_v3.py       v1 vs v3 quick-comparison report
        sample_calibration_2k.py  sample 2k records for v3 calibration
        prep_full_v3_input.py     build full re-extraction input list
        verify_setup.py        sanity checks (no inference)
    out/                       extraction outputs and eval reports
    logs/                      vLLM server + runner logs
```

## One-time setup (already done)

- Conda env `qwen-vllm` at `/home/yp6443/miniconda3/envs/qwen-vllm`
  (Python 3.11, vLLM 0.19.1, PyTorch 2.10 + CUDA 12.8)
- Model cached at `/home/yp6443/.cache/huggingface/hub/Qwen--Qwen3.6-35B-A3B`
  (26 safetensor shards, 67 GB, bf16)

## Serve the model

One replica on GPUs 0+1:

```bash
GPUS=0,1 PORT=8000 bash event_extraction/scripts/serve_qwen.sh
```

First load takes a few minutes (weights into VRAM + CUDA graph capture). Once
`Uvicorn running on http://0.0.0.0:8000` appears, the server is ready.

## Run the v3 extraction

The runner accepts flags that select the prompt/schema generation. Defaults
match v1 for backwards compatibility; pass the v3 flags for the current
schema.

```bash
/home/yp6443/miniconda3/envs/qwen-vllm/bin/python \
    event_extraction/scripts/extract_vllm.py \
        --input-records event_extraction/out/full_corpus_v3_input.jsonl \
        --out event_extraction/out/full_corpus_v3.jsonl \
        --resume \
        --system-file system_v3.txt \
        --fewshot-file few_shot_v3.json \
        --guided-json schema_v3.json \
        --concurrency 16 \
        --max-tokens 4096
```

Key flags:

- `--guided-json schema_v3.json` — forces output to validate against the
  JSON Schema. Closed vocabularies are mechanically enforced; the model
  cannot invent strings.
- `--system-file` / `--fewshot-file` — pick which prompt set to use.
  v3 files (`system_v3.txt`, `few_shot_v3.json`) are the current schema; the
  unsuffixed v1 files are kept for reproducing the v1 baseline.
- `--input-records <jsonl>` — explicit list of records to extract, one per
  line with `record_id`, `source`, `text`. Use this when running on a
  curated subset (e.g. the 56k structured-supervision records or a 2k
  calibration sample).
- `--resume` — append to `--out` and skip records whose `(source, record_id)`
  is already present and `ok=true`.

Output is JSONL, one object per narrative:

```json
{"record_id": "...", "source": "NTSB_ASRS:Pre2008", "ok": true,
 "nodes": [...], "edges": [...],
 "latency_s": 1.78, "input_tokens": 6473, "output_tokens": 412}
```

## Reproduce the v1 baseline (for calibration)

```bash
/home/yp6443/miniconda3/envs/qwen-vllm/bin/python \
    event_extraction/scripts/extract_vllm.py \
        --corpus data/corpus/corpus.jsonl \
        --pilot-size 500 \
        --out event_extraction/out/v1_pilot.jsonl
```

Without `--guided-json` the runner falls back to v1's free-form prompt. The
existing 45k-record v1 baseline is preserved at
[`out/full_corpus_events.jsonl`](out/full_corpus_events.jsonl) — do not
overwrite.

## Evaluate

Two evals are available:

1. **`eval_stage1.py`** — fast token-overlap proxy. Use for sanity checks
   during prompt iteration. Caveat: under-reports recall by ~50 pts because
   HFACS phrasing in NTSB Findings doesn't tokenize-overlap with narrative
   trigger phrases.
2. **`semantic_eval.py`** — LLM-as-judge using the same Qwen via vLLM,
   constrained-JSON output. This is the right matcher for HFACS-vs-narrative
   paraphrase mismatch. Slower (~3.5/s on concurrency 4 → ~10/s on
   concurrency 16) but produces the headline numbers.

Example:

```bash
# Filter v1 baseline to the 2k calibration set
python3 -c '
import json
ids = {json.loads(l)["record_id"] for l in open("event_extraction/out/calibration_2k_input.jsonl")}
with open("event_extraction/out/full_corpus_events.jsonl") as f, \
     open("event_extraction/out/calibration_2k_v1.jsonl","w") as g:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get("record_id") in ids and r.get("ok"):
            g.write(line)
'

# Score v1 and v3 with the LLM-as-judge
python event_extraction/scripts/semantic_eval.py \
    --extraction event_extraction/out/calibration_2k_v1.jsonl \
    --tag v1 --out event_extraction/out/semantic_eval_v1.jsonl

python event_extraction/scripts/semantic_eval.py \
    --extraction event_extraction/out/calibration_2k_v3.jsonl \
    --tag v3 --out event_extraction/out/semantic_eval_v3.jsonl
```

Summaries are written to `*.summary.json`. The current numbers:

| Metric | v1 | v3 |
|---|---:|---:|
| Recall | 73.86 % | **80.52 %** |
| Precision | 21.31 % | **29.87 %** |
| Schema violations | 53 % | **0 %** |

See [docs/2026-04-27-v3-schema-redesign.md](../docs/2026-04-27-v3-schema-redesign.md)
for the full delta and design history.

## Build the Layer-1 aggregate KG

After (or alongside) extraction, merge all per-narrative subgraphs into a
single aggregate knowledge graph. The builder is incremental-safe — re-run
any time on the partial output.

```bash
python event_extraction/scripts/build_kg_layer1.py \
    --extraction event_extraction/out/full_corpus_v3.jsonl \
    --enriched   data/corpus/corpus_enriched.jsonl \
    --out-dir    event_extraction/out/aggregate_kg
```

Outputs (under `event_extraction/out/aggregate_kg/`):

| File | Use |
|---|---|
| `aggregate_kg.gpickle` | NetworkX MultiDiGraph, fast Python reload |
| `aggregate_kg.graphml` | Cytoscape / Gephi / yEd visualisation |
| `aggregate_edges.csv` | `src,edge_type,dst,count` — input for Stage-3 PC algorithm and LLM causal-order priors |
| `per_category/*.csv` | Same edge format, one CSV per CICTT category (LOC-I, CFIT, SCF-PP, MAC, ICE, etc.) |
| `aggregate_kg.summary.md` | Human-readable: top event_types, top edges, per-category subgraph sizes |
| `aggregate_kg.summary.json` | Machine-readable summary stats |

Aggregate-node id format:
- `EVT:<event_type>` — one node per Stage-3 event_type (42 max)
- `COND:<condition_type>` — one per condition type (10 max)
- `ENT:<entity_type>` — one per entity type (10 max)

Aggregate-edge data:
- `type` — one of the 17 edge types from the v3 schema
- `count` — number of distinct narratives carrying this triple (deduped within record)

CICTT categorisation uses NTSB `Occurrence_Code` → CAST/ICAO mapping when
available (Pre2008), HFACS category number for avall, and FAA AIDS general
cause category for AIDS records. Records without a mapping fall into
`OTHER:NTSB-<code>` or `UNCATEGORIZED:<source>` buckets.

## Knobs worth tuning later

- `--concurrency` — 16 per replica is the working number. vLLM's continuous
  batching can take more once warm.
- `--max-tokens` — 4096 is comfortable; 6144 was used for v1 and rarely
  saturated.
- Few-shot examples — `few_shot_v3.json` carries one. Adding more would
  cost prefix-cache slots; better to iterate the system prompt.
