# Event extraction (Qwen3.6-35B-A3B via vLLM)

Local OSS runner for pulling event sequences + causal/temporal relations out of
the cleaned aviation safety corpus (`data/corpus/corpus.jsonl`).

## Layout

```
event_extraction/
    prompts/
        schema.md        schema documentation (human-readable)
        system.txt       system prompt sent to the model
        few_shot.json    two worked examples embedded as chat turns
    scripts/
        serve_qwen.sh    launches one vLLM OpenAI-compatible replica
        extract_vllm.py  pilot runner: samples narratives, calls vLLM, writes JSONL
        verify_setup.py  sanity checks (no inference)
    out/                 extraction results
    logs/                vLLM server logs
```

## One-time setup (already done)

- Conda env `qwen-vllm` at `/home/yp6443/miniconda3/envs/qwen-vllm`
  (Python 3.11, vLLM 0.19.1, PyTorch 2.10 + CUDA 12.8)
- Model cached at `/home/yp6443/.cache/huggingface/hub/Qwen--Qwen3.6-35B-A3B`
  (26 safetensor shards, 67 GB, bf16)

Re-run any time:

```bash
/home/yp6443/miniconda3/envs/qwen-vllm/bin/python \
    event_extraction/scripts/verify_setup.py
```

## Serve the model

One replica on GPUs 0+1:

```bash
GPUS=0,1 PORT=8000 bash event_extraction/scripts/serve_qwen.sh
```

Optional second replica on GPUs 2+3 for ~2× throughput:

```bash
GPUS=2,3 PORT=8001 bash event_extraction/scripts/serve_qwen.sh
```

First load takes a few minutes (weights into VRAM + CUDA graph capture).
Once `Uvicorn running on http://0.0.0.0:8000` appears, the server is ready.

## Run the pilot

From the repo root, with the `qwen-vllm` env's Python:

```bash
/home/yp6443/miniconda3/envs/qwen-vllm/bin/python \
    event_extraction/scripts/extract_vllm.py \
        --corpus data/corpus/corpus.jsonl \
        --pilot-size 500 \
        --out event_extraction/out/pilot.jsonl \
        --endpoints http://localhost:8000/v1
```

Two replicas: repeat `--endpoints http://localhost:8001/v1` and bump
`--concurrency` (try 16).

Output is JSONL with one line per narrative:

```json
{"record_id": "...", "source": "BEA", "ok": true, "events": [...], "relations": [...], "latency_s": 3.4, "input_tokens": 742, "output_tokens": 418, ...}
```

## Knobs worth tuning later

- `--pilot-size` — start at 500; calibrate schema on output, then scale.
- `--concurrency` — 8 per replica is conservative. Bump to 32+ once the
  server is warm; vLLM's continuous batching handles it.
- `MAX_MODEL_LEN` in `serve_qwen.sh` — 32K by default. Set higher only if
  you need to extract from a full NTSB report without chunking.
- Thinking mode — disabled via `chat_template_kwargs.enable_thinking: False`
  in `extract_vllm.py`. Flip to `True` if you want the model to reason
  before answering (slower, costs output tokens, may or may not help).
