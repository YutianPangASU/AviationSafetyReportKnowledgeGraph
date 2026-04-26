"""Pilot event-extraction runner against a local vLLM server.

The server speaks the OpenAI Chat Completions API (see serve_qwen.sh). We
send each narrative as a single chat turn and parse the JSON response into
`events` and `relations`. Output is JSONL, one object per narrative.

Basic usage (after starting the server on port 8000):

    python -m event_extraction.scripts.extract_vllm \
        --corpus data/corpus/corpus.jsonl \
        --pilot-size 500 \
        --out event_extraction/out/pilot.jsonl

Multi-replica throughput: pass --endpoints http://localhost:8000/v1
--endpoints http://localhost:8001/v1 and requests round-robin across them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompts(version: str) -> tuple[str, list[dict[str, Any]]]:
    """Load the system prompt and few-shot examples for a given schema version.

    ``version`` selects the file suffix: ``"v1"`` reads ``system.txt``;
    ``"v2"`` reads ``system_v2.txt`` + ``few_shot_v2.json``.
    """
    if version == "v1":
        sys_path = PROMPTS_DIR / "system.txt"
        few_path = PROMPTS_DIR / "few_shot.json"
    else:
        sys_path = PROMPTS_DIR / f"system_{version}.txt"
        few_path = PROMPTS_DIR / f"few_shot_{version}.json"
    system = sys_path.read_text(encoding="utf-8").strip()
    fewshot = json.loads(few_path.read_text(encoding="utf-8"))
    return system, fewshot


def build_messages(system: str, fewshot: list[dict[str, Any]], narrative: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in fewshot:
        msgs.append({"role": "user", "content": f"NARRATIVE:\n{ex['narrative']}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["extraction"])})
    msgs.append({"role": "user", "content": f"NARRATIVE:\n{narrative}"})
    return msgs


@dataclass
class ExtractionResult:
    record_id: str
    source: str
    ok: bool
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    raw: Optional[str]
    error: Optional[str]
    latency_s: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]


def _parse_model_output(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse the model's JSON output.

    Supports both schema versions:
      v1 — ``{"events": [...], "relations": [...]}``
      v2 — ``{"nodes": [...], "edges": [...]}`` where nodes have ``kind`` ∈
            {event, entity, condition} and edges may reference any node id.

    Returns ``(nodes, edges)`` where v1 inputs are normalized so each event is
    treated as a node with ``kind="event"`` for downstream compatibility.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = text[: -3]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in response")
    obj = json.loads(text[start : end + 1])

    if "nodes" in obj or "edges" in obj:
        nodes = obj.get("nodes") or []
        edges = obj.get("edges") or []
    else:
        # v1 → normalize to v2-style.
        nodes = [{**e, "kind": "event"} for e in (obj.get("events") or [])]
        edges = obj.get("relations") or []

    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes / edges must be arrays")

    ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    for e in edges:
        if not isinstance(e, dict):
            raise ValueError("edge is not an object")
        for key in ("src", "dst"):
            if e.get(key) not in ids:
                raise ValueError(f"edge {key}={e.get(key)!r} not in nodes")
    return nodes, edges


async def _one_call(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen 3.x thinking mode wastes output tokens on reasoning traces.
        # Disable via the chat-template extra kwargs that vLLM forwards.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = await client.post(f"{endpoint}/chat/completions", json=body, timeout=600)
    resp.raise_for_status()
    payload = resp.json()
    content = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage") or {}
    return content, usage


async def _worker(
    idx: int,
    endpoints: list[str],
    records: list[dict[str, Any]],
    out_fp,
    out_lock: asyncio.Lock,
    model: str,
    system: str,
    fewshot: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    sem: asyncio.Semaphore,
    pbar_state: dict[str, int],
) -> None:
    ep = endpoints[idx % len(endpoints)]
    async with httpx.AsyncClient() as client:
        for rec in records:
            async with sem:
                t0 = time.time()
                messages = build_messages(system, fewshot, rec["text"])
                try:
                    raw, usage = await _one_call(
                        client, ep, model, messages, max_tokens, temperature
                    )
                    nodes, edges = _parse_model_output(raw)
                    result = ExtractionResult(
                        record_id=rec["record_id"],
                        source=rec["source"],
                        ok=True,
                        nodes=nodes,
                        edges=edges,
                        raw=raw,
                        error=None,
                        latency_s=time.time() - t0,
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                    )
                except Exception as e:
                    result = ExtractionResult(
                        record_id=rec["record_id"],
                        source=rec["source"],
                        ok=False,
                        nodes=[],
                        edges=[],
                        raw=None,
                        error=f"{type(e).__name__}: {e}",
                        latency_s=time.time() - t0,
                        input_tokens=None,
                        output_tokens=None,
                    )
                async with out_lock:
                    out_fp.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    pbar_state["done"] += 1
                    if result.ok:
                        pbar_state["ok"] += 1
                    if pbar_state["done"] % 10 == 0 or pbar_state["done"] == pbar_state["total"]:
                        print(
                            f"[{pbar_state['done']}/{pbar_state['total']}] "
                            f"ok={pbar_state['ok']} "
                            f"last_latency={result.latency_s:.1f}s "
                            f"source={rec['source']}",
                            flush=True,
                        )


def stratified_sample(
    corpus_path: Path,
    pilot_size: int,
    seed: int,
    min_words: int,
    max_words: int,
) -> list[dict[str, Any]]:
    """Per-source even sampling. Skips narratives outside the [min,max] word range."""
    rng = random.Random(seed)
    by_source: Dict[str, list[dict[str, Any]]] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            wc = len(r.get("text", "").split())
            if wc < min_words or wc > max_words:
                continue
            by_source.setdefault(r["source"], []).append(r)
    per_source = max(1, pilot_size // max(1, len(by_source)))
    picked: list[dict[str, Any]] = []
    for src, rows in by_source.items():
        rng.shuffle(rows)
        picked.extend(rows[:per_source])
    rng.shuffle(picked)
    return picked[:pilot_size]


async def run(args: argparse.Namespace) -> int:
    system, fewshot = _load_prompts(args.prompt_version)
    corpus = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_records:
        records = [json.loads(l) for l in Path(args.input_records).open()]
    else:
        records = stratified_sample(
            corpus,
            pilot_size=args.pilot_size,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
        )
    print(f"loaded {len(records)} records to extract")

    # Split records across workers; each worker owns a slice so the output
    # order is determinate given a seed + concurrency level.
    n_workers = max(1, args.concurrency)
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, r in enumerate(records):
        chunks[i % n_workers].append(r)

    sem = asyncio.Semaphore(args.concurrency)
    pbar_state = {"done": 0, "ok": 0, "total": len(records)}
    out_lock = asyncio.Lock()
    with open(out_path, "w", encoding="utf-8") as fp:
        t0 = time.time()
        await asyncio.gather(
            *[
                _worker(
                    i,
                    args.endpoints,
                    chunks[i],
                    fp,
                    out_lock,
                    args.model,
                    system,
                    fewshot,
                    args.max_tokens,
                    args.temperature,
                    sem,
                    pbar_state,
                )
                for i in range(n_workers)
            ]
        )
    print(
        f"done in {time.time()-t0:.1f}s  ok={pbar_state['ok']}/{pbar_state['total']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus/corpus.jsonl", type=Path)
    ap.add_argument(
        "--input-records",
        default=None,
        type=Path,
        help="explicit JSONL of records to extract (overrides sampling)",
    )
    ap.add_argument("--out", default="event_extraction/out/pilot.jsonl", type=Path)
    ap.add_argument("--pilot-size", default=500, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--min-words", default=30, type=int)
    ap.add_argument("--max-words", default=2000, type=int)
    ap.add_argument(
        "--endpoints",
        action="append",
        default=None,
        help="vLLM server base URL, repeat for replicas",
    )
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--max-tokens", default=3072, type=int)
    ap.add_argument("--temperature", default=0.2, type=float)
    ap.add_argument("--concurrency", default=8, type=int)
    ap.add_argument(
        "--prompt-version",
        default="v1",
        choices=["v1", "v2"],
        help="schema version: v1 = events/relations, v2 = nodes/edges with HAEM+HFACS+STAMP+AcciMap",
    )
    args = ap.parse_args(argv)
    if not args.endpoints:
        args.endpoints = ["http://localhost:8000/v1"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
