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


def _load_prompts() -> tuple[str, list[dict[str, Any]]]:
    """Load the canonical system prompt and few-shot examples."""
    system = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()
    fewshot = json.loads((PROMPTS_DIR / "few_shot.json").read_text(encoding="utf-8"))
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
    """Parse the model's JSON output: ``{"nodes": [...], "edges": [...]}``.

    Nodes carry ``kind`` ∈ {event, entity, condition}; edges may reference any
    node id. The model occasionally wraps JSON in ```json fences or prefixes
    with prose — strip the outer fence and locate the outermost object.
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

    nodes = obj.get("nodes") or []
    edges = obj.get("edges") or []
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


def _load_full_corpus(
    corpus_path: Path, min_words: int, max_words: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            wc = len(r.get("text", "").split())
            if wc < min_words or wc > max_words:
                continue
            out.append(r)
    return out


def _load_done_keys(out_path: Path) -> set[tuple[str, str]]:
    """Read existing output JSONL and return the set of (source, record_id)
    pairs already extracted. Used to skip records on resume."""
    if not out_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r.get("source", ""), r.get("record_id", "")))
    return done


async def run(args: argparse.Namespace) -> int:
    system, fewshot = _load_prompts()
    corpus = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_records:
        records = [json.loads(l) for l in Path(args.input_records).open()]
    elif args.full_corpus:
        records = _load_full_corpus(
            corpus, min_words=args.min_words, max_words=args.max_words
        )
    else:
        records = stratified_sample(
            corpus,
            pilot_size=args.pilot_size,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
        )

    # Resume: drop records whose (source, record_id) is already in out_path.
    if args.resume:
        done = _load_done_keys(out_path)
        if done:
            before = len(records)
            records = [
                r for r in records if (r["source"], r["record_id"]) not in done
            ]
            print(f"resume: skipping {before - len(records)} already-done records")
    else:
        # Truncate output if not resuming, so we don't append to a stale file.
        if out_path.exists():
            out_path.unlink()

    print(f"loaded {len(records)} records to extract")
    if not records:
        # Nothing to do; create an empty file if missing, then exit.
        if not out_path.exists():
            out_path.touch()
        return 0

    # Split records across workers; each worker owns a slice so the output
    # order is determinate given a seed + concurrency level.
    n_workers = max(1, args.concurrency)
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, r in enumerate(records):
        chunks[i % n_workers].append(r)

    sem = asyncio.Semaphore(args.concurrency)
    pbar_state = {"done": 0, "ok": 0, "total": len(records)}
    out_lock = asyncio.Lock()
    # Append mode: truncation is handled in run() when not resuming.
    with open(out_path, "a", encoding="utf-8") as fp:
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
    ap.add_argument("--max-tokens", default=6144, type=int)
    ap.add_argument("--temperature", default=0.2, type=float)
    ap.add_argument("--concurrency", default=16, type=int)
    ap.add_argument(
        "--full-corpus",
        action="store_true",
        help="extract every record in the corpus that meets [min,max] words "
        "instead of running a stratified pilot sample.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="append to --out and skip records whose (source, record_id) is "
        "already present. Without this flag the output file is truncated.",
    )
    args = ap.parse_args(argv)
    if not args.endpoints:
        args.endpoints = ["http://localhost:8000/v1"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
