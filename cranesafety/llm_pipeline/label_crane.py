"""LLM sentence labeler for the crane safety corpus, against a local vLLM server.

For every record we determine which cohort(s) the human annotators worked on
(a record is almost always labeled for exactly one cohort) and ask the model to
label each sentence IN / OUT / MAYBE for that cohort, using the same written
guidelines the humans used. The full report is shown for context; the model must
return a value for every sentence index.

Server: an OpenAI-compatible vLLM endpoint hosting Qwen3.6-35B-A3B
(see ../../event_extraction/scripts/serve_qwen.sh).

Usage:
    python label_crane.py \
        --corpus ../labeled_crane_corpus_sample.jsonl \
        --out out/llm_labels.jsonl \
        --endpoints http://localhost:8000/v1 --concurrency 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from guidelines import COHORTS, system_prompt

VALUES = {"IN", "OUT", "MAYBE"}

# vLLM guided decoding: force the shape of the response. We can't fix the array
# length here, so we validate/backfill indices in code after parsing.
GUIDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "value": {"type": "string", "enum": ["IN", "OUT", "MAYBE"]},
                },
                "required": ["index", "value"],
            },
        }
    },
    "required": ["labels"],
}


def target_cohorts(record: dict[str, Any]) -> list[str]:
    """Cohorts the humans actually annotated in this record (preserves guide order)."""
    seen = {lb["label"] for s in record["sentences"] for lb in s["labels"]}
    return [c for c in COHORTS if c in seen]


def build_user_prompt(record: dict[str, Any]) -> str:
    lines = [f"{s['index']}. {s['text']}" for s in record["sentences"]]
    return "REPORT:\n" + "\n".join(lines)


def parse_labels(raw: str, valid_indices: list[int]) -> dict[int, str]:
    """Parse model JSON into {index: value}; backfill any missing index as OUT."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = text[:-3]
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start : end + 1])
    got: dict[int, str] = {}
    for item in obj.get("labels", []):
        try:
            idx = int(item["index"])
            val = str(item["value"]).upper()
        except (KeyError, TypeError, ValueError):
            continue
        if idx in valid_indices and val in VALUES:
            got[idx] = val
    # Backfill: any sentence the model skipped is treated as OUT (guide default).
    return {i: got.get(i, "OUT") for i in valid_indices}


async def one_call(
    client: httpx.AsyncClient, endpoint: str, model: str,
    system: str, user: str, temperature: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 2048,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": GUIDED_SCHEMA,
    }
    resp = await client.post(f"{endpoint}/chat/completions", json=body, timeout=600)
    resp.raise_for_status()
    payload = resp.json()
    return payload["choices"][0]["message"]["content"], payload.get("usage") or {}


async def worker(
    tasks: list[tuple[dict, str]], endpoint: str, model: str, temperature: float,
    out_fp, lock: asyncio.Lock, sem: asyncio.Semaphore, state: dict[str, int],
) -> None:
    async with httpx.AsyncClient() as client:
        for record, cohort in tasks:
            async with sem:
                t0 = time.time()
                indices = [s["index"] for s in record["sentences"]]
                user = build_user_prompt(record)
                try:
                    raw, usage = await one_call(
                        client, endpoint, model, system_prompt(cohort), user, temperature
                    )
                    labels = parse_labels(raw, indices)
                    result = {
                        "record_id": record["record_id"], "cohort": cohort,
                        "ok": True, "labels": labels, "error": None,
                        "latency_s": round(time.time() - t0, 2),
                        "output_tokens": usage.get("completion_tokens"),
                    }
                except Exception as e:  # noqa: BLE001 - log & continue
                    result = {
                        "record_id": record["record_id"], "cohort": cohort,
                        "ok": False, "labels": {}, "error": f"{type(e).__name__}: {e}",
                        "latency_s": round(time.time() - t0, 2), "output_tokens": None,
                    }
                async with lock:
                    out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    state["done"] += 1
                    state["ok"] += int(result["ok"])
                    if state["done"] % 25 == 0 or state["done"] == state["total"]:
                        print(f"[{state['done']}/{state['total']}] ok={state['ok']} "
                              f"last={result['latency_s']}s", flush=True)


def load_done(out_path: Path) -> set[tuple[Any, str]]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            done.add((r["record_id"], r["cohort"]))
    return done


async def run(args: argparse.Namespace) -> int:
    records = [json.loads(l) for l in Path(args.corpus).open()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[dict, str]] = []
    for r in records:
        for cohort in target_cohorts(r):
            tasks.append((r, cohort))

    if args.resume:
        done = load_done(out_path)
        before = len(tasks)
        tasks = [t for t in tasks if (t[0]["record_id"], t[1]) not in done]
        print(f"resume: skipping {before - len(tasks)} done; {len(tasks)} remain")
        mode = "a"
    else:
        if out_path.exists():
            out_path.unlink()
        mode = "w"

    print(f"{len(tasks)} (record, cohort) labeling tasks")
    if not tasks:
        out_path.touch()
        return 0

    endpoints = args.endpoints
    n = max(1, args.concurrency)
    chunks: list[list] = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        chunks[i % n].append(t)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    state = {"done": 0, "ok": 0, "total": len(tasks)}
    t0 = time.time()
    with out_path.open(mode, encoding="utf-8") as out_fp:
        await asyncio.gather(*[
            worker(chunks[i], endpoints[i % len(endpoints)], args.model,
                   args.temperature, out_fp, lock, sem, state)
            for i in range(n)
        ])
    print(f"done: {state['ok']}/{state['total']} ok in {time.time()-t0:.0f}s -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="../labeled_crane_corpus_sample.jsonl")
    p.add_argument("--out", default="out/llm_labels.jsonl")
    p.add_argument("--model", default="qwen3.6-35b-a3b")
    p.add_argument("--endpoints", nargs="+", default=["http://localhost:8000/v1"])
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
