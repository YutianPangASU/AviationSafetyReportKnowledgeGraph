"""Document-level LLM labeler: ONE IN/OUT/MAYBE per (report, cohort).

This is the fair, apples-to-apples task: the human annotators labeled each
report as a whole (their sentence labels are uniform within a report), so we ask
the model the same single question per report and cohort. Runs against the local
vLLM server (Qwen). Emits the same schema an API/subagent labeler produces, so
all models are scored identically:

    {"record_id": ..., "cohort": ..., "label": "IN"|"OUT"|"MAYBE", "evidence": "..."}

Usage:
    python doc_label.py --out out/doc_qwen.jsonl --endpoints http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from guidelines import system_prompt_doc
from label_crane import build_user_prompt, target_cohorts

GUIDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["IN", "OUT", "MAYBE"]},
        "evidence": {"type": "string"},
    },
    "required": ["label"],
}


def parse(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = text[:-3]
    obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    label = str(obj.get("label", "OUT")).upper()
    if label not in {"IN", "OUT", "MAYBE"}:
        label = "OUT"
    return label, str(obj.get("evidence", ""))[:200]


async def one_call(client, endpoint, model, system, user, temperature):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 256, "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": GUIDED_SCHEMA,
    }
    resp = await client.post(f"{endpoint}/chat/completions", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def worker(tasks, endpoint, model, temperature, out_fp, lock, sem, state):
    async with httpx.AsyncClient() as client:
        for record, cohort in tasks:
            async with sem:
                t0 = time.time()
                user = build_user_prompt(record)
                try:
                    raw = await one_call(client, endpoint, model,
                                         system_prompt_doc(cohort), user, temperature)
                    label, evidence = parse(raw)
                    res = {"record_id": record["record_id"], "cohort": cohort,
                           "label": label, "evidence": evidence, "ok": True,
                           "latency_s": round(time.time() - t0, 2)}
                except Exception as e:  # noqa: BLE001
                    res = {"record_id": record["record_id"], "cohort": cohort,
                           "label": "OUT", "evidence": "", "ok": False,
                           "error": f"{type(e).__name__}: {e}",
                           "latency_s": round(time.time() - t0, 2)}
                async with lock:
                    out_fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    state["done"] += 1
                    state["ok"] += int(res["ok"])
                    if state["done"] % 50 == 0 or state["done"] == state["total"]:
                        print(f"[{state['done']}/{state['total']}] ok={state['ok']}",
                              flush=True)


async def run(args):
    records = [json.loads(l) for l in Path(args.corpus).open()]
    tasks = [(r, c) for r in records for c in target_cohorts(r)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    print(f"{len(tasks)} (record, cohort) document-level tasks")

    n = max(1, args.concurrency)
    chunks: list[list] = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        chunks[i % n].append(t)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    state = {"done": 0, "ok": 0, "total": len(tasks)}
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out_fp:
        await asyncio.gather(*[
            worker(chunks[i], args.endpoints[i % len(args.endpoints)], args.model,
                   args.temperature, out_fp, lock, sem, state)
            for i in range(n)
        ])
    print(f"done {state['ok']}/{state['total']} in {time.time()-t0:.0f}s -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="../labeled_crane_corpus_sample.jsonl")
    p.add_argument("--out", default="out/doc_qwen.jsonl")
    p.add_argument("--model", default="qwen3.6-35b-a3b")
    p.add_argument("--endpoints", nargs="+", default=["http://localhost:8000/v1"])
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
