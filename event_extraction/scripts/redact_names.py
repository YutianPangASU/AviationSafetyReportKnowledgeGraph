"""Screen the released chains for named individuals and redact them (review 2026-10-10).

Every chain node quotes narrative text (trigger, label, and the evidence of
each parent), and some FAA and NTSB narratives name the people involved.
This script

  1. collects every quote in the released chain files that contains a
     capitalized word pair, the shape of a personal name,
  2. asks the local Qwen judge model to list the spans that name a private
     person, as opposed to an airport, place, aircraft model or company,
  3. keeps only spans that occur verbatim in their quote, and
  4. after a person reviews the spans, replaces each confirmed name with
     "[name]" in the quote fields of every released file, leaving all other
     fields unchanged.

The model over-flags, returning job titles, pronouns and airports named after
people, so step 4 reads a reviewed list, one span per line, from
event_extraction/out/redaction/names_confirmed.txt. The screen cache and the
list hold the names themselves and are not released.

Use:
    python event_extraction/scripts/redact_names.py screen --endpoints http://localhost:8000/v1
    python event_extraction/scripts/redact_names.py redact
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "event_extraction/out"
CACHE = OUT / "redaction/name_screen.jsonl"
CONFIRMED = OUT / "redaction/names_confirmed.txt"
SUMMARY = OUT / "redaction/redaction_summary.json"
JSONL_FILES = [
    OUT / "causation_kg/per_accident_chains.jsonl.gz",
    OUT / "full_corpus_v4.jsonl.gz",
    *sorted((OUT / "selfconsistency").glob("calibration_2k_v4_narrative_*.jsonl")),
]
FENCED_FILES = sorted((ROOT / "event_extraction/gold496/outputs").glob("*/*.json"))
QUOTE_KEYS = ("trigger", "label", "evidence")
SHAPE = re.compile(r"\b[A-Z][a-z]+\.?\s+(?:[A-Z]\.\s+)?[A-Z][a-z]{2,}")
BATCH = 25

SYSTEM = """You screen quoted snippets from aviation accident reports for the names of private individuals before the snippets are published.

For each snippet, list every span that names a specific person: a pilot, passenger, witness, victim, mechanic, controller, inspector, owner, homeowner or anyone else. Include a title that is part of the span, such as "Mr." or "Capt.". Copy each span exactly as it appears in the snippet.

Do not list a name that is part of a place, airport, airfield, mountain, lake, building, company, aircraft model or product, such as "Ronald Reagan Washington National", "Ted Stevens Anchorage", "Capt. Ben Smith Airfield", "Mt. Ben Stewart" or "Steven James model". Do not list job titles without a name.

Return JSON {"names": {"<snippet index>": ["span", ...]}} with entries only for snippets that contain a name."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["names"],
    "properties": {"names": {"type": "object", "additionalProperties": {
        "type": "array", "items": {"type": "string"}}}},
}


def read_jsonl(path: Path) -> list[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", compresslevel=9) as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fenced_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def quotes(chain) -> list[str]:
    out = []
    for n in chain or []:
        if not isinstance(n, dict):
            continue
        out += [n[k] for k in QUOTE_KEYS if isinstance(n.get(k), str)]
        for c in n.get("caused_by") or []:
            if isinstance(c, dict) and isinstance(c.get("evidence"), str):
                out.append(c["evidence"])
    return out


def candidate_quotes() -> list[str]:
    seen = set()
    for p in JSONL_FILES:
        for r in read_jsonl(p):
            seen.update(q for q in quotes(r.get("chain")) if SHAPE.search(q))
    for p in FENCED_FILES:
        obj = fenced_json(p.read_text())
        if obj:
            seen.update(q for q in quotes(obj.get("chain")) if SHAPE.search(q))
    return sorted(seen)


async def screen(endpoints: list[str], model: str, concurrency: int) -> None:
    cands = candidate_quotes()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        with CACHE.open() as f:
            for line in f:
                done.add(json.loads(line)["quote"])
    todo = [q for q in cands if q not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(cands)} candidate quotes, {len(todo)} pending in {len(batches)} batches")
    sems = [asyncio.Semaphore(concurrency) for _ in endpoints]
    lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with CACHE.open("a") as out:
            async def run(k: int, batch: list[str]) -> None:
                j = k % len(endpoints)
                user = "\n".join(f"[{i}] {q}" for i, q in enumerate(batch))
                body = {"model": model, "max_tokens": 1024, "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": user}],
                        "response_format": {"type": "json_schema", "json_schema": {
                            "name": "names", "schema": SCHEMA, "strict": True}}}
                async with sems[j]:
                    resp = await client.post(f"{endpoints[j]}/chat/completions", json=body,
                                             timeout=300)
                    resp.raise_for_status()
                names = json.loads(resp.json()["choices"][0]["message"]["content"])["names"]
                async with lock:
                    for i, q in enumerate(batch):
                        spans = [s for s in names.get(str(i), []) if s and s in q]
                        out.write(json.dumps({"quote": q, "names": spans}) + "\n")
                    out.flush()
            await asyncio.gather(*[run(k, b) for k, b in enumerate(batches)])


def redact() -> None:
    names = {ln.strip() for ln in CONFIRMED.read_text().splitlines() if ln.strip()}
    # a one-word span such as a bare surname would also hit place names when
    # replaced everywhere, so only spans of two or more words are redacted
    single = sorted(n for n in names if len(n.split()) < 2)
    if single:
        raise SystemExit(f"one-word spans in {CONFIRMED.name}: {single}")
    # longest first, so "Mr. Jane Example Doe" goes before "Jane Example"
    order = sorted(names, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(n) for n in order)) if order else None

    def clean(s: str) -> str:
        return pat.sub("[name]", s) if pat else s

    def clean_chain(chain) -> int:
        k = 0
        for n in chain or []:
            if not isinstance(n, dict):
                continue
            for key in QUOTE_KEYS:
                if isinstance(n.get(key), str) and (c := clean(n[key])) != n[key]:
                    n[key], k = c, k + 1
            for c in n.get("caused_by") or []:
                if isinstance(c, dict) and isinstance(c.get("evidence"), str):
                    if (e := clean(c["evidence"])) != c["evidence"]:
                        c["evidence"], k = e, k + 1
        return k

    per_file = {}
    for p in JSONL_FILES:
        rows = read_jsonl(p)
        k = sum(clean_chain(r.get("chain")) for r in rows)
        if k:
            write_jsonl(p, rows)
        per_file[str(p.relative_to(ROOT))] = k
    for p in FENCED_FILES:
        text = p.read_text()
        obj = fenced_json(text)
        if obj and clean_chain(obj.get("chain")):
            new = clean(text)  # same spans, applied to the raw file to keep its format
            p.write_text(new)
            per_file[str(p.relative_to(ROOT))] = 1
    SUMMARY.write_text(json.dumps({"distinct_names": len(names),
                                   "quotes_changed_per_file": per_file}, indent=2))
    print(json.dumps({"distinct_names": len(names),
                      "files_changed": sum(1 for v in per_file.values() if v)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["screen", "redact"])
    ap.add_argument("--endpoints", nargs="+", default=["http://localhost:8000/v1"])
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--concurrency", default=8, type=int)
    args = ap.parse_args()
    if args.step == "screen":
        asyncio.run(screen(args.endpoints, args.model, args.concurrency))
    else:
        redact()


if __name__ == "__main__":
    main()
