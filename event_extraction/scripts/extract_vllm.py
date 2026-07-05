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

# validate_extraction.py lives in the same directory; scripts/ is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_extraction import load_def_enums, Report, validate_v4  # noqa: E402


# --------------------------------------------------------------------------
# v4 grounded mode — supervision block rendering
# --------------------------------------------------------------------------
MAX_SEQ_ENTRIES = 30
MAX_FINDINGS = 15


def render_supervision_block(structured: dict | None) -> str | None:
    """Render NTSB / FAA AIDS structured supervision into the STRUCTURED
    FINDINGS prompt block. Returns None when there is nothing usable.

    The format here is a contract with few_shot_v4.json (built via
    build_fewshot_v4.py using this same function) and system_v4.txt — change
    all three together."""
    if not structured:
        return None
    lines: list[str] = []

    pct = (structured.get("primary_cause_text") or "").strip()
    if pct:
        lines.append(f"Primary cause: {pct}")
    sct = (structured.get("secondary_cause_text") or "").strip() if structured.get("secondary_cause_text") else ""
    if sct:
        lines.append(f"Secondary cause: {sct}")

    occs = structured.get("occurrences") or []
    if occs:
        lines.append("Occurrences:")
        for i, o in enumerate(occs, 1):
            phase = (o.get("phase_text") or "").strip()
            lines.append(f"  {i}. {(o.get('code_text') or '?').strip()}"
                         + (f" (phase: {phase})" if phase else ""))

    seq = structured.get("seq_of_events") or []
    if seq:
        lines.append("Sequence of events:")
        for e in seq[:MAX_SEQ_ENTRIES]:
            subj = (e.get("subj_text") or "").strip()
            if not subj or subj == "None":
                continue
            cf = (e.get("cause_factor") or "").strip() or "-"
            mod = (e.get("modifier_text") or "").strip()
            lines.append(f"  [occ {e.get('occurrence_no', '?')}] [{cf}] {subj}"
                         + (f" - {mod}" if mod else ""))
        if len(seq) > MAX_SEQ_ENTRIES:
            lines.append(f"  ... ({len(seq) - MAX_SEQ_ENTRIES} more entries omitted)")

    findings = structured.get("findings") or []
    if findings:
        lines.append("Findings:")
        for fi in findings[:MAX_FINDINGS]:
            cf = (fi.get("cause_factor") or "").strip() or "-"
            lines.append(f"  [{cf}] {(fi.get('description') or '?').strip()}")
        if len(findings) > MAX_FINDINGS:
            lines.append(f"  ... ({len(findings) - MAX_FINDINGS} more findings omitted)")

    inj = structured.get("injury") or {}
    ac = structured.get("aircraft") or {}
    tail_bits = []
    if inj:
        tail_bits.append("Injury: " + ", ".join(
            f"{inj.get(k, 0) or 0} {k}" for k in ("fatal", "serious", "minor", "none")))
    dmg = (ac.get("damage") or "").strip()
    if dmg:
        tail_bits.append(f"Aircraft damage: {dmg}")
    if tail_bits:
        lines.append(". ".join(tail_bits) + ".")

    # Occurrence/injury metadata alone (no causes, no sequence) isn't worth a block.
    if not pct and not seq and not findings:
        return None
    return "STRUCTURED FINDINGS (official investigation):\n" + "\n".join(lines)


def build_user_content(narrative: str, structured: dict | None, supervision: str) -> str:
    content = f"NARRATIVE:\n{narrative}"
    if supervision == "grounded":
        block = render_supervision_block(structured)
        if block:
            content += f"\n\n{block}"
    return content


def _load_prompts(
    system_file: str = "system_v4.txt",
    fewshot_file: str = "few_shot_v4.json",
) -> tuple[str, list[dict[str, Any]]]:
    """Load the canonical system prompt and few-shot examples."""
    system = (PROMPTS_DIR / system_file).read_text(encoding="utf-8").strip()
    fewshot = json.loads((PROMPTS_DIR / fewshot_file).read_text(encoding="utf-8"))
    return system, fewshot


def _load_guided_schema(schema_file: str | None) -> dict[str, Any] | None:
    """Load a JSON Schema for vLLM `guided_json` constrained generation."""
    if not schema_file:
        return None
    return json.loads((PROMPTS_DIR / schema_file).read_text(encoding="utf-8"))


def build_messages(system: str, fewshot: list[dict[str, Any]], user_content: str) -> list[dict[str, Any]]:
    """v4 few-shot examples carry a prebuilt 'user' field (narrative +
    optional STRUCTURED FINDINGS block); v1/v3 examples carry 'narrative'."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in fewshot:
        content = ex.get("user") or f"NARRATIVE:\n{ex['narrative']}"
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": json.dumps(ex["extraction"])})
    msgs.append({"role": "user", "content": user_content})
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


def _parse_chain_output(raw: str) -> dict[str, Any]:
    """Parse the v4 chain-mode JSON: {"chain": [...], "outcome_severity": "..."}."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in response")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj.get("chain"), list):
        raise ValueError("chain must be an array")
    return obj


def _validate_chain(obj: dict[str, Any], enums: dict[str, dict[str, set]]) -> list[str]:
    """Run the structural checks guided_json cannot express (index
    monotonicity, back-reference range, outcome-as-cause) plus enum
    membership. Returns the problem list (empty = valid)."""
    rep = Report()
    return validate_v4(obj, enums, rep)


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
    guided_schema: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen 3.x thinking mode wastes output tokens on reasoning traces.
        # Disable via the chat-template extra kwargs that vLLM forwards.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if guided_schema is not None:
        # IMPORTANT: this vLLM version (0.19.x) silently IGNORES the legacy
        # `guided_json` body field — verified 2026-07-05 by sending an
        # out-of-enum value, which came back unconstrained. That is the root
        # cause of the v3 corpus drift (82% of records carried enum
        # violations). The OpenAI-standard `response_format json_schema` IS
        # enforced, so use that.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": guided_schema,
                            "strict": True},
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
    guided_schema: dict[str, Any] | None = None,
    supervision: str = "none",
) -> None:
    ep = endpoints[idx % len(endpoints)]
    chain_mode = bool((guided_schema or {}).get("properties", {}).get("chain"))
    chain_enums = load_def_enums(PROMPTS_DIR / "schema_v4.json") if chain_mode else None
    async with httpx.AsyncClient() as client:
        for rec in records:
            async with sem:
                t0 = time.time()
                user_content = build_user_content(
                    rec["text"], rec.get("_structured"), supervision)
                messages = build_messages(system, fewshot, user_content)
                try:
                    raw, usage = await _one_call(
                        client, ep, model, messages, max_tokens, temperature,
                        guided_schema=guided_schema,
                    )
                    in_tok = usage.get("prompt_tokens")
                    out_tok = usage.get("completion_tokens")
                    if chain_mode:
                        obj = _parse_chain_output(raw)
                        problems = _validate_chain(obj, chain_enums)
                        retried = False
                        if problems:
                            # One repair round-trip: show the model its own
                            # output and the specific violations.
                            repair = (
                                "Your extraction violated these structural rules:\n- "
                                + "\n- ".join(problems[:12])
                                + "\nRe-emit the FULL corrected JSON object. Remember: "
                                "every caused_by src must be STRICTLY LESS than the "
                                "node's own idx (reorder the chain if needed), idx must "
                                "equal array position, and outcome nodes may only be "
                                "cited as src by other outcome nodes."
                            )
                            messages = messages + [
                                {"role": "assistant", "content": raw},
                                {"role": "user", "content": repair},
                            ]
                            raw, usage2 = await _one_call(
                                client, ep, model, messages, max_tokens, temperature,
                                guided_schema=guided_schema,
                            )
                            in_tok = (in_tok or 0) + (usage2.get("prompt_tokens") or 0)
                            out_tok = (out_tok or 0) + (usage2.get("completion_tokens") or 0)
                            obj = _parse_chain_output(raw)
                            problems = _validate_chain(obj, chain_enums)
                            retried = True
                        sanitized = False
                        if problems:
                            # Same-type caused_by links are legitimate at
                            # instance level (two successive CONTROL_INPUT_
                            # IMPROPER events) and only collapse to self-loops
                            # after type aggregation — the KG builder drops
                            # them anyway. If they are the ONLY residual
                            # problem, strip the links instead of discarding
                            # the record.
                            fatal = [p for p in problems
                                     if not p.startswith("type_level_self_loop")]
                            if fatal:
                                raise ValueError(
                                    "validation failed after retry: " + "; ".join(fatal[:8]))
                            by_idx = {n.get("idx"): n for n in obj["chain"]
                                      if isinstance(n, dict)}
                            for n in obj["chain"]:
                                n["caused_by"] = [
                                    l for l in (n.get("caused_by") or [])
                                    if (by_idx.get(l.get("src")) or {}).get("factor_type")
                                    != n.get("factor_type")]
                            sanitized = True
                        row = {
                            "record_id": rec["record_id"],
                            "source": rec["source"],
                            "ok": True,
                            "chain": obj["chain"],
                            "outcome_severity": obj.get("outcome_severity", "unknown"),
                            "retried": retried,
                            "sanitized": sanitized,
                            "error": None,
                            "latency_s": time.time() - t0,
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                        }
                    else:
                        nodes, edges = _parse_model_output(raw)
                        row = ExtractionResult(
                            record_id=rec["record_id"],
                            source=rec["source"],
                            ok=True,
                            nodes=nodes,
                            edges=edges,
                            raw=raw,
                            error=None,
                            latency_s=time.time() - t0,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                        ).__dict__
                except Exception as e:
                    row = {
                        "record_id": rec["record_id"],
                        "source": rec["source"],
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "latency_s": time.time() - t0,
                    }
                    if not chain_mode:
                        row.update({"nodes": [], "edges": [], "raw": None,
                                    "input_tokens": None, "output_tokens": None})
                async with out_lock:
                    out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    pbar_state["done"] += 1
                    if row["ok"]:
                        pbar_state["ok"] += 1
                    if row.get("retried"):
                        pbar_state["retried"] = pbar_state.get("retried", 0) + 1
                    if pbar_state["done"] % 10 == 0 or pbar_state["done"] == pbar_state["total"]:
                        print(
                            f"[{pbar_state['done']}/{pbar_state['total']}] "
                            f"ok={pbar_state['ok']} "
                            f"retried={pbar_state.get('retried', 0)} "
                            f"last_latency={row['latency_s']:.1f}s "
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
    pairs that were SUCCESSFULLY extracted (ok=true). Failed records (ok=false
    — typically connection errors when the server died) are NOT considered
    done, so resume will retry them."""
    if not out_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok"):
                continue
            done.add((r.get("source", ""), r.get("record_id", "")))
    return done


async def run(args: argparse.Namespace) -> int:
    system, fewshot = _load_prompts(args.system_file, args.fewshot_file)
    guided_schema = _load_guided_schema(args.guided_json)
    if guided_schema is not None:
        print(f"guided_json: enforcing schema {args.guided_json}")
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

    # Grounded mode: join structured supervision onto the selected records.
    # Only the needed records' blobs are kept in memory.
    if args.supervision == "grounded":
        wanted = {r["record_id"] for r in records}
        structured_by_id: dict[str, dict] = {}
        with open(args.enriched, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("record_id") in wanted and r.get("structured"):
                    structured_by_id[r["record_id"]] = r["structured"]
        n_hit = 0
        for r in records:
            s = structured_by_id.get(r["record_id"])
            if s is not None:
                r["_structured"] = s
                n_hit += 1
        print(f"supervision=grounded: structured data joined for "
              f"{n_hit}/{len(records)} records (others run narrative-only)")

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
                    guided_schema=guided_schema,
                    supervision=args.supervision,
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
    ap.add_argument(
        "--system-file",
        default="system_v4.txt",
        help="filename in event_extraction/prompts/ to use as the system prompt. "
        "Default is the current v4 chain prompt; pass system_v3.txt to "
        "reproduce the v3 baseline.",
    )
    ap.add_argument(
        "--fewshot-file",
        default="few_shot_v4.json",
        help="filename in event_extraction/prompts/ for few-shot examples "
        "(few_shot_v3.json for the v3 baseline).",
    )
    ap.add_argument(
        "--guided-json",
        default="schema_v4.json",
        help="filename in event_extraction/prompts/ of a JSON Schema for "
        "vLLM constrained generation (sent as response_format json_schema; "
        "the legacy guided_json body field is ignored by vLLM 0.19.x). "
        "Schemas with a top-level 'chain' property switch the runner to "
        "v4 chain mode (structural validation + one repair retry). "
        "Pass an empty string to disable constrained generation.",
    )
    ap.add_argument(
        "--supervision",
        choices=["none", "grounded"],
        default="none",
        help="grounded: append the STRUCTURED FINDINGS block (NTSB Findings / "
        "seq_of_events / occurrences from --enriched) to each record's prompt. "
        "Records without structured data fall back to narrative-only.",
    )
    ap.add_argument(
        "--enriched",
        default=Path("data/corpus/corpus_enriched.jsonl"),
        type=Path,
        help="enriched corpus JSONL carrying the 'structured' supervision "
        "field, used by --supervision grounded.",
    )
    args = ap.parse_args(argv)
    if not args.endpoints:
        args.endpoints = ["http://localhost:8000/v1"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
