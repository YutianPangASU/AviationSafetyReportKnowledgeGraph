"""Score gold-496 extractions against the investigation's own conclusions.

Two layers:

1. Structural (no LLM): output counts, chain length stats, severity agreement
   with the coded injury level where available.

2. Judge-based (cross-family judge, local Qwen via an OpenAI-compatible
   endpoint — the extractors are Claude models, so a Qwen judge avoids
   same-family judging):
   a. Gold decomposition (cached, once per report, shared by all models):
      the judge splits the probable cause statement and findings into discrete
      causal elements, each tagged causal=true/false (certification recitals
      and non-causal findings are excluded from recall).
   b. Coverage: for each (model, report), the judge marks each causal gold
      element covered/uncovered by the extracted chain, and each extracted
      non-outcome node as supported/unsupported by the gold. Yields
      per-report recall and precision; macro-averaged per model.

Usage:
  python3 event_extraction/gold496/score_gold.py                 # structural only
  python3 event_extraction/gold496/score_gold.py \
      --judge-endpoint http://localhost:8000/v1 --judge-model qwen3.6-35b-a3b
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "judge_cache"

def sev_from_injury(inj) -> str | None:
    """Worst severity from the gold injury counts dict."""
    if not isinstance(inj, dict):
        return None
    if inj.get("fatal"):
        return "fatal"
    if inj.get("serious"):
        return "serious_injury"
    if inj.get("minor"):
        return "minor_injury"
    return "aircraft_damage_only"

DECOMPOSE_PROMPT = """You are preparing an evaluation gold standard. Below are the official conclusions of an NTSB investigation: the probable cause statement and (possibly) a list of findings.

Split them into discrete causal elements. One element = one cause, contributing factor, causal condition, or causal event. Mark each element causal=true if it names something that caused or contributed to the accident/incident, causal=false if it is background, a certification recital, a survival-factors note, or a recommendation.

Return ONLY a JSON object: {"elements": [{"text": "<short paraphrase>", "causal": true|false}, ...]}

PROBABLE CAUSE:
%s

FINDINGS:
%s
"""

COVERAGE_PROMPT = """You are grading an automated causal-chain extraction against gold causal elements from the official investigation.

GOLD CAUSAL ELEMENTS:
%s

EXTRACTED CHAIN (idx, factor_type, label):
%s

For each gold element, decide whether the extracted chain contains a node (or clearly implied combination of nodes) expressing the same causal factor. Semantic match suffices; wording may differ.
For each extracted node, decide whether it is supported: it corresponds to a gold element OR to an evidently real part of the accident sequence (an intermediate event or the outcome). Mark unsupported only nodes that are fabricated, irrelevant, or contradict the gold.

Return ONLY a JSON object:
{"gold_covered": [true|false, ... one per gold element, same order],
 "node_supported": [true|false, ... one per extracted node, same order]}
"""


def chat(endpoint: str, model: str, prompt: str, timeout: float = 240.0) -> dict:
    import httpx

    last_err = None
    for attempt, temp in enumerate((0.0, 0.2)):
        r = httpx.post(
            f"{endpoint}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": 8192,
                # Qwen3.x templates default to thinking mode; disable it so the
                # budget goes to the JSON verdict (same as extract_vllm.py).
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = txt.split("\n", 1)[1] if txt.startswith("json") else txt
        start, end = txt.find("{"), txt.rfind("}")
        try:
            return json.loads(txt[start : end + 1])
        except json.JSONDecodeError as e:
            last_err = e
    raise last_err


def gold_elements(rec: dict, args) -> list[dict] | None:
    """Judge-decomposed causal elements, cached per report."""
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{rec['report_number']}.json"
    if fp.exists():
        return json.loads(fp.read_text())["elements"]
    g = rec["gold"]
    findings = "\n".join(f"- {x}" for x in g["findings_items"][:40]) or g["conclusions_text"][:6000] or "(none)"
    pc = g["probable_cause_text"][:4000] or g["coded_primary_cause"] or "(none)"
    if pc == "(none)" and findings == "(none)":
        return None
    out = chat(args.judge_endpoint, args.judge_model, DECOMPOSE_PROMPT % (pc, findings))
    fp.write_text(json.dumps(out))
    return out["elements"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-endpoint", default=None)
    ap.add_argument("--judge-model", default="qwen3.6-35b-a3b")
    ap.add_argument("--models", nargs="*", default=["opus", "sonnet", "haiku"])
    ap.add_argument("--limit", type=int, default=None, help="score only the first N reports (smoke test)")
    args = ap.parse_args()

    gold = {r["report_number"]: r for r in map(json.loads, open(HERE / "gold.jsonl"))}
    results = {}

    for model in args.models:
        outdir = HERE / "outputs" / model
        recs = {}
        for fp in sorted(outdir.glob("*.json")) if outdir.exists() else []:
            try:
                raw = fp.read_text().strip()
                # tolerate accidental markdown fences (same as validate_outputs)
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    raw = raw.split("\n", 1)[1] if raw.startswith("json") else raw
                recs[fp.stem] = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
        row = {"n_outputs": len(recs)}
        if recs:
            lens = [len(r.get("chain", [])) for r in recs.values()]
            row["chain_len_median"] = statistics.median(lens)
            row["chain_len_mean"] = round(statistics.fmean(lens), 2)
            # severity agreement vs coded injury
            agree = tot = 0
            for rn, r in recs.items():
                inj = (gold.get(rn, {}).get("gold", {}) or {}).get("injury")
                want = sev_from_injury(inj)
                if want:
                    tot += 1
                    agree += int(r.get("outcome_severity") == want)
            row["severity_agreement"] = round(agree / tot, 3) if tot else None

        if args.judge_endpoint and recs:
            recalls, precisions, judged = [], [], 0
            items = sorted(recs.items())[: args.limit] if args.limit else sorted(recs.items())
            for rn, r in items:
                grec = gold.get(rn)
                if grec is None or not r.get("chain"):
                    continue
                try:
                    elems = gold_elements(grec, args)
                except Exception as e:  # noqa: BLE001
                    print(f"decompose failed {rn}: {e}")
                    continue
                if not elems:
                    continue
                causal = [e for e in elems if e.get("causal")]
                if not causal:
                    continue
                chain_txt = "\n".join(
                    f"{n['idx']}. [{n.get('factor_type','?')}] {n.get('label','')}"
                    for n in r["chain"]
                )
                gold_txt = "\n".join(f"{i}. {e['text']}" for i, e in enumerate(causal))
                try:
                    v = chat(args.judge_endpoint, args.judge_model, COVERAGE_PROMPT % (gold_txt, chain_txt))
                    cov = v["gold_covered"][: len(causal)]
                    sup = v["node_supported"][: len(r["chain"])]
                    recalls.append(sum(map(bool, cov)) / len(causal))
                    if sup:
                        precisions.append(sum(map(bool, sup)) / len(sup))
                    judged += 1
                except Exception as e:  # noqa: BLE001
                    print(f"judge failed {model}/{rn}: {e}")
            if judged:
                row["judged_reports"] = judged
                row["cause_recall_macro"] = round(statistics.fmean(recalls), 3)
                row["node_support_macro"] = round(statistics.fmean(precisions), 3)
        results[model] = row

    out = HERE / "scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
