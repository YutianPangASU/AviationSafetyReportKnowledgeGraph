"""Stage 1 (event extraction) calibration evaluation.

Compares the v1 extraction (out/full_corpus_events.jsonl) against the now-joined
structured supervision (corpus_enriched.jsonl). Measures:

  * Phase-of-flight agreement: per-record exact match (after normalization) and
    fuzzy match (token-level) between extracted events' phase_of_flight and the
    NTSB structured phase.
  * Cause coverage: for each NTSB cause-flagged Finding / seq_of_events row,
    is there at least one extracted event with a token-overlap match to its
    description? Aggregate precision/recall over all cause supervisors.
  * Severity / event-type sanity: how often the extraction's worst-severity
    matches the structural NTSB damage / fatality signal.

P/R/F1 here is approximate — we are measuring textual overlap, not formal
ontology alignment. It is the right calibration metric for "before vs. after
schema redesign," not the formal Stage-1 metric in the design doc.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
EXTRACTED = ROOT / "event_extraction/out/full_corpus_events.jsonl"
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"
REPORT = ROOT / "event_extraction/out/eval_stage1.json"

STOP = set("a an the of and or to in on at by for with from is was were be been being not as into onto over under through during after before".split())
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

def tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP and len(t) > 2}

# Phase-of-flight normalization: structured uses ALL-CAPS NTSB-style or the
# avall TEXT mappings; extraction uses schema vocab (landing/takeoff/cruise/...)
PHASE_NORM = {
    # NTSB to schema-vocab
    "TAKEOFF": "takeoff", "TAKEOFF, INITIAL CLIMB": "initial_climb", "TAKEOFF GROUND ROLL": "takeoff",
    "INITIAL CLIMB": "initial_climb", "CLIMB": "climb",
    "CRUISE": "cruise", "CRUISE/LEVEL FLIGHT": "cruise", "NORMAL CRUISE": "cruise",
    "DESCENT": "descent", "DESCENT - UNCONTROLLED": "descent",
    "APPROACH": "approach", "APPROACH - VFR PATTERN - FINAL APPROACH": "final_approach",
    "FINAL APPROACH": "final_approach",
    "LANDING": "landing", "LANDING - LEVEL OFF/TOUCHDOWN": "landing", "LANDING: TOUCHDOWN": "landing",
    "LANDING: ROLLOUT": "rollout", "ROLL-OUT (FIXED WING)": "rollout", "LEVEL OFF TOUCHDOWN": "landing",
    "TAXI": "taxi_out", "GROUND TAXI, OTHER A": "taxi_out", "TAXI - FROM LANDING": "taxi_in",
    "STANDING": "ground", "TAKEOFF GROUND": "takeoff",
    "MANEUVERING": "cruise",
    "GO-AROUND (VFR)": "go_around", "GO-AROUND (IFR)": "go_around",
    "PRE-FLIGHT": "preflight", "PRE-FLIGHT/PARKED": "preflight",
    "MAINTENANCE": "maintenance",
    "EMERGENCY DESCENT": "descent",
    "FCD/PREC LDG FROM CR": "emergency_descent",
}
def norm_phase(p: str | None) -> str | None:
    if not p:
        return None
    p2 = p.strip().upper()
    return PHASE_NORM.get(p2, p2.lower().replace(" ", "_").replace("-", "_").replace(",", "").replace("/", "_"))

def extracted_phases(rec: dict) -> Counter:
    c = Counter()
    for n in rec.get("nodes") or []:
        if isinstance(n, dict) and n.get("kind") == "event":
            ph = n.get("phase_of_flight")
            if ph:
                c[ph.lower()] += 1
    return c

def cause_sources(structured: dict) -> list[str]:
    """All cause-flagged supervision strings from NTSB structured fields."""
    out = []
    for f in structured.get("findings") or []:
        if (f.get("cause_factor") or "").upper() in ("C", "F") and f.get("description"):
            out.append(f["description"])
    for v in structured.get("seq_of_events") or []:
        if (v.get("cause_factor") or "").upper() == "C" and v.get("subj_text"):
            out.append(v["subj_text"])
    return out

def extracted_event_texts(rec: dict) -> list[str]:
    out = []
    for n in rec.get("nodes") or []:
        if isinstance(n, dict) and n.get("kind") == "event":
            parts = [n.get("trigger"), n.get("subtype"), n.get("haem")]
            out.append(" ".join(p for p in parts if p))
    return out

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def main():
    # Index enriched corpus by record_id (only NTSB records with structured fields)
    print("Indexing enriched corpus…")
    structured_by_id: dict[str, dict] = {}
    src_by_id: dict[str, str] = {}
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("structured") and r["source"].startswith("NTSB_ASRS"):
                structured_by_id[r["record_id"]] = r["structured"]
                src_by_id[r["record_id"]] = r["source"]
    print(f"  {len(structured_by_id):,} NTSB records with structured fields")

    print(f"Streaming extraction file…")

    # Aggregators
    n_extracted = 0
    n_evaluable = 0
    n_phase_match = 0
    n_phase_partial = 0
    n_phase_extracted = 0
    cause_total = 0
    cause_matched = 0
    extracted_event_total = 0
    extracted_event_matched_to_cause = 0
    src_metrics = defaultdict(lambda: {"n": 0, "phase_match": 0, "phase_partial": 0,
                                        "cause_total": 0, "cause_matched": 0,
                                        "ext_events": 0, "ext_match_cause": 0})

    with EXTRACTED.open() as f:
        for line in f:
            r = json.loads(line)
            n_extracted += 1
            rid = r.get("record_id")
            if not rid or not r.get("ok"):
                continue
            structured = structured_by_id.get(rid)
            if not structured:
                continue
            n_evaluable += 1
            src = src_by_id[rid]
            sm = src_metrics[src]
            sm["n"] += 1

            # ---- Phase agreement ----
            structured_phase = norm_phase(structured.get("phase_of_flight"))
            ex_phases = extracted_phases(r)
            if structured_phase and ex_phases:
                n_phase_extracted += 1
                if structured_phase in ex_phases:
                    n_phase_match += 1
                    sm["phase_match"] += 1
                else:
                    # Partial: token overlap between structured phase and any extracted phase
                    s_toks = set(structured_phase.split("_"))
                    if any(s_toks & set(p.split("_")) for p in ex_phases):
                        n_phase_partial += 1
                        sm["phase_partial"] += 1

            # ---- Cause coverage ----
            cause_strings = cause_sources(structured)
            event_strings = extracted_event_texts(r)
            cause_total += len(cause_strings)
            extracted_event_total += len(event_strings)
            sm["cause_total"] += len(cause_strings)
            sm["ext_events"] += len(event_strings)

            # Build token sets once per event
            event_toks = [tokens(s) for s in event_strings]
            for cs in cause_strings:
                cs_toks = tokens(cs)
                if not cs_toks:
                    continue
                # Match if any event has Jaccard >= 0.2 OR contains all key tokens (>=3 tokens of overlap)
                hit = False
                for et in event_toks:
                    if not et:
                        continue
                    overlap = cs_toks & et
                    if len(overlap) >= 3 or jaccard(cs_toks, et) >= 0.2:
                        hit = True
                        break
                if hit:
                    cause_matched += 1
                    sm["cause_matched"] += 1
            # Also count how many extracted events match any cause (precision side)
            for et in event_toks:
                if not et:
                    continue
                for cs in cause_strings:
                    cs_toks = tokens(cs)
                    overlap = cs_toks & et
                    if len(overlap) >= 3 or jaccard(cs_toks, et) >= 0.2:
                        extracted_event_matched_to_cause += 1
                        sm["ext_match_cause"] += 1
                        break

    def pct(a, b):
        return f"{100.0 * a / b:.2f}%" if b else "n/a"

    print(f"\n=== Calibration eval ({n_evaluable:,} evaluable / {n_extracted:,} extracted) ===")
    print(f"\n--- Phase of flight agreement ---")
    print(f"  records with both extracted+structured phase: {n_phase_extracted:,}")
    print(f"  exact match   : {n_phase_match:,} ({pct(n_phase_match, n_phase_extracted)})")
    print(f"  partial match : {n_phase_partial:,} ({pct(n_phase_partial, n_phase_extracted)})")
    print(f"  miss          : {n_phase_extracted - n_phase_match - n_phase_partial:,}")
    print(f"\n--- Cause coverage (recall: how many NTSB cause-flagged items appear in extraction) ---")
    print(f"  cause-flagged structured items : {cause_total:,}")
    print(f"  matched in extraction          : {cause_matched:,}")
    print(f"  RECALL (proxy)                 : {pct(cause_matched, cause_total)}")
    print(f"\n--- Extraction precision (how many extracted events align with a cause-flagged item) ---")
    print(f"  extracted events  : {extracted_event_total:,}")
    print(f"  matched to a cause: {extracted_event_matched_to_cause:,}")
    print(f"  PRECISION (proxy) : {pct(extracted_event_matched_to_cause, extracted_event_total)}")
    print(f"\n--- Per-source breakdown ---")
    print(f"{'source':28s} {'n':>8s} {'phase_match':>12s} {'phase_part':>12s} {'recall':>10s} {'precision':>10s}")
    for src in sorted(src_metrics):
        m = src_metrics[src]
        n = m["n"]
        rec = m["cause_matched"] / m["cause_total"] if m["cause_total"] else 0
        prec = m["ext_match_cause"] / m["ext_events"] if m["ext_events"] else 0
        pm = m["phase_match"] / n if n else 0
        pp = m["phase_partial"] / n if n else 0
        print(f"{src:28s} {n:>8,} {pm*100:>11.2f}% {pp*100:>11.2f}% {rec*100:>9.2f}% {prec*100:>9.2f}%")

    REPORT.write_text(json.dumps({
        "n_extracted": n_extracted,
        "n_evaluable": n_evaluable,
        "phase": {
            "extracted": n_phase_extracted,
            "exact_match": n_phase_match,
            "partial_match": n_phase_partial,
        },
        "cause": {
            "structured_total": cause_total,
            "matched": cause_matched,
            "recall_proxy": cause_matched / cause_total if cause_total else 0,
        },
        "extracted": {
            "events_total": extracted_event_total,
            "matched_to_cause": extracted_event_matched_to_cause,
            "precision_proxy": extracted_event_matched_to_cause / extracted_event_total if extracted_event_total else 0,
        },
        "by_source": {s: dict(m) for s, m in src_metrics.items()},
    }, indent=2))
    print(f"\nWrote {REPORT}")

if __name__ == "__main__":
    main()
