"""Compare v1 vs v3 extraction on the 10 LOC-I cases.

Reports:
  * Schema compliance (HAEM distribution; closed-vocab violations)
  * cause_role distribution (v3 only)
  * NTSB cause-coverage (token-overlap proxy, same metric as eval_stage1.py)
  * Per-case node/edge counts side-by-side
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
LOCI_CASES = ROOT / "event_extraction/out/loci_feasibility_cases.jsonl"
V1_FULL = ROOT / "event_extraction/out/full_corpus_events.jsonl"
V3 = ROOT / "event_extraction/out/loci_v3.jsonl"

STOP = set("a an the of and or to in on at by for with from is was were be been being not as into onto over under through during after before".split())
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
def tokens(t):
    if not t:
        return set()
    return {x.lower() for x in TOKEN_RE.findall(t) if x.lower() not in STOP and len(x) > 2}
def jaccard(a, b):
    return (len(a & b) / len(a | b)) if (a and b) else 0.0

def cause_strings(struct):
    out = []
    for f in struct.get("findings") or []:
        if (f.get("cause_factor") or "").upper() in ("C", "F") and f.get("description"):
            out.append(f["description"])
    for v in struct.get("seq_of_events") or []:
        if (v.get("cause_factor") or "").upper() == "C" and v.get("subj_text"):
            out.append(v["subj_text"])
    return out

def event_text(n):
    return " ".join(p for p in (n.get("trigger"), n.get("subtype"), n.get("event_type"), n.get("haem")) if p)

def coverage(extracted, cause_list):
    """Return (n_causes_matched, n_events_matching_a_cause)."""
    cause_toks = [tokens(c) for c in cause_list]
    event_toks = [tokens(event_text(n)) for n in (extracted.get("nodes") or [])
                  if isinstance(n, dict) and n.get("kind") == "event"]
    matched_causes = 0
    for ct in cause_toks:
        if not ct:
            continue
        for et in event_toks:
            if not et:
                continue
            if len(ct & et) >= 3 or jaccard(ct, et) >= 0.2:
                matched_causes += 1
                break
    matched_events = 0
    for et in event_toks:
        if not et:
            continue
        for ct in cause_toks:
            if not ct:
                continue
            if len(ct & et) >= 3 or jaccard(ct, et) >= 0.2:
                matched_events += 1
                break
    return matched_causes, matched_events, len(event_toks), len([c for c in cause_toks if c])

def main():
    # Load LOC-I cases (have structured + v1 extraction embedded)
    cases = []
    with LOCI_CASES.open() as f:
        for line in f:
            cases.append(json.loads(line))
    rid_to_case = {c["record_id"]: c for c in cases}

    # Load v3 extraction
    v3_by_rid = {}
    with V3.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok"):
                v3_by_rid[r["record_id"]] = r

    # v1 distributions across 10 cases
    v1_haem = Counter()
    v1_event_types = Counter()
    v3_haem = Counter()
    v3_event_types = Counter()
    v3_cause_roles = Counter()
    v1_total_events = 0
    v3_total_events = 0
    v3_violations = 0

    v1_cause_match = v3_cause_match = 0
    v1_cause_total = v3_cause_total = 0
    v1_event_match = v3_event_match = 0
    v1_event_total = v3_event_total = 0

    print(f"{'#':>2}  {'record_id':<22} {'v1 nodes/edges':>14}  {'v3 nodes/edges':>14}  {'v1 cov':>8} {'v3 cov':>8}")
    for c in cases:
        rid = c["record_id"]
        v1 = c["v1_extraction"]
        v3 = v3_by_rid.get(rid)
        if not v3:
            continue

        # v1 node accounting
        for n in v1.get("nodes") or []:
            if isinstance(n, dict) and n.get("kind") == "event":
                v1_total_events += 1
                v1_haem[n.get("haem", "?")] += 1
                v1_event_types[n.get("subtype", "?")] += 1

        # v3 node accounting + violation detection
        for n in v3.get("nodes") or []:
            if isinstance(n, dict) and n.get("kind") == "event":
                v3_total_events += 1
                h = n.get("haem", "?")
                if h not in {"H", "A", "E", "M", "UCA", "unknown"}:
                    v3_violations += 1
                v3_haem[h] += 1
                v3_event_types[n.get("event_type", "?")] += 1
                v3_cause_roles[n.get("cause_role", "?")] += 1

        cause_list = cause_strings(c)
        m1, e1, n1e, ncs = coverage(v1, cause_list)
        m3, e3, n3e, _   = coverage(v3, cause_list)
        v1_cause_match += m1; v3_cause_match += m3
        v1_cause_total += ncs; v3_cause_total += ncs
        v1_event_match += e1; v3_event_match += e3
        v1_event_total += n1e; v3_event_total += n3e

        print(f"{c['case_no']:>2}  {rid:<22} {len(v1.get('nodes') or []):>5}/{len(v1.get('edges') or []):<8} {len(v3.get('nodes') or []):>5}/{len(v3.get('edges') or []):<8}  {m1}/{ncs:<5}  {m3}/{ncs}")

    def pct(a, b): return f"{100*a/b:.1f}%" if b else "n/a"

    print()
    print(f"=== Schema compliance ===")
    print(f"v1 events: {v1_total_events}, OPERATIONAL HAEM: {v1_haem.get('OPERATIONAL', 0)} ({pct(v1_haem.get('OPERATIONAL', 0), v1_total_events)})")
    print(f"v3 events: {v3_total_events}, schema violations: {v3_violations}")

    print()
    print(f"=== HAEM distribution ===")
    for h in sorted(set(list(v1_haem) + list(v3_haem))):
        v1c = v1_haem.get(h, 0); v3c = v3_haem.get(h, 0)
        print(f"  {h:>14}  v1={v1c:>4} ({pct(v1c, v1_total_events):>6})   v3={v3c:>4} ({pct(v3c, v3_total_events):>6})")

    print()
    print(f"=== v3 cause_role distribution ===")
    for k in ("primary", "contributing", "outcome", "context"):
        c = v3_cause_roles.get(k, 0)
        print(f"  {k:>14}: {c:>4} ({pct(c, v3_total_events):>6})")

    print()
    print(f"=== v3 event_type usage (top 12) ===")
    for et, cnt in v3_event_types.most_common(12):
        print(f"  {et:<40}  {cnt}")

    print()
    print(f"=== Cause coverage delta (proxy: token-overlap vs NTSB cause-flagged factors) ===")
    print(f"  v1 cause recall : {v1_cause_match}/{v1_cause_total} = {pct(v1_cause_match, v1_cause_total)}")
    print(f"  v3 cause recall : {v3_cause_match}/{v3_cause_total} = {pct(v3_cause_match, v3_cause_total)}")
    print(f"  v1 event-precision: {v1_event_match}/{v1_event_total} = {pct(v1_event_match, v1_event_total)}")
    print(f"  v3 event-precision: {v3_event_match}/{v3_event_total} = {pct(v3_event_match, v3_event_total)}")

if __name__ == "__main__":
    main()
