"""Shared helpers for Stage 3 modules.

Provides the Stage-3 factor vocabulary (v4: the 47 CAUSAL factors from
schema_v4.json — outcomes are the target layer, not DAG nodes), the NTSB
Occurrence_Code -> CICTT mapping (kept in sync with build_kg_v4.py), and the
categorize() helper.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

_SCHEMA_V4 = Path(__file__).resolve().parent.parent.parent / "prompts" / "schema_v4.json"

# Outcomes are excluded from the causal-DAG node set: they are the target
# variable (see build_kg_v4.py outcome layer), and as near-universal terminal
# nodes they swamp PC's conditional-independence tests.
OUTCOME_FACTORS = {
    "GROUND_IMPACT", "WATER_IMPACT", "MIDAIR_COLLISION", "GROUND_COLLISION",
    "RUNWAY_EXCURSION_OR_OVERRUN", "INFLIGHT_BREAKUP", "EMERGENCY_LANDING",
    "SUCCESSFUL_RECOVERY", "INJURY_OR_FATALITY",
}

def _load_vocab() -> list[str]:
    raw = json.loads(_SCHEMA_V4.read_text())
    enum = raw["$defs"]["ChainNode"]["properties"]["factor_type"]["enum"]
    return [v for v in enum if v != "unknown" and v not in OUTCOME_FACTORS]

# v4 causal-factor vocabulary (47 = 35 events + 12 conditions), schema order.
# Matrix row/column ordering everywhere in Stage 3 — do not reorder without
# regenerating every per-category matrix.
EVENT_VOCAB = _load_vocab()
assert len(EVENT_VOCAB) == 47, f"expected 47 causal factors, got {len(EVENT_VOCAB)}"
VOCAB_INDEX = {et: i for i, et in enumerate(EVENT_VOCAB)}

# NTSB Occurrence_Code -> CICTT mapping (mirrors build_kg_layer1.py)
# NOTE (2026-09-30): the keys are legacy pre-2008 NTSB occurrence codes (ct_seqevt), and several
# labels do not match their meaning. The "SCF-PP" key holds 130 airframe/component/system failure,
# 131 propeller failure and 140 decompression, so the paper calls it the airframe failure category;
# 350 loss of engine power lands in "ARC" and 380 roll over in "LOC-I". Kept as is so the published
# per-category networks stay reproducible; remap before using any other category.
NTSB_OCC_TO_CICTT = {
    100: "LOC-I", 110: "LOC-I", 120: "OTHM", 130: "SCF-PP", 131: "SCF-PP",
    140: "SCF-PP", 160: "FUEL", 180: "USOS", 200: "ICE", 210: "F-NI",
    220: "WSTRW", 230: "CFIT", 240: "WX", 250: "LOC-I", 260: "LOC-G",
    270: "CTOL", 280: "GCOL", 290: "ARC", 300: "MAC", 310: "MAC",
    320: "GCOL", 330: "ARC", 340: "RE", 350: "ARC", 370: "ARC",
    380: "LOC-I", 390: "BIRD", 400: "ADRM", 410: "EVAC", 420: "ATM",
    430: "RI", 440: "FUEL", 540: "OTHM", 560: "EXTL", 580: "MED", 600: "SEC",
}

def categorize(record: dict) -> str:
    s = record.get("structured") or {}
    src = record.get("source", "?")
    occs = s.get("occurrences") or []
    for o in occs:
        oc = o.get("occurrence_code")
        if isinstance(oc, int) and oc in NTSB_OCC_TO_CICTT:
            return NTSB_OCC_TO_CICTT[oc]
    if occs:
        oc = occs[0].get("occurrence_code")
        if oc is not None:
            return f"OTHER:NTSB-{oc}"
    findings = s.get("findings") or []
    if findings:
        cat = findings[0].get("category")
        if cat:
            return f"HFACS_CAT_{cat}"
    cgc = s.get("general_cause_category")
    if cgc:
        return f"AIDS_{cgc.strip().split(',')[0].split()[0].upper()}"
    return f"UNCATEGORIZED:{src.split(':')[0]}"

def stream_records_by_category(extraction_path: Path, enriched_path: Path) -> Iterator[tuple[str, dict]]:
    """Yield (category, extraction_record) pairs."""
    cat_by_id: dict[str, str] = {}
    with enriched_path.open() as f:
        for line in f:
            r = json.loads(line)
            cat_by_id[r["record_id"]] = categorize(r)
    with extraction_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok"):
                continue
            yield cat_by_id.get(r["record_id"], "UNCATEGORIZED:?"), r

def event_indices_in_record(record: dict) -> dict:
    """Map local node key -> EVENT_VOCAB index for in-vocabulary factors.

    v4 chain format: key is the chain node's integer idx, factor from
    factor_type. v3 node/edge format (legacy): key is the node's string id,
    factor from event_type."""
    out: dict = {}
    if "chain" in record:
        for n in (record.get("chain") or []):
            if not isinstance(n, dict):
                continue
            ft = n.get("factor_type")
            if ft in VOCAB_INDEX:
                out[n.get("idx")] = VOCAB_INDEX[ft]
        return out
    for n in (record.get("nodes") or []):
        if not isinstance(n, dict) or n.get("kind") != "event":
            continue
        et = n.get("event_type")
        if et in VOCAB_INDEX:
            out[n.get("id")] = VOCAB_INDEX[et]
    return out


def precedence_pairs_in_record(record: dict, local_to_idx: dict) -> set[tuple[int, int]]:
    """Type-level (src_vocab_idx, dst_vocab_idx) causal-precedence pairs for
    one record, deduped. v4: from chain caused_by back-references (all
    strengths — both are causal). v3 legacy: from typed edges."""
    pairs: set[tuple[int, int]] = set()
    if "chain" in record:
        for n in (record.get("chain") or []):
            if not isinstance(n, dict):
                continue
            di = local_to_idx.get(n.get("idx"))
            if di is None:
                continue
            for link in (n.get("caused_by") or []):
                si = local_to_idx.get(link.get("src"))
                if si is None or si == di:
                    continue
                pairs.add((si, di))
        return pairs
    v3_types = {"CAUSES", "CONTRIBUTES_TO", "TRIGGERS", "ENABLES", "PRECEDES"}
    for e in (record.get("edges") or []):
        if not isinstance(e, dict) or e.get("type") not in v3_types:
            continue
        si = local_to_idx.get(e.get("src"))
        di = local_to_idx.get(e.get("dst"))
        if si is None or di is None or si == di:
            continue
        pairs.add((si, di))
    return pairs
