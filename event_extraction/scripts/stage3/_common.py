"""Shared helpers for Stage 3 modules.

Provides the 42-event vocabulary, NTSB Occurrence_Code -> CICTT mapping (kept in
sync with build_kg_layer1.py), and the categorize() helper.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

# 42-value Stage-3 vocabulary, frozen order — used as row/column index in matrices
EVENT_VOCAB = [
    # Pilot acts (HFACS Level 1)
    "DECISION_INAPPROPRIATE", "CONTROL_INPUT_IMPROPER", "PROCEDURE_NOT_FOLLOWED",
    "PERCEPTION_FAILURE", "SPATIAL_DISORIENTATION", "PILOT_INCAPACITATION_OR_IMPAIRMENT",
    "TRAINING_OR_CURRENCY_GAP", "CREW_COORDINATION_FAILURE",
    # External operators
    "ATC_OR_DISPATCH_INADEQUATE", "GROUND_PERSONNEL_ERROR",
    # Aircraft systems
    "ENGINE_FAILURE", "FUEL_SYSTEM_ANOMALY", "AIRFRAME_STRUCTURAL_FAILURE",
    "CONTROL_SURFACE_ANOMALY", "INSTRUMENT_OR_AVIONICS_FAILURE", "LANDING_GEAR_ANOMALY",
    "AUTOMATION_ANOMALY", "INFLIGHT_FIRE_OR_SMOKE",
    # Environmental
    "ICING_ENCOUNTER", "TURBULENCE_ENCOUNTER", "THUNDERSTORM_OR_CONVECTIVE",
    "LOW_VISIBILITY_OR_IMC", "WIND_SHEAR_OR_GUST", "BIRD_OR_WILDLIFE_STRIKE",
    "TERRAIN_OR_OBSTACLE_PROXIMITY", "RUNWAY_CONDITION_HAZARD",
    # Organizational
    "MAINTENANCE_INADEQUATE", "ORG_OR_REGULATORY_INADEQUATE", "DESIGN_DEFECT_LATENT",
    # Aerodynamic states
    "STALL", "LOSS_OF_CONTROL_INFLIGHT", "LOSS_OF_CONTROL_GROUND",
    "STRUCTURAL_OVERLOAD", "ALTITUDE_DEVIATION_UNCONTROLLED",
    # Outcomes
    "GROUND_IMPACT", "WATER_IMPACT", "MIDAIR_COLLISION", "RUNWAY_EXCURSION_OR_OVERRUN",
    "INFLIGHT_BREAKUP", "SUCCESSFUL_RECOVERY", "EMERGENCY_LANDING", "INJURY_OR_FATALITY",
]
assert len(EVENT_VOCAB) == 42
VOCAB_INDEX = {et: i for i, et in enumerate(EVENT_VOCAB)}

# NTSB Occurrence_Code -> CICTT mapping (mirrors build_kg_layer1.py)
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

def event_indices_in_record(record: dict) -> dict[str, int]:
    """Map local node id -> EVENT_VOCAB index, only for event nodes whose
    event_type is in the vocabulary."""
    out: dict[str, int] = {}
    for n in (record.get("nodes") or []):
        if not isinstance(n, dict) or n.get("kind") != "event":
            continue
        et = n.get("event_type")
        if et in VOCAB_INDEX:
            out[n.get("id")] = VOCAB_INDEX[et]
    return out
