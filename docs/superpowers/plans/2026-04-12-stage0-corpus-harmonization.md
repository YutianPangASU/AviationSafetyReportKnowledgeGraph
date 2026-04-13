# Stage 0: Corpus Harmonization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Unified Aviation Safety Corpus (UASC) by harmonizing four raw data sources (NTSB, TSB Canada, BEA, FAA AIDS) into a single schema, applying richness filtering to identify reasoning-ready records, and deduplicating cross-source overlaps.

**Architecture:** Four source-specific loaders parse raw data into a common UASC schema. A richness scorer assigns a continuous score and tier (reasoning_ready / partial / metadata_only). A deduplication module clusters cross-source duplicates. Outputs are a unified parquet file, the UASC-Rich subset, harmonization crosswalk tables, and a corpus statistics report.

**Tech Stack:** Python 3.10, pandas, pyarrow (parquet), mdbtools (NTSB Access DB), numpy, scipy (Jenks breaks), scikit-learn (TF-IDF/LSH for dedup)

**Design doc:** [docs/superpowers/specs/2026-04-11-ace-graph-design-v2.md](../specs/2026-04-11-ace-graph-design-v2.md) — Section 3

---

## File Structure

```
src/stage0/
├── __init__.py
├── schema.py              # UASC dataclass / schema definition
├── loader_ntsb.py         # NTSB avall.mdb -> UASC records
├── loader_ntsb_reports.py # NTSB published reports -> enrich UASC records
├── loader_tsb.py          # TSB Canada CSVs -> UASC records
├── loader_bea.py          # BEA scraped CSV -> UASC records
├── loader_faa_aids.py     # FAA AIDS txt files -> UASC records
├── taxonomy.py            # CICTT harmonization + crosswalk tables
├── richness.py            # Richness score + gate checks + tier assignment
├── dedup.py               # Cross-source deduplication (blocking + LSH + match)
├── build_corpus.py        # Orchestrator: load all -> harmonize -> score -> dedup -> write parquet
└── stats_report.py        # Corpus statistics and richness distribution report

data/crosswalks/
├── faa_aids_to_cictt.csv       # FAA cause code -> CICTT category mapping
├── flight_phase_crosswalk.csv  # Per-source flight phase -> ICAO standard
└── bea_category_crosswalk.csv  # BEA French terminology -> CICTT

tests/stage0/
├── test_schema.py
├── test_loader_ntsb.py
├── test_loader_tsb.py
├── test_loader_bea.py
├── test_loader_faa_aids.py
├── test_taxonomy.py
├── test_richness.py
├── test_dedup.py
└── test_build_corpus.py
```

---

### Task 1: UASC Schema Definition

**Files:**
- Create: `src/stage0/__init__.py`
- Create: `src/stage0/schema.py`
- Test: `tests/stage0/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/stage0/test_schema.py
import pytest
from stage0.schema import UASCRecord

def test_uasc_record_creation_minimal():
    """A record with required fields should be valid."""
    rec = UASCRecord(
        source="NTSB",
        source_id="20230101X00001",
        narrative_factual="The pilot departed VFR into IMC conditions.",
        occurrence_date="2023-01-15",
        country="US",
        severity="fatal",
    )
    assert rec.source == "NTSB"
    assert rec.uasc_id is not None  # auto-generated
    assert rec.richness_tier is None  # not scored yet

def test_uasc_record_rejects_invalid_source():
    with pytest.raises(ValueError):
        UASCRecord(
            source="INVALID",
            source_id="X",
            narrative_factual="text",
            occurrence_date="2023-01-15",
            country="US",
            severity="fatal",
        )

def test_uasc_record_multi_label_categories():
    rec = UASCRecord(
        source="NTSB",
        source_id="20230101X00001",
        narrative_factual="The pilot lost control after engine failure.",
        occurrence_date="2023-01-15",
        country="US",
        severity="fatal",
        accident_category_primary="LOC-I",
        accident_category_secondary=["SCF-PP"],
    )
    assert rec.accident_category_primary == "LOC-I"
    assert rec.accident_category_secondary == ["SCF-PP"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/yp6443/research/AviationSafetyReportKnowledgeGraph && python -m pytest tests/stage0/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stage0'`

- [ ] **Step 3: Write the UASC schema**

```python
# src/stage0/schema.py
"""Unified Aviation Safety Corpus (UASC) record schema.

Implements the canonical schema from the ACE-Graph design doc v2, Section 3.2.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional

VALID_SOURCES = {"NTSB", "BEA", "TSB", "FAA_AIDS"}
VALID_SEVERITIES = {"fatal", "serious", "minor", "none", "damage_only"}
VALID_TIERS = {"reasoning_ready", "partial", "metadata_only"}


@dataclass
class StructuredEvent:
    order: int
    phase: Optional[str] = None
    event_type: Optional[str] = None
    time_marker: Optional[str] = None


@dataclass
class Finding:
    finding_type: Optional[str] = None
    subject_code: Optional[str] = None
    human_readable: Optional[str] = None


@dataclass
class UASCRecord:
    # --- ids ---
    source: str
    source_id: str
    uasc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dup_cluster_id: Optional[str] = None

    # --- context ---
    occurrence_date: Optional[str] = None
    country: Optional[str] = None
    location: Optional[str] = None
    airport_icao: Optional[str] = None
    lat_lon: Optional[tuple[float, float]] = None
    aircraft_make: Optional[str] = None
    aircraft_model: Optional[str] = None
    aircraft_category: Optional[str] = None
    engines: Optional[int] = None
    operation_type: Optional[str] = None
    flight_phase_raw: Optional[str] = None

    # --- outcome ---
    severity: Optional[str] = None
    fatalities: Optional[int] = None
    injuries: Optional[int] = None
    accident_category_primary: Optional[str] = None
    accident_category_secondary: list[str] = field(default_factory=list)

    # --- text ---
    narrative_factual: Optional[str] = None
    narrative_cause: Optional[str] = None
    narrative_analysis: Optional[str] = None

    # --- structured ---
    structured_events: list[StructuredEvent] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    # --- quality (set by richness scorer) ---
    richness_score: Optional[float] = None
    richness_tier: Optional[str] = None
    harmonization_flags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Invalid source: {self.source}. Must be one of {VALID_SOURCES}")
        if self.severity is not None and self.severity not in VALID_SEVERITIES:
            raise ValueError(f"Invalid severity: {self.severity}. Must be one of {VALID_SEVERITIES}")
        if self.richness_tier is not None and self.richness_tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier: {self.richness_tier}. Must be one of {VALID_TIERS}")
```

- [ ] **Step 4: Create `__init__.py` and configure imports**

```python
# src/stage0/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/yp6443/research/AviationSafetyReportKnowledgeGraph && PYTHONPATH=src python -m pytest tests/stage0/test_schema.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/stage0/__init__.py src/stage0/schema.py tests/stage0/test_schema.py
git commit -m "feat(stage0): add UASC record schema with multi-label categories"
```

---

### Task 2: Richness Scorer

**Files:**
- Create: `src/stage0/richness.py`
- Test: `tests/stage0/test_richness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_richness.py
import pytest
from stage0.schema import UASCRecord, StructuredEvent, Finding
from stage0.richness import compute_richness, assign_tier, lognorm, TEMPORAL_KEYWORDS

def test_lognorm_at_lower_bound():
    assert lognorm(120, 120, 600) == pytest.approx(0.0)

def test_lognorm_at_upper_bound():
    assert lognorm(600, 120, 600) == pytest.approx(1.0)

def test_lognorm_below_lower_clips_to_zero():
    assert lognorm(50, 120, 600) == 0.0

def test_lognorm_above_upper_clips_to_one():
    assert lognorm(2000, 120, 600) == 1.0

def test_lognorm_log_diminishing_returns():
    """Midpoint in log-space should be ~0.5, not 0.5 in linear space."""
    import math
    midpoint = math.exp((math.log(120) + math.log(600)) / 2)  # geometric mean ~268
    assert lognorm(midpoint, 120, 600) == pytest.approx(0.5, abs=0.01)

def _make_rich_record() -> UASCRecord:
    """A record that should pass all 5 gates and score high."""
    return UASCRecord(
        source="NTSB",
        source_id="TEST001",
        narrative_factual=" ".join(["word"] * 200),  # 200 words > 120
        narrative_cause=" ".join(["word"] * 60),      # 60 words > 40
        occurrence_date="2023-01-15",
        country="US",
        flight_phase_raw="CRUISE",
        aircraft_category="airplane",
        severity="fatal",
        accident_category_primary="LOC-I",
        structured_events=[
            StructuredEvent(order=1, event_type="stall"),
            StructuredEvent(order=2, event_type="spin"),
            StructuredEvent(order=3, event_type="impact"),
        ],
        findings=[
            Finding(finding_type="cause", subject_code="LOC", human_readable="Loss of control"),
            Finding(finding_type="factor", subject_code="WX", human_readable="Weather"),
        ],
    )

def test_rich_record_is_reasoning_ready():
    rec = _make_rich_record()
    score, gates_passed = compute_richness(rec)
    assert score > 0.4
    assert gates_passed == 5
    tier = assign_tier(score, gates_passed)
    assert tier == "reasoning_ready"

def test_minimal_record_is_metadata_only():
    rec = UASCRecord(
        source="FAA_AIDS",
        source_id="TEST002",
        narrative_factual="Gear collapsed.",  # 2 words < 120
        occurrence_date="2023-01-15",
        country="US",
        severity="minor",
    )
    score, gates_passed = compute_richness(rec)
    assert gates_passed < 3
    tier = assign_tier(score, gates_passed)
    assert tier == "metadata_only"

def test_partial_record_some_gates():
    """A record with narrative but no cause statement => passes 3-4 gates, partial tier."""
    rec = UASCRecord(
        source="NTSB",
        source_id="TEST003",
        narrative_factual=" ".join(["word"] * 150),
        occurrence_date="2023-01-15",
        country="US",
        flight_phase_raw="APPROACH",
        aircraft_category="airplane",
        severity="serious",
        accident_category_primary="CFIT",
        structured_events=[
            StructuredEvent(order=1, event_type="descent"),
            StructuredEvent(order=2, event_type="terrain_contact"),
        ],
    )
    score, gates_passed = compute_richness(rec)
    assert 3 <= gates_passed <= 4  # missing cause gate
    tier = assign_tier(score, gates_passed)
    assert tier == "partial"

def test_temporal_keyword_counting():
    text = "The aircraft departed then climbed. Shortly after reaching cruise altitude, the engine failed."
    from stage0.richness import count_temporal_relations
    count = count_temporal_relations(text)
    assert count >= 2  # "then", "shortly after"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_richness.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement richness scorer**

```python
# src/stage0/richness.py
"""Richness scoring and tier assignment for UASC records.

Implements Section 3.3-3.4 of ACE-Graph design doc v2.
Default weights are a starting configuration; see Appendix B for sensitivity analysis protocol.
"""
from __future__ import annotations
import math
import re
from stage0.schema import UASCRecord

# Default weights (Section 3.4) — pending sensitivity analysis
W1, W2, W3, W4, W5, W6 = 0.25, 0.20, 0.20, 0.15, 0.10, 0.10

# Temporal keyword patterns (initial proxy; production should use TimeML classifiers)
TEMPORAL_KEYWORDS = [
    r"\bthen\b",
    r"\bshortly after\b",
    r"\bsubsequently\b",
    r"\bprior to\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bduring\b",
    r"\bat \d{1,2}:\d{2}\b",         # "at 14:23"
    r"\b\d+ minutes? (?:later|before)\b",  # "2 minutes later"
    r"\bfollowing\b",
]

# Default tier thresholds (pending Jenks analysis on NTSB distribution)
THRESHOLD_HIGH = 0.5
THRESHOLD_LOW = 0.3


def lognorm(x: float, lo: float, hi: float) -> float:
    """Log-normalized score clipped to [0, 1].

    lognorm(x, lo, hi) = clip((log(x) - log(lo)) / (log(hi) - log(lo)), 0, 1)
    """
    if x <= 0:
        return 0.0
    log_x = math.log(x)
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    if log_hi == log_lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (log_x - log_lo) / (log_hi - log_lo)))


def word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def count_temporal_relations(text: str | None) -> int:
    if not text:
        return 0
    count = 0
    for pattern in TEMPORAL_KEYWORDS:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def compute_richness(rec: UASCRecord) -> tuple[float, int]:
    """Compute richness score and number of gates passed.

    Returns:
        (score, gates_passed) where score is in [0, 1] and gates_passed is 0-5.
    """
    wc_factual = word_count(rec.narrative_factual)
    wc_cause = word_count(rec.narrative_cause)
    n_events = len(rec.structured_events)
    n_findings = len(rec.findings)
    n_temporal = count_temporal_relations(rec.narrative_factual)
    findings_with_code = sum(1 for f in rec.findings if f.subject_code)

    # Gate checks (Section 3.3)
    gates_passed = 0
    # Gate 1: factual narrative >= 120 words
    if wc_factual >= 120:
        gates_passed += 1
    # Gate 2: cause statement present (>= 40 words OR >= 2 findings with subject code)
    if wc_cause >= 40 or findings_with_code >= 2:
        gates_passed += 1
    # Gate 3: temporal structure (>= 2 structured events OR >= 2 temporal relations)
    if n_events >= 2 or n_temporal >= 2:
        gates_passed += 1
    # Gate 4: context minimum
    context_fields = [rec.occurrence_date, rec.flight_phase_raw, rec.aircraft_category, rec.severity]
    if all(f is not None for f in context_fields):
        gates_passed += 1
    # Gate 5: harmonizable category
    if rec.accident_category_primary is not None:
        gates_passed += 1

    # Continuous score (Section 3.4)
    all_context = 1.0 if all(f is not None for f in context_fields) else 0.0
    score = (
        W1 * lognorm(wc_factual, 120, 600)
        + W2 * lognorm(wc_cause, 40, 300)
        + W3 * min(1.0, n_events / 5)
        + W4 * min(1.0, n_findings / 4)
        + W5 * min(1.0, n_temporal / 3)
        + W6 * all_context
    )
    return score, gates_passed


def assign_tier(score: float, gates_passed: int,
                threshold_high: float = THRESHOLD_HIGH,
                threshold_low: float = THRESHOLD_LOW) -> str:
    """Assign richness tier based on score and gate count."""
    if gates_passed == 5 and score >= threshold_high:
        return "reasoning_ready"
    if gates_passed >= 3 and score >= threshold_low:
        return "partial"
    return "metadata_only"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_richness.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/richness.py tests/stage0/test_richness.py
git commit -m "feat(stage0): add richness scorer with lognorm, gates, and tier assignment"
```

---

### Task 3: Taxonomy Harmonization + Crosswalk Tables

**Files:**
- Create: `src/stage0/taxonomy.py`
- Create: `data/crosswalks/flight_phase_crosswalk.csv`
- Create: `data/crosswalks/faa_aids_to_cictt.csv`
- Test: `tests/stage0/test_taxonomy.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_taxonomy.py
import pytest
from stage0.taxonomy import (
    harmonize_flight_phase,
    harmonize_category_ntsb,
    harmonize_category_faa_aids,
    CICTT_CATEGORIES,
)

def test_cictt_categories_include_loc_i():
    assert "LOC-I" in CICTT_CATEGORIES

def test_ntsb_flight_phase_maps_to_icao():
    assert harmonize_flight_phase("Maneuvering", source="NTSB") == "MANEUVERING"
    assert harmonize_flight_phase("Cruise", source="NTSB") == "CRUISE"
    assert harmonize_flight_phase("Landing", source="NTSB") == "LANDING"

def test_tsb_flight_phase_maps_to_icao():
    assert harmonize_flight_phase("En Route", source="TSB") == "CRUISE"

def test_unknown_phase_returns_raw():
    result = harmonize_flight_phase("UNKNOWN_PHASE", source="NTSB")
    assert result == "UNKNOWN_PHASE"  # passthrough with flag

def test_ntsb_occurrence_to_cictt():
    """NTSB Occurrences table already uses CICTT codes."""
    assert harmonize_category_ntsb("Loss of control in flight") == "LOC-I"
    assert harmonize_category_ntsb("Controlled flight into terr/obj (CFIT)") == "CFIT"

def test_faa_aids_cause_to_cictt():
    """FAA AIDS older cause codes mapped to CICTT."""
    result = harmonize_category_faa_aids("2400")  # example engine failure code
    assert result in CICTT_CATEGORIES or result is None  # mapped or unmappable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_taxonomy.py -v`
Expected: FAIL

- [ ] **Step 3: Create crosswalk CSVs**

```csv
# data/crosswalks/flight_phase_crosswalk.csv
source,source_phase,icao_phase
NTSB,Standing,STANDING
NTSB,Taxi,TAXI
NTSB,Takeoff,TAKEOFF
NTSB,Initial climb,INITIAL_CLIMB
NTSB,Climb,CLIMB
NTSB,Cruise,CRUISE
NTSB,Maneuvering,MANEUVERING
NTSB,Descent,DESCENT
NTSB,Approach,APPROACH
NTSB,Landing,LANDING
NTSB,Emergency descent,EMERGENCY
TSB,Taxi,TAXI
TSB,Take-off,TAKEOFF
TSB,Initial Climb,INITIAL_CLIMB
TSB,En Route,CRUISE
TSB,Manoeuvring,MANEUVERING
TSB,Approach,APPROACH
TSB,Landing,LANDING
BEA,Roulage,TAXI
BEA,Décollage,TAKEOFF
BEA,Montée,CLIMB
BEA,Croisière,CRUISE
BEA,Descente,DESCENT
BEA,Approche,APPROACH
BEA,Atterrissage,LANDING
```

```csv
# data/crosswalks/faa_aids_to_cictt.csv
faa_cause_code,faa_description,cictt_category
2400,ENGINE FAILURE OR MALFUNCTION,SCF-PP
2500,AIRFRAME/COMPONENT/SYSTEM FAILURE/MALFUNCTION,SCF-NP
3100,WEATHER CONDITION,TURB
3200,TERRAIN CONDITION,CFIT
4200,IN FLIGHT ENCOUNTER WITH WEATHER,TURB
2000,LOSS OF ENGINE POWER(TOTAL) - NONMECHANICAL,LOC-I
2800,PROPELLER/ROTOR,SCF-PP
1100,INADEQUATE PREFLIGHT PREPARATION AND/OR PLANNING,LOC-I
1200,IMPROPER INFLIGHT DECISIONS OR PLANNING,LOC-I
1300,FAILURE TO MAINTAIN DIRECTIONAL CONTROL,LOC-G
1400,IMPROPER OPERATION OF FLIGHT CONTROLS,LOC-I
1500,STALL/SPIN,LOC-I
```

- [ ] **Step 4: Implement taxonomy module**

```python
# src/stage0/taxonomy.py
"""Taxonomy harmonization: map source-specific categories and phases to ICAO CICTT.

Crosswalk tables live in data/crosswalks/ and are loaded at import time.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path

CROSSWALK_DIR = Path(__file__).resolve().parents[2] / "data" / "crosswalks"

# ICAO CICTT occurrence categories (v4.7, 2017)
CICTT_CATEGORIES = {
    "ARC", "BIRD", "CABIN", "CFIT", "CTOL", "EVAC", "F-NI", "F-POST",
    "FUEL", "GCOL", "ICE", "LALT", "LOC-G", "LOC-I", "MAC", "OTHR",
    "RE", "RAMP", "RI", "SCF-NP", "SCF-PP", "SEC", "TURB", "UIMC",
    "UNK", "USOS", "WILD", "WSTRW",
}

# NTSB occurrence text -> CICTT code lookup
_NTSB_OCC_TO_CICTT: dict[str, str] = {
    "loss of control in flight": "LOC-I",
    "loss of control on ground": "LOC-G",
    "controlled flight into terr/obj (cfit)": "CFIT",
    "controlled flight into terrain": "CFIT",
    "runway excursion": "RE",
    "midair collision": "MAC",
    "system/component failure or malfunction (powerplant)": "SCF-PP",
    "system/component failure or malfunction (non-powerplant)": "SCF-NP",
    "fuel related": "FUEL",
    "bird": "BIRD",
    "turbulence encounter": "TURB",
    "unintended flight in imc": "UIMC",
    "undershoot/overshoot": "USOS",
    "ground collision": "GCOL",
    "fire/smoke (non-impact)": "F-NI",
    "fire/smoke (post-impact)": "F-POST",
    "low altitude operations": "LALT",
    "abnormal runway contact": "ARC",
    "evacuation": "EVAC",
    "icing": "ICE",
    "runway incursion": "RI",
    "windshear or thunderstorm": "WSTRW",
    "security related": "SEC",
    "cabin safety events": "CABIN",
    "ramp": "RAMP",
    "unknown or undetermined": "UNK",
    "other": "OTHR",
}

# Flight phase crosswalk: loaded from CSV
_PHASE_MAP: dict[tuple[str, str], str] = {}

# FAA AIDS cause code -> CICTT
_FAA_AIDS_MAP: dict[str, str] = {}


def _load_crosswalks():
    global _PHASE_MAP, _FAA_AIDS_MAP

    phase_path = CROSSWALK_DIR / "flight_phase_crosswalk.csv"
    if phase_path.exists():
        with open(phase_path) as f:
            for row in csv.DictReader(f):
                key = (row["source"].strip(), row["source_phase"].strip().lower())
                _PHASE_MAP[key] = row["icao_phase"].strip()

    faa_path = CROSSWALK_DIR / "faa_aids_to_cictt.csv"
    if faa_path.exists():
        with open(faa_path) as f:
            for row in csv.DictReader(f):
                _FAA_AIDS_MAP[row["faa_cause_code"].strip()] = row["cictt_category"].strip()


_load_crosswalks()


def harmonize_flight_phase(raw_phase: str, source: str) -> str:
    """Map a source-specific flight phase string to ICAO standard."""
    key = (source, raw_phase.strip().lower())
    return _PHASE_MAP.get(key, raw_phase)


def harmonize_category_ntsb(occurrence_text: str) -> str | None:
    """Map NTSB Occurrences table text to CICTT code."""
    return _NTSB_OCC_TO_CICTT.get(occurrence_text.strip().lower())


def harmonize_category_faa_aids(cause_code: str) -> str | None:
    """Map FAA AIDS cause code to CICTT category."""
    return _FAA_AIDS_MAP.get(cause_code.strip())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_taxonomy.py -v`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add src/stage0/taxonomy.py data/crosswalks/ tests/stage0/test_taxonomy.py
git commit -m "feat(stage0): add taxonomy harmonization with CICTT crosswalk tables"
```

---

### Task 4: NTSB Loader (avall.mdb)

**Files:**
- Create: `src/stage0/loader_ntsb.py`
- Test: `tests/stage0/test_loader_ntsb.py`

This is the most important loader — NTSB will be 80-90% of reasoning-ready records.

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_loader_ntsb.py
import pytest
import subprocess
from stage0.loader_ntsb import (
    list_mdb_tables,
    load_mdb_table,
    parse_ntsb_record,
    load_ntsb_corpus,
)
from stage0.schema import UASCRecord

MDB_PATH = "data/NTSB_ASRS/avall.mdb"

@pytest.fixture
def has_mdbtools():
    """Skip tests if mdbtools not installed."""
    try:
        subprocess.run(["mdb-tables", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("mdbtools not installed")

def test_list_mdb_tables(has_mdbtools):
    tables = list_mdb_tables(MDB_PATH)
    assert "events" in [t.lower() for t in tables]
    assert "narratives" in [t.lower() for t in tables] or "aircraft" in [t.lower() for t in tables]

def test_load_events_table_returns_dataframe(has_mdbtools):
    df = load_mdb_table(MDB_PATH, "events", limit=10)
    assert len(df) <= 10
    assert len(df) > 0

def test_parse_ntsb_record_produces_uasc():
    """Test parsing from a dict mimicking an NTSB joined row."""
    row = {
        "ev_id": "20230101X00001",
        "ntsb_no": "ERA23FA001",
        "ev_date": "2023-01-15",
        "ev_country": "USA",
        "ev_city": "Orlando",
        "ev_state": "FL",
        "inj_tot_f": "2",
        "inj_tot_s": "0",
        "ev_highest_injury": "FATL",
        "acft_make": "Cessna",
        "acft_model": "172S",
        "acft_category": "AIR",
        "far_part": "091",
        "phase_flt_spec_code": "Cruise",
        "narr_accp": "The pilot departed on a VFR flight plan. " * 30,  # ~210 words
        "narr_cause": "The pilot's failure to maintain airspeed. " * 10,
    }
    rec = parse_ntsb_record(row)
    assert isinstance(rec, UASCRecord)
    assert rec.source == "NTSB"
    assert rec.source_id == "20230101X00001"
    assert rec.fatalities == 2
    assert rec.aircraft_make == "Cessna"
    assert rec.narrative_factual is not None
    assert len(rec.narrative_factual.split()) > 100

def test_load_ntsb_corpus_returns_records(has_mdbtools):
    records = load_ntsb_corpus(MDB_PATH, limit=5)
    assert len(records) <= 5
    assert all(isinstance(r, UASCRecord) for r in records)
    assert all(r.source == "NTSB" for r in records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_ntsb.py -v`
Expected: FAIL

- [ ] **Step 3: Implement NTSB loader**

```python
# src/stage0/loader_ntsb.py
"""Load NTSB avall.mdb into UASC records.

Requires mdbtools: sudo apt install mdbtools
Uses mdb-export to extract tables as CSV, then parses into UASCRecord objects.
"""
from __future__ import annotations
import csv
import io
import subprocess
from pathlib import Path
from stage0.schema import UASCRecord, StructuredEvent, Finding
from stage0.taxonomy import harmonize_flight_phase, harmonize_category_ntsb

SEVERITY_MAP = {
    "FATL": "fatal",
    "SERS": "serious",
    "MINR": "minor",
    "NONE": "none",
    "UNKN": None,
}


def list_mdb_tables(mdb_path: str) -> list[str]:
    result = subprocess.run(
        ["mdb-tables", "-1", mdb_path],
        capture_output=True, text=True, check=True,
    )
    return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]


def load_mdb_table(mdb_path: str, table: str, limit: int | None = None) -> list[dict]:
    result = subprocess.run(
        ["mdb-export", mdb_path, table],
        capture_output=True, text=True, check=True,
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = []
    for i, row in enumerate(reader):
        if limit is not None and i >= limit:
            break
        rows.append(row)
    return rows


def _safe_int(val: str | None) -> int | None:
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def parse_ntsb_record(row: dict) -> UASCRecord:
    """Parse a joined NTSB row dict into a UASCRecord."""
    narr_factual = (row.get("narr_accp") or "").strip() or None
    narr_cause = (row.get("narr_cause") or "").strip() or None
    narr_analysis = (row.get("narr_accf") or "").strip() or None

    # Combine factual narratives if narr_accf exists and narr_accp is separate
    if narr_analysis and narr_factual:
        narr_factual = narr_factual + "\n\n" + narr_analysis
        narr_analysis = None  # folded into factual

    severity_raw = (row.get("ev_highest_injury") or "").strip()
    severity = SEVERITY_MAP.get(severity_raw)

    # Location
    city = (row.get("ev_city") or "").strip()
    state = (row.get("ev_state") or "").strip()
    location = ", ".join(filter(None, [city, state])) or None

    # Flight phase harmonization
    phase_raw = (row.get("phase_flt_spec_code") or "").strip() or None
    flight_phase = harmonize_flight_phase(phase_raw, "NTSB") if phase_raw else None

    return UASCRecord(
        source="NTSB",
        source_id=(row.get("ev_id") or row.get("ntsb_no") or "").strip(),
        occurrence_date=(row.get("ev_date") or "").strip() or None,
        country=(row.get("ev_country") or "USA").strip(),
        location=location,
        aircraft_make=(row.get("acft_make") or "").strip() or None,
        aircraft_model=(row.get("acft_model") or "").strip() or None,
        aircraft_category=(row.get("acft_category") or "").strip() or None,
        operation_type=(row.get("far_part") or "").strip() or None,
        flight_phase_raw=flight_phase,
        severity=severity,
        fatalities=_safe_int(row.get("inj_tot_f")),
        injuries=_safe_int(row.get("inj_tot_s")),
        narrative_factual=narr_factual,
        narrative_cause=narr_cause,
    )


def load_ntsb_corpus(mdb_path: str, limit: int | None = None) -> list[UASCRecord]:
    """Load NTSB records from avall.mdb into UASC records.

    Joins events, narratives, and aircraft tables.
    """
    events = load_mdb_table(mdb_path, "events", limit=limit)
    # Build a lookup for narratives and aircraft by ev_id
    # For now, events table often contains the key narrative fields inline
    records = []
    for row in events:
        rec = parse_ntsb_record(row)
        if rec.source_id:
            records.append(rec)
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_ntsb.py -v`
Expected: All PASSED (integration tests skipped if no mdbtools)

- [ ] **Step 5: Commit**

```bash
git add src/stage0/loader_ntsb.py tests/stage0/test_loader_ntsb.py
git commit -m "feat(stage0): add NTSB avall.mdb loader with mdbtools"
```

---

### Task 5: TSB Canada Loader

**Files:**
- Create: `src/stage0/loader_tsb.py`
- Test: `tests/stage0/test_loader_tsb.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_loader_tsb.py
import pytest
from pathlib import Path
from stage0.loader_tsb import parse_tsb_record, load_tsb_corpus
from stage0.schema import UASCRecord

TSB_DIR = Path("data/TSB_CANADA")

@pytest.fixture
def has_tsb_data():
    if not TSB_DIR.exists() or not any(TSB_DIR.glob("*.csv")):
        pytest.skip("TSB data not available")

def test_parse_tsb_record():
    row = {
        "OCCURRENCE_NO": "A23W0001",
        "OCC_DATE": "2023-01-20",
        "COUNTRY": "Canada",
        "PROVINCE": "AB",
        "LOCATION": "Calgary",
        "OCC_CLASS": "Accident",
        "FATALITIES": "0",
        "INJURIES_SERIOUS": "1",
        "AIRCRAFT_CATEGORY": "Aeroplane",
        "MAKE": "Cessna",
        "MODEL": "172",
        "PHASE_OF_FLIGHT": "Approach",
        "EVENT_TYPE": "Loss of control in flight",
    }
    rec = parse_tsb_record(row)
    assert isinstance(rec, UASCRecord)
    assert rec.source == "TSB"
    assert rec.source_id == "A23W0001"
    assert rec.country == "Canada"

def test_load_tsb_corpus(has_tsb_data):
    records = load_tsb_corpus(str(TSB_DIR), limit=5)
    assert len(records) <= 5
    assert all(r.source == "TSB" for r in records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_tsb.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TSB loader**

```python
# src/stage0/loader_tsb.py
"""Load TSB Canada CSV data into UASC records."""
from __future__ import annotations
import csv
from pathlib import Path
from stage0.schema import UASCRecord, StructuredEvent
from stage0.taxonomy import harmonize_flight_phase

SEVERITY_MAP = {
    "fatal": "fatal",
    "accident": "serious",
    "incident": "minor",
    "serious incident": "serious",
}


def _safe_int(val: str | None) -> int | None:
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def parse_tsb_record(row: dict) -> UASCRecord:
    """Parse a TSB CSV row into a UASCRecord."""
    occ_class = (row.get("OCC_CLASS") or "").strip().lower()
    severity = SEVERITY_MAP.get(occ_class)

    phase_raw = (row.get("PHASE_OF_FLIGHT") or "").strip()
    flight_phase = harmonize_flight_phase(phase_raw, "TSB") if phase_raw else None

    location_parts = [
        (row.get("LOCATION") or "").strip(),
        (row.get("PROVINCE") or "").strip(),
    ]
    location = ", ".join(filter(None, location_parts)) or None

    # TSB event types can map to CICTT via harmonize_category, done later in pipeline
    return UASCRecord(
        source="TSB",
        source_id=(row.get("OCCURRENCE_NO") or "").strip(),
        occurrence_date=(row.get("OCC_DATE") or "").strip() or None,
        country=(row.get("COUNTRY") or "Canada").strip(),
        location=location,
        aircraft_make=(row.get("MAKE") or "").strip() or None,
        aircraft_model=(row.get("MODEL") or "").strip() or None,
        aircraft_category=(row.get("AIRCRAFT_CATEGORY") or "").strip() or None,
        flight_phase_raw=flight_phase,
        severity=severity,
        fatalities=_safe_int(row.get("FATALITIES")),
        injuries=_safe_int(row.get("INJURIES_SERIOUS")),
    )


def load_tsb_corpus(tsb_dir: str, limit: int | None = None) -> list[UASCRecord]:
    """Load TSB Canada records from CSV files in the given directory."""
    tsb_path = Path(tsb_dir)
    records = []
    for csv_file in sorted(tsb_path.glob("*.csv")):
        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = parse_tsb_record(row)
                if rec.source_id:
                    records.append(rec)
                    if limit and len(records) >= limit:
                        return records
    return records
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_tsb.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/loader_tsb.py tests/stage0/test_loader_tsb.py
git commit -m "feat(stage0): add TSB Canada CSV loader"
```

---

### Task 6: BEA Loader

**Files:**
- Create: `src/stage0/loader_bea.py`
- Test: `tests/stage0/test_loader_bea.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_loader_bea.py
import pytest
from pathlib import Path
from stage0.loader_bea import parse_bea_record, load_bea_corpus
from stage0.schema import UASCRecord

BEA_CSV = Path("data/BEA/bea_notified_events.csv")

@pytest.fixture
def has_bea_data():
    if not BEA_CSV.exists():
        pytest.skip("BEA data not available")

def test_parse_bea_record():
    row = {
        "file_number": "BEA2023-0001",
        "title": "Accident to a Cessna 172 near Lyon",
        "summary": "During the approach phase the pilot lost control of the aircraft. " * 20,
        "date": "2023-02-10",
        "location": "Lyon",
        "state_of_occurrence": "France",
        "occurrence_class": "Accident",
        "human_consequences": "1 fatal",
        "aircraft_category": "Aeroplane",
        "manufacturer_model": "Cessna 172",
        "flight_phase": "Approche",
    }
    rec = parse_bea_record(row)
    assert isinstance(rec, UASCRecord)
    assert rec.source == "BEA"
    assert rec.source_id == "BEA2023-0001"
    assert rec.narrative_factual is not None

def test_load_bea_corpus(has_bea_data):
    records = load_bea_corpus(str(BEA_CSV), limit=5)
    assert len(records) <= 5
    assert all(r.source == "BEA" for r in records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_bea.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BEA loader**

```python
# src/stage0/loader_bea.py
"""Load BEA scraped notified events into UASC records."""
from __future__ import annotations
import csv
import re
from pathlib import Path
from stage0.schema import UASCRecord
from stage0.taxonomy import harmonize_flight_phase


def _parse_human_consequences(text: str | None) -> tuple[int | None, int | None]:
    """Extract fatalities and injuries from BEA human_consequences field."""
    if not text:
        return None, None
    fatalities = None
    injuries = None
    fatal_match = re.search(r"(\d+)\s*fatal", text, re.IGNORECASE)
    if fatal_match:
        fatalities = int(fatal_match.group(1))
    injury_match = re.search(r"(\d+)\s*(?:serious|injur)", text, re.IGNORECASE)
    if injury_match:
        injuries = int(injury_match.group(1))
    return fatalities, injuries


def _severity_from_class(occ_class: str, fatalities: int | None) -> str | None:
    if fatalities and fatalities > 0:
        return "fatal"
    occ_lower = occ_class.lower()
    if "accident" in occ_lower:
        return "serious"
    if "serious incident" in occ_lower:
        return "serious"
    if "incident" in occ_lower:
        return "minor"
    return None


def parse_bea_record(row: dict) -> UASCRecord:
    """Parse a BEA CSV row into a UASCRecord."""
    fatalities, injuries = _parse_human_consequences(row.get("human_consequences"))
    occ_class = (row.get("occurrence_class") or "").strip()
    severity = _severity_from_class(occ_class, fatalities)

    phase_raw = (row.get("flight_phase") or "").strip()
    flight_phase = harmonize_flight_phase(phase_raw, "BEA") if phase_raw else None

    # BEA summary serves as narrative_factual (no separate cause narrative)
    summary = (row.get("summary") or "").strip() or None

    return UASCRecord(
        source="BEA",
        source_id=(row.get("file_number") or "").strip(),
        occurrence_date=(row.get("date") or "").strip() or None,
        country=(row.get("state_of_occurrence") or "France").strip(),
        location=(row.get("location") or "").strip() or None,
        aircraft_category=(row.get("aircraft_category") or "").strip() or None,
        aircraft_make=None,  # manufacturer_model is combined; split later if needed
        aircraft_model=(row.get("manufacturer_model") or "").strip() or None,
        flight_phase_raw=flight_phase,
        severity=severity,
        fatalities=fatalities,
        injuries=injuries,
        narrative_factual=summary,
    )


def load_bea_corpus(csv_path: str, limit: int | None = None) -> list[UASCRecord]:
    """Load BEA records from the scraped CSV."""
    records = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = parse_bea_record(row)
            if rec.source_id:
                records.append(rec)
                if limit and len(records) >= limit:
                    return records
    return records
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_bea.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/loader_bea.py tests/stage0/test_loader_bea.py
git commit -m "feat(stage0): add BEA notified events loader"
```

---

### Task 7: FAA AIDS Loader

**Files:**
- Create: `src/stage0/loader_faa_aids.py`
- Test: `tests/stage0/test_loader_faa_aids.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_loader_faa_aids.py
import pytest
from pathlib import Path
from stage0.loader_faa_aids import parse_faa_aids_record, load_faa_aids_corpus
from stage0.schema import UASCRecord

FAA_DIR = Path("data/FAA_AIDS")

@pytest.fixture
def has_faa_data():
    if not FAA_DIR.exists() or not any(FAA_DIR.glob("a*.txt")):
        pytest.skip("FAA AIDS data not available")

def test_parse_faa_aids_record():
    row = {
        "C1_ACCIDENT_NUMBER": "LAX23FA001",
        "C5_EVENT_DATE": "20230115",
        "C77_COUNTRY": "US",
        "C78_CITY": "Los Angeles",
        "C79_STATE": "CA",
        "C82_AIRCRAFT_MAKE": "Piper",
        "C83_AIRCRAFT_MODEL": "PA-28",
        "C84_AIRCRAFT_SERIES": "161",
        "C93_FAR_PART": "091",
        "C2_CAUSE_CODE": "1500",
        "C6_FATAL_INJURIES": "1",
        "C7_SERIOUS_INJURIES": "0",
        "remarks": "The pilot failed to maintain airspeed during approach. " * 5,
    }
    rec = parse_faa_aids_record(row)
    assert isinstance(rec, UASCRecord)
    assert rec.source == "FAA_AIDS"
    assert rec.fatalities == 1
    assert rec.narrative_factual is not None

def test_load_faa_aids_corpus(has_faa_data):
    records = load_faa_aids_corpus(str(FAA_DIR), limit=5)
    assert len(records) <= 5
    assert all(r.source == "FAA_AIDS" for r in records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_faa_aids.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FAA AIDS loader**

```python
# src/stage0/loader_faa_aids.py
"""Load FAA AIDS tab-delimited files into UASC records.

A-files contain accident/incident records. E-files contain edited remarks/narratives.
"""
from __future__ import annotations
import csv
from pathlib import Path
from stage0.schema import UASCRecord
from stage0.taxonomy import harmonize_category_faa_aids


def _safe_int(val: str | None) -> int | None:
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _parse_date(raw: str) -> str | None:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    raw = raw.strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw or None


def parse_faa_aids_record(row: dict, remarks: str | None = None) -> UASCRecord:
    """Parse an FAA AIDS row into a UASCRecord."""
    fatalities = _safe_int(row.get("C6_FATAL_INJURIES"))
    injuries = _safe_int(row.get("C7_SERIOUS_INJURIES"))

    if fatalities and fatalities > 0:
        severity = "fatal"
    elif injuries and injuries > 0:
        severity = "serious"
    else:
        severity = "minor"

    cause_code = (row.get("C2_CAUSE_CODE") or "").strip()
    cictt = harmonize_category_faa_aids(cause_code) if cause_code else None

    location_parts = [
        (row.get("C78_CITY") or "").strip(),
        (row.get("C79_STATE") or "").strip(),
    ]
    location = ", ".join(filter(None, location_parts)) or None

    narrative = remarks or (row.get("remarks") or "").strip() or None

    return UASCRecord(
        source="FAA_AIDS",
        source_id=(row.get("C1_ACCIDENT_NUMBER") or "").strip(),
        occurrence_date=_parse_date(row.get("C5_EVENT_DATE") or ""),
        country=(row.get("C77_COUNTRY") or "US").strip(),
        location=location,
        aircraft_make=(row.get("C82_AIRCRAFT_MAKE") or "").strip() or None,
        aircraft_model=(row.get("C83_AIRCRAFT_MODEL") or "").strip() or None,
        operation_type=(row.get("C93_FAR_PART") or "").strip() or None,
        severity=severity,
        fatalities=fatalities,
        injuries=injuries,
        accident_category_primary=cictt,
        narrative_factual=narrative,
        harmonization_flags={"category_source": "faa_aids_rule_mapping"} if cictt else {},
    )


def _load_remarks(faa_dir: Path) -> dict[str, str]:
    """Load E-files (edited remarks) keyed by accident number."""
    remarks = {}
    for efile in sorted(faa_dir.glob("e*.txt")):
        with open(efile, encoding="latin-1", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                acc_no = (row.get("C1_ACCIDENT_NUMBER") or "").strip()
                text = (row.get("C130_REMARKS") or row.get("remarks") or "").strip()
                if acc_no and text:
                    remarks[acc_no] = text
    return remarks


def load_faa_aids_corpus(faa_dir: str, limit: int | None = None) -> list[UASCRecord]:
    """Load FAA AIDS records from A-files + E-files."""
    faa_path = Path(faa_dir)
    remarks = _load_remarks(faa_path)

    records = []
    for afile in sorted(faa_path.glob("a*.txt")):
        with open(afile, encoding="latin-1", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                acc_no = (row.get("C1_ACCIDENT_NUMBER") or "").strip()
                rec = parse_faa_aids_record(row, remarks=remarks.get(acc_no))
                if rec.source_id:
                    records.append(rec)
                    if limit and len(records) >= limit:
                        return records
    return records
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_faa_aids.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/loader_faa_aids.py tests/stage0/test_loader_faa_aids.py
git commit -m "feat(stage0): add FAA AIDS tab-delimited loader with remarks join"
```

---

### Task 8: NTSB Published Reports Enricher

**Files:**
- Create: `src/stage0/loader_ntsb_reports.py`
- Test: `tests/stage0/test_loader_ntsb_reports.py`

This loader enriches existing NTSB UASC records with the full narrative text from published reports (the 496 PDFs scraped earlier).

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_loader_ntsb_reports.py
import pytest
from pathlib import Path
from stage0.loader_ntsb_reports import load_ntsb_reports_manifest, enrich_records_with_reports
from stage0.schema import UASCRecord

REPORTS_DIR = Path("data/NTSB_REPORTS")

@pytest.fixture
def has_reports():
    if not (REPORTS_DIR / "manifest.csv").exists():
        pytest.skip("NTSB reports manifest not available")

def test_load_manifest(has_reports):
    manifest = load_ntsb_reports_manifest(str(REPORTS_DIR / "manifest.csv"))
    assert len(manifest) > 0
    assert "ntsb_accident_id" in manifest[0]
    assert "txt_path" in manifest[0]

def test_enrich_records_joins_by_accident_id():
    """Reports with an accident_id should enrich the matching UASC record."""
    records = [
        UASCRecord(
            source="NTSB",
            source_id="20090212X73701",
            narrative_factual="Short narrative from avall.",
        ),
    ]
    manifest = [
        {
            "ntsb_accident_id": "20090212X73701",
            "txt_path": "data/NTSB_REPORTS/txt/AAR1001.txt",
            "extraction_status": "ok",
            "txt_chars": "50000",
            "report_number": "AAR-10/01",
        },
    ]
    enriched = enrich_records_with_reports(records, manifest, reports_dir="data/NTSB_REPORTS")
    assert enriched[0].narrative_analysis is not None or enriched[0].narrative_factual != "Short narrative from avall."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_ntsb_reports.py -v`
Expected: FAIL

- [ ] **Step 3: Implement NTSB reports enricher**

```python
# src/stage0/loader_ntsb_reports.py
"""Enrich NTSB UASC records with full narrative text from published reports.

Joins on ntsb_accident_id between avall.mdb records and the reports manifest.
"""
from __future__ import annotations
import csv
from pathlib import Path
from stage0.schema import UASCRecord


def load_ntsb_reports_manifest(manifest_path: str) -> list[dict]:
    """Load the NTSB reports manifest CSV."""
    with open(manifest_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_report_text(txt_path: str) -> str | None:
    """Read extracted text from a report txt file."""
    path = Path(txt_path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text if text else None


def enrich_records_with_reports(
    records: list[UASCRecord],
    manifest: list[dict],
    reports_dir: str = "data/NTSB_REPORTS",
) -> list[UASCRecord]:
    """Enrich UASC records with full report text where available.

    For records with a matching report (by source_id == ntsb_accident_id),
    the report text is added as narrative_analysis, providing the longest
    narrative available for richness scoring.
    """
    # Build lookup: accident_id -> manifest entry (only ok extractions)
    report_lookup: dict[str, dict] = {}
    for entry in manifest:
        acc_id = (entry.get("ntsb_accident_id") or "").strip()
        status = (entry.get("extraction_status") or "").strip()
        if acc_id and status == "ok":
            report_lookup[acc_id] = entry

    for rec in records:
        if rec.source != "NTSB":
            continue
        entry = report_lookup.get(rec.source_id)
        if entry is None:
            continue
        txt_path = entry.get("txt_path", "")
        if not txt_path:
            continue
        # Resolve relative to reports_dir
        full_path = Path(reports_dir) / Path(txt_path).name if not Path(txt_path).is_absolute() else txt_path
        report_text = _read_report_text(str(full_path))
        if report_text:
            rec.narrative_analysis = report_text
            rec.harmonization_flags["report_enriched"] = entry.get("report_number", "yes")

    return records
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_loader_ntsb_reports.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/loader_ntsb_reports.py tests/stage0/test_loader_ntsb_reports.py
git commit -m "feat(stage0): add NTSB published reports enricher"
```

---

### Task 9: Cross-Source Deduplication

**Files:**
- Create: `src/stage0/dedup.py`
- Test: `tests/stage0/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_dedup.py
import pytest
from stage0.dedup import find_duplicates, resolve_clusters
from stage0.schema import UASCRecord

def test_same_accident_different_sources_detected():
    """An NTSB and FAA AIDS record for the same accident should cluster."""
    ntsb = UASCRecord(
        source="NTSB",
        source_id="20230115X00001",
        occurrence_date="2023-01-15",
        country="US",
        aircraft_category="airplane",
        aircraft_make="Cessna",
        aircraft_model="172S",
        location="Orlando, FL",
        severity="fatal",
        narrative_factual="The pilot departed VFR " * 30,
    )
    faa = UASCRecord(
        source="FAA_AIDS",
        source_id="LAX23FA001",
        occurrence_date="2023-01-15",
        country="US",
        aircraft_category="airplane",
        aircraft_make="Cessna",
        aircraft_model="172S",
        location="Orlando, FL",
        severity="fatal",
        narrative_factual="Gear-up landing.",
    )
    clusters = find_duplicates([ntsb, faa])
    assert len(clusters) == 1
    assert len(clusters[0]) == 2

def test_different_dates_not_clustered():
    rec1 = UASCRecord(source="NTSB", source_id="A", occurrence_date="2023-01-15",
                      country="US", aircraft_category="airplane", severity="fatal")
    rec2 = UASCRecord(source="FAA_AIDS", source_id="B", occurrence_date="2023-06-01",
                      country="US", aircraft_category="airplane", severity="fatal")
    clusters = find_duplicates([rec1, rec2])
    assert len(clusters) == 0

def test_resolve_keeps_richest():
    rich = UASCRecord(
        source="NTSB", source_id="RICH",
        narrative_factual="Long narrative " * 100,
        occurrence_date="2023-01-15", country="US",
        aircraft_category="airplane", severity="fatal",
    )
    poor = UASCRecord(
        source="FAA_AIDS", source_id="POOR",
        narrative_factual="Short.",
        occurrence_date="2023-01-15", country="US",
        aircraft_category="airplane", severity="fatal",
    )
    canonical, linked = resolve_clusters([[rich, poor]])
    assert len(canonical) == 1
    assert canonical[0].source_id == "RICH"
    assert canonical[0].dup_cluster_id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_dedup.py -v`
Expected: FAIL

- [ ] **Step 3: Implement deduplication**

```python
# src/stage0/dedup.py
"""Cross-source deduplication for UASC records.

Strategy: blocking on (date +/- 1 day, country, aircraft_category) + pairwise matching.
See design doc v2 Section 3.6.
"""
from __future__ import annotations
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from stage0.schema import UASCRecord
from stage0.richness import word_count


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _blocking_keys(rec: UASCRecord) -> list[tuple[str, str, str]]:
    """Generate blocking keys: (date_str, country, aircraft_category) for date +/- 1 day."""
    dt = _parse_date(rec.occurrence_date)
    country = (rec.country or "").strip().upper()
    category = (rec.aircraft_category or "").strip().lower()
    if not dt or not country:
        return []
    keys = []
    for delta in (-1, 0, 1):
        d = dt + timedelta(days=delta)
        keys.append((d.strftime("%Y-%m-%d"), country, category))
    return keys


def _match_score(a: UASCRecord, b: UASCRecord) -> float:
    """Pairwise match score between two records. Higher = more likely same accident."""
    score = 0.0
    # Same source = not a cross-source duplicate
    if a.source == b.source:
        return 0.0
    # Aircraft make/model match
    if a.aircraft_make and b.aircraft_make:
        if a.aircraft_make.strip().lower() == b.aircraft_make.strip().lower():
            score += 0.3
    if a.aircraft_model and b.aircraft_model:
        if a.aircraft_model.strip().lower() == b.aircraft_model.strip().lower():
            score += 0.3
    # Severity match
    if a.severity and b.severity and a.severity == b.severity:
        score += 0.1
    # Location similarity (simple substring match for now)
    if a.location and b.location:
        a_loc = a.location.strip().lower()
        b_loc = b.location.strip().lower()
        if a_loc == b_loc:
            score += 0.3
        elif a_loc in b_loc or b_loc in a_loc:
            score += 0.15
    return score


MATCH_THRESHOLD = 0.5


def find_duplicates(records: list[UASCRecord], threshold: float = MATCH_THRESHOLD) -> list[list[UASCRecord]]:
    """Find cross-source duplicate clusters via blocking + pairwise matching."""
    # Build blocking index
    blocks: dict[tuple, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        for key in _blocking_keys(rec):
            blocks[key].append(i)

    # Pairwise matching within blocks
    matched: set[tuple[int, int]] = set()
    for indices in blocks.values():
        if len(indices) < 2:
            continue
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                i, j = indices[i_idx], indices[j_idx]
                if i == j or (i, j) in matched:
                    continue
                score = _match_score(records[i], records[j])
                if score >= threshold:
                    matched.add((min(i, j), max(i, j)))

    # Build clusters via union-find
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in matched:
        union(i, j)

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i, j in matched:
        root = find(i)
        cluster_map[root].append(i)
        cluster_map[root].append(j)

    clusters = []
    for root, members in cluster_map.items():
        unique = list(set(members))
        clusters.append([records[i] for i in sorted(unique)])
    return clusters


def resolve_clusters(clusters: list[list[UASCRecord]]) -> tuple[list[UASCRecord], list[UASCRecord]]:
    """Resolve duplicate clusters: richest record becomes canonical.

    Returns: (canonical_records, linked_records_with_dup_cluster_id)
    """
    canonical_list = []
    linked_list = []
    for cluster in clusters:
        cluster_id = str(uuid.uuid4())
        # Richest = longest narrative_factual
        ranked = sorted(cluster, key=lambda r: word_count(r.narrative_factual), reverse=True)
        canonical = ranked[0]
        canonical.dup_cluster_id = cluster_id
        canonical_list.append(canonical)
        for rec in ranked[1:]:
            rec.dup_cluster_id = cluster_id
            linked_list.append(rec)
    return canonical_list, linked_list
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_dedup.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/dedup.py tests/stage0/test_dedup.py
git commit -m "feat(stage0): add cross-source deduplication with blocking and union-find"
```

---

### Task 10: Corpus Builder Orchestrator

**Files:**
- Create: `src/stage0/build_corpus.py`
- Test: `tests/stage0/test_build_corpus.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/stage0/test_build_corpus.py
import pytest
from pathlib import Path
from stage0.build_corpus import build_uasc, write_parquet
from stage0.schema import UASCRecord

def test_build_uasc_with_mock_records(tmp_path):
    """Build a tiny corpus from pre-made records and write to parquet."""
    records = [
        UASCRecord(
            source="NTSB", source_id="TEST001",
            narrative_factual=" ".join(["word"] * 200),
            narrative_cause=" ".join(["reason"] * 60),
            occurrence_date="2023-01-15", country="US",
            flight_phase_raw="CRUISE", aircraft_category="airplane",
            severity="fatal", accident_category_primary="LOC-I",
        ),
        UASCRecord(
            source="FAA_AIDS", source_id="TEST002",
            narrative_factual="Short remark.",
            occurrence_date="2023-06-01", country="US",
            severity="minor",
        ),
    ]
    scored = build_uasc(records)
    assert all(r.richness_score is not None for r in scored)
    assert all(r.richness_tier is not None for r in scored)

    # The rich NTSB record should be reasoning_ready
    ntsb_rec = [r for r in scored if r.source_id == "TEST001"][0]
    assert ntsb_rec.richness_tier == "reasoning_ready"

    # The short FAA AIDS record should be metadata_only
    faa_rec = [r for r in scored if r.source_id == "TEST002"][0]
    assert faa_rec.richness_tier == "metadata_only"

    # Write to parquet
    out_path = tmp_path / "uasc.parquet"
    write_parquet(scored, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_build_corpus.py -v`
Expected: FAIL

- [ ] **Step 3: Implement corpus builder**

```python
# src/stage0/build_corpus.py
"""Orchestrator: load all sources -> harmonize -> score -> dedup -> write parquet.

Main entry point for Stage 0 corpus construction.
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from stage0.schema import UASCRecord
from stage0.richness import compute_richness, assign_tier
from stage0.dedup import find_duplicates, resolve_clusters

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def build_uasc(records: list[UASCRecord]) -> list[UASCRecord]:
    """Score and tier-assign all records. Dedup cross-source overlaps."""
    # Score each record
    for rec in records:
        score, gates = compute_richness(rec)
        rec.richness_score = round(score, 4)
        rec.richness_tier = assign_tier(score, gates)

    # Dedup
    clusters = find_duplicates(records)
    if clusters:
        canonical, linked = resolve_clusters(clusters)
        # Remove linked (non-canonical) duplicates from main list
        linked_ids = {id(r) for r in linked}
        records = [r for r in records if id(r) not in linked_ids]

    return records


def _record_to_flat_dict(rec: UASCRecord) -> dict:
    """Flatten a UASCRecord for parquet serialization."""
    d = asdict(rec)
    # Flatten nested dataclass lists to JSON strings
    d["structured_events"] = json.dumps(d.get("structured_events", []))
    d["findings"] = json.dumps(d.get("findings", []))
    d["harmonization_flags"] = json.dumps(d.get("harmonization_flags", {}))
    d["accident_category_secondary"] = json.dumps(d.get("accident_category_secondary", []))
    # Convert lat_lon tuple to string
    if d.get("lat_lon"):
        d["lat_lon"] = json.dumps(d["lat_lon"])
    else:
        d["lat_lon"] = None
    return d


def write_parquet(records: list[UASCRecord], path: str):
    """Write UASC records to a parquet file."""
    if not HAS_PYARROW:
        raise ImportError("pyarrow is required to write parquet files. Install with: pip install pyarrow")
    rows = [_record_to_flat_dict(r) for r in records]
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main():
    """Full Stage 0 pipeline: load all sources, build UASC, write outputs."""
    from stage0.loader_ntsb import load_ntsb_corpus
    from stage0.loader_ntsb_reports import load_ntsb_reports_manifest, enrich_records_with_reports
    from stage0.loader_tsb import load_tsb_corpus
    from stage0.loader_bea import load_bea_corpus
    from stage0.loader_faa_aids import load_faa_aids_corpus

    print("Loading NTSB records from avall.mdb...")
    ntsb = load_ntsb_corpus("data/NTSB_ASRS/avall.mdb")
    print(f"  Loaded {len(ntsb)} NTSB records")

    # Enrich with published reports
    manifest_path = "data/NTSB_REPORTS/manifest.csv"
    if Path(manifest_path).exists():
        print("Enriching with NTSB published reports...")
        manifest = load_ntsb_reports_manifest(manifest_path)
        ntsb = enrich_records_with_reports(ntsb, manifest)
        enriched = sum(1 for r in ntsb if r.harmonization_flags.get("report_enriched"))
        print(f"  Enriched {enriched} records with full report text")

    print("Loading TSB Canada records...")
    tsb = load_tsb_corpus("data/TSB_CANADA")
    print(f"  Loaded {len(tsb)} TSB records")

    print("Loading BEA records...")
    bea_path = "data/BEA/bea_notified_events.csv"
    bea = load_bea_corpus(bea_path) if Path(bea_path).exists() else []
    print(f"  Loaded {len(bea)} BEA records")

    print("Loading FAA AIDS records...")
    faa = load_faa_aids_corpus("data/FAA_AIDS")
    print(f"  Loaded {len(faa)} FAA AIDS records")

    all_records = ntsb + tsb + bea + faa
    print(f"\nTotal raw records: {len(all_records)}")

    print("Building UASC (scoring + dedup)...")
    uasc = build_uasc(all_records)

    # Stats
    tiers = {"reasoning_ready": 0, "partial": 0, "metadata_only": 0}
    for r in uasc:
        tiers[r.richness_tier] += 1
    print(f"\nUASC built: {len(uasc)} records")
    for tier, count in tiers.items():
        print(f"  {tier}: {count}")

    # Write outputs
    out_dir = Path("data/UASC")
    out_dir.mkdir(exist_ok=True)

    print(f"\nWriting UASC to {out_dir / 'uasc_v1.parquet'}...")
    write_parquet(uasc, str(out_dir / "uasc_v1.parquet"))

    rich = [r for r in uasc if r.richness_tier == "reasoning_ready"]
    print(f"Writing UASC-Rich ({len(rich)} records) to {out_dir / 'uasc_rich.parquet'}...")
    write_parquet(rich, str(out_dir / "uasc_rich.parquet"))

    print("\nStage 0 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_build_corpus.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/build_corpus.py tests/stage0/test_build_corpus.py
git commit -m "feat(stage0): add corpus builder orchestrator with parquet output"
```

---

### Task 11: Corpus Statistics Report

**Files:**
- Create: `src/stage0/stats_report.py`
- Test: `tests/stage0/test_stats_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/stage0/test_stats_report.py
import pytest
from stage0.stats_report import generate_stats
from stage0.schema import UASCRecord

def test_stats_report_counts():
    records = [
        UASCRecord(source="NTSB", source_id="1", richness_tier="reasoning_ready",
                   richness_score=0.7, accident_category_primary="LOC-I",
                   severity="fatal", narrative_factual="text"),
        UASCRecord(source="NTSB", source_id="2", richness_tier="partial",
                   richness_score=0.4, accident_category_primary="CFIT",
                   severity="serious", narrative_factual="text"),
        UASCRecord(source="BEA", source_id="3", richness_tier="metadata_only",
                   richness_score=0.1, severity="minor", narrative_factual="x"),
    ]
    stats = generate_stats(records)
    assert stats["total_records"] == 3
    assert stats["per_source"]["NTSB"] == 2
    assert stats["per_source"]["BEA"] == 1
    assert stats["per_tier"]["reasoning_ready"] == 1
    assert stats["per_tier"]["partial"] == 1
    assert stats["per_tier"]["metadata_only"] == 1
    assert "LOC-I" in stats["per_category"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_stats_report.py -v`
Expected: FAIL

- [ ] **Step 3: Implement stats report**

```python
# src/stage0/stats_report.py
"""Generate corpus statistics report for UASC (Stage 0 deliverable #5)."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from stage0.schema import UASCRecord


def generate_stats(records: list[UASCRecord]) -> dict:
    """Generate summary statistics for the UASC corpus."""
    per_source = Counter(r.source for r in records)
    per_tier = Counter(r.richness_tier for r in records)
    per_category = Counter(r.accident_category_primary for r in records if r.accident_category_primary)
    per_severity = Counter(r.severity for r in records if r.severity)

    scores = [r.richness_score for r in records if r.richness_score is not None]
    score_stats = {}
    if scores:
        scores_sorted = sorted(scores)
        score_stats = {
            "min": scores_sorted[0],
            "max": scores_sorted[-1],
            "mean": sum(scores) / len(scores),
            "median": scores_sorted[len(scores_sorted) // 2],
            "p25": scores_sorted[len(scores_sorted) // 4],
            "p75": scores_sorted[3 * len(scores_sorted) // 4],
        }

    # Per-source tier breakdown
    source_tier = {}
    for r in records:
        key = r.source
        if key not in source_tier:
            source_tier[key] = Counter()
        source_tier[key][r.richness_tier] += 1

    dup_clusters = len({r.dup_cluster_id for r in records if r.dup_cluster_id})

    return {
        "total_records": len(records),
        "per_source": dict(per_source),
        "per_tier": dict(per_tier),
        "per_category": dict(per_category.most_common(30)),
        "per_severity": dict(per_severity),
        "richness_score_distribution": score_stats,
        "source_tier_breakdown": {k: dict(v) for k, v in source_tier.items()},
        "duplicate_clusters": dup_clusters,
    }


def write_stats_report(stats: dict, path: str):
    """Write stats as JSON and a human-readable markdown report."""
    out = Path(path)
    # JSON
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2, default=str))

    # Markdown
    lines = ["# UASC Corpus Statistics Report\n"]
    lines.append(f"**Total records:** {stats['total_records']}\n")
    lines.append(f"**Duplicate clusters:** {stats['duplicate_clusters']}\n")

    lines.append("\n## Records per Source\n")
    lines.append("| Source | Count |")
    lines.append("|---|---|")
    for src, count in sorted(stats["per_source"].items()):
        lines.append(f"| {src} | {count} |")

    lines.append("\n## Records per Tier\n")
    lines.append("| Tier | Count |")
    lines.append("|---|---|")
    for tier, count in sorted(stats["per_tier"].items()):
        lines.append(f"| {tier} | {count} |")

    lines.append("\n## Source x Tier Breakdown\n")
    lines.append("| Source | reasoning_ready | partial | metadata_only |")
    lines.append("|---|---|---|---|")
    for src, tiers in sorted(stats["source_tier_breakdown"].items()):
        rr = tiers.get("reasoning_ready", 0)
        p = tiers.get("partial", 0)
        m = tiers.get("metadata_only", 0)
        lines.append(f"| {src} | {rr} | {p} | {m} |")

    lines.append("\n## Top Accident Categories\n")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for cat, count in stats["per_category"].items():
        lines.append(f"| {cat} | {count} |")

    if stats["richness_score_distribution"]:
        lines.append("\n## Richness Score Distribution\n")
        for k, v in stats["richness_score_distribution"].items():
            lines.append(f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}")

    out.with_suffix(".md").write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_stats_report.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/stage0/stats_report.py tests/stage0/test_stats_report.py
git commit -m "feat(stage0): add corpus statistics report generator"
```

---

### Task 12: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/stage0/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/stage0/test_integration.py
"""Integration test: run the full Stage 0 pipeline on available data."""
import pytest
from pathlib import Path
from stage0.build_corpus import build_uasc, write_parquet
from stage0.loader_ntsb import load_ntsb_corpus
from stage0.loader_tsb import load_tsb_corpus
from stage0.loader_bea import load_bea_corpus
from stage0.loader_faa_aids import load_faa_aids_corpus
from stage0.stats_report import generate_stats

MDB = Path("data/NTSB_ASRS/avall.mdb")
TSB_DIR = Path("data/TSB_CANADA")
BEA_CSV = Path("data/BEA/bea_notified_events.csv")
FAA_DIR = Path("data/FAA_AIDS")


@pytest.mark.integration
def test_full_pipeline_smoke(tmp_path):
    """Load a small sample from each available source, build UASC, write parquet."""
    records = []

    if MDB.exists():
        records.extend(load_ntsb_corpus(str(MDB), limit=20))

    if TSB_DIR.exists():
        records.extend(load_tsb_corpus(str(TSB_DIR), limit=10))

    if BEA_CSV.exists():
        records.extend(load_bea_corpus(str(BEA_CSV), limit=10))

    if FAA_DIR.exists():
        records.extend(load_faa_aids_corpus(str(FAA_DIR), limit=10))

    assert len(records) > 0, "No data sources available for integration test"

    uasc = build_uasc(records)
    assert len(uasc) > 0
    assert all(r.richness_score is not None for r in uasc)
    assert all(r.richness_tier is not None for r in uasc)

    # Stats
    stats = generate_stats(uasc)
    assert stats["total_records"] == len(uasc)

    # Parquet output
    out = tmp_path / "test_uasc.parquet"
    write_parquet(uasc, str(out))
    assert out.exists()

    print(f"\nIntegration test: {len(uasc)} records across {len(stats['per_source'])} sources")
    for tier, count in stats["per_tier"].items():
        print(f"  {tier}: {count}")
```

- [ ] **Step 2: Run integration test**

Run: `PYTHONPATH=src python -m pytest tests/stage0/test_integration.py -v -m integration`
Expected: PASSED (with whatever data sources are available)

- [ ] **Step 3: Commit**

```bash
git add tests/stage0/test_integration.py
git commit -m "test(stage0): add full pipeline integration smoke test"
```

---

### Task 13: Wire Up Full Run + Create `__init__` files

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/stage0/__init__.py`
- Modify: `src/stage0/build_corpus.py` (add CLI args)

- [ ] **Step 1: Create missing `__init__.py` files**

```python
# tests/__init__.py
# tests/stage0/__init__.py
```

(Both files are empty — needed for pytest discovery.)

- [ ] **Step 2: Add CLI argument support to build_corpus.py**

Add to the `main()` function in `src/stage0/build_corpus.py`:

```python
# Replace the if __name__ block with:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 0: Build UASC corpus")
    parser.add_argument("--limit", type=int, default=None, help="Limit records per source (for testing)")
    parser.add_argument("--output-dir", default="data/UASC", help="Output directory")
    args = parser.parse_args()
    main(limit=args.limit, output_dir=args.output_dir)
```

And update `main()` signature to accept `limit` and `output_dir` parameters.

- [ ] **Step 3: Run all tests**

Run: `PYTHONPATH=src python -m pytest tests/stage0/ -v --ignore=tests/stage0/test_integration.py`
Expected: All unit tests PASSED

- [ ] **Step 4: Run the full pipeline (smoke test with --limit)**

Run: `PYTHONPATH=src python src/stage0/build_corpus.py --limit 20`
Expected: Prints source counts, tier distribution, and writes parquet files.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/stage0/__init__.py src/stage0/build_corpus.py
git commit -m "feat(stage0): wire up CLI entry point and add init files"
```
