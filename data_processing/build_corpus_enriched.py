"""Build corpus_enriched.jsonl by joining structured supervision fields onto
data/corpus/corpus.jsonl.

Sources joined:
  - NTSB_ASRS:Pre2008  (record_id = ev_id_AircraftKey)  -> Pre2008.mdb tables
  - NTSB_ASRS:avall    (record_id = ev_id_AircraftKey)  -> avall.mdb tables
  - NTSB_REPORT        (record_id = report_number)      -> manifest.csv -> .mdb via ntsb_no
  - FAA_AIDS:*         (record_id = c5 control number)  -> a*.txt extracted JSONL
  - BEA, TSB_CANADA    -> structured fields are null (no upstream source provides them)

Schema-additions to each record:
  structured: {
    primary_cause_text, secondary_cause_text, additional_causes[],
    cause_codes[],
    findings: [{code, description, category, cause_factor}, ...],   # avall NTSB
    occurrences: [{occurrence_code, code_text, phase_code, phase_text}, ...],   # Pre2008 NTSB
    seq_of_events: [{order, subj_code, subj_text, cause_factor}, ...],          # Pre2008 NTSB
    phase_of_flight,
    flying_condition,                # VFR/IFR
    light_condition,
    aircraft: {make, model, damage, num_eng},
    injury: {fatal:n, serious:n, minor:n, none:n},
    event_type,                      # accident vs incident
    cictt_category,                  # null until classifier runs
    supervision_source,              # which structured source the fields came from
  }
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yp6443/research/AviationSafetyReportKnowledgeGraph")
DATA = ROOT / "data"
NTSB_EXTRACT = DATA / "NTSB_ASRS" / "extracted"
FAA_EXTRACT = DATA / "FAA_AIDS" / "extracted"
NTSB_REPORTS = DATA / "NTSB_REPORTS"
CORPUS = DATA / "corpus" / "corpus.jsonl"
OUT = DATA / "corpus" / "corpus_enriched.jsonl"
STATS_OUT = DATA / "corpus" / "corpus_enriched.stats.json"

def load_jsonl(p: Path):
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_seqevt_codes() -> dict[int, str]:
    """ct_seqevt has integer code -> human-readable meaning. Used for
    Subj_Code, Occurrence_Code, Phase_of_Flight."""
    codes: dict[int, str] = {}
    for db in ("Pre2008", "avall"):
        for r in load_jsonl(NTSB_EXTRACT / db / "ct_seqevt.jsonl"):
            c = r.get("code")
            if isinstance(c, int):
                codes[c] = r.get("meaning") or codes.get(c, "")
    return codes

def load_ntsb_tables(db: str):
    """Returns dict of indices for one .mdb dump:
        events_by_evid, aircraft_by_evid_key, narratives_by_evid_key,
        findings_by_evid_key (list), occurrences_by_evid_key (list),
        seqevt_by_evid_key (list), injuries_by_evid_key (list)
    """
    p = NTSB_EXTRACT / db
    events = {r["ev_id"]: r for r in load_jsonl(p / "events.jsonl")}
    aircraft = {(r["ev_id"], r["Aircraft_Key"]): r for r in load_jsonl(p / "aircraft.jsonl")}
    narratives = {(r["ev_id"], r["Aircraft_Key"]): r for r in load_jsonl(p / "narratives.jsonl")}
    findings = defaultdict(list)
    for r in load_jsonl(p / "Findings.jsonl"):
        findings[(r["ev_id"], r["Aircraft_Key"])].append(r)
    occurrences = defaultdict(list)
    for r in load_jsonl(p / "Occurrences.jsonl"):
        occurrences[(r["ev_id"], r["Aircraft_Key"])].append(r)
    seqevt = defaultdict(list)
    for r in load_jsonl(p / "seq_of_events.jsonl"):
        seqevt[(r["ev_id"], r["Aircraft_Key"])].append(r)
    injuries = defaultdict(list)
    for r in load_jsonl(p / "injury.jsonl"):
        injuries[(r["ev_id"], r["Aircraft_Key"])].append(r)
    # Build ntsb_no -> (ev_id, Aircraft_Key) reverse index for the NTSB_REPORT join
    by_ntsb_no: dict[str, tuple[str, int]] = {}
    for (ev_id, ak), ac in aircraft.items():
        nn = ac.get("ntsb_no")
        if nn:
            by_ntsb_no[nn.strip()] = (ev_id, ak)
    return {
        "events": events, "aircraft": aircraft, "narratives": narratives,
        "findings": findings, "occurrences": occurrences, "seqevt": seqevt,
        "injuries": injuries, "by_ntsb_no": by_ntsb_no,
    }

def load_faa_aids():
    out: dict[str, dict] = {}
    for p in sorted(FAA_EXTRACT.glob("a*.jsonl")):
        for r in load_jsonl(p):
            out[r["record_id"]] = r
    return out

def load_ntsb_report_manifest():
    """report_number -> {ntsb_accident_id, txt_path, ...}"""
    out: dict[str, dict] = {}
    with (NTSB_REPORTS / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            out[row["report_number"]] = row
    return out

# Severity mapping for FAA AIDS event_type and pilot_killed flags
def faa_severity(rec: dict) -> str | None:
    ev = (rec.get("event_type") or "").upper()
    pk = (rec.get("pilot_killed_flag") or "").strip().upper()
    if pk in ("Y", "1", "T"):
        return "fatal"
    if ev == "A":
        return "accident"
    if ev == "I":
        return "incident"
    return None

INJURY_LEVEL_MAP = {"FATL": "fatal", "SERS": "serious", "MINR": "minor", "NONE": "none"}

def aggregate_injuries(rows: list[dict]) -> dict:
    out = {"fatal": 0, "serious": 0, "minor": 0, "none": 0}
    for r in rows:
        lvl = INJURY_LEVEL_MAP.get((r.get("injury_level") or "").upper())
        if lvl:
            out[lvl] += int(r.get("inj_person_count") or 0)
    return out

def ntsb_structured(db: dict, ev_id: str, ak: int, codes: dict[int, str]) -> dict:
    """Build the per-record structured payload for an NTSB record."""
    s = {
        "primary_cause_text": None,
        "findings": [],
        "occurrences": [],
        "seq_of_events": [],
        "phase_of_flight": None,
        "aircraft": None,
        "injury": None,
        "event_type": None,
    }
    ev = db["events"].get(ev_id)
    if ev:
        s["event_type"] = (ev.get("ev_type") or "").upper()
    ac = db["aircraft"].get((ev_id, ak))
    if ac:
        s["aircraft"] = {
            "make": ac.get("acft_make"),
            "model": ac.get("acft_model"),
            "damage": ac.get("damage"),
            "num_eng": ac.get("num_eng"),
            "category": ac.get("acft_category"),
            "regis_no": ac.get("regis_no"),
            "ntsb_no": ac.get("ntsb_no"),
        }
    for f in db["findings"].get((ev_id, ak), []):
        s["findings"].append({
            "code": f.get("finding_code"),
            "description": f.get("finding_description"),
            "category": f.get("category_no"),
            "cause_factor": f.get("Cause_Factor"),
        })
    seen_phases = []
    for o in db["occurrences"].get((ev_id, ak), []):
        oc = o.get("Occurrence_Code")
        ph = o.get("Phase_of_Flight")
        s["occurrences"].append({
            "occurrence_code": oc,
            "code_text": codes.get(oc) if isinstance(oc, int) else None,
            "phase_code": ph,
            "phase_text": codes.get(ph) if isinstance(ph, int) else None,
        })
        if ph and codes.get(ph):
            seen_phases.append(codes[ph])
    if seen_phases:
        s["phase_of_flight"] = seen_phases[0]
    for sv in sorted(db["seqevt"].get((ev_id, ak), []),
                     key=lambda r: (r.get("Occurrence_No") or 0, r.get("seq_event_no") or 0)):
        sc = sv.get("Subj_Code")
        s["seq_of_events"].append({
            "order": sv.get("seq_event_no"),
            "occurrence_no": sv.get("Occurrence_No"),
            "subj_code": sc,
            "subj_text": codes.get(sc) if isinstance(sc, int) else None,
            "modifier_code": sv.get("Modifier_Code"),
            "modifier_text": codes.get(sv.get("Modifier_Code")) if isinstance(sv.get("Modifier_Code"), int) else None,
            "cause_factor": sv.get("Cause_Factor"),
            "person_code": sv.get("Person_Code"),
        })
    inj = db["injuries"].get((ev_id, ak), [])
    if inj:
        s["injury"] = aggregate_injuries(inj)
    # Pull primary cause from first cause-flagged finding (avall) or seq_of_events (Pre2008)
    cause_findings = [f for f in s["findings"] if (f.get("cause_factor") or "").upper() in ("C", "F")]
    if cause_findings:
        s["primary_cause_text"] = cause_findings[0]["description"]
    else:
        cause_seq = [v for v in s["seq_of_events"] if (v.get("cause_factor") or "").upper() == "C"]
        if cause_seq and cause_seq[0].get("subj_text"):
            s["primary_cause_text"] = cause_seq[0]["subj_text"]
    return s

def faa_structured(rec: dict) -> dict:
    return {
        "primary_cause_text": rec.get("cause_primary_text"),
        "secondary_cause_text": rec.get("cause_secondary_text"),
        "additional_causes": [c for c in (rec.get("cause_additional_text"), rec.get("cause_2nd_additional_text")) if c],
        "general_cause_category": rec.get("cause_general_category_text"),
        "phase_of_flight": rec.get("phase_of_flight_text"),
        "flying_condition": rec.get("flying_condition_primary"),
        "light_condition": rec.get("light_condition_text"),
        "event_type": (rec.get("event_type") or "").upper() or None,
        "aircraft": {
            "make": rec.get("aircraft_make"),
            "model": rec.get("aircraft_model"),
            "engine_make": rec.get("engine_make"),
            "engine_model": rec.get("engine_model"),
        },
        "severity": faa_severity(rec),
    }

def main():
    print("Loading code lookup table…")
    codes = load_seqevt_codes()
    print(f"  {len(codes):,} codes")
    print("Loading Pre2008.mdb extracted tables…")
    pre2008 = load_ntsb_tables("Pre2008")
    print(f"  events:{len(pre2008['events']):,} aircraft:{len(pre2008['aircraft']):,} "
          f"findings:{sum(len(v) for v in pre2008['findings'].values()):,} "
          f"occurrences:{sum(len(v) for v in pre2008['occurrences'].values()):,} "
          f"seq:{sum(len(v) for v in pre2008['seqevt'].values()):,}")
    print("Loading avall.mdb extracted tables…")
    avall = load_ntsb_tables("avall")
    print(f"  events:{len(avall['events']):,} aircraft:{len(avall['aircraft']):,} "
          f"findings:{sum(len(v) for v in avall['findings'].values()):,}")
    print("Loading FAA AIDS extracted records…")
    faa = load_faa_aids()
    print(f"  {len(faa):,} records")
    print("Loading NTSB report manifest…")
    manifest = load_ntsb_report_manifest()
    print(f"  {len(manifest):,} reports")

    # Stats
    n_total = 0
    src_counts = defaultdict(int)
    src_with_struct = defaultdict(int)
    src_with_cause = defaultdict(int)

    print(f"Joining onto {CORPUS} -> {OUT} …")
    with CORPUS.open() as f, OUT.open("w") as outf:
        for line in f:
            r = json.loads(line)
            n_total += 1
            src = r.get("source", "?")
            src_counts[src] += 1
            structured = None
            sup_source = None
            if src == "NTSB_ASRS:Pre2008":
                rid = r["record_id"]  # "<ev_id>_<AircraftKey>"
                if "_" in rid:
                    ev_id, ak_s = rid.rsplit("_", 1)
                    try:
                        ak = int(ak_s)
                        structured = ntsb_structured(pre2008, ev_id, ak, codes)
                        sup_source = "NTSB_eADMS_Pre2008"
                    except ValueError:
                        pass
            elif src == "NTSB_ASRS:avall":
                rid = r["record_id"]
                if "_" in rid:
                    ev_id, ak_s = rid.rsplit("_", 1)
                    try:
                        ak = int(ak_s)
                        structured = ntsb_structured(avall, ev_id, ak, codes)
                        sup_source = "NTSB_eADMS_avall"
                    except ValueError:
                        pass
            elif src == "NTSB_REPORT":
                m = manifest.get(r["record_id"])
                if m and m.get("ntsb_accident_id"):
                    nn = m["ntsb_accident_id"].strip()
                    hit = pre2008["by_ntsb_no"].get(nn) or avall["by_ntsb_no"].get(nn)
                    if hit:
                        ev_id, ak = hit
                        db = pre2008 if nn in pre2008["by_ntsb_no"] else avall
                        structured = ntsb_structured(db, ev_id, ak, codes)
                        sup_source = "NTSB_REPORT_via_ntsb_no"
                if structured is None:
                    # No .mdb match; fall back to manifest metadata
                    if m:
                        structured = {
                            "ntsb_accident_id": m.get("ntsb_accident_id"),
                            "report_type": m.get("report_type"),
                            "txt_path": m.get("txt_path"),
                        }
                        sup_source = "NTSB_REPORT_manifest_only"
            elif src.startswith("FAA_AIDS:"):
                rec = faa.get(r["record_id"])
                if rec:
                    structured = faa_structured(rec)
                    sup_source = "FAA_AIDS"
            # BEA, TSB_CANADA: no structured supervision available
            r["structured"] = structured
            r["supervision_source"] = sup_source
            outf.write(json.dumps(r, ensure_ascii=False) + "\n")
            if structured:
                src_with_struct[src] += 1
                if structured.get("primary_cause_text"):
                    src_with_cause[src] += 1

    # Write stats
    stats = {
        "total": n_total,
        "by_source": {
            s: {
                "count": src_counts[s],
                "with_structured": src_with_struct.get(s, 0),
                "with_primary_cause": src_with_cause.get(s, 0),
            }
            for s in src_counts
        },
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2))
    print(f"\nDone. {n_total:,} records -> {OUT}")
    print(f"Stats:\n{json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()
