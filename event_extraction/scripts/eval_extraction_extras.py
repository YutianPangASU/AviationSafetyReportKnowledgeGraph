"""Additional extraction-evaluation measures from the stored judge outputs
(review 2026-09-14, comment 5). No model calls: everything is read from the
judge outputs already on disk.

  1. Precision by node role. The main evaluation scores every chain node
     against the cause-flagged findings, so intermediates and outcomes can
     never match. Here nodes with cause_role primary or contributing are
     scored against the cause flags, and event-class nodes are scored
     against the coded occurrence sequence (from the chain-order judge).
  2. Recall by finding group. NTSB avall findings carry a category code
     (01 aircraft, 02 personnel, 03 environment, 04 organizational, 05 not
     determined). Pre2008 sequence entries carry no category, so they are
     grouped by subject keywords and reported separately.
  3. Adjudication sample. 100 records (seed 0) exported with the narrative,
     the cause-flagged findings, the extracted nodes and the judge's
     matches, for two human annotators; agreement with the judge and between
     annotators is then computed by score_adjudication.py.

Output: event_extraction/out/eval_extras_v4_narrative.json and
        event_extraction/out/adjudication/sample_100.jsonl (+ .csv)
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENRICHED = ROOT / "data/corpus/corpus_enriched.jsonl"
EXTRACTION = ROOT / "event_extraction/out/calibration_2k_v4_narrative.jsonl"
JUDGE = ROOT / "event_extraction/out/semantic_eval_v4_narrative.jsonl"
ORDER = ROOT / "event_extraction/out/chain_order_v4_narrative.jsonl"
OUT = ROOT / "event_extraction/out/eval_extras_v4_narrative.json"
ADJ_DIR = ROOT / "event_extraction/out/adjudication"

AVALL_CAT = {"01": "aircraft", "02": "personnel", "03": "environment",
             "04": "organizational", "05": "not determined"}
KEYWORDS = [  # Pre2008 subject text, first match wins
    ("organizational", ("MAINTENANCE", "INSPECTION", "COMPANY", "MANAGEMENT",
                        "FAA", "ATC", "SUPERVISION", "MANUFACTURER", "PROCEDURE INADEQUATE",
                        "TRAINING")),
    ("environment", ("WEATHER", "WIND", "TERRAIN", "VISIBILITY", "LIGHT CONDITION",
                     "ICING", "TURBULENCE", "OBJECT", "RUNWAY", "ANIMAL", "BIRD",
                     "DENSITY ALTITUDE", "CLOUDS", "FOG", "RAIN", "SNOW", "CROSSWIND",
                     "TAILWIND", "GUSTS", "CARBURETOR ICING CONDITIONS")),
    ("aircraft", ("ENGINE", "FUEL SYSTEM", "LANDING GEAR", "FLIGHT CONTROL", "PROPELLER",
                  "AIRFRAME", "SYSTEM", "COMPONENT", "CARBURETOR", "MAGNETO", "CYLINDER",
                  "WING", "BRAKE", "TIRE", "ELECTRICAL", "HYDRAULIC", "FUEL TANK",
                  "FUEL LINE", "EXHAUST", "OIL")),
    ("personnel", ("PILOT", "CREW", "INSTRUCTOR", "PASSENGER", "JUDGMENT", "DECISION",
                   "PROCEDURE", "INADEQUATE", "IMPROPER", "FAILURE TO", "NOT MAINTAINED",
                   "MISJUDGED", "DELAYED", "EXCEEDED", "SELECTED", "PLANNING",
                   "PREFLIGHT", "FUEL EXHAUSTION", "FUEL STARVATION", "ALTITUDE",
                   "AIRSPEED", "STALL", "COMPENSATION", "ATTENTION", "REMEDIAL ACTION")),
]


def cause_entries(structured: dict) -> list[dict]:
    """Same order as semantic_eval.cause_strings, with a group per entry."""
    out = []
    for f in (structured.get("findings") or []):
        if (f.get("cause_factor") or "").upper() in ("C", "F") and f.get("description"):
            cat = (f.get("category") or "")[:2]
            out.append({"text": f["description"], "group": AVALL_CAT.get(cat),
                        "source": "avall"})
    for v in (structured.get("seq_of_events") or []):
        if (v.get("cause_factor") or "").upper() == "C" and v.get("subj_text"):
            txt = v["subj_text"]
            if v.get("modifier_text") and v["modifier_text"] != "None":
                txt = f"{txt} ({v['modifier_text']})"
            up = txt.upper()
            grp = next((g for g, kws in KEYWORDS if any(k in up for k in kws)), "unclassified")
            out.append({"text": txt, "group": grp, "source": "pre2008_keyword"})
    return out


def main() -> None:
    import argparse
    global EXTRACTION, JUDGE, ORDER, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="narrative", choices=["narrative", "grounded"])
    args = ap.parse_args()
    if args.tag == "grounded":
        EXTRACTION = ROOT / "event_extraction/out/calibration_2k_v4_grounded.jsonl"
        JUDGE = ROOT / "event_extraction/out/semantic_eval_v4_grounded.jsonl"
        ORDER = ROOT / "event_extraction/out/chain_order_v4_grounded.jsonl"
        OUT = ROOT / "event_extraction/out/eval_extras_v4_grounded.json"
    chains = {}
    with open(EXTRACTION) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok"):
                chains[r["record_id"]] = r
    judge = {}
    with open(JUDGE) as f:
        for line in f:
            r = json.loads(line)
            judge[r["record_id"]] = r["matches"]
    order = {}
    if ORDER.exists():
        with open(ORDER) as f:
            for line in f:
                r = json.loads(line)
                order[r["record_id"]] = r["matches"]
    enriched = {}
    with open(ENRICHED) as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in judge:
                enriched[r["record_id"]] = r

    # --- 1. precision by node role ---------------------------------------------
    role_tot, role_hit = defaultdict(int), defaultdict(int)
    cls_tot, cls_hit = defaultdict(int), defaultdict(int)
    for rid, m in judge.items():
        rec = chains.get(rid)
        if not rec:
            continue
        matched = {e for lst in m.values() for e in lst}
        for k, n in enumerate(rec["chain"]):
            role, cls = n.get("cause_role") or "none", n.get("class") or "none"
            role_tot[role] += 1
            cls_tot[cls] += 1
            if k in matched:
                role_hit[role] += 1
                cls_hit[cls] += 1
    prec_role = {r: {"nodes": role_tot[r], "matched_to_cause_flag": role_hit[r],
                     "precision": round(role_hit[r] / role_tot[r], 4)} for r in role_tot}
    prec_cls = {c: {"nodes": cls_tot[c], "matched": cls_hit[c],
                    "precision": round(cls_hit[c] / cls_tot[c], 4)} for c in cls_tot}
    cause_nodes = role_tot["primary"] + role_tot["contributing"]
    cause_hits = role_hit["primary"] + role_hit["contributing"]

    # event-class nodes against the coded occurrence sequence
    ev_tot = ev_hit = 0
    n_order_records = 0
    for rid, m in order.items():
        rec = chains.get(rid)
        if not rec:
            continue
        n_order_records += 1
        matched = {e for lst in m.values() for e in lst}
        for k, n in enumerate(rec["chain"]):
            if n.get("class") == "event":
                ev_tot += 1
                ev_hit += k in matched

    # --- 2. recall by finding group ------------------------------------------------
    grp_tot, grp_hit = defaultdict(int), defaultdict(int)
    for rid, m in judge.items():
        s = (enriched.get(rid) or {}).get("structured") or {}
        entries = cause_entries(s)
        if len(entries) != len(m):
            continue  # ordering mismatch; skip rather than misattribute
        for i, e in enumerate(entries):
            key = (e["source"], e["group"])
            grp_tot[key] += 1
            grp_hit[key] += bool(m.get(str(i)))
    recall_group = {f"{src}:{grp}": {"findings": grp_tot[(src, grp)],
                                     "recalled": grp_hit[(src, grp)],
                                     "recall": round(grp_hit[(src, grp)] / grp_tot[(src, grp)], 4)}
                    for (src, grp) in sorted(grp_tot)}

    # --- 3. adjudication sample ---------------------------------------------------
    ADJ_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    ids = sorted(rid for rid in judge if rid in chains and rid in enriched
                 and cause_entries(enriched[rid].get("structured") or {}))
    sample = rng.sample(ids, min(100, len(ids)))
    with open(ADJ_DIR / "sample_100.jsonl", "w") as fj, \
            open(ADJ_DIR / "sample_100.csv", "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["record_id", "finding_idx", "finding_text", "node_idx", "factor_type",
                    "cause_role", "trigger", "judge_match", "annotator_match", "annotator_note"])
        for rid in sample:
            rec, m, s = chains[rid], judge[rid], enriched[rid].get("structured") or {}
            entries = cause_entries(s)
            fj.write(json.dumps({"record_id": rid, "narrative": enriched[rid].get("text"),
                                 "findings": [e["text"] for e in entries],
                                 "nodes": [{"idx": n["idx"], "factor_type": n["factor_type"],
                                            "cause_role": n.get("cause_role"),
                                            "trigger": n.get("trigger")} for n in rec["chain"]],
                                 "judge_matches": m}) + "\n")
            for i, e in enumerate(entries):
                for n in rec["chain"]:
                    w.writerow([rid, i, e["text"], n["idx"], n["factor_type"], n.get("cause_role"),
                                n.get("trigger"), int(n["idx"] in m.get(str(i), [])), "", ""])

    out = {
        "n_records_judged": len(judge),
        "precision_all_nodes": round(sum(role_hit.values()) / sum(role_tot.values()), 4),
        "precision_by_cause_role": prec_role,
        "precision_cause_role_nodes": {"nodes": cause_nodes, "matched": cause_hits,
                                       "precision": round(cause_hits / cause_nodes, 4)},
        "precision_by_class": prec_cls,
        "event_nodes_vs_coded_sequence": {"records_with_sequence": n_order_records,
                                          "event_nodes": ev_tot, "matched": ev_hit,
                                          "precision": round(ev_hit / ev_tot, 4) if ev_tot else None},
        "recall_by_finding_group": recall_group,
        "adjudication_sample": {"n": len(sample), "path": str(ADJ_DIR / "sample_100.jsonl")},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
