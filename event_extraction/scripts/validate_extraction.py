"""Post-extraction structural + enum validator.

Validates an extraction JSONL against the closed vocabularies of its schema
and the structural invariants that `guided_json` cannot express. Works for
both the v3 node/edge format and the v4 chain format (Phase-1 of the v4 plan,
docs/2026-07-05-v4-chain-schema-plan.md).

Checks (v3 format):
  * JSON parse / ok-flag failures
  * enum membership for every closed field (pulled from the schema file, so
    the validator never drifts out of sync with the contract)
  * edge endpoints reference existing node ids
  * edge type validity (catches PERFORMED_BY / AFFECTED_BY / AFFECTS drift)
  * type-level self-loops (same event_type on both ends of an edge) — these
    collapse to self-loops in the aggregate KG
  * events-per-record cap (schema says <= 20)

Checks (v4 chain format):
  * enum membership (factor_type, class, cause_role, phase_of_flight,
    outcome_severity)
  * idx equals list position
  * caused_by.src strictly less than own idx (acyclicity by construction)
  * caused_by.src references an existing chain position
  * outcome-class nodes never cited as cause of a non-outcome node
  * same-factor_type caused_by links (type-level self-loop)

Usage:
    python event_extraction/scripts/validate_extraction.py \
        --extraction event_extraction/out/full_corpus_v3.jsonl \
        --schema event_extraction/prompts/schema_v3.json \
        --out event_extraction/out/validation_v3.summary.json \
        --flagged event_extraction/out/validation_v3.flagged.jsonl

Exit code 0 = clean, 2 = violations found (summary still written).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------- schema ---
def load_def_enums(schema_path: Path) -> dict[str, dict[str, set]]:
    """Walk a JSON Schema's $defs and return {def_name: {field: set(enum)}}.

    Only leaf 'enum' keywords are collected (including enums nested one level
    inside oneOf branches, as used for nullable enum fields)."""
    schema = json.loads(schema_path.read_text())
    out: dict[str, dict[str, set]] = {}
    for def_name, definition in (schema.get("$defs") or {}).items():
        fields: dict[str, set] = {}
        for fname, fdef in (definition.get("properties") or {}).items():
            if not isinstance(fdef, dict):
                continue
            if "enum" in fdef:
                fields[fname] = set(fdef["enum"])
            elif "oneOf" in fdef:
                vals: set = set()
                for branch in fdef["oneOf"]:
                    if isinstance(branch, dict) and "enum" in branch:
                        vals.update(branch["enum"])
                if vals:
                    fields[fname] = vals
        out[def_name] = fields
    # top-level properties too (v4 keeps outcome_severity at the top level)
    top: dict[str, set] = {}
    for fname, fdef in (schema.get("properties") or {}).items():
        if isinstance(fdef, dict) and "enum" in fdef:
            top[fname] = set(fdef["enum"])
    if top:
        out["_top"] = top
    return out


def detect_format(record: dict) -> str:
    if "chain" in record:
        return "v4"
    if "nodes" in record or "edges" in record:
        return "v3"
    return "unknown"


# ------------------------------------------------------------ validators ---
class Report:
    def __init__(self):
        self.n_lines = 0
        self.n_parse_errors = 0
        self.n_not_ok = 0
        self.n_validated = 0
        self.n_records_with_violations = 0
        self.violations = Counter()            # check name -> count
        self.oov = defaultdict(Counter)        # field -> Counter(bad value)
        self.flagged: list[dict] = []          # {record_id, problems: [...]}

    def add(self, record_id: str, problems: list[str]):
        if problems:
            self.n_records_with_violations += 1
            self.flagged.append({"record_id": record_id, "problems": problems})

    def summary(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "n_parse_errors": self.n_parse_errors,
            "n_not_ok": self.n_not_ok,
            "n_validated": self.n_validated,
            "n_records_with_violations": self.n_records_with_violations,
            "violation_counts": dict(self.violations.most_common()),
            "oov_values": {f: dict(c.most_common(25)) for f, c in self.oov.items()},
        }


def check_enum(rep: Report, problems: list, enums: dict[str, set],
               field: str, value, check_name: str):
    allowed = enums.get(field)
    if allowed is None or value is None:
        return
    if value not in allowed:
        rep.violations[check_name] += 1
        rep.oov[check_name][str(value)] += 1
        problems.append(f"{check_name}: {value!r}")


MAX_EVENTS_PER_RECORD = 20


def validate_v3(rec: dict, enums: dict[str, dict[str, set]], rep: Report) -> list[str]:
    problems: list[str] = []
    ev_enums = enums.get("Event", {})
    ent_enums = enums.get("Entity", {})
    cond_enums = enums.get("Condition", {})
    edge_enums = enums.get("Edge", {})

    nodes = rec.get("nodes") or []
    edges = rec.get("edges") or []
    node_by_id: dict[str, dict] = {}
    n_events = 0
    for n in nodes:
        if not isinstance(n, dict):
            rep.violations["node_not_object"] += 1
            problems.append("node_not_object")
            continue
        nid = n.get("id")
        if nid in node_by_id:
            rep.violations["duplicate_node_id"] += 1
            problems.append(f"duplicate_node_id: {nid}")
        node_by_id[nid] = n
        kind = n.get("kind")
        if kind == "event":
            n_events += 1
            check_enum(rep, problems, ev_enums, "event_type", n.get("event_type"), "event_type_oov")
            check_enum(rep, problems, ev_enums, "haem", n.get("haem"), "haem_oov")
            check_enum(rep, problems, ev_enums, "cause_role", n.get("cause_role"), "cause_role_oov")
            check_enum(rep, problems, ev_enums, "phase_of_flight", n.get("phase_of_flight"), "phase_oov")
            check_enum(rep, problems, ev_enums, "severity", n.get("severity"), "severity_oov")
        elif kind == "entity":
            check_enum(rep, problems, ent_enums, "type", n.get("type"), "entity_type_oov")
        elif kind == "condition":
            check_enum(rep, problems, cond_enums, "type", n.get("type"), "condition_type_oov")
            check_enum(rep, problems, cond_enums, "scope", n.get("scope"), "condition_scope_oov")
        else:
            rep.violations["node_kind_invalid"] += 1
            rep.oov["node_kind_invalid"][str(kind)] += 1
            problems.append(f"node_kind_invalid: {kind!r}")

    if n_events > MAX_EVENTS_PER_RECORD:
        rep.violations["too_many_events"] += 1
        problems.append(f"too_many_events: {n_events}")

    for e in edges:
        if not isinstance(e, dict):
            rep.violations["edge_not_object"] += 1
            problems.append("edge_not_object")
            continue
        check_enum(rep, problems, edge_enums, "type", e.get("type"), "edge_type_oov")
        src, dst = node_by_id.get(e.get("src")), node_by_id.get(e.get("dst"))
        if src is None or dst is None:
            rep.violations["edge_dangling_endpoint"] += 1
            problems.append(f"edge_dangling_endpoint: {e.get('src')}->{e.get('dst')}")
            continue
        if (src.get("kind") == "event" and dst.get("kind") == "event"
                and src.get("event_type") == dst.get("event_type")):
            rep.violations["type_level_self_loop"] += 1
            rep.oov["type_level_self_loop"][str(src.get("event_type"))] += 1
            problems.append(f"type_level_self_loop: {src.get('event_type')}")
    return problems


def validate_v4(rec: dict, enums: dict[str, dict[str, set]], rep: Report) -> list[str]:
    problems: list[str] = []
    node_enums = enums.get("ChainNode", {})
    top_enums = enums.get("_top", {})

    chain = rec.get("chain") or []
    check_enum(rep, problems, top_enums, "outcome_severity",
               rec.get("outcome_severity"), "outcome_severity_oov")

    classes: list[str | None] = []
    types: list[str | None] = []
    for pos, n in enumerate(chain):
        if not isinstance(n, dict):
            rep.violations["chain_item_not_object"] += 1
            problems.append("chain_item_not_object")
            classes.append(None)
            types.append(None)
            continue
        classes.append(n.get("class"))
        types.append(n.get("factor_type"))
        if n.get("idx") != pos:
            rep.violations["idx_mismatch"] += 1
            problems.append(f"idx_mismatch: pos {pos} has idx {n.get('idx')}")
        check_enum(rep, problems, node_enums, "class", n.get("class"), "class_oov")
        check_enum(rep, problems, node_enums, "factor_type", n.get("factor_type"), "factor_type_oov")
        check_enum(rep, problems, node_enums, "cause_role", n.get("cause_role"), "cause_role_oov")
        check_enum(rep, problems, node_enums, "phase_of_flight", n.get("phase_of_flight"), "phase_oov")

        for link in (n.get("caused_by") or []):
            if not isinstance(link, dict):
                rep.violations["caused_by_not_object"] += 1
                problems.append("caused_by_not_object")
                continue
            src = link.get("src")
            check_enum(rep, problems, node_enums, "strength", link.get("strength"), "strength_oov")
            if not isinstance(src, int) or src < 0 or src >= len(chain):
                rep.violations["caused_by_out_of_range"] += 1
                problems.append(f"caused_by_out_of_range: {src} at pos {pos}")
                continue
            if src >= pos:
                rep.violations["caused_by_forward_reference"] += 1
                problems.append(f"caused_by_forward_reference: {src} >= {pos}")
            if src < len(classes) and classes[src] == "outcome" and n.get("class") != "outcome":
                rep.violations["outcome_as_cause"] += 1
                problems.append(f"outcome_as_cause: pos {src} -> pos {pos}")
            if src < len(types) and types[src] is not None and types[src] == n.get("factor_type"):
                rep.violations["type_level_self_loop"] += 1
                rep.oov["type_level_self_loop"][str(n.get("factor_type"))] += 1
                problems.append(f"type_level_self_loop: {n.get('factor_type')}")
    return problems


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", required=True, type=Path)
    ap.add_argument("--schema", type=Path,
                    default=Path("event_extraction/prompts/schema_v4.json"),
                    help="schema whose enums to validate against (pass "
                    "schema_v3.json when checking v3-format files)")
    ap.add_argument("--format", choices=["auto", "v3", "v4"], default="auto")
    ap.add_argument("--out", type=Path, default=None,
                    help="summary JSON path (default: <extraction>.validation.json)")
    ap.add_argument("--flagged", type=Path, default=None,
                    help="optional JSONL of flagged records with their problems")
    ap.add_argument("--max-flagged", type=int, default=5000,
                    help="cap on flagged records kept in memory / written out")
    args = ap.parse_args()

    enums = load_def_enums(args.schema)
    rep = Report()
    fmt = args.format

    with args.extraction.open() as f:
        for line in f:
            rep.n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rep.n_parse_errors += 1
                continue
            if not rec.get("ok", True):
                rep.n_not_ok += 1
                continue
            rfmt = detect_format(rec) if fmt == "auto" else fmt
            if rfmt == "v3":
                problems = validate_v3(rec, enums, rep)
            elif rfmt == "v4":
                problems = validate_v4(rec, enums, rep)
            else:
                rep.violations["unknown_format"] += 1
                problems = ["unknown_format"]
            rep.n_validated += 1
            if problems and len(rep.flagged) < args.max_flagged:
                rep.add(rec.get("record_id", f"line{rep.n_lines}"), problems)
            elif problems:
                rep.n_records_with_violations += 1

    summary = rep.summary()
    out_path = args.out or args.extraction.with_suffix(".validation.json")
    out_path.write_text(json.dumps(summary, indent=2))
    if args.flagged:
        with args.flagged.open("w") as g:
            for row in rep.flagged:
                g.write(json.dumps(row) + "\n")

    total_viol = sum(summary["violation_counts"].values())
    print(json.dumps(summary, indent=2))
    print(f"\nSummary written -> {out_path}")
    if args.flagged:
        print(f"Flagged records  -> {args.flagged} ({len(rep.flagged)} rows)")
    sys.exit(0 if total_viol == 0 else 2)


if __name__ == "__main__":
    main()
