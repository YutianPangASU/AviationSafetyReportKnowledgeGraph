"""Layer-1 aggregate knowledge graph builder.

Reads the per-narrative graphs in `event_extraction/out/full_corpus_v3.jsonl`
(streaming, resumable / partial-input safe) and produces:

  * a single aggregate MultiDiGraph keyed by node-type (event_type for events,
    type for entities/conditions) — all 56k narratives merged into one graph
  * per-CICTT-category subgraphs (LOC-I, CFIT, SCF, WX, MAC, RE, etc.)
  * a markdown summary report with top event_types, edges, and per-category
    breakdowns
  * a CSV of edge-counts for downstream Stage-3 work (PC algorithm input,
    LLM causal-order priors)

This is Layer 1 — descriptive aggregation. Layer 2 (per-category causal
discovery with LLM priors) consumes these CSVs.

Usage:
    python event_extraction/scripts/build_kg_layer1.py \\
        --extraction event_extraction/out/full_corpus_v3.jsonl \\
        --enriched   data/corpus/corpus_enriched.jsonl \\
        --out-dir    event_extraction/out/aggregate_kg
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

# --- CICTT-equivalent mapping for NTSB Pre2008 Occurrence_Code ---
# Source: NTSB ct_seqevt + CAST/ICAO Common Taxonomy Team v4.7 mapping.
# Codes not listed pass through as 'OTHER:<code>' so we don't lose information.
NTSB_OCC_TO_CICTT = {
    100: "LOC-I",   # ABRUPT MANEUVER (loosely LOC-I)
    110: "LOC-I",   # ALTITUDE DEVIATION,UNCONTROLLED
    120: "OTHM",    # CARGO SHIFT (Other)
    130: "SCF-PP",  # AIRFRAME/COMPONENT/SYSTEM FAILURE/MALFUNCTION
    131: "SCF-PP",  # PROPELLER FAILURE
    140: "SCF-PP",  # POWERPLANT FAILURE
    160: "FUEL",    # FUEL RELATED
    180: "USOS",    # UNDERSHOOT/OVERSHOOT
    200: "ICE",     # ICING
    210: "F-NI",    # FIRE/SMOKE (NON-IMPACT)
    220: "WSTRW",   # WINDSHEAR/THUNDERSTORM
    230: "CFIT",    # IN FLIGHT COLLISION WITH TERRAIN/WATER
    240: "WX",      # IN FLIGHT ENCOUNTER WITH WEATHER
    250: "LOC-I",   # LOSS OF CONTROL - IN FLIGHT
    260: "LOC-G",   # LOSS OF CONTROL - ON GROUND/WATER
    270: "CTOL",    # COLLISION WITH OBJECT (FLIGHT)
    280: "GCOL",    # ON GROUND COLLISION WITH OBJECT/TERRAIN/VEHICLE
    290: "ARC",     # HARD LANDING
    300: "MAC",     # MIDAIR COLLISION
    310: "MAC",     # NEAR MIDAIR COLLISION
    320: "GCOL",    # ON GROUND COLLISION WITH AIRCRAFT
    330: "ARC",     # NOSE OVER
    340: "RE",      # OVERRUN
    350: "ARC",     # SHORT LANDING
    370: "ARC",     # GEAR COLLAPSED
    380: "LOC-I",   # STALL
    390: "BIRD",    # BIRD/WILDLIFE STRIKE
    400: "ADRM",    # AERODROME (taxi/runway events)
    410: "EVAC",    # EVACUATION
    420: "ATM",     # AIR TRAFFIC MANAGEMENT
    430: "RI",      # RUNWAY INCURSION
    440: "FUEL",    # FUEL EXHAUSTION/STARVATION
    540: "OTHM",    # NON-AIRCRAFT (other)
    560: "EXTL",    # EXTERNAL CONDITIONS (other)
    580: "MED",     # MEDICAL (incapacitation)
    600: "SEC",     # SECURITY-RELATED
}

def categorize(record: dict) -> str:
    """Return a short CICTT-like category for a record. Falls back to source-
    derived buckets when structured supervision cannot be mapped."""
    s = record.get("structured") or {}
    src = record.get("source", "?")
    # Pre2008: prefer the FIRST occurrence with a known mapping; else first occ as 'OTHER:<code>'
    occs = s.get("occurrences") or []
    for o in occs:
        oc = o.get("occurrence_code")
        if isinstance(oc, int) and oc in NTSB_OCC_TO_CICTT:
            return NTSB_OCC_TO_CICTT[oc]
    if occs:
        oc = occs[0].get("occurrence_code")
        if oc is not None:
            return f"OTHER:NTSB-{oc}"
    # avall: first finding's HFACS category number
    findings = s.get("findings") or []
    if findings:
        cat = findings[0].get("category")
        if cat:
            return f"HFACS_CAT_{cat}"
    # FAA AIDS: cause_general_category
    cgc = s.get("general_cause_category")
    if cgc:
        # truncate FAA's 20-char fields to a stable bucket name
        return f"AIDS_{cgc.strip().split(',')[0].split()[0].upper()}"
    # No structured signal
    return f"UNCATEGORIZED:{src.split(':')[0]}"

# --- aggregate-node id helpers ---
def evt_node(event_type: str) -> str:
    return f"EVT:{event_type or 'unknown'}"
def cond_node(cond_type: str) -> str:
    return f"COND:{cond_type or 'unknown'}"
def ent_node(entity_type: str) -> str:
    return f"ENT:{entity_type or 'unknown'}"

def node_id_from_local(node: dict, local_index: dict[str, dict]) -> str | None:
    """Map a per-narrative node id to its aggregate-graph node id."""
    if not isinstance(node, dict):
        return None
    k = node.get("kind")
    if k == "event":
        return evt_node(node.get("event_type"))
    if k == "condition":
        return cond_node(node.get("type"))
    if k == "entity":
        return ent_node(node.get("type"))
    return None

# --- main aggregation ---
def aggregate(extraction_path: Path, enriched_path: Path) -> tuple[nx.MultiDiGraph, dict[str, nx.MultiDiGraph], dict]:
    print(f"Indexing {enriched_path.name} for category lookup…")
    cat_by_id: dict[str, str] = {}
    with enriched_path.open() as f:
        for line in f:
            r = json.loads(line)
            cat_by_id[r["record_id"]] = categorize(r)
    print(f"  {len(cat_by_id):,} records indexed")

    G = nx.MultiDiGraph()                 # full aggregate KG
    perCat: dict[str, nx.MultiDiGraph] = defaultdict(nx.MultiDiGraph)

    n_records = n_ok = 0
    n_events_total = n_edges_total = 0
    n_records_per_cat: Counter = Counter()
    event_type_counts: Counter = Counter()
    edge_triple_counts: Counter = Counter()  # (src_id, type, dst_id) -> count
    cause_role_per_event: defaultdict = defaultdict(Counter)  # EVT:X -> Counter({primary:n,...})
    phase_per_event: defaultdict = defaultdict(Counter)        # EVT:X -> Counter({phase:n,...})
    severity_per_event: defaultdict = defaultdict(Counter)
    haem_per_event: defaultdict = defaultdict(Counter)

    print(f"Streaming {extraction_path.name}…")
    with extraction_path.open() as f:
        for line in f:
            n_records += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok"):
                continue
            n_ok += 1
            rid = r.get("record_id", "")
            cat = cat_by_id.get(rid, "UNCATEGORIZED:?")
            n_records_per_cat[cat] += 1

            local = {n.get("id"): n for n in (r.get("nodes") or []) if isinstance(n, dict) and n.get("id")}
            seen_nodes_in_record: set[str] = set()
            seen_edges_in_record: set[tuple[str, str, str]] = set()

            # Aggregate nodes
            for n in (r.get("nodes") or []):
                if not isinstance(n, dict):
                    continue
                nid = node_id_from_local(n, local)
                if not nid:
                    continue
                seen_nodes_in_record.add(nid)
                if n.get("kind") == "event":
                    n_events_total += 1
                    event_type_counts[nid] += 1
                    cause_role_per_event[nid][n.get("cause_role") or "?"] += 1
                    phase_per_event[nid][n.get("phase_of_flight") or "?"] += 1
                    severity_per_event[nid][n.get("severity") or "?"] += 1
                    haem_per_event[nid][n.get("haem") or "?"] += 1

            # Aggregate edges
            for e in (r.get("edges") or []):
                if not isinstance(e, dict):
                    continue
                src_local = local.get(e.get("src"))
                dst_local = local.get(e.get("dst"))
                if not src_local or not dst_local:
                    continue
                src_agg = node_id_from_local(src_local, local)
                dst_agg = node_id_from_local(dst_local, local)
                if not src_agg or not dst_agg:
                    continue
                etype = e.get("type") or "?"
                key = (src_agg, etype, dst_agg)
                # dedup within a single narrative — we count per-record
                # presence, not multiple mentions of the same triple
                if key in seen_edges_in_record:
                    continue
                seen_edges_in_record.add(key)
                edge_triple_counts[key] += 1
                n_edges_total += 1
                # also write into per-category subgraph
                pG = perCat[cat]
                if not pG.has_node(src_agg):
                    pG.add_node(src_agg, kind=src_agg.split(":", 1)[0])
                if not pG.has_node(dst_agg):
                    pG.add_node(dst_agg, kind=dst_agg.split(":", 1)[0])

    # Materialise the full graph
    print(f"\nMaterialising aggregate graph…")
    for nid, c in event_type_counts.items():
        G.add_node(nid, kind="event",
                   count=c,
                   cause_role=dict(cause_role_per_event[nid]),
                   phase=dict(phase_per_event[nid]),
                   severity=dict(severity_per_event[nid]),
                   haem=dict(haem_per_event[nid]))
    # add condition / entity nodes that appeared
    for (src, etype, dst), c in edge_triple_counts.items():
        for nid in (src, dst):
            if not G.has_node(nid):
                kind = nid.split(":", 1)[0].lower()
                G.add_node(nid, kind=kind, count=0)
        G.add_edge(src, dst, key=etype, type=etype, count=c)
    # write per-category edge counts too
    for (src, etype, dst), _ in edge_triple_counts.items():
        # we'll re-iterate from scratch with proper per-cat counts below
        pass

    # Re-walk to populate per-category edge counts (cheap second pass; we
    # already kept the seed nodes during the first pass)
    print(f"Computing per-category edges…")
    perCat_edges: defaultdict[str, Counter] = defaultdict(Counter)
    with extraction_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok"):
                continue
            cat = cat_by_id.get(r.get("record_id", ""), "UNCATEGORIZED:?")
            local = {n.get("id"): n for n in (r.get("nodes") or []) if isinstance(n, dict) and n.get("id")}
            seen_in_record: set[tuple[str, str, str]] = set()
            for e in (r.get("edges") or []):
                if not isinstance(e, dict):
                    continue
                src_local = local.get(e.get("src"))
                dst_local = local.get(e.get("dst"))
                if not src_local or not dst_local:
                    continue
                src_agg = node_id_from_local(src_local, local)
                dst_agg = node_id_from_local(dst_local, local)
                if not src_agg or not dst_agg:
                    continue
                k = (src_agg, e.get("type") or "?", dst_agg)
                if k in seen_in_record:
                    continue
                seen_in_record.add(k)
                perCat_edges[cat][k] += 1
    for cat, ctr in perCat_edges.items():
        pG = perCat[cat]
        for (src, etype, dst), c in ctr.items():
            if not pG.has_node(src):
                pG.add_node(src, kind=src.split(":", 1)[0])
            if not pG.has_node(dst):
                pG.add_node(dst, kind=dst.split(":", 1)[0])
            pG.add_edge(src, dst, key=etype, type=etype, count=c)

    summary = {
        "n_records_in_extraction": n_records,
        "n_records_ok": n_ok,
        "n_event_node_instances": n_events_total,
        "n_edge_instances": n_edges_total,
        "n_aggregate_nodes": G.number_of_nodes(),
        "n_aggregate_edges": G.number_of_edges(),
        "records_per_category": dict(n_records_per_cat.most_common()),
    }
    return G, dict(perCat), summary

# --- output writers ---
def write_summary_md(G: nx.MultiDiGraph, perCat: dict[str, nx.MultiDiGraph], summary: dict, out_path: Path):
    lines = []
    lines.append(f"# Aggregate Knowledge Graph (Layer 1)\n")
    lines.append(f"Generated from `full_corpus_v3.jsonl`. {summary['n_records_ok']:,} records contributed.\n")
    lines.append(f"## Topline\n")
    lines.append(f"- Records (ok): **{summary['n_records_ok']:,}** of {summary['n_records_in_extraction']:,} total")
    lines.append(f"- Aggregate nodes: **{summary['n_aggregate_nodes']:,}**")
    lines.append(f"- Aggregate edges (distinct (src,type,dst) triples): **{summary['n_aggregate_edges']:,}**")
    lines.append(f"- Total event-instance count across narratives: {summary['n_event_node_instances']:,}")
    lines.append(f"- Total edge-instance count across narratives: {summary['n_edge_instances']:,}\n")

    lines.append(f"## Top 25 event_types by frequency\n")
    lines.append("| Rank | event_type | Count | Top cause_role | Top phase | Top HAEM |")
    lines.append("|---|---|---:|---|---|---|")
    evt_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("kind") == "event"]
    evt_nodes.sort(key=lambda kv: -kv[1].get("count", 0))
    for i, (n, d) in enumerate(evt_nodes[:25], 1):
        cr = max(d.get("cause_role", {}).items(), key=lambda kv: kv[1], default=("-", 0))
        ph = max(d.get("phase", {}).items(), key=lambda kv: kv[1], default=("-", 0))
        hm = max(d.get("haem", {}).items(), key=lambda kv: kv[1], default=("-", 0))
        et = n.split(":", 1)[1]
        lines.append(f"| {i} | `{et}` | {d.get('count', 0):,} | {cr[0]} ({cr[1]}) | {ph[0]} ({ph[1]}) | {hm[0]} ({hm[1]}) |")

    lines.append(f"\n## Top 30 edges by frequency (event ↔ event)\n")
    lines.append("| Rank | src | edge | dst | Count |")
    lines.append("|---|---|---|---|---:|")
    edge_rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u.startswith("EVT:") and v.startswith("EVT:"):
            edge_rows.append((u, k, v, d.get("count", 0)))
    edge_rows.sort(key=lambda r: -r[3])
    for i, (u, etype, v, c) in enumerate(edge_rows[:30], 1):
        lines.append(f"| {i} | `{u.split(':', 1)[1]}` | {etype} | `{v.split(':', 1)[1]}` | {c:,} |")

    lines.append(f"\n## Records per CICTT category\n")
    lines.append("| Category | Records |")
    lines.append("|---|---:|")
    for cat, c in summary["records_per_category"].items():
        lines.append(f"| `{cat}` | {c:,} |")

    lines.append(f"\n## Per-category subgraph sizes\n")
    lines.append("| Category | Nodes | Edges | Top edge |")
    lines.append("|---|---:|---:|---|")
    for cat in sorted(perCat, key=lambda c: -perCat[c].number_of_edges()):
        pG = perCat[cat]
        if pG.number_of_edges() == 0:
            continue
        top = max(((u, v, k, d) for u, v, k, d in pG.edges(keys=True, data=True)),
                  key=lambda r: r[3].get("count", 0), default=None)
        top_str = ""
        if top:
            u, v, k, d = top
            top_str = f"`{u.split(':',1)[1]}` --{k}--> `{v.split(':',1)[1]}` ({d.get('count', 0)})"
        lines.append(f"| `{cat}` | {pG.number_of_nodes()} | {pG.number_of_edges()} | {top_str} |")

    out_path.write_text("\n".join(lines))

def write_edges_csv(G: nx.MultiDiGraph, out_path: Path):
    """Edge-count CSV — fuel for Stage-3 causal discovery."""
    with out_path.open("w") as f:
        f.write("src,edge_type,dst,count\n")
        for u, v, k, d in G.edges(keys=True, data=True):
            f.write(f"{u},{k},{v},{d.get('count', 0)}\n")

def write_per_cat_edges_csv(perCat: dict[str, nx.MultiDiGraph], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat, pG in perCat.items():
        if pG.number_of_edges() == 0:
            continue
        safe = cat.replace(":", "_").replace("/", "_")
        with (out_dir / f"{safe}.csv").open("w") as f:
            f.write("src,edge_type,dst,count\n")
            for u, v, k, d in pG.edges(keys=True, data=True):
                f.write(f"{u},{k},{v},{d.get('count', 0)}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", default="event_extraction/out/full_corpus_v3.jsonl", type=Path)
    ap.add_argument("--enriched", default="data/corpus/corpus_enriched.jsonl", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg", type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    G, perCat, summary = aggregate(args.extraction, args.enriched)

    # write outputs
    pickle_path = args.out_dir / "aggregate_kg.gpickle"
    graphml_path = args.out_dir / "aggregate_kg.graphml"
    edges_csv = args.out_dir / "aggregate_edges.csv"
    summary_md = args.out_dir / "aggregate_kg.summary.md"
    summary_json = args.out_dir / "aggregate_kg.summary.json"
    per_cat_csv_dir = args.out_dir / "per_category"

    print(f"Writing graph -> {pickle_path}")
    import pickle
    with pickle_path.open("wb") as f:
        pickle.dump(G, f)
    print(f"Writing GraphML -> {graphml_path}")
    # GraphML can't store dicts as attrs — flatten to JSON strings
    Gflat = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}
        Gflat.add_node(n, **flat)
    for u, v, k, d in G.edges(keys=True, data=True):
        flat = {kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in d.items()}
        Gflat.add_edge(u, v, key=k, **flat)
    nx.write_graphml(Gflat, str(graphml_path))

    print(f"Writing edges CSV -> {edges_csv}")
    write_edges_csv(G, edges_csv)
    print(f"Writing per-category CSVs -> {per_cat_csv_dir}/")
    write_per_cat_edges_csv(perCat, per_cat_csv_dir)
    print(f"Writing summary JSON -> {summary_json}")
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"Writing summary MD -> {summary_md}")
    write_summary_md(G, perCat, summary, summary_md)

    print(f"\nDone. {summary['n_records_ok']:,} records -> {G.number_of_nodes()} aggregate nodes, "
          f"{G.number_of_edges()} aggregate edges, {len(perCat)} per-category subgraphs.")

if __name__ == "__main__":
    main()
