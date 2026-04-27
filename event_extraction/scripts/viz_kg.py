"""Render the Layer-1 aggregate KG to interactive HTML via pyvis.

Produces:
  * <out_dir>/aggregate_kg.html              - full aggregate graph
  * <out_dir>/per_category/<CAT>.html        - one HTML per CICTT category
  * <out_dir>/index.html                     - landing page with links

Each HTML is a single self-contained file with vis.js inlined: draggable
nodes, hover tooltips with counts and cause-role distribution, physics-based
auto-layout, search box.

Usage:
    python event_extraction/scripts/viz_kg.py \\
        --in-dir  event_extraction/out/aggregate_kg \\
        --out-dir event_extraction/out/aggregate_kg/html
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network

# --- CICTT (Common Aviation Taxonomy Team) short -> long names ---
CICTT_LONG = {
    "LOC-I":  "Loss of Control – In-flight",
    "LOC-G":  "Loss of Control – On Ground / Water",
    "CFIT":   "Controlled Flight Into / Toward Terrain",
    "SCF-PP": "System / Component Failure – Powerplant",
    "SCF-NP": "System / Component Failure – Non-Powerplant",
    "ARC":    "Abnormal Runway Contact",
    "RE":     "Runway Excursion",
    "RI":     "Runway Incursion",
    "USOS":   "Undershoot / Overshoot",
    "MAC":    "Mid-Air Collision (or Near-Miss)",
    "GCOL":   "Ground Collision (Aircraft–Aircraft)",
    "CTOL":   "Collision With Object (in flight)",
    "WSTRW":  "Windshear / Thunderstorm",
    "WX":     "In-flight Weather Encounter",
    "ICE":    "Icing",
    "BIRD":   "Bird / Wildlife Strike",
    "FUEL":   "Fuel Related (exhaustion / contamination)",
    "F-NI":   "Fire / Smoke (Non-Impact)",
    "EVAC":   "Evacuation",
    "ATM":    "Air Traffic Management Event",
    "ADRM":   "Aerodrome",
    "MED":    "Medical (incapacitation)",
    "SEC":    "Security-Related",
    "EXTL":   "External Conditions",
    "OTHM":   "Other / Miscellaneous",
}
def cictt_full(short: str) -> str:
    """Return 'SHORT — Long Name' if known, else the bucket as-is."""
    if short in CICTT_LONG:
        return f"{short} — {CICTT_LONG[short]}"
    if short.startswith("OTHER:NTSB-"):
        return f"{short} (NTSB Occurrence Code, no CICTT mapping)"
    if short.startswith("HFACS_CAT_"):
        return f"{short} (HFACS finding category — post-2008 NTSB CAST taxonomy)"
    if short.startswith("AIDS_"):
        return f"{short} (FAA AIDS legacy cause category)"
    if short.startswith("UNCATEGORIZED"):
        return f"{short} (no structured supervision)"
    return short

# --- visual encodings ---
NODE_COLOR = {
    "EVT": "#4f7cff",   # event - blue
    "COND": "#3aa56a",  # condition - green
    "ENT": "#9aa0a6",   # entity - gray
}
EDGE_COLOR = {
    # causal/temporal
    "CAUSES":          "#d62728",   # red
    "CONTRIBUTES_TO":  "#ff7f0e",   # orange
    "TRIGGERS":        "#e377c2",   # pink
    "ENABLES":         "#bcbd22",   # olive
    "PREVENTS":        "#2ca02c",   # green (good)
    "MITIGATES":       "#17becf",   # teal
    "DETECTS":         "#9467bd",   # purple
    "RESPONDS_TO":     "#8c564b",   # brown
    "PRECEDES":        "#7f7f7f",   # gray
    "CONCURRENT":      "#c7c7c7",   # light gray
    "CO_OCCURS":       "#c7c7c7",
    "IF_THEN":         "#aec7e8",   # light blue
    "INHIBITS":        "#98df8a",   # light green
    # condition / structural
    "UNDER_CONDITION":     "#1f77b4",
    "INDUCED_CONDITION":   "#1f77b4",
    "MASKED_BY_CONDITION": "#1f77b4",
    "SUPERVISED_BY":       "#7f7f7f",
}
DEFAULT_EDGE_COLOR = "#bbb"

def render(G: nx.MultiDiGraph, out_path: Path, *, title: str, top_n: int | None = None) -> None:
    """Render `G` (or its top-N edges by count) to a single HTML file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if top_n and G.number_of_edges() > top_n:
        # keep the top_n edges by count + every node those edges touch
        edges = sorted(
            ((u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)),
            key=lambda r: -r[3].get("count", 0),
        )[:top_n]
        keep_nodes: set[str] = set()
        for u, v, _, _ in edges:
            keep_nodes.add(u)
            keep_nodes.add(v)
        H = nx.MultiDiGraph()
        for n in keep_nodes:
            H.add_node(n, **G.nodes[n])
        for u, v, k, d in edges:
            H.add_edge(u, v, key=k, **d)
        G = H

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="#e0e0e0",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    # physics tuned for ~50–500 nodes
    net.barnes_hut(
        gravity=-30000, central_gravity=0.4, spring_length=120,
        spring_strength=0.02, damping=0.6, overlap=0.0,
    )
    net.toggle_physics(True)

    # add nodes
    max_count = max((d.get("count", 1) for _, d in G.nodes(data=True)), default=1) or 1
    for n, d in G.nodes(data=True):
        kind_prefix = n.split(":", 1)[0]
        color = NODE_COLOR.get(kind_prefix, "#888")
        count = d.get("count", 0)
        size = 12 + 28 * math.log1p(count) / max(1.0, math.log1p(max_count))
        label = n.split(":", 1)[1] if ":" in n else n
        # Tooltip: cause_role + phase + severity distribution
        tooltip_parts = [f"<b>{n}</b>", f"count: {count}"]
        for field in ("cause_role", "phase", "severity", "haem"):
            dist = d.get(field)
            if isinstance(dist, dict) and dist:
                top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
                tooltip_parts.append(
                    f"{field}: " + ", ".join(f"{k}={v}" for k, v in top)
                )
        net.add_node(
            n, label=label, color=color, size=size,
            title="<br>".join(tooltip_parts),
            shape="dot" if kind_prefix == "EVT" else ("diamond" if kind_prefix == "COND" else "box"),
        )

    # add edges
    max_e_count = max((d.get("count", 1) for _, _, _, d in G.edges(keys=True, data=True)), default=1) or 1
    for u, v, k, d in G.edges(keys=True, data=True):
        c = d.get("count", 0)
        width = 0.5 + 5.0 * math.log1p(c) / max(1.0, math.log1p(max_e_count))
        color = EDGE_COLOR.get(k, DEFAULT_EDGE_COLOR)
        title = f"<b>{k}</b><br>count: {c}"
        net.add_edge(
            u, v, title=title, color=color, width=width, label=str(c),
            font={"size": 10, "color": "#aaa", "strokeWidth": 0},
            arrows="to",
        )

    # control panel
    net.show_buttons(filter_=["physics", "edges", "nodes"])

    html = net.generate_html(notebook=False)
    # inject a small header banner
    banner = (
        f'<div style="background:#222;color:#eee;padding:10px 16px;'
        f'font-family:system-ui,sans-serif;border-bottom:1px solid #333;">'
        f'<b>{title}</b> &mdash; {G.number_of_nodes()} nodes, {G.number_of_edges()} edges'
        f'</div>'
    )
    html = html.replace("<body>", "<body>" + banner, 1)
    out_path.write_text(html)

def write_index(per_cat_files: list[tuple[str, int, int, Path]], main_file: Path, out_path: Path) -> None:
    """Index page linking to the main view + per-category subgraphs."""
    def row(cat: str, n: int, e: int, p: Path) -> str:
        full = cictt_full(cat)
        # split short / long for display
        if " — " in full:
            short, long_name = full.split(" — ", 1)
            label = f'<b>{short}</b> &mdash; {long_name}'
        else:
            label = full
        return (
            f'<li><a href="per_category/{p.name}">{label}</a> '
            f'<span class="meta">({n} nodes, {e} edges)</span></li>'
        )
    rows = "\n".join(row(*r) for r in per_cat_files)
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>ACE-Graph aggregate KG</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  .meta {{ color: #888; font-size: 0.9em; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: 0.3em 0; border-bottom: 1px solid #eee; }}
  .main {{ font-size: 1.1em; padding: 1em; background: #f4f7ff; border-radius: 6px; margin: 1em 0; }}
  .legend {{ background: #fafaf6; padding: 1em; border-radius: 6px; margin: 1em 0; font-size: 0.95em; }}
  .legend table {{ border-collapse: collapse; margin: 0.5em 0; width: 100%; }}
  .legend td, .legend th {{ padding: 0.3em 0.6em; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }}
  code {{ background: #eee; padding: 0 0.3em; border-radius: 3px; font-size: 0.95em; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>ACE-Graph aggregate KG</h1>
<p class="meta">Layer-1 aggregate produced by <code>build_kg_layer1.py</code>.</p>

<div class="legend">
<h3 style="margin-top:0;">What this graph shows</h3>
<p>Each per-category subgraph is built from <b>all accidents in that category</b>. The graph is a <b>type-level summary</b> of how accidents in this category typically unfold — it is <b>not</b> a single accident, and it is <b>not</b> a list of root causes.</p>

<table>
<tr><th>Symbol</th><th>What it means</th></tr>
<tr><td><span class="swatch" style="background:#4f7cff"></span><b>Blue dot — Event type</b></td>
    <td>One of the 42 fixed event types from the v3 schema (e.g. <code>CONTROL_INPUT_IMPROPER</code>, <code>STALL</code>, <code>GROUND_IMPACT</code>). The node represents the <i>concept</i>, not an individual occurrence — every accident that contained an event of this type contributes to it. Node size = log(count).</td></tr>
<tr><td><span class="swatch" style="background:#3aa56a"></span><b>Green diamond — Condition type</b></td>
    <td>State-like factor: <code>weather</code>, <code>mechanical</code>, <code>cognitive</code>, <code>regulatory</code>, etc. Same aggregation rule.</td></tr>
<tr><td><span class="swatch" style="background:#9aa0a6"></span><b>Gray box — Entity type</b></td>
    <td><code>aircraft</code>, <code>person</code>, <code>component</code>, etc. — present mostly to anchor participation, rarely the most informative part of the graph.</td></tr>
<tr><td><b>Edge with count = N</b></td>
    <td>The LLM extracted this <code>(src, edge_type, dst)</code> triple in <b>N distinct accidents</b> in this category (deduped within each accident). Edge thickness = log(N). Edge color = edge type (red <code>CAUSES</code>, orange <code>CONTRIBUTES_TO</code>, pink <code>TRIGGERS</code>, etc.).</td></tr>
</table>

<p><b>How to read an edge.</b> An edge <code>X --CAUSES--> Y</code> with count 112 in the LOC-I subgraph means: across the 323 LOC-I accidents in the corpus, 112 of them contained a narrative passage where the LLM extracted "X caused Y." It is <b>empirical co-occurrence with directionality</b>, not a verified causal claim. Stage 3 of the pipeline turns these counts into causal-discovery weights via the PC algorithm + LLM priors; this Layer-1 graph is the input, not the conclusion.</p>

<p><b>Reasons-for-the-category vs. mechanisms-within-the-category.</b> The category itself (e.g. "LOC-I") is the accident <i>type</i>. The graph shows the <b>internal mechanism patterns</b> typical of that type: which event types tend to precede which others, which conditions tend to be present, what the typical terminal events are. The "primary cause" of an individual accident is captured at a different layer — see <code>cause_role: primary</code> on each event in the original per-narrative extractions.</p>

<p><b>Why the high-count edges are sometimes uninformative.</b> The single most frequent edge in many subgraphs is <code>GROUND_IMPACT --CAUSES--> INJURY_OR_FATALITY</code>. That is mechanically correct ("crashes hurt people") but trivial. The interesting edges are the <i>upstream</i> ones — the chains leading <i>into</i> the terminal events, which is what Stage 3 will turn into a causal DAG.</p>
</div>

<div class="main">
  <a href="{main_file.name}"><b>Full aggregate graph</b></a>
  <span class="meta"> &mdash; every event type across every category, top-400 edges by count</span>
</div>

<h2>Per-CICTT-category subgraphs</h2>
<p class="meta">CICTT = ICAO/CAST <i>Common Aviation Taxonomy Team</i> occurrence categories. Each category captures a distinct accident type with its own characteristic causal mechanism — Stage 3 builds one DAG per category.</p>
<ul>
{rows}
</ul>

<h2>Files on disk</h2>
<ul>
  <li><code>aggregate_kg.gpickle</code> &mdash; NetworkX MultiDiGraph for Python</li>
  <li><code>aggregate_kg.graphml</code> &mdash; for Cytoscape / Gephi / yEd</li>
  <li><code>aggregate_edges.csv</code> &mdash; <code>src,edge_type,dst,count</code> for Stage-3 PC algorithm</li>
  <li><code>per_category/*.csv</code> &mdash; same format, one CSV per category</li>
  <li><code>aggregate_kg.summary.md</code> &mdash; top events / edges / per-category tables</li>
</ul>

</body></html>"""
    out_path.write_text(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="event_extraction/out/aggregate_kg", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/html", type=Path)
    ap.add_argument("--top-n-edges", default=400, type=int,
                    help="cap the full-graph view at the top-N edges by count "
                         "(default 400; per-category views are uncapped)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_category").mkdir(parents=True, exist_ok=True)

    # main aggregate
    pickle_path = args.in_dir / "aggregate_kg.gpickle"
    print(f"Loading {pickle_path}…")
    with pickle_path.open("rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    main_file = args.out_dir / "aggregate_kg.html"
    print(f"Rendering full aggregate -> {main_file} (top {args.top_n_edges} edges)")
    render(G, main_file, title="ACE-Graph &mdash; full aggregate", top_n=args.top_n_edges)

    # per-category from edge CSVs in per_category/
    per_cat_files = []
    pc_dir = args.in_dir / "per_category"
    for csv in sorted(pc_dir.glob("*.csv")):
        cat = csv.stem
        # rebuild a small subgraph from the CSV + node attributes from G
        sub = nx.MultiDiGraph()
        with csv.open() as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 4:
                    continue
                src, etype, dst, count = parts[0], parts[1], parts[2], int(parts[3])
                # carry node attrs from main G if present
                if not sub.has_node(src):
                    sub.add_node(src, **(G.nodes.get(src) or {"count": 0}))
                if not sub.has_node(dst):
                    sub.add_node(dst, **(G.nodes.get(dst) or {"count": 0}))
                sub.add_edge(src, dst, key=etype, type=etype, count=count)
        if sub.number_of_edges() == 0:
            continue
        out_html = args.out_dir / "per_category" / f"{cat}.html"
        render(sub, out_html, title=f"ACE-Graph &mdash; {cictt_full(cat)}")
        per_cat_files.append((cat, sub.number_of_nodes(), sub.number_of_edges(), out_html))
        print(f"  {cat:20s}  {sub.number_of_nodes():3d} nodes  {sub.number_of_edges():4d} edges  -> {out_html.name}")

    # index
    per_cat_files.sort(key=lambda r: -r[2])  # by edge count desc
    index_path = args.out_dir / "index.html"
    write_index(per_cat_files, main_file, index_path)
    print(f"\nIndex page: {index_path}")
    print(f"\nOpen this in your browser:")
    print(f"  file://{index_path.resolve()}")

if __name__ == "__main__":
    main()
