"""Piece 6 — render Stage-3 per-category DAGs to fancy HTML.

Different from viz_kg.py:
  * Input is per_category_dag/{cat}.gpickle (the PC output) — NetworkX DiGraph
    with edge attrs direction_score, support_count, llm_prior, laplacian_prior,
    bootstrap_stability.
  * Edge width  ∝ bootstrap_stability  (PC + bootstraps)
  * Edge opacity ∝ direction_score   (LLM + PC orientation)
  * Node colour by HFACS family (same palette as viz_kg.py)
  * Hierarchical layout (top-down causal flow) by default; fcose fallback

Run:
    python event_extraction/scripts/stage3/viz_dag.py \\
        --in-dir event_extraction/out/aggregate_kg/per_category_dag \\
        --out-dir event_extraction/out/aggregate_kg/html_dag
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from viz_kg import EVENT_FAMILY, CICTT_LONG, cictt_full

def to_elements(G: nx.DiGraph) -> list[dict]:
    elements: list[dict] = []
    presence = [d.get("presence", 1) for _, d in G.nodes(data=True)]
    max_pres = max(presence) if presence else 1
    for n, d in G.nodes(data=True):
        family = EVENT_FAMILY.get(n, "misc")
        pres = d.get("presence", 0)
        size = 14 + 24 * (math.log1p(pres) / max(1.0, math.log1p(max_pres)))
        elements.append({
            "data": {"id": n, "label": n, "family": family,
                     "kind": "event", "presence": pres, "size": size},
            "classes": f"node-event family-{family}",
        })
    for u, v, d in G.edges(data=True):
        boot = d.get("bootstrap_stability") or 0.0
        dirs = d.get("direction_score") or 0.0
        sup = d.get("support_count") or 0
        llm = d.get("llm_prior") or 0.0
        # edge width via bootstrap stability; floor at 0.5px so weak edges still show
        width = 0.7 + 5.0 * boot
        # opacity by direction_score
        opacity = 0.25 + 0.7 * dirs
        elements.append({
            "data": {"id": f"{u}__{v}", "source": u, "target": v,
                     "etype": "PC_DIRECTED",
                     "support_count": sup,
                     "bootstrap_stability": float(boot),
                     "direction_score": float(dirs),
                     "llm_prior": float(llm),
                     "width": width,
                     "opacity": opacity},
            "classes": "edge edge-PC",
        })
    return elements

HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  :root {{
    --bg: #0a0e1a; --panel: rgba(20, 26, 41, 0.85); --panel-border: #1f2a44;
    --fg: #e6edf3; --muted: #8b96b0; --accent: #38bdf8;
    --c-pilot: #ff6b6b; --c-pilot-pre: #ffa94d; --c-external: #ffd43b;
    --c-aircraft: #22d3ee; --c-env: #10b981; --c-org: #a78bfa;
    --c-aero: #ec4899; --c-outcome: #94a3b8; --c-misc: #6b7280;
    --e-PC: #f87171;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; height: 100%; width: 100%;
    background: var(--bg);
    background-image:
      radial-gradient(circle at 20% 0%, rgba(56, 189, 248, 0.08), transparent 40%),
      radial-gradient(circle at 80% 100%, rgba(236, 72, 153, 0.08), transparent 40%);
    color: var(--fg); font-family: 'Inter', sans-serif; overflow: hidden;
  }}
  #cy {{ position: absolute; inset: 0; z-index: 1; }}
  .topbar {{ position: absolute; top: 0; left: 0; right: 0; z-index: 10; padding: 14px 20px;
    background: linear-gradient(to bottom, rgba(10,14,26,0.95), rgba(10,14,26,0.7) 70%, transparent);
    backdrop-filter: blur(8px); display: flex; gap: 16px; }}
  .topbar h1 {{ margin: 0; font-size: 17px; font-weight: 700; line-height: 1.2; }}
  .topbar .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
  .topbar .stats {{ margin-left: auto; display: flex; gap: 12px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--panel-border); border-radius: 6px;
    padding: 6px 10px; font-size: 11px; color: var(--muted); }}
  .stat b {{ color: var(--fg); font-size: 13px; font-weight: 600; margin-right: 4px; }}
  .controls {{ position: absolute; top: 76px; left: 20px; z-index: 9;
    display: flex; flex-direction: column; gap: 10px; max-width: 280px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 10px 12px; backdrop-filter: blur(8px); }}
  .panel h3 {{ margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); font-weight: 600; }}
  .pill {{ display: inline-flex; padding: 3px 9px; margin: 3px 4px 3px 0; border-radius: 999px;
    font-size: 11px; cursor: pointer; user-select: none;
    background: rgba(255,255,255,0.05); color: var(--muted); border: 1px solid transparent; }}
  .pill.active {{ background: rgba(255,255,255,0.10); color: var(--fg); border-color: rgba(255,255,255,0.15); }}
  .detail {{ position: absolute; top: 76px; right: 20px; z-index: 9; width: 320px;
    max-height: calc(100% - 120px); overflow-y: auto;
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 14px 16px; backdrop-filter: blur(8px); font-size: 13px; display: none; }}
  .detail.shown {{ display: block; }}
  .detail h2 {{ margin: 0; font-size: 16px; font-weight: 700; }}
  .detail .row {{ display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; }}
  .detail .row b {{ font-family: 'JetBrains Mono', monospace; color: var(--accent); }}
  .detail h4 {{ margin: 12px 0 4px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); font-weight: 600; }}
  .legend {{ position: absolute; bottom: 14px; right: 20px; z-index: 9;
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 8px 12px; font-size: 11px; color: var(--muted); backdrop-filter: blur(8px); }}
  .legend a {{ color: var(--accent); text-decoration: none; font-weight: 500; }}
  .slider-row {{ display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--muted); margin: 6px 0; }}
  .slider-row input {{ flex: 1; }}
  .slider-row b {{ color: var(--fg); font-family: 'JetBrains Mono', monospace; min-width: 32px; text-align: right; }}
</style>
</head>
<body>
<div id="cy"></div>
<div class="topbar">
  <div>
    <h1>{title}</h1>
    <div class="sub">{subtitle}</div>
  </div>
  <div class="stats">
    <div class="stat"><b>{n_nodes}</b>nodes</div>
    <div class="stat"><b>{n_edges}</b>edges</div>
  </div>
</div>
<div class="controls">
  <div class="panel">
    <h3>edge filters</h3>
    <div class="slider-row"><span>min stability</span><input type="range" id="min-boot" min="0" max="1" step="0.05" value="0.0"><b id="min-boot-val">0.00</b></div>
    <div class="slider-row"><span>min direction</span><input type="range" id="min-dir" min="0" max="1" step="0.05" value="0.0"><b id="min-dir-val">0.00</b></div>
    <div class="slider-row"><span>min support</span><input type="range" id="min-sup" min="0" max="500" step="5" value="0"><b id="min-sup-val">0</b></div>
  </div>
  <div class="panel">
    <h3>layout</h3>
    <span class="pill active" data-layout="dagre">hierarchy (dagre)</span>
    <span class="pill" data-layout="fcose">force (fcose)</span>
    <span class="pill" data-layout="concentric">concentric</span>
  </div>
</div>
<div class="detail" id="detail">
  <h2 id="d-label"></h2>
  <div id="d-meta" style="color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:0.08em;"></div>
  <div id="d-stats"></div>
  <h4>Top causal predecessors</h4><div id="d-in"></div>
  <h4>Top causal successors</h4><div id="d-out"></div>
</div>
<div class="legend"><a href="../html/index.html">← back to Layer-1 KG</a></div>

<script>
const ELEMENTS = {elements_json};
const STYLE = [
  {{ selector: "node", style: {{
    "label": "data(label)", "color": "#e6edf3", "font-family": "Inter, sans-serif",
    "font-size": 11, "font-weight": 500, "text-outline-width": 2, "text-outline-color": "#0a0e1a",
    "text-margin-y": 6, "text-valign": "bottom", "text-halign": "center",
    "width": "data(size)", "height": "data(size)", "border-width": 1.5, "border-color": "rgba(255,255,255,0.15)",
  }} }},
  {{ selector: "node.family-pilot",     style: {{ "background-color": "var(--c-pilot)",     "shadow-color": "var(--c-pilot)",     "shadow-blur": 24, "shadow-opacity": 0.6 }} }},
  {{ selector: "node.family-pilot_pre", style: {{ "background-color": "var(--c-pilot-pre)", "shadow-color": "var(--c-pilot-pre)", "shadow-blur": 22, "shadow-opacity": 0.55 }} }},
  {{ selector: "node.family-external",  style: {{ "background-color": "var(--c-external)",  "shadow-color": "var(--c-external)",  "shadow-blur": 20, "shadow-opacity": 0.5 }} }},
  {{ selector: "node.family-aircraft",  style: {{ "background-color": "var(--c-aircraft)",  "shadow-color": "var(--c-aircraft)",  "shadow-blur": 24, "shadow-opacity": 0.6 }} }},
  {{ selector: "node.family-env",       style: {{ "background-color": "var(--c-env)",       "shadow-color": "var(--c-env)",       "shadow-blur": 22, "shadow-opacity": 0.55 }} }},
  {{ selector: "node.family-org",       style: {{ "background-color": "var(--c-org)",       "shadow-color": "var(--c-org)",       "shadow-blur": 20, "shadow-opacity": 0.5 }} }},
  {{ selector: "node.family-aero",      style: {{ "background-color": "var(--c-aero)",      "shadow-color": "var(--c-aero)",      "shadow-blur": 32, "shadow-opacity": 0.7, "border-width": 2 }} }},
  {{ selector: "node.family-outcome",   style: {{ "background-color": "var(--c-outcome)",   "shape": "round-rectangle" }} }},
  {{ selector: "node.family-misc",      style: {{ "background-color": "var(--c-misc)" }} }},
  {{ selector: "edge", style: {{
    "width": "data(width)", "opacity": "data(opacity)",
    "curve-style": "bezier", "control-point-step-size": 60,
    "target-arrow-shape": "triangle",
    "line-color": "var(--e-PC)", "target-arrow-color": "var(--e-PC)",
    "shadow-color": "var(--e-PC)", "shadow-blur": 8, "shadow-opacity": 0.35,
    "arrow-scale": 1.0,
  }} }},
  {{ selector: ".faded", style: {{ "opacity": 0.06 }} }},
  {{ selector: ".highlight", style: {{ "border-color": "#fff", "border-width": 2.5 }} }},
];

const cy = cytoscape({{
  container: document.getElementById("cy"),
  elements: ELEMENTS,
  style: STYLE,
  wheelSensitivity: 0.25,
  layout: {{ name: "dagre", rankDir: "TB", nodeSep: 30, edgeSep: 12, rankSep: 80, animate: true, animationDuration: 600 }},
}});

// sliders
function applyEdgeFilters() {{
  const mb = parseFloat(document.getElementById("min-boot").value);
  const md = parseFloat(document.getElementById("min-dir").value);
  const ms = parseInt(document.getElementById("min-sup").value);
  document.getElementById("min-boot-val").textContent = mb.toFixed(2);
  document.getElementById("min-dir-val").textContent  = md.toFixed(2);
  document.getElementById("min-sup-val").textContent  = ms;
  cy.batch(() => {{
    cy.edges().forEach(e => {{
      const ok = (e.data("bootstrap_stability") >= mb)
              && (e.data("direction_score") >= md)
              && (e.data("support_count") >= ms);
      e.style("display", ok ? "element" : "none");
    }});
    // hide nodes with zero visible edges? Keep all nodes for context.
  }});
}}
["min-boot","min-dir","min-sup"].forEach(id => {{
  document.getElementById(id).addEventListener("input", applyEdgeFilters);
}});
// layout switch
document.querySelectorAll(".pill[data-layout]").forEach(p => {{
  p.onclick = () => {{
    document.querySelectorAll(".pill[data-layout]").forEach(x => x.classList.remove("active"));
    p.classList.add("active");
    const cfg = {{
      dagre:      {{ name: "dagre", rankDir: "TB", nodeSep: 30, edgeSep: 12, rankSep: 80, animate: true, animationDuration: 600 }},
      fcose:      {{ name: "fcose", animate: true, animationDuration: 700, nodeRepulsion: 6000, idealEdgeLength: 75 }},
      concentric: {{ name: "concentric", animate: true, animationDuration: 600, concentric: n => n.degree(false), levelWidth: () => 2 }},
    }}[p.dataset.layout];
    cy.layout(cfg).run();
  }};
}});
// detail panel
const detail = document.getElementById("detail");
function topEdges(n, dir) {{
  const eds = (dir === "in" ? n.incomers("edge") : n.outgoers("edge"))
    .toArray()
    .filter(e => e.style("display") !== "none")
    .sort((a, b) => (b.data("bootstrap_stability") || 0) - (a.data("bootstrap_stability") || 0))
    .slice(0, 6);
  if (!eds.length) return "<div style='color:var(--muted)'>—</div>";
  return eds.map(e => {{
    const other = dir === "in" ? cy.getElementById(e.data("source")).data("label") : cy.getElementById(e.data("target")).data("label");
    return `<div class="row"><span>${{other}}</span><b>boot=${{e.data("bootstrap_stability").toFixed(2)}} sup=${{e.data("support_count")}}</b></div>`;
  }}).join("");
}}
cy.on("tap", "node", (ev) => {{
  const n = ev.target;
  cy.elements().addClass("faded").removeClass("highlight");
  n.removeClass("faded").addClass("highlight");
  n.closedNeighborhood().removeClass("faded");
  document.getElementById("d-label").textContent = n.data("label");
  document.getElementById("d-meta").textContent = `event · family ${{n.data("family")}}`;
  document.getElementById("d-stats").innerHTML = `
    <div class="row"><span>presence (accidents)</span><b>${{n.data("presence")}}</b></div>
    <div class="row"><span>in-degree</span><b>${{n.indegree(false)}}</b></div>
    <div class="row"><span>out-degree</span><b>${{n.outdegree(false)}}</b></div>`;
  document.getElementById("d-in").innerHTML = topEdges(n, "in");
  document.getElementById("d-out").innerHTML = topEdges(n, "out");
  detail.classList.add("shown");
}});
cy.on("tap", (ev) => {{ if (ev.target === cy) {{ cy.elements().removeClass("faded highlight"); detail.classList.remove("shown"); }} }});
</script>
</body></html>
"""

def render_one(G: nx.DiGraph, out_path: Path, title: str, subtitle: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    elements = to_elements(G)
    out_path.write_text(HTML_TEMPLATE.format(
        title=title, subtitle=subtitle,
        n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges(),
        elements_json=json.dumps(elements),
    ))

def write_index(per_cat: list[tuple[str, int, int, Path]], out_path: Path):
    rows = []
    for cat, nn, ne, p in per_cat:
        full = cictt_full(cat)
        if " — " in full:
            short, long_name = full.split(" — ", 1)
            label = f'<span class="cat-short">{short}</span><span class="cat-long">{long_name}</span>'
        else:
            label = full
        rows.append(f'<a href="{p.name}" class="card"><div class="card-title">{label}</div>'
                    f'<div class="card-stats"><span class="stat-pill">{nn} nodes</span>'
                    f'<span class="stat-pill">{ne} edges</span></div></a>')
    cards = "\n".join(rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Stage 3 — Per-category causal DAGs</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  body {{ margin: 0; padding: 40px 24px; min-height: 100vh; color: #e6edf3;
    background: #0a0e1a;
    background-image: radial-gradient(circle at 15% 0%, rgba(56, 189, 248, 0.10), transparent 35%),
                      radial-gradient(circle at 85% 100%, rgba(236, 72, 153, 0.10), transparent 35%);
    font-family: 'Inter', sans-serif; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 36px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px 0;
    background: linear-gradient(135deg, #fff 0%, #38bdf8 60%, #ec4899 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent; }}
  .lede {{ color: #8b96b0; font-size: 14px; max-width: 720px; line-height: 1.55; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 12px; }}
  .card {{ display: block; padding: 14px 16px; border-radius: 10px;
    background: rgba(20,26,41,0.85); border: 1px solid #1f2a44; color: #e6edf3; text-decoration: none;
    transition: transform 0.15s, border-color 0.15s; }}
  .card:hover {{ transform: translateY(-2px); border-color: #38bdf8; }}
  .card-title {{ font-weight: 600; font-size: 14px; margin-bottom: 8px; }}
  .cat-short {{ display: inline-block; padding: 2px 7px; border-radius: 4px;
    background: rgba(56,189,248,0.12); color: #38bdf8; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; margin-right: 8px; }}
  .stat-pill {{ background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px;
    font-size: 11px; color: #8b96b0; font-family: 'JetBrains Mono', monospace; margin-right: 6px; }}
  .legend-box {{ background: rgba(20,26,41,0.85); border: 1px solid #1f2a44; border-radius: 10px;
    padding: 18px 22px; margin: 20px 0; font-size: 13px; line-height: 1.6; }}
  .legend-box code {{ background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; }}
  a {{ color: #38bdf8; }}
</style></head><body><div class="container">
<h1>Stage-3 per-category causal DAGs</h1>
<p class="lede">Output of the constraint-based PC algorithm seeded with LLM causal-order priors and Laplacian-similarity priors. Edges are pruned by per-category support, oriented by LLM judgments where PC produced an undirected CPDAG, and weighted by bootstrap stability.</p>
<div class="legend-box">
<b>How to read these graphs</b><br>
Edge thickness = <code>bootstrap_stability</code> (fraction of resamples that preserved the edge). Edge opacity = <code>direction_score</code> (PC orientation strength × LLM agreement). Use the sliders to filter weak / unstable / unsupported edges. Default layout is hierarchical (top-down causal flow) via dagre; switch to fcose for an organic force-directed view.<br><br>
<a href="../html/index.html">← Layer-1 aggregate KG (descriptive)</a>
</div>
<div class="grid">{cards}</div>
</div></body></html>"""
    out_path.write_text(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="event_extraction/out/aggregate_kg/per_category_dag", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/html_dag", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_cat: list[tuple[str, int, int, Path]] = []
    for pkl in sorted(args.in_dir.glob("*.gpickle")):
        cat = pkl.stem.replace("_", ":") if pkl.stem.startswith("OTHER_") else pkl.stem
        # the stem may have had ':' replaced with '_'; we don't fully invert because the Layer-1 lookup
        # is purely by short name in CICTT_LONG. cictt_full handles unknowns gracefully.
        short = pkl.stem
        with pkl.open("rb") as f:
            G: nx.DiGraph = pickle.load(f)
        if G.number_of_edges() == 0:
            continue
        out_html = args.out_dir / f"{short}.html"
        full = cictt_full(short)
        render_one(G, out_html,
                   title=f"Stage-3 DAG — {full}",
                   subtitle=f"{G.number_of_nodes()} event types, {G.number_of_edges()} causal edges (PC + LLM prior + bootstrap)")
        per_cat.append((short, G.number_of_nodes(), G.number_of_edges(), out_html))
        print(f"  {short:20s}  {G.number_of_nodes():3d} nodes  {G.number_of_edges():4d} edges  -> {out_html.name}")
    write_index(sorted(per_cat, key=lambda r: -r[2]), args.out_dir / "index.html")
    print(f"\nIndex: {args.out_dir / 'index.html'}")

if __name__ == "__main__":
    main()
