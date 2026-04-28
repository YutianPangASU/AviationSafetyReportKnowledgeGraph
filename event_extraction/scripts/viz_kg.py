"""Render the Layer-1 aggregate KG to interactive HTML using Cytoscape.js + fCoSE.

Replaces the earlier pyvis/vis.js renderer with a more modern, fancy look:
  * dark theme with a neon palette
  * event types coloured by HFACS family (pilot acts / aircraft / env / org / aero / outcome / external)
  * edge style varies by edge type — CAUSES is a glowing red flow, CONTRIBUTES_TO is a softer orange,
    UNDER_CONDITION / TRIGGERS / PRECEDES each get distinct treatments
  * fCoSE force-directed layout with tuned parameters for organic clustering
  * sidebar with category title, stats, and per-node detail card on click
  * top bar with edge-type filter pills, family filter pills, and a search box

Outputs:
  * <out_dir>/aggregate_kg.html         - full aggregate (top-N edges by count)
  * <out_dir>/per_category/<CAT>.html   - one HTML per CICTT category
  * <out_dir>/index.html                - landing page

Cytoscape.js + fCoSE are loaded from CDN (~700 KB total). HTML files are
~tens of KB each (data is the only thing embedded).

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

import networkx as nx

# --- CICTT short -> long ---
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

# --- HFACS / family grouping for event types ---
EVENT_FAMILY = {
    # pilot acts (HFACS Level 1)
    "DECISION_INAPPROPRIATE":             "pilot",
    "CONTROL_INPUT_IMPROPER":             "pilot",
    "PROCEDURE_NOT_FOLLOWED":             "pilot",
    "PERCEPTION_FAILURE":                 "pilot",
    "SPATIAL_DISORIENTATION":             "pilot",
    "CREW_COORDINATION_FAILURE":          "pilot",
    # pilot preconditions
    "PILOT_INCAPACITATION_OR_IMPAIRMENT": "pilot_pre",
    "TRAINING_OR_CURRENCY_GAP":           "pilot_pre",
    # external operators
    "ATC_OR_DISPATCH_INADEQUATE":         "external",
    "GROUND_PERSONNEL_ERROR":             "external",
    # aircraft systems
    "ENGINE_FAILURE":                     "aircraft",
    "FUEL_SYSTEM_ANOMALY":                "aircraft",
    "AIRFRAME_STRUCTURAL_FAILURE":        "aircraft",
    "CONTROL_SURFACE_ANOMALY":            "aircraft",
    "INSTRUMENT_OR_AVIONICS_FAILURE":     "aircraft",
    "LANDING_GEAR_ANOMALY":               "aircraft",
    "AUTOMATION_ANOMALY":                 "aircraft",
    "INFLIGHT_FIRE_OR_SMOKE":             "aircraft",
    # environmental
    "ICING_ENCOUNTER":                    "env",
    "TURBULENCE_ENCOUNTER":               "env",
    "THUNDERSTORM_OR_CONVECTIVE":         "env",
    "LOW_VISIBILITY_OR_IMC":              "env",
    "WIND_SHEAR_OR_GUST":                 "env",
    "BIRD_OR_WILDLIFE_STRIKE":            "env",
    "TERRAIN_OR_OBSTACLE_PROXIMITY":      "env",
    "RUNWAY_CONDITION_HAZARD":            "env",
    # organizational
    "MAINTENANCE_INADEQUATE":             "org",
    "ORG_OR_REGULATORY_INADEQUATE":       "org",
    "DESIGN_DEFECT_LATENT":               "org",
    # aerodynamic states (failure-mode hubs)
    "STALL":                              "aero",
    "LOSS_OF_CONTROL_INFLIGHT":           "aero",
    "LOSS_OF_CONTROL_GROUND":             "aero",
    "STRUCTURAL_OVERLOAD":                "aero",
    "ALTITUDE_DEVIATION_UNCONTROLLED":    "aero",
    # outcomes (terminal events)
    "GROUND_IMPACT":                      "outcome",
    "WATER_IMPACT":                       "outcome",
    "MIDAIR_COLLISION":                   "outcome",
    "RUNWAY_EXCURSION_OR_OVERRUN":        "outcome",
    "INFLIGHT_BREAKUP":                   "outcome",
    "SUCCESSFUL_RECOVERY":                "outcome",
    "EMERGENCY_LANDING":                  "outcome",
    "INJURY_OR_FATALITY":                 "outcome",
}

# --- conversion: NetworkX -> Cytoscape elements ---
def to_elements(G: nx.MultiDiGraph) -> list[dict]:
    elements: list[dict] = []
    counts = [d.get("count", 1) for _, d in G.nodes(data=True)]
    max_c = max(counts) if counts else 1
    e_counts = [d.get("count", 1) for _, _, _, d in G.edges(keys=True, data=True)]
    max_ec = max(e_counts) if e_counts else 1

    for n, d in G.nodes(data=True):
        kind_prefix, _, label = n.partition(":")
        kind = {"EVT": "event", "COND": "condition", "ENT": "entity"}.get(kind_prefix, "other")
        family = EVENT_FAMILY.get(label, "misc") if kind == "event" else kind
        count = d.get("count", 0)
        # node radius via log scale; ~12px floor so labels stay readable,
        # ~38px cap so high-count hubs don't dominate
        size = 12 + 26 * (math.log1p(count) / max(1.0, math.log1p(max_c)))
        # tooltip distributions
        cause_role = d.get("cause_role") or {}
        phase = d.get("phase") or {}
        severity = d.get("severity") or {}
        haem = d.get("haem") or {}
        elements.append({
            "data": {
                "id": n,
                "label": label,
                "kind": kind,
                "family": family,
                "count": count,
                "size": size,
                "cause_role": cause_role,
                "phase": phase,
                "severity": severity,
                "haem": haem,
            },
            "classes": f"node-{kind} family-{family}",
        })

    for u, v, k, d in G.edges(keys=True, data=True):
        c = d.get("count", 0)
        # edge width via log scale, capped
        width = 1.0 + 6.5 * (math.log1p(c) / max(1.0, math.log1p(max_ec)))
        elements.append({
            "data": {
                "id": f"{u}__{k}__{v}",
                "source": u,
                "target": v,
                "etype": k,
                "count": c,
                "width": width,
            },
            "classes": f"edge edge-{k}",
        })
    return elements

# --- HTML template ---
HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<style>
  :root {{
    --bg: #0a0e1a;
    --bg-2: #0e1422;
    --panel: rgba(20, 26, 41, 0.85);
    --panel-border: #1f2a44;
    --fg: #e6edf3;
    --muted: #8b96b0;
    --accent: #38bdf8;
    /* family colours */
    --c-pilot: #ff6b6b;
    --c-pilot-pre: #ffa94d;
    --c-external: #ffd43b;
    --c-aircraft: #22d3ee;
    --c-env: #10b981;
    --c-org: #a78bfa;
    --c-aero: #ec4899;
    --c-outcome: #94a3b8;
    --c-misc: #6b7280;
    --c-condition: #34d399;
    --c-entity: #6b7280;
    /* edge colours */
    --e-CAUSES: #ef4444;
    --e-CONTRIBUTES_TO: #f97316;
    --e-TRIGGERS: #ec4899;
    --e-ENABLES: #facc15;
    --e-PREVENTS: #22c55e;
    --e-MITIGATES: #14b8a6;
    --e-DETECTS: #a78bfa;
    --e-RESPONDS_TO: #fb923c;
    --e-PRECEDES: #64748b;
    --e-CONCURRENT: #475569;
    --e-CO_OCCURS: #475569;
    --e-IF_THEN: #60a5fa;
    --e-INHIBITS: #4ade80;
    --e-UNDER_CONDITION: #38bdf8;
    --e-INDUCED_CONDITION: #38bdf8;
    --e-MASKED_BY_CONDITION: #38bdf8;
    --e-SUPERVISED_BY: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0; height: 100%; width: 100%;
    background: var(--bg);
    background-image:
      radial-gradient(circle at 20% 0%, rgba(56, 189, 248, 0.08), transparent 40%),
      radial-gradient(circle at 80% 100%, rgba(236, 72, 153, 0.08), transparent 40%);
    color: var(--fg);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: hidden;
  }}
  #cy {{
    position: absolute; inset: 0; z-index: 1;
  }}
  /* top bar */
  .topbar {{
    position: absolute; top: 0; left: 0; right: 0;
    z-index: 10; padding: 14px 20px;
    background: linear-gradient(to bottom, rgba(10, 14, 26, 0.95), rgba(10, 14, 26, 0.7) 70%, transparent);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    display: flex; align-items: flex-start; gap: 16px;
  }}
  .topbar h1 {{
    margin: 0; font-size: 17px; font-weight: 700; letter-spacing: 0.01em; line-height: 1.2;
  }}
  .topbar .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; font-weight: 500; }}
  .topbar .stats {{ margin-left: auto; display: flex; gap: 12px; }}
  .stat {{
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 6px;
    padding: 6px 10px; font-size: 11px; color: var(--muted);
  }}
  .stat b {{ color: var(--fg); font-size: 13px; font-weight: 600; margin-right: 4px; }}
  /* filters */
  .filters {{
    position: absolute; top: 76px; left: 20px; z-index: 9;
    display: flex; flex-direction: column; gap: 10px; max-width: 280px;
  }}
  .filter-group {{
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 10px 12px; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  }}
  .filter-group h3 {{
    margin: 0 0 8px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); font-weight: 600;
  }}
  .pill {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; margin: 3px 4px 3px 0; border-radius: 999px;
    font-size: 11px; font-weight: 500; cursor: pointer; user-select: none;
    background: rgba(255,255,255,0.05); color: var(--muted);
    border: 1px solid transparent; transition: all 0.15s;
  }}
  .pill .swatch {{
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
  }}
  .pill.active {{
    background: rgba(255,255,255,0.10); color: var(--fg); border-color: rgba(255,255,255,0.15);
  }}
  .pill:hover {{ color: var(--fg); }}
  /* detail panel (right) */
  .detail {{
    position: absolute; top: 76px; right: 20px; z-index: 9;
    width: 320px; max-height: calc(100% - 120px); overflow-y: auto;
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 14px 16px; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    font-size: 13px;
    display: none;
  }}
  .detail.shown {{ display: block; }}
  .detail h2 {{ margin: 0 0 4px 0; font-size: 16px; font-weight: 700; line-height: 1.25; }}
  .detail .kind {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
  .detail .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; margin: 12px 0; }}
  .detail .stats-grid div {{ display: flex; justify-content: space-between; font-size: 12px; }}
  .detail .stats-grid b {{ font-family: 'JetBrains Mono', monospace; }}
  .detail h4 {{ margin: 12px 0 4px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }}
  .detail .dist-row {{
    display: flex; justify-content: space-between; padding: 2px 0; font-size: 12px;
  }}
  .detail .dist-row b {{ font-family: 'JetBrains Mono', monospace; color: var(--accent); }}
  .detail .neighborhood {{ font-size: 12px; }}
  .detail .neighborhood div {{ padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
  .detail .close {{
    position: absolute; top: 10px; right: 12px; cursor: pointer; color: var(--muted);
    font-size: 18px; line-height: 1; padding: 2px 6px; border-radius: 4px;
  }}
  .detail .close:hover {{ background: rgba(255,255,255,0.05); color: var(--fg); }}
  /* search box */
  .search {{
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 6px;
    padding: 6px 10px; color: var(--fg); font-family: inherit; font-size: 12px;
    width: 200px;
  }}
  .search:focus {{ outline: none; border-color: var(--accent); }}
  /* legend at bottom */
  .legend {{
    position: absolute; bottom: 14px; right: 20px; z-index: 9;
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px;
    padding: 8px 12px; font-size: 11px; color: var(--muted);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  }}
  .legend a {{ color: var(--accent); text-decoration: none; font-weight: 500; }}
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
    <div class="stat"><b id="stat-nodes">{n_nodes}</b>nodes</div>
    <div class="stat"><b id="stat-edges">{n_edges}</b>edges</div>
    <input type="text" class="search" id="search" placeholder="search node…">
  </div>
</div>

<div class="filters">
  <div class="filter-group">
    <h3>family</h3>
    <div id="family-pills"></div>
  </div>
  <div class="filter-group">
    <h3>edge type</h3>
    <div id="edge-pills"></div>
  </div>
  <div class="filter-group">
    <h3>layout</h3>
    <span class="pill active" data-layout="fcose">fcose</span>
    <span class="pill" data-layout="concentric">concentric</span>
    <span class="pill" data-layout="cose">cose</span>
    <span class="pill" data-layout="circle">circle</span>
  </div>
</div>

<div class="detail" id="detail">
  <span class="close" id="close-detail">×</span>
  <h2 id="d-label"></h2>
  <div class="kind" id="d-kind"></div>
  <div class="stats-grid" id="d-stats"></div>
  <h4>Cause role distribution</h4><div id="d-causerole"></div>
  <h4>Phase of flight</h4><div id="d-phase"></div>
  <h4>Severity</h4><div id="d-severity"></div>
  <h4>HAEM</h4><div id="d-haem"></div>
  <h4>Top incoming edges</h4><div id="d-in" class="neighborhood"></div>
  <h4>Top outgoing edges</h4><div id="d-out" class="neighborhood"></div>
</div>

<div class="legend">
  <a href="../index.html">← back to index</a>
</div>

<script>
const ELEMENTS = {elements_json};
const FAMILIES = ["pilot","pilot_pre","external","aircraft","env","org","aero","outcome","misc","condition","entity"];
const FAMILY_LABELS = {{
  pilot: "Pilot acts",
  pilot_pre: "Pilot precond.",
  external: "ATC / ground",
  aircraft: "Aircraft",
  env: "Environmental",
  org: "Org / regulatory",
  aero: "Aerodynamic state",
  outcome: "Outcome",
  misc: "Other",
  condition: "Condition",
  entity: "Entity",
}};
const FAMILY_COLOR = {{
  pilot: "var(--c-pilot)", pilot_pre: "var(--c-pilot-pre)",
  external: "var(--c-external)", aircraft: "var(--c-aircraft)",
  env: "var(--c-env)", org: "var(--c-org)", aero: "var(--c-aero)",
  outcome: "var(--c-outcome)", misc: "var(--c-misc)",
  condition: "var(--c-condition)", entity: "var(--c-entity)",
}};
const EDGE_TYPES = [
  "CAUSES","CONTRIBUTES_TO","TRIGGERS","ENABLES","PREVENTS","MITIGATES","DETECTS","RESPONDS_TO",
  "PRECEDES","CONCURRENT","CO_OCCURS","IF_THEN","INHIBITS",
  "UNDER_CONDITION","INDUCED_CONDITION","MASKED_BY_CONDITION","SUPERVISED_BY",
];

// Cytoscape stylesheet
const STYLE = [
  // generic node
  {{
    selector: "node",
    style: {{
      "label": "data(label)",
      "color": "#e6edf3",
      "font-family": "Inter, sans-serif",
      "font-size": 11,
      "font-weight": 500,
      "text-outline-width": 2,
      "text-outline-color": "#0a0e1a",
      "text-outline-opacity": 0.9,
      "text-margin-y": 6,
      "text-valign": "bottom",
      "text-halign": "center",
      "width": "data(size)",
      "height": "data(size)",
      "border-width": 1.5,
      "border-color": "rgba(255,255,255,0.15)",
      "background-opacity": 0.95,
      "transition-property": "background-color, border-color, opacity, width, height",
      "transition-duration": "200ms",
    }},
  }},
  // family colours
  {{ selector: "node.family-pilot",     style: {{ "background-color": "var(--c-pilot)",     "shadow-color": "var(--c-pilot)",     "shadow-blur": 28, "shadow-opacity": 0.6 }} }},
  {{ selector: "node.family-pilot_pre", style: {{ "background-color": "var(--c-pilot-pre)", "shadow-color": "var(--c-pilot-pre)", "shadow-blur": 24, "shadow-opacity": 0.55 }} }},
  {{ selector: "node.family-external",  style: {{ "background-color": "var(--c-external)",  "shadow-color": "var(--c-external)",  "shadow-blur": 22, "shadow-opacity": 0.5 }} }},
  {{ selector: "node.family-aircraft",  style: {{ "background-color": "var(--c-aircraft)",  "shadow-color": "var(--c-aircraft)",  "shadow-blur": 28, "shadow-opacity": 0.6 }} }},
  {{ selector: "node.family-env",       style: {{ "background-color": "var(--c-env)",       "shadow-color": "var(--c-env)",       "shadow-blur": 24, "shadow-opacity": 0.55 }} }},
  {{ selector: "node.family-org",       style: {{ "background-color": "var(--c-org)",       "shadow-color": "var(--c-org)",       "shadow-blur": 22, "shadow-opacity": 0.5 }} }},
  {{ selector: "node.family-aero",      style: {{ "background-color": "var(--c-aero)",      "shadow-color": "var(--c-aero)",      "shadow-blur": 36, "shadow-opacity": 0.7, "border-color": "rgba(236,72,153,0.5)", "border-width": 2 }} }},
  {{ selector: "node.family-outcome",   style: {{ "background-color": "var(--c-outcome)",   "shape": "round-rectangle", "shadow-color": "var(--c-outcome)", "shadow-blur": 20, "shadow-opacity": 0.4 }} }},
  {{ selector: "node.family-misc",      style: {{ "background-color": "var(--c-misc)" }} }},
  {{ selector: "node.node-condition",   style: {{ "shape": "diamond", "background-color": "var(--c-condition)", "border-color": "rgba(52,211,153,0.5)" }} }},
  {{ selector: "node.node-entity",      style: {{ "shape": "round-rectangle", "background-color": "var(--c-entity)", "opacity": 0.7 }} }},
  // edges
  {{
    selector: "edge",
    style: {{
      "width": "data(width)",
      "curve-style": "bezier",
      "control-point-step-size": 60,
      "target-arrow-shape": "triangle",
      "target-arrow-color": "data(color)",
      "line-color": "data(color)",
      "arrow-scale": 0.9,
      "opacity": 0.65,
      "transition-property": "opacity, line-color, target-arrow-color, width",
      "transition-duration": "200ms",
    }},
  }},
];
// Inject per-edge-type colours via class selectors
EDGE_TYPES.forEach(t => {{
  STYLE.push({{ selector: `edge.edge-${{t}}`, style: {{
    "line-color": `var(--e-${{t}})`,
    "target-arrow-color": `var(--e-${{t}})`,
  }}}});
}});
// Causes / contributes get a glow
STYLE.push({{ selector: "edge.edge-CAUSES",         style: {{ "shadow-color": "var(--e-CAUSES)",         "shadow-blur": 12, "shadow-opacity": 0.5, "line-style": "solid" }} }});
STYLE.push({{ selector: "edge.edge-CONTRIBUTES_TO", style: {{ "shadow-color": "var(--e-CONTRIBUTES_TO)", "shadow-blur": 10, "shadow-opacity": 0.4, "line-style": "solid" }} }});
STYLE.push({{ selector: "edge.edge-TRIGGERS",       style: {{ "line-style": "solid" }} }});
STYLE.push({{ selector: "edge.edge-PRECEDES",       style: {{ "line-style": "dashed", "line-dash-pattern": [6, 4] }} }});
STYLE.push({{ selector: "edge.edge-UNDER_CONDITION,edge.edge-INDUCED_CONDITION,edge.edge-MASKED_BY_CONDITION", style: {{ "line-style": "dashed", "line-dash-pattern": [4, 4], "opacity": 0.5 }} }});
// hover / selected
STYLE.push({{ selector: ".faded",       style: {{ "opacity": 0.08 }} }});
STYLE.push({{ selector: ".highlight",   style: {{ "opacity": 1.0, "z-compound-depth": "top", "border-color": "#fff", "border-width": 2.5 }} }});
STYLE.push({{ selector: "edge.highlight", style: {{ "opacity": 1.0, "width": "mapData(width, 0, 8, 2, 10)", "z-index": 999 }} }});

const cy = cytoscape({{
  container: document.getElementById("cy"),
  elements: ELEMENTS,
  style: STYLE,
  wheelSensitivity: 0.25,
  layout: {{
    name: "fcose", quality: "default", randomize: true, animate: true, animationDuration: 800,
    nodeRepulsion: 6000, idealEdgeLength: 75, edgeElasticity: 0.45,
    gravity: 0.4, gravityRange: 3.8, numIter: 2500, nodeSeparation: 60, tile: false,
  }},
}});

// --- filter pills ---
const familyPills = document.getElementById("family-pills");
FAMILIES.forEach(f => {{
  const span = document.createElement("span");
  span.className = "pill active";
  span.dataset.family = f;
  span.innerHTML = `<span class="swatch" style="background:${{FAMILY_COLOR[f]}}"></span>${{FAMILY_LABELS[f]}}`;
  span.onclick = () => {{
    span.classList.toggle("active");
    applyFilters();
  }};
  familyPills.appendChild(span);
}});
const edgePills = document.getElementById("edge-pills");
EDGE_TYPES.forEach(t => {{
  const span = document.createElement("span");
  span.className = "pill active";
  span.dataset.etype = t;
  span.style.borderColor = `var(--e-${{t}})`;
  span.innerHTML = `<span class="swatch" style="background:var(--e-${{t}})"></span>${{t}}`;
  span.onclick = () => {{
    span.classList.toggle("active");
    applyFilters();
  }};
  edgePills.appendChild(span);
}});
function applyFilters() {{
  const activeFams = new Set(Array.from(document.querySelectorAll(".pill[data-family].active")).map(p => p.dataset.family));
  const activeEdges = new Set(Array.from(document.querySelectorAll(".pill[data-etype].active")).map(p => p.dataset.etype));
  cy.batch(() => {{
    cy.nodes().forEach(n => {{
      const fam = n.data("family");
      n.style("display", activeFams.has(fam) ? "element" : "none");
    }});
    cy.edges().forEach(e => {{
      const t = e.data("etype");
      const showEdge = activeEdges.has(t)
        && cy.getElementById(e.data("source")).visible()
        && cy.getElementById(e.data("target")).visible();
      e.style("display", showEdge ? "element" : "none");
    }});
  }});
}}
// --- layout switch ---
document.querySelectorAll(".pill[data-layout]").forEach(p => {{
  p.onclick = () => {{
    document.querySelectorAll(".pill[data-layout]").forEach(x => x.classList.remove("active"));
    p.classList.add("active");
    runLayout(p.dataset.layout);
  }};
}});
function runLayout(name) {{
  const cfg = {{
    fcose:      {{ name: "fcose", animate: true, animationDuration: 700, nodeRepulsion: 6000, idealEdgeLength: 75, gravity: 0.4, numIter: 2500 }},
    concentric: {{ name: "concentric", animate: true, animationDuration: 700, concentric: n => n.degree(false), levelWidth: () => 2, minNodeSpacing: 30 }},
    cose:       {{ name: "cose", animate: true, animationDuration: 700, idealEdgeLength: 80, nodeRepulsion: 400000 }},
    circle:     {{ name: "circle", animate: true, animationDuration: 700 }},
  }}[name];
  cy.layout(cfg).run();
}}
// --- search ---
document.getElementById("search").oninput = (ev) => {{
  const q = ev.target.value.toLowerCase().trim();
  if (!q) {{
    cy.elements().removeClass("faded highlight");
    return;
  }}
  const matched = cy.nodes().filter(n => n.data("label").toLowerCase().includes(q));
  cy.elements().addClass("faded").removeClass("highlight");
  matched.removeClass("faded").addClass("highlight");
  matched.connectedEdges().removeClass("faded");
  matched.neighborhood().nodes().removeClass("faded");
}};
// --- click to focus ---
const detail = document.getElementById("detail");
function distRows(distObj) {{
  const items = Object.entries(distObj || {{}}).sort((a,b) => b[1]-a[1]).slice(0,5);
  if (items.length === 0) return "<div class='dist-row' style='color:var(--muted)'>—</div>";
  return items.map(([k,v]) => `<div class="dist-row"><span>${{k}}</span><b>${{v}}</b></div>`).join("");
}}
function topEdges(node, dir) {{
  const edges = (dir === "in" ? node.incomers("edge") : node.outgoers("edge"))
    .toArray()
    .sort((a, b) => (b.data("count") || 0) - (a.data("count") || 0))
    .slice(0, 5);
  if (edges.length === 0) return "<div style='color:var(--muted)'>—</div>";
  return edges.map(e => {{
    const otherId = dir === "in" ? e.data("source") : e.data("target");
    const otherLabel = cy.getElementById(otherId).data("label");
    const t = e.data("etype");
    return `<div><span style="color:var(--e-${{t}})">${{t}}</span> &middot; ${{otherLabel}} <b style="float:right;font-family:'JetBrains Mono',monospace">${{e.data("count")}}</b></div>`;
  }}).join("");
}}
cy.on("tap", "node", (ev) => {{
  const n = ev.target;
  cy.elements().addClass("faded").removeClass("highlight");
  n.removeClass("faded").addClass("highlight");
  const nbh = n.closedNeighborhood();
  nbh.removeClass("faded");
  nbh.edges().addClass("highlight");
  // populate detail panel
  document.getElementById("d-label").textContent = n.data("label");
  document.getElementById("d-kind").textContent = `${{n.data("kind")}} · family: ${{n.data("family")}}`;
  document.getElementById("d-stats").innerHTML = `
    <div><span>count</span><b>${{n.data("count")}}</b></div>
    <div><span>degree</span><b>${{n.degree(false)}}</b></div>`;
  document.getElementById("d-causerole").innerHTML = distRows(n.data("cause_role"));
  document.getElementById("d-phase").innerHTML = distRows(n.data("phase"));
  document.getElementById("d-severity").innerHTML = distRows(n.data("severity"));
  document.getElementById("d-haem").innerHTML = distRows(n.data("haem"));
  document.getElementById("d-in").innerHTML = topEdges(n, "in");
  document.getElementById("d-out").innerHTML = topEdges(n, "out");
  detail.classList.add("shown");
}});
cy.on("tap", (ev) => {{
  if (ev.target === cy) {{
    cy.elements().removeClass("faded highlight");
    detail.classList.remove("shown");
  }}
}});
document.getElementById("close-detail").onclick = () => {{
  cy.elements().removeClass("faded highlight");
  detail.classList.remove("shown");
}};
</script>
</body></html>
"""

def render(G: nx.MultiDiGraph, out_path: Path, *, title: str, subtitle: str, top_n: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if top_n and G.number_of_edges() > top_n:
        edges = sorted(
            ((u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)),
            key=lambda r: -r[3].get("count", 0),
        )[:top_n]
        keep_nodes: set[str] = set()
        for u, v, _, _ in edges:
            keep_nodes.add(u); keep_nodes.add(v)
        H = nx.MultiDiGraph()
        for n in keep_nodes:
            H.add_node(n, **G.nodes[n])
        for u, v, k, d in edges:
            H.add_edge(u, v, key=k, **d)
        G = H
    elements = to_elements(G)
    html = HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        elements_json=json.dumps(elements),
    )
    out_path.write_text(html)

def write_index(per_cat_files: list[tuple[str, int, int, int, Path]], main_file: Path, out_path: Path) -> None:
    """Index page; per_cat_files items are (cat, n_records, n_nodes, n_edges, path)."""
    def row(cat, nrec, n, e, p):
        full = cictt_full(cat)
        if " — " in full:
            short, long_name = full.split(" — ", 1)
            label = f'<span class="cat-short">{short}</span><span class="cat-long">{long_name}</span>'
        else:
            label = f'<span class="cat-short">{full}</span>'
        return f'''
<a href="per_category/{p.name}" class="card">
  <div class="card-title">{label}</div>
  <div class="card-stats">
    <span class="stat-pill">{nrec} records</span>
    <span class="stat-pill">{n} nodes</span>
    <span class="stat-pill">{e} edges</span>
  </div>
</a>'''
    cards = "\n".join(row(*r) for r in per_cat_files)
    main_file_name = main_file.name
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>ACE-Graph aggregate KG</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0e1a; --bg-2: #0e1422; --panel: rgba(20, 26, 41, 0.85);
    --panel-border: #1f2a44; --fg: #e6edf3; --muted: #8b96b0; --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px 24px; min-height: 100vh;
    background: var(--bg);
    background-image:
      radial-gradient(circle at 15% 0%, rgba(56, 189, 248, 0.10), transparent 35%),
      radial-gradient(circle at 85% 100%, rgba(236, 72, 153, 0.10), transparent 35%);
    color: var(--fg);
    font-family: 'Inter', -apple-system, sans-serif;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{
    font-size: 36px; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px 0;
    background: linear-gradient(135deg, #fff 0%, #38bdf8 60%, #ec4899 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .lede {{ color: var(--muted); font-size: 14px; max-width: 700px; line-height: 1.55; margin-bottom: 32px; }}
  h2 {{ font-size: 18px; font-weight: 700; margin: 30px 0 12px 0; color: var(--fg); }}
  h2 .meta {{ color: var(--muted); font-weight: 500; font-size: 12px; margin-left: 8px; }}
  .main-card {{
    display: block; padding: 22px 24px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(236, 72, 153, 0.05));
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: var(--fg); text-decoration: none;
    transition: transform 0.15s, border-color 0.15s;
  }}
  .main-card:hover {{ transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.5); }}
  .main-card .t {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; }}
  .main-card .s {{ color: var(--muted); font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 12px; }}
  .card {{
    display: block; padding: 14px 16px; border-radius: 10px;
    background: var(--panel); border: 1px solid var(--panel-border);
    color: var(--fg); text-decoration: none;
    transition: transform 0.15s, border-color 0.15s, background 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); border-color: var(--accent); background: rgba(56,189,248,0.05); }}
  .card-title {{ font-weight: 600; font-size: 14px; margin-bottom: 8px; }}
  .cat-short {{ display: inline-block; padding: 2px 7px; border-radius: 4px; background: rgba(56,189,248,0.12); color: var(--accent); font-family: 'JetBrains Mono', monospace; font-size: 12px; margin-right: 8px; }}
  .cat-long {{ color: var(--fg); }}
  .card-stats {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .stat-pill {{
    background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px;
    font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace;
  }}
  .legend-box {{
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 10px;
    padding: 18px 22px; margin: 24px 0; font-size: 13px; line-height: 1.6;
  }}
  .legend-box h3 {{ margin: 0 0 10px 0; font-size: 14px; }}
  .legend-box code {{ background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }}
  .legend-box p {{ margin: 8px 0; color: #cbd5e1; }}
  .legend-box .muted {{ color: var(--muted); font-size: 12px; }}
  .files {{ font-size: 12px; color: var(--muted); }}
  .files li {{ padding: 2px 0; }}
  .files code {{ background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 3px; color: #cbd5e1; }}
  ul {{ list-style: none; padding: 0; margin: 8px 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>ACE-Graph aggregate KG</h1>
  <p class="lede">Layer-1 aggregation of per-narrative causal subgraphs across all NTSB / FAA AIDS records that pass v3 schema-constrained extraction. Each per-CICTT-category subgraph below is the type-level summary of how that kind of accident typically unfolds.</p>

  <div class="legend-box">
    <h3>What you are looking at</h3>
    <p>Each per-category graph is built from <b>all accidents in that category</b> (e.g. all Loss-of-Control-In-flight events). Nodes are <b>event types</b> from a fixed 42-value vocabulary; each event type's node aggregates every accident in the category that contained an event of that type.</p>
    <p>An edge <code>X — CAUSES → Y</code> with count <code>N</code> means: in <code>N</code> distinct accidents in this category the LLM extracted that triple from the narrative. It is empirical co-occurrence with directionality, <b>not</b> a verified causal claim — Stage 3 (PC algorithm + LLM priors) turns these counts into a real causal DAG.</p>
    <p class="muted">The single most-frequent edge in many subgraphs is <code>GROUND_IMPACT → INJURY_OR_FATALITY</code>, which is mechanically true but uninteresting. The interesting edges are the chains <i>upstream</i> of terminal events.</p>
  </div>

  <h2>Full aggregate</h2>
  <a class="main-card" href="{main_file_name}">
    <div class="t">All categories combined</div>
    <div class="s">Top-400 edges by count across the entire corpus, every event type rendered.</div>
  </a>

  <h2>Per-CICTT-category subgraphs <span class="meta">CICTT = ICAO/CAST Common Aviation Taxonomy Team</span></h2>
  <div class="grid">
{cards}
  </div>

  <h2>Files on disk</h2>
  <ul class="files">
    <li><code>aggregate_kg.gpickle</code> — NetworkX MultiDiGraph for Python</li>
    <li><code>aggregate_kg.graphml</code> — for Cytoscape / Gephi / yEd</li>
    <li><code>aggregate_edges.csv</code> — <code>src,edge_type,dst,count</code> for Stage-3 PC algorithm</li>
    <li><code>per_category/*.csv</code> — same format, one CSV per category</li>
    <li><code>aggregate_kg.summary.md</code> — top events / edges / per-category tables</li>
  </ul>
</div>
</body></html>"""
    out_path.write_text(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="event_extraction/out/aggregate_kg", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/aggregate_kg/html", type=Path)
    ap.add_argument("--top-n-edges", default=400, type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_category").mkdir(parents=True, exist_ok=True)

    pickle_path = args.in_dir / "aggregate_kg.gpickle"
    print(f"Loading {pickle_path}…")
    with pickle_path.open("rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    summary = {}
    summary_path = args.in_dir / "aggregate_kg.summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    rec_per_cat = summary.get("records_per_category", {})

    main_file = args.out_dir / "aggregate_kg.html"
    print(f"Rendering full aggregate -> {main_file} (top {args.top_n_edges} edges)")
    render(G, main_file,
           title="ACE-Graph — full aggregate",
           subtitle=f"Top-{args.top_n_edges} edges by count across the entire corpus",
           top_n=args.top_n_edges)

    per_cat_files: list[tuple[str, int, int, int, Path]] = []
    pc_dir = args.in_dir / "per_category"
    for csv in sorted(pc_dir.glob("*.csv")):
        cat = csv.stem
        sub = nx.MultiDiGraph()
        with csv.open() as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 4:
                    continue
                src, etype, dst, count = parts[0], parts[1], parts[2], int(parts[3])
                if not sub.has_node(src):
                    sub.add_node(src, **(G.nodes.get(src) or {"count": 0}))
                if not sub.has_node(dst):
                    sub.add_node(dst, **(G.nodes.get(dst) or {"count": 0}))
                sub.add_edge(src, dst, key=etype, type=etype, count=count)
        if sub.number_of_edges() == 0:
            continue
        out_html = args.out_dir / "per_category" / f"{cat}.html"
        n_records = rec_per_cat.get(cat, "?")
        full = cictt_full(cat)
        render(sub, out_html,
               title=f"ACE-Graph — {full}",
               subtitle=f"{n_records} accidents in this category" if isinstance(n_records, int) else cat)
        per_cat_files.append((cat, n_records if isinstance(n_records, int) else 0,
                              sub.number_of_nodes(), sub.number_of_edges(), out_html))
        print(f"  {cat:20s}  {sub.number_of_nodes():3d} nodes  {sub.number_of_edges():4d} edges  -> {out_html.name}")

    per_cat_files.sort(key=lambda r: -r[3])
    index_path = args.out_dir / "index.html"
    write_index(per_cat_files, main_file, index_path)
    print(f"\nIndex page: {index_path}")
    print(f"\nOpen:")
    print(f"  file://{index_path.resolve()}")

if __name__ == "__main__":
    main()
