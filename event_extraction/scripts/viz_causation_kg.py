"""Interactive HTML viewer for the v4 causation KG.

Reads causation_edges.csv + outcome_layer.csv + the per-category do()
intervention JSONs and emits a self-contained HTML page (Cytoscape via CDN,
same as the Stage-3 DAG viewers):

  * nodes: 47 causal factors — color = family (categorical, fixed order),
    size ~ sqrt(record support), always-visible labels
  * edges: direction src->dst, width ~ sqrt(support); hover for support,
    P(dst|src), lift, direct share
  * controls: min-support slider, lift>=1 toggle, layout switch
  * click a node: side panel with risk ratio (outcome layer) + top in/out edges
  * collapsible sortable edge TABLE (accessibility fallback — three palette
    slots are below 3:1 contrast, so identity is never color-alone here)

Palette: dataviz reference categorical palette, slots 1-7 in fixed order
(pre-validated: worst adjacent CVD dE 24.2 >= 12; low-contrast slots covered
by labels + table view). Dark mode uses the documented dark-stepped variants.

Run:
    python3 event_extraction/scripts/viz_causation_kg.py
    # -> event_extraction/out/causation_kg/html/causation_kg.html
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

OUT_DIR = Path("event_extraction/out/causation_kg")
HTML_DIR = OUT_DIR / "html"

# factor family assignment (mirrors schema_v4 family grouping)
FAMILIES = {
    "Pilot acts": ["DECISION_INAPPROPRIATE", "CONTROL_INPUT_IMPROPER",
                   "PROCEDURE_NOT_FOLLOWED", "PERCEPTION_FAILURE",
                   "SPATIAL_DISORIENTATION", "PILOT_INCAPACITATION_OR_IMPAIRMENT",
                   "TRAINING_OR_CURRENCY_GAP", "CREW_COORDINATION_FAILURE"],
    "External ops": ["ATC_OR_DISPATCH_INADEQUATE", "GROUND_PERSONNEL_ERROR"],
    "Aircraft systems": ["ENGINE_FAILURE", "FUEL_SYSTEM_ANOMALY",
                         "AIRFRAME_STRUCTURAL_FAILURE", "CONTROL_SURFACE_ANOMALY",
                         "INSTRUMENT_OR_AVIONICS_FAILURE", "LANDING_GEAR_ANOMALY",
                         "AUTOMATION_ANOMALY", "INFLIGHT_FIRE_OR_SMOKE",
                         "OTHER_SYSTEM_FAILURE"],
    "Environment": ["ICING_ENCOUNTER", "TURBULENCE_ENCOUNTER",
                    "THUNDERSTORM_OR_CONVECTIVE", "LOW_VISIBILITY_OR_IMC",
                    "WIND_SHEAR_OR_GUST", "BIRD_OR_WILDLIFE_STRIKE",
                    "TERRAIN_OR_OBSTACLE_PROXIMITY", "RUNWAY_CONDITION_HAZARD"],
    "Organizational": ["MAINTENANCE_INADEQUATE", "ORG_OR_REGULATORY_INADEQUATE",
                       "DESIGN_DEFECT_LATENT"],
    "Aero states": ["STALL", "LOSS_OF_CONTROL_INFLIGHT", "LOSS_OF_CONTROL_GROUND",
                    "STRUCTURAL_OVERLOAD", "ALTITUDE_DEVIATION_UNCONTROLLED"],
    "Conditions": ["FUEL_EXHAUSTION_OR_STARVATION", "FUEL_CONTAMINATION",
                   "CARBURETOR_OR_INDUCTION_ICING", "ADVERSE_WIND_CONDITION",
                   "HIGH_DENSITY_ALTITUDE", "DARK_NIGHT_OR_LOW_LIGHT",
                   "MOUNTAINOUS_OR_RISING_TERRAIN",
                   "UNSUITABLE_TERRAIN_FOR_FORCED_LANDING",
                   "AIRCRAFT_WEIGHT_OR_BALANCE_OUT_OF_LIMITS",
                   "LATENT_MECHANICAL_DEFECT", "LOW_ALTITUDE_OPERATION",
                   "PILOT_FATIGUE_OR_PHYSIOLOGICAL_STATE"],
}
# dataviz reference categorical palette, slots 1-7 fixed order (light, dark)
PALETTE = {
    "Pilot acts":       ("#2a78d6", "#3987e5"),
    "External ops":     ("#1baf7a", "#199e70"),
    "Aircraft systems": ("#eda100", "#c98500"),
    "Environment":      ("#008300", "#008300"),
    "Organizational":   ("#4a3aa7", "#9085e9"),
    "Aero states":      ("#e34948", "#e66767"),
    "Conditions":       ("#e87ba4", "#d55181"),
}
FAMILY_OF = {f: fam for fam, fs in FAMILIES.items() for f in fs}


def main():
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    edges = list(csv.DictReader((OUT_DIR / "causation_edges.csv").open()))
    outcome = {r["factor"]: r for r in
               csv.DictReader((OUT_DIR / "outcome_layer.csv").open())}
    summary = json.loads((OUT_DIR / "causation_kg.summary.json").read_text())

    node_support = {}
    for r in edges:
        for k in ("src", "dst"):
            node_support.setdefault(r[k], 0)
    for f, r in outcome.items():
        node_support[f] = int(r["n_records"])
    for f, c in summary.get("top_factors", {}).items():
        if f in node_support and node_support[f] == 0:
            node_support[f] = int(c)

    nodes_js = [{
        "id": f, "label": f.replace("_", "\n"), "support": n,
        "family": FAMILY_OF.get(f, "Conditions"),
        "risk_ratio": float(outcome[f]["risk_ratio"]) if f in outcome else None,
        "p_serious": float(outcome[f]["p_serious_or_fatal_given_factor"]) if f in outcome else None,
    } for f, n in node_support.items() if f in FAMILY_OF]

    edges_js = [{
        "src": r["src"], "dst": r["dst"], "support": int(r["support"]),
        "p": float(r["p_dst_given_src"]), "lift": float(r["lift"]),
        "direct": float(r["direct_share"]),
    } for r in edges if r["src"] in FAMILY_OF and r["dst"] in FAMILY_OF]

    interventions = {}
    for p in sorted(glob.glob(str(OUT_DIR / "bn_do_smoke_*.json"))):
        d = json.loads(Path(p).read_text())
        interventions[d["category"]] = {
            "baseline": d["baseline_p_serious"], "n": d["n_records"],
            "rows": [r for r in d["interventions"] if r["n_present"] >= 50][:10],
        }

    data_blob = json.dumps({
        "nodes": nodes_js, "edges": edges_js,
        "palette": PALETTE, "families": list(FAMILIES),
        "baseline": summary.get("baseline_p_serious_or_fatal"),
        "n_records": summary.get("n_records"),
        "interventions": interventions,
    })

    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aviation Causation KG (v4)</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>
<style>
:root{--bg:#fff;--panel:#f6f6f4;--ink:#1a1a19;--ink2:#5f5e56;--line:#d9d8d2;--mode:0}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--panel:#242422;--ink:#fff;--ink2:#c3c2b7;--line:#3a3936;--mode:1}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:16px;margin:0}header .sub{color:var(--ink2);font-size:12px}
#controls{display:flex;gap:16px;align-items:center;padding:8px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap;font-size:12px}
#controls label{display:flex;gap:6px;align-items:center}
#wrap{display:flex;height:calc(100vh - 170px);min-height:420px}
#cy{flex:1;min-width:0}
#side{width:290px;border-left:1px solid var(--line);padding:10px 12px;overflow-y:auto;background:var(--panel);font-size:12px}
#side h2{font-size:13px;margin:4px 0 8px}#side .rr{font-size:22px;font-weight:700}
#legend{display:flex;gap:12px;flex-wrap:wrap;padding:6px 16px;border-bottom:1px solid var(--line);font-size:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:-1px}
#tbl{padding:10px 16px}#tbl table{border-collapse:collapse;font-size:12px;width:100%}
#tbl th,#tbl td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left}
#tbl th{cursor:pointer;color:var(--ink2)}#tbl td.num,#tbl th.num{text-align:right}
details{margin:8px 16px}summary{cursor:pointer;color:var(--ink2)}
select,input[type=range]{accent-color:#2a78d6}
.muted{color:var(--ink2)}
#ivx{padding:10px 16px}#ivx table{border-collapse:collapse;font-size:12px}
#ivx th,#ivx td{padding:3px 8px;border-bottom:1px solid var(--line)}#ivx td.num{text-align:right}
</style></head><body>
<header><h1>Aviation Causation KG — v4</h1>
<span class="sub" id="topline"></span></header>
<div id="legend"></div>
<div id="controls">
 <label>min edge support <input type="range" id="minsup" min="30" max="1000" step="10" value="150"><span id="minsupv">150</span></label>
 <label><input type="checkbox" id="liftonly"> lift &ge; 1 only</label>
 <label>layout <select id="layout"><option value="cose">force</option><option value="concentric">concentric (by support)</option><option value="circle">circle</option></select></label>
 <span class="muted">node size = record support &middot; edge width = co-support &middot; click a node for details</span>
</div>
<div id="wrap"><div id="cy"></div>
<div id="side"><h2>Causation KG</h2><p class="muted">Click a factor node. Hover edges for support, P(dst|src), lift, direct share.</p></div></div>
<details><summary>Edge table (sortable, full data)</summary><div id="tbl"></div></details>
<details open><summary>Per-category do(X) intervention rankings (backdoor-adjusted severity effects)</summary><div id="ivx"></div></details>
<script>
const D = __DATA__;
const dark = matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 1 : 0;
const col = f => D.palette[f][dark];
document.getElementById('topline').textContent =
  `${D.n_records.toLocaleString()} accident chains · ${D.nodes.length} causal factors · ${D.edges.length} distinct edges · baseline P(serious/fatal)=${D.baseline}`;
document.getElementById('legend').innerHTML = D.families.map(f =>
  `<span><span class="sw" style="background:${col(f)}"></span>${f}</span>`).join('');
const nodeById = Object.fromEntries(D.nodes.map(n => [n.id, n]));
const sz = s => 12 + 3.2*Math.sqrt(s/50);
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: [],
  style: [
    {selector:'node', style:{
      'background-color': e => col(e.data('family')),
      'width': e => sz(e.data('support')), 'height': e => sz(e.data('support')),
      'label':'data(label)','font-size':7,'text-wrap':'wrap','text-max-width':70,
      'color': dark ? '#fff' : '#1a1a19','text-valign':'bottom','text-margin-y':3}},
    {selector:'edge', style:{
      'curve-style':'bezier','target-arrow-shape':'triangle','arrow-scale':.7,
      'width': e => Math.max(1, 1.4*Math.sqrt(e.data('support')/150)),
      'line-color': dark ? '#4a4946' : '#c9c8c2','target-arrow-color': dark ? '#4a4946' : '#c9c8c2',
      'opacity':.85}},
    {selector:'edge.hi', style:{'line-color':'#2a78d6','target-arrow-color':'#2a78d6','opacity':1}},
    {selector:'node.dim', style:{'opacity':.25}}, {selector:'edge.dim', style:{'opacity':.08}},
  ],
});
function rebuild(){
  const ms = +document.getElementById('minsup').value;
  const lo = document.getElementById('liftonly').checked;
  document.getElementById('minsupv').textContent = ms;
  const es = D.edges.filter(e => e.support >= ms && (!lo || e.lift >= 1));
  const keep = new Set(es.flatMap(e => [e.src, e.dst]));
  cy.elements().remove();
  cy.add(D.nodes.filter(n => keep.has(n.id)).map(n => ({group:'nodes', data:n})));
  cy.add(es.map((e,i) => ({group:'edges', data:{id:'e'+i, source:e.src, target:e.dst, ...e}})));
  relayout();
}
function relayout(){
  const kind = document.getElementById('layout').value;
  const opt = kind==='cose' ? {name:'cose', animate:false, nodeRepulsion:9e5, idealEdgeLength:90}
    : kind==='concentric' ? {name:'concentric', animate:false, concentric:n=>n.data('support'), levelWidth:()=>1500}
    : {name:'circle', animate:false};
  cy.layout(opt).run();
}
cy.on('tap','node', ev => {
  const n = ev.target; const d = n.data();
  cy.elements().removeClass('dim hi');
  cy.elements().not(n.closedNeighborhood()).addClass('dim');
  n.connectedEdges().addClass('hi');
  const ins  = D.edges.filter(e=>e.dst===d.id).sort((a,b)=>b.support-a.support).slice(0,8);
  const outs = D.edges.filter(e=>e.src===d.id).sort((a,b)=>b.support-a.support).slice(0,8);
  const rr = d.risk_ratio ? `<div class="rr">RR ${(+d.risk_ratio).toFixed(2)}</div>
    <div class="muted">P(serious/fatal | factor) = ${(+d.p_serious).toFixed(3)} vs baseline ${D.baseline}</div>` : '';
  document.getElementById('side').innerHTML = `<h2>${d.id}</h2>
    <div class="muted">${d.family} · ${d.support.toLocaleString()} records</div>${rr}
    <h2>caused by</h2>${ins.map(e=>`<div>${e.src} <span class="muted">(${e.support.toLocaleString()}, lift ${e.lift.toFixed(2)})</span></div>`).join('')||'<div class="muted">—</div>'}
    <h2>causes</h2>${outs.map(e=>`<div>${e.dst} <span class="muted">(${e.support.toLocaleString()}, P ${e.p.toFixed(2)}, lift ${e.lift.toFixed(2)})</span></div>`).join('')||'<div class="muted">—</div>'}`;
});
cy.on('tap', ev => { if(ev.target===cy){ cy.elements().removeClass('dim hi'); }});
cy.on('mouseover','edge', ev => {
  const e = ev.target.data();
  ev.target.addClass('hi');
  document.getElementById('side').innerHTML = `<h2>${e.source} &rarr; ${e.target}</h2>
   <div>support <b>${e.support.toLocaleString()}</b> records</div>
   <div>P(dst | src) = <b>${e.p.toFixed(3)}</b></div><div>lift = <b>${e.lift.toFixed(2)}</b></div>
   <div>direct-strength share = <b>${(e.direct*100).toFixed(0)}%</b></div>`;
});
cy.on('mouseout','edge', ev => ev.target.removeClass('hi'));
document.getElementById('minsup').oninput = rebuild;
document.getElementById('liftonly').onchange = rebuild;
document.getElementById('layout').onchange = relayout;
// edge table
(function(){
  let rows = [...D.edges].sort((a,b)=>b.support-a.support), key='support', asc=false;
  const render = () => {
    document.getElementById('tbl').innerHTML =
    `<table><thead><tr><th data-k="src">src</th><th data-k="dst">dst</th>
     <th class="num" data-k="support">support</th><th class="num" data-k="p">P(dst|src)</th>
     <th class="num" data-k="lift">lift</th><th class="num" data-k="direct">direct</th></tr></thead><tbody>` +
    rows.map(e=>`<tr><td>${e.src}</td><td>${e.dst}</td><td class="num">${e.support.toLocaleString()}</td>
     <td class="num">${e.p.toFixed(3)}</td><td class="num">${e.lift.toFixed(2)}</td><td class="num">${(e.direct*100).toFixed(0)}%</td></tr>`).join('')
    + '</tbody></table>';
    document.querySelectorAll('#tbl th').forEach(th => th.onclick = () => {
      const k = th.dataset.k; asc = (k===key) ? !asc : false; key=k;
      rows.sort((a,b)=> (a[k]>b[k]?1:-1) * (asc?1:-1)); render();
    });
  }; render();
})();
// intervention rankings
document.getElementById('ivx').innerHTML = Object.entries(D.interventions).map(([cat,v]) =>
 `<h3>${cat} <span class="muted">n=${v.n.toLocaleString()}, baseline P(serious/fatal)=${v.baseline}</span></h3>
  <table><thead><tr><th>factor</th><th class="num">n</th><th class="num">P(Y|do=0)</th><th class="num">P(Y|do=1)</th><th class="num">effect</th><th>adjusted on</th></tr></thead><tbody>`+
  v.rows.map(r=>`<tr><td>${r.factor}</td><td class="num">${r.n_present.toLocaleString()}</td>
   <td class="num">${r.p_serious_do_absent.toFixed(3)}</td><td class="num">${r.p_serious_do_present.toFixed(3)}</td>
   <td class="num"><b>${r.effect_do_present_vs_absent>0?'+':''}${r.effect_do_present_vs_absent.toFixed(3)}</b></td>
   <td class="muted">${(r.adjustment_set||[]).join(', ')||'—'}</td></tr>`).join('')+'</tbody></table>').join('');
rebuild();
</script></body></html>"""
    html = html.replace("__DATA__", data_blob)
    out = HTML_DIR / "causation_kg.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, "
          f"{len(nodes_js)} nodes, {len(edges_js)} edges, "
          f"{len(interventions)} intervention tables)")


if __name__ == "__main__":
    main()
