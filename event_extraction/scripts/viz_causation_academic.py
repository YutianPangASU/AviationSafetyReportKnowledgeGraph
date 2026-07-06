"""Academic (publication-style) causal-DAG figures via Graphviz.

Renders layered top-down causal diagrams in the style of safety-science
papers: factors as rounded serif boxes arranged in causal tiers
(organizational -> latent/ambient conditions -> environmental encounters ->
crew & external actions -> aircraft system failures -> aerodynamic states),
thin directed edges weighted by record support, line style by bootstrap
stability (solid >= 0.7, dashed >= 0.5), grayscale-safe.

Two modes:
  * per-category Stage-3 DAGs (genuinely acyclic; the causal "tree" the
    counterfactual analysis runs on):
        python3 viz_causation_academic.py --category LOC-I CFIT ...
  * aggregate causation KG (filtered to lift >= 1 and a support floor):
        python3 viz_causation_academic.py --aggregate --min-support 300

Outputs SVG + PNG + .gv source under out/causation_kg/figures/.
Requires the graphviz `dot` binary (conda env `ntsb`).
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

DOT = "/home/yp6443/miniconda3/envs/ntsb/bin/dot"
OUT_DIR = Path("event_extraction/out/causation_kg/figures")
DAG_DIR = Path("event_extraction/out/aggregate_kg/per_category_dag")
EDGES_CSV = Path("event_extraction/out/causation_kg/causation_edges.csv")
OUTCOME_CSV = Path("event_extraction/out/causation_kg/outcome_layer.csv")

TIERS = [
    ("Organizational", ["MAINTENANCE_INADEQUATE", "ORG_OR_REGULATORY_INADEQUATE",
                        "DESIGN_DEFECT_LATENT"]),
    ("Latent / ambient conditions",
     ["LATENT_MECHANICAL_DEFECT", "FUEL_CONTAMINATION",
      "FUEL_EXHAUSTION_OR_STARVATION", "CARBURETOR_OR_INDUCTION_ICING",
      "AIRCRAFT_WEIGHT_OR_BALANCE_OUT_OF_LIMITS", "HIGH_DENSITY_ALTITUDE",
      "ADVERSE_WIND_CONDITION", "DARK_NIGHT_OR_LOW_LIGHT",
      "MOUNTAINOUS_OR_RISING_TERRAIN", "UNSUITABLE_TERRAIN_FOR_FORCED_LANDING",
      "LOW_ALTITUDE_OPERATION", "PILOT_FATIGUE_OR_PHYSIOLOGICAL_STATE"]),
    ("Environmental encounters",
     ["ICING_ENCOUNTER", "TURBULENCE_ENCOUNTER", "THUNDERSTORM_OR_CONVECTIVE",
      "LOW_VISIBILITY_OR_IMC", "WIND_SHEAR_OR_GUST", "BIRD_OR_WILDLIFE_STRIKE",
      "TERRAIN_OR_OBSTACLE_PROXIMITY", "RUNWAY_CONDITION_HAZARD"]),
    ("Crew & external actions",
     ["DECISION_INAPPROPRIATE", "CONTROL_INPUT_IMPROPER", "PROCEDURE_NOT_FOLLOWED",
      "PERCEPTION_FAILURE", "SPATIAL_DISORIENTATION",
      "PILOT_INCAPACITATION_OR_IMPAIRMENT", "TRAINING_OR_CURRENCY_GAP",
      "CREW_COORDINATION_FAILURE", "ATC_OR_DISPATCH_INADEQUATE",
      "GROUND_PERSONNEL_ERROR"]),
    ("Aircraft system failures",
     ["ENGINE_FAILURE", "FUEL_SYSTEM_ANOMALY", "AIRFRAME_STRUCTURAL_FAILURE",
      "CONTROL_SURFACE_ANOMALY", "INSTRUMENT_OR_AVIONICS_FAILURE",
      "LANDING_GEAR_ANOMALY", "AUTOMATION_ANOMALY", "INFLIGHT_FIRE_OR_SMOKE",
      "OTHER_SYSTEM_FAILURE"]),
    ("Aerodynamic states",
     ["STALL", "LOSS_OF_CONTROL_INFLIGHT", "LOSS_OF_CONTROL_GROUND",
      "STRUCTURAL_OVERLOAD", "ALTITUDE_DEVIATION_UNCONTROLLED"]),
]
TIER_OF = {f: i for i, (_, fs) in enumerate(TIERS) for f in fs}

PRETTY = {
    "LOW_VISIBILITY_OR_IMC": "Low visibility / IMC",
    "PILOT_INCAPACITATION_OR_IMPAIRMENT": "Pilot incapacitation\n/ impairment",
    "AIRCRAFT_WEIGHT_OR_BALANCE_OUT_OF_LIMITS": "Weight / balance\nout of limits",
    "UNSUITABLE_TERRAIN_FOR_FORCED_LANDING": "Unsuitable terrain for\nforced landing",
    "PILOT_FATIGUE_OR_PHYSIOLOGICAL_STATE": "Pilot fatigue /\nphysiological state",
    "CARBURETOR_OR_INDUCTION_ICING": "Carburetor /\ninduction icing",
    "ORG_OR_REGULATORY_INADEQUATE": "Organizational /\nregulatory inadequate",
    "ATC_OR_DISPATCH_INADEQUATE": "ATC / dispatch\ninadequate",
    "LOSS_OF_CONTROL_INFLIGHT": "Loss of control\nin flight",
    "LOSS_OF_CONTROL_GROUND": "Loss of control\non ground",
    "ALTITUDE_DEVIATION_UNCONTROLLED": "Uncontrolled\naltitude deviation",
    "TERRAIN_OR_OBSTACLE_PROXIMITY": "Terrain / obstacle\nproximity",
    "FUEL_EXHAUSTION_OR_STARVATION": "Fuel exhaustion\n/ starvation",
    "DARK_NIGHT_OR_LOW_LIGHT": "Dark night /\nlow light",
    "MOUNTAINOUS_OR_RISING_TERRAIN": "Mountainous /\nrising terrain",
    "THUNDERSTORM_OR_CONVECTIVE": "Thunderstorm /\nconvective wx",
    "BIRD_OR_WILDLIFE_STRIKE": "Bird / wildlife strike",
    "INSTRUMENT_OR_AVIONICS_FAILURE": "Instrument /\navionics failure",
}


def pretty(f: str) -> str:
    if f in PRETTY:
        return PRETTY[f]
    words = f.replace("_", " ").lower()
    words = words[0].upper() + words[1:]
    # wrap at ~16 chars
    out, line = [], ""
    for w in words.split():
        if len(line) + len(w) > 15 and line:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    out.append(line)
    return "\n".join(out)


def render(name: str, title: str, edges: list[dict], support_of: dict,
           min_stability: float | None, out_dir: Path):
    """edges: [{src,dst,support,stability(optional),extra_label(optional)}]"""
    used = {e["src"] for e in edges} | {e["dst"] for e in edges}
    max_sup = max((e["support"] for e in edges), default=1)

    g = ['digraph G {',
         '  rankdir=TB; splines=spline; nodesep=0.28; ranksep=0.85;',
         '  bgcolor="white";',
         f'  labelloc="t"; label=<<font face="Times-Roman" point-size="15">{title}</font>>;',
         '  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7",'
         ' color="#333333", penwidth=0.8, fontname="Times-Roman", fontsize=10,'
         ' margin="0.10,0.05"];',
         '  edge [color="#4d4d4d", arrowsize=0.55, fontname="Times-Roman",'
         ' fontsize=8, fontcolor="#4d4d4d"];']

    # tier rank groups + left-margin tier captions
    prev_anchor = None
    for i, (tier_name, factors) in enumerate(TIERS):
        members = [f for f in factors if f in used]
        if not members:
            continue
        anchor = f"tier{i}"
        g.append(f'  {anchor} [shape=plaintext, style="", fillcolor=none,'
                 f' fontname="Times-Italic", fontsize=10, fontcolor="#808080",'
                 f' label="{tier_name}"];')
        row = " ".join(f'"{f}"' for f in members)
        g.append(f'  {{ rank=same; {anchor}; {row} }}')
        if prev_anchor:
            g.append(f'  {prev_anchor} -> {anchor} [style=invis];')
        prev_anchor = anchor

    for f in sorted(used):
        n = support_of.get(f)
        sub = f'<br/><font point-size="8" color="#808080">n = {n:,}</font>' if n else ""
        lbl = pretty(f).replace("\n", "<br/>")
        g.append(f'  "{f}" [label=<{lbl}{sub}>];')

    for e in edges:
        pw = 0.5 + 1.8 * math.sqrt(e["support"] / max_sup)
        style = "solid"
        if min_stability is not None:
            stab = e.get("stability", 0.0)
            style = "solid" if stab >= 0.7 else "dashed"
        tip = f'{e["src"]} -> {e["dst"]}: support {e["support"]:,}'
        if e.get("stability") is not None:
            tip += f', stability {e["stability"]:.2f}'
        if e.get("lift") is not None:
            tip += f', lift {e["lift"]:.2f}'
        g.append(f'  "{e["src"]}" -> "{e["dst"]}" [penwidth={pw:.2f},'
                 f' style={style}, tooltip="{tip}"];')
    g.append("}")

    out_dir.mkdir(parents=True, exist_ok=True)
    gv = out_dir / f"{name}.gv"
    gv.write_text("\n".join(g))
    for fmt in ("svg", "png"):
        subprocess.run([DOT, f"-T{fmt}",
                        *( ["-Gdpi=180"] if fmt == "png" else []),
                        "-o", str(out_dir / f"{name}.{fmt}"), str(gv)],
                       check=True)
    print(f"  {name}: {len(used)} nodes, {len(edges)} edges -> "
          f"{out_dir / (name + '.svg')}")


def per_category(cat: str, min_stability: float, min_support: int):
    path = DAG_DIR / f"{cat}.csv"
    rows = list(csv.DictReader(path.open()))
    edges, support_of = [], {}
    for r in rows:
        stab = float(r.get("bootstrap_stability") or 0)
        sup = int(r["support_count"])
        if stab < min_stability or sup < min_support:
            continue
        edges.append({"src": r["src"], "dst": r["dst"], "support": sup,
                      "stability": stab})
    # node n = within-category presence counts (Stage-3 precedence matrices)
    import json as _json
    import numpy as _np
    stem = cat.replace(":", "_").replace("/", "_")
    prec_dir = Path("event_extraction/out/aggregate_kg/per_category_precedence")
    meta = _json.loads((prec_dir / f"{stem}.json").read_text())
    presence = _np.load(prec_dir / f"{stem}.npz")["presence"]
    support_of = {f: int(presence[i]) for i, f in enumerate(meta["vocab"])}
    title = (f"Causal structure — {cat} (PC + LLM priors, reoriented; "
             f"solid: stability &#8805; 0.7, dashed: &#8805; {min_stability}; "
             f"line weight &#8733; record support)")
    render(f"dag_{cat.replace(':', '_')}", title, edges, support_of,
           min_stability, OUT_DIR)


def aggregate(min_support: int, min_lift: float):
    rows = list(csv.DictReader(EDGES_CSV.open()))
    support_of = {r["factor"]: int(r["n_records"])
                  for r in csv.DictReader(OUTCOME_CSV.open())}
    edges = []
    for r in rows:
        sup, lift = int(r["support"]), float(r["lift"])
        if sup < min_support or lift < min_lift:
            continue
        if r["src"] not in TIER_OF or r["dst"] not in TIER_OF:
            continue
        edges.append({"src": r["src"], "dst": r["dst"], "support": sup,
                      "lift": lift})
    title = (f"Aggregate causation KG, 55,940 NTSB/FAA accident chains "
             f"(edges: support &#8805; {min_support}, lift &#8805; {min_lift}; "
             f"line weight &#8733; support)")
    render("causation_kg_aggregate", title, edges, support_of, None, OUT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", nargs="*", default=[])
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--min-stability", type=float, default=0.5)
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--agg-min-support", type=int, default=300)
    ap.add_argument("--agg-min-lift", type=float, default=1.0)
    args = ap.parse_args()
    for cat in args.category:
        per_category(cat, args.min_stability, args.min_support)
    if args.aggregate:
        aggregate(args.agg_min_support, args.agg_min_lift)


if __name__ == "__main__":
    main()
