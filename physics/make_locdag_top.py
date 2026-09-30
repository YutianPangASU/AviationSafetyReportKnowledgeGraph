"""Decluttered LOC-I learned-network figure for the paper: keep only the
top-N edges by within-category record support (the Table-1 scale), rendered
with the same publication styling as viz_causation_academic. Outputs
paper/figs/dag_LOC-I_top.pdf."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "event_extraction" / "scripts"))
import viz_causation_academic as viz  # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--category", default="LOC-I")
_ap.add_argument("--top-n", type=int, default=12)
_ap.add_argument("--min-stability", type=float, default=0.5,
                 help="draw only edges the backdoor adjustment sets use")
_args = _ap.parse_args()

TOP_N = _args.top_n
CAT = _args.category
DAG_CSV = ROOT / f"event_extraction/out/aggregate_kg/per_category_dag/{CAT}.csv"
PREC_DIR = ROOT / "event_extraction/out/aggregate_kg/per_category_precedence"
OUT_DIR = ROOT / "event_extraction/out/causation_kg/figures"
PAPER_PDF = ROOT / f"paper/figs/dag_{CAT}_top.pdf"

rows = [r for r in csv.DictReader(DAG_CSV.open())
        if float(r.get("bootstrap_stability") or 0) >= _args.min_stability]
rows.sort(key=lambda r: -int(r["support_count"]))
edges = [{"src": r["src"], "dst": r["dst"],
          "support": int(r["support_count"]),
          "stability": float(r.get("bootstrap_stability") or 0)}
         for r in rows[:TOP_N]]
print(f"kept {len(edges)}/{len(rows)} edges; support range "
      f"{edges[-1]['support']}..{edges[0]['support']}")

meta = json.loads((PREC_DIR / f"{CAT}.json").read_text())
presence = np.load(PREC_DIR / f"{CAT}.npz")["presence"]
support_of = {f: int(presence[i]) for i, f in enumerate(meta["vocab"])}

# No on-canvas title: the figure caption in the manuscript carries the
# description (publication plotting rule).
# min_stability=None -> all edges solid; direction is the learned orientation.
viz.render(f"dag_{CAT}_top", "", edges, support_of, None, OUT_DIR)

gv = OUT_DIR / f"dag_{CAT}_top.gv"
subprocess.run([viz.DOT, "-Tpdf", "-o", str(PAPER_PDF), str(gv)], check=True)
print("wrote", PAPER_PDF)
