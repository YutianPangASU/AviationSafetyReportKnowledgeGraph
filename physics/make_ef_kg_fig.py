"""Relation diagram for Failure Study II: contributing factors, the
engine-failure mode, and the outcome layer, mirroring make_loc_kg_fig.py.
Factors and fitted strengths are read from physics/out/ef_risk.json; arrow
width scales with the contribution Delta_i.
Output: paper/figs/fig_ef_kg.pdf."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

from kg_fig_style import (WIDTH_IN, FS_MAIN, INK_SUB, factor_box, hub_box,
                          outcome_box, edge, header, tier_legend)

OUT = "paper/figs/fig_ef_kg.pdf"
D = json.load(open("physics/out/ef_risk.json"))

NICE = {"FUEL_EXHAUSTION_OR_STARVATION": "Fuel exhaustion / starvation",
        "LATENT_MECHANICAL_DEFECT": "Latent mechanical defect",
        "PROCEDURE_NOT_FOLLOWED": "Procedure not followed",
        "OTHER_SYSTEM_FAILURE": "Other system failure",
        "CARBURETOR_OR_INDUCTION_ICING": "Carburetor / induction icing",
        "FUEL_SYSTEM_ANOMALY": "Fuel system anomaly",
        "FUEL_CONTAMINATION": "Fuel contamination",
        "MAINTENANCE_INADEQUATE": "Inadequate maintenance",
        "AIRFRAME_STRUCTURAL_FAILURE": "Airframe structural failure",
        "CONTROL_INPUT_IMPROPER": "Improper control input",
        "DECISION_INAPPROPRIATE": "Inappropriate decision"}
TAG = {"FUEL_EXHAUSTION_OR_STARVATION": "endurance mass balance",
       "CARBURETOR_OR_INDUCTION_ICING": "icing thermodynamics",
       "FUEL_SYSTEM_ANOMALY": "component reliability",
       "FUEL_CONTAMINATION": "component reliability",
       "AIRFRAME_STRUCTURAL_FAILURE": "Pratt gust and V-n envelope"}
TIER = {"physics": "phys", "surrogate": "surr", "data": "data"}

FACTORS = [(NICE.get(r["factor"], r["factor"]), TAG.get(r["factor"]), r["support"],
            r["p"], TIER[r["tier"]], r["contribution"]) for r in D["factors"]]
OUTCOMES = [("Ground impact", 6343), ("Emergency landing", 630),
            ("Water impact", 426)]

k = len(FACTORS)
top = 1.02 * k + 0.9
fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.49 * (top + 0.3)))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, top)
ax.axis("off")

XF, W, H = 1.62, 2.95, 0.92
ys = np.linspace(top - 1.0, 0.62, k)
max_w = max(c for *_, c in FACTORS)

XL, WH, HH = 6.0, 2.55, 1.30
YL = float(ys.mean())
hub_box(ax, XL, YL, WH, HH, "ENGINE\nFAILURE",
        r"$n$ = 16,406   $P(\mathrm{serious/fatal})$ = 0.24")
ax.text(XL, YL + HH / 2 + 0.50,
        r"$P(\mathrm{EF}\mid O)=1-(1-\ell)\prod_i\left(1-\pi_i(O)\,p_i\right)$",
        ha="center", fontsize=FS_MAIN, color=INK_SUB)

off = np.linspace(0.72, -0.72, k) * HH / 2
for (label, tag, n, p, tier, contrib), y, dy in zip(FACTORS, ys, off):
    factor_box(ax, XF, y, W, H, label, tag, n, p, tier, stat_name="p")
    edge(ax, (XF + W / 2, y), (XL - WH / 2, YL + dy), contrib, max_w)

XO, WO, HO = 9.32, 1.95, 0.86
yso = [YL + 1.7, YL, YL - 1.7]
max_no = max(n for _, n in OUTCOMES)
offo = np.linspace(0.5, -0.5, len(OUTCOMES)) * HH / 2
for (label, n), y, dy in zip(OUTCOMES, yso, offo):
    outcome_box(ax, XO, y, WO, HO, label, n)
    edge(ax, (XL + WH / 2, YL + dy), (XO - WO / 2, y), n, max_no)

header(ax, XF, top - 0.2, "Contributing factors (learned from data)")
header(ax, XL, top - 0.2, "Failure mode")
header(ax, XO, top - 0.2, "Outcome layer (data)")
tier_legend(ax)

fig.tight_layout(pad=0.25)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
