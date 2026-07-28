"""Relation diagram for Failure Study 2: contributing factors, the
engine-failure mode, and the outcome layer, mirroring make_loc_kg_fig.py.
Edge statistics sit inside the factor boxes; arrow width scales with support.
Output: paper/figs/fig_ef_kg.pdf."""
import os
import numpy as np
import matplotlib.pyplot as plt

from kg_fig_style import (WIDTH_IN, FS_MAIN, INK_SUB, factor_box, hub_box,
                          outcome_box, edge, header, tier_legend)

OUT = "paper/figs/fig_ef_kg.pdf"

# (label, model tag or None, n, q, tier) -- from ef_risk.json / causation KG
FACTORS = [
    ("Fuel exhaustion / starvation", "endurance mass balance", 5051, 0.93, "phys"),
    ("Latent mechanical defect", None, 2331, 0.28, "data"),
    ("Procedure not followed", None, 1137, 0.07, "data"),
    ("Other system failure", None, 1035, 0.28, "data"),
    ("Carburetor / induction icing", "icing thermodynamics", 971, 0.67, "phys"),
    ("Fuel system anomaly", "component reliability", 954, 0.57, "surr"),
    ("Fuel contamination", "component reliability", 826, 0.75, "surr"),
    ("Inadequate maintenance", None, 649, 0.12, "data"),
]
OUTCOMES = [("Ground impact", 6343), ("Emergency landing", 630),
            ("Water impact", 426)]

fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 9.5)
ax.axis("off")

XF, W, H = 1.62, 2.95, 0.92
ys = np.linspace(8.55, 0.60, len(FACTORS))
max_n = max(n for _, _, n, _, _ in FACTORS)

XL, YL, WH, HH = 6.0, 4.55, 2.55, 1.30
hub_box(ax, XL, YL, WH, HH, "ENGINE\nFAILURE",
        r"$n$ = 16,406   $P(\mathrm{serious/fatal})$ = 0.24")
ax.text(XL, YL + HH / 2 + 0.50,
        r"$P(\mathrm{EF}\mid O)=1-\prod_i\left(1-\pi_i(O)\,q_i\right)$",
        ha="center", fontsize=FS_MAIN, color=INK_SUB)

off = np.linspace(0.72, -0.72, len(FACTORS)) * HH / 2
for (label, tag, n, q, tier), y, dy in zip(FACTORS, ys, off):
    factor_box(ax, XF, y, W, H, label, tag, n, q, tier)
    edge(ax, (XF + W / 2, y), (XL - WH / 2, YL + dy), n, max_n)

XO, WO, HO = 9.32, 1.95, 0.86
yso = [6.10, 4.55, 3.00]
max_no = max(n for _, n in OUTCOMES)
offo = np.linspace(0.5, -0.5, len(OUTCOMES)) * HH / 2
for (label, n), y, dy in zip(OUTCOMES, yso, offo):
    outcome_box(ax, XO, y, WO, HO, label, n)
    edge(ax, (XL + WH / 2, YL + dy), (XO - WO / 2, y), n, max_no)

header(ax, XF, 9.32, "Contributing factors (learned from data)")
header(ax, XL, 9.32, "Failure mode")
header(ax, XO, 9.32, "Outcome layer (data)")
tier_legend(ax)

fig.tight_layout(pad=0.25)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
