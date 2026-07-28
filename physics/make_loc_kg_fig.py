"""Relation diagram: the data-learned contributing factors, the loss-of-control
failure mode, and the outcome layer, with the occurrence-model tier of each
factor marked. Edge statistics (support n, triggering strength q) sit inside
the factor boxes; arrow width scales with support. Styling in kg_fig_style.py.
Output: paper/figs/fig_loc_kg.pdf."""
import os
import numpy as np
import matplotlib.pyplot as plt

from kg_fig_style import (WIDTH_IN, FS_MAIN, INK_SUB, factor_box, hub_box,
                          outcome_box, edge, header, tier_legend)

OUT = "paper/figs/fig_loc_kg.pdf"

# (label, model tag or None, n, q, tier) -- tier: phys | surr | data
FACTORS = [
    ("Stall", "JSBSim 6-DOF aerodynamics", 2428, 0.37, "phys"),
    ("Improper control input", None, 1430, 0.05, "data"),
    ("Engine failure", "hazard rate and icing path", 1016, 0.06, "surr"),
    ("Inappropriate decision", None, 697, 0.03, "data"),
    ("Control-surface anomaly", "component reliability", 624, 0.35, "surr"),
    ("Airframe structural failure", "Pratt gust and V--n envelope", 560, 0.21, "phys"),
    ("Spatial disorientation", None, 448, 0.48, "data"),
    ("Pilot incapacitation", None, 360, 0.27, "data"),
    ("Turbulence", "Dryden gust spectrum", 334, 0.23, "phys"),
    ("Wind shear / gust", "shear margin exceedance", 235, 0.11, "phys"),
]
OUTCOMES = [("Ground impact", 7355), ("Water impact", 457),
            ("Ground collision", 453)]

fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.3))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 10.9)
ax.set_aspect("auto")
ax.axis("off")

# --- left column: contributing factors -------------------------------------
XF, W, H = 1.62, 2.95, 0.92
ys = np.linspace(9.9, 0.62, len(FACTORS))
max_n = max(n for _, _, n, _, _ in FACTORS)

# --- center: failure-mode hub ----------------------------------------------
XL, YL, WH, HH = 6.0, 5.25, 2.55, 1.30
hub_box(ax, XL, YL, WH, HH, "LOSS OF CONTROL\nIN FLIGHT",
        r"$n$ = 9,813   $P(\mathrm{serious/fatal})$ = 0.59")
ax.text(XL, YL + HH / 2 + 0.52,
        r"$P(\mathrm{LOC}\mid O)=1-\prod_i\left(1-\pi_i(O)\,q_i\right)$",
        ha="center", fontsize=FS_MAIN, color=INK_SUB)

# factor -> hub edges: distributed anchors along the hub's left edge
off = np.linspace(0.72, -0.72, len(FACTORS)) * HH / 2
for (label, tag, n, q, tier), y, dy in zip(FACTORS, ys, off):
    factor_box(ax, XF, y, W, H, label, tag, n, q, tier)
    edge(ax, (XF + W / 2, y), (XL - WH / 2, YL + dy), n, max_n)

# --- right column: outcomes ------------------------------------------------
XO, WO, HO = 9.32, 1.95, 0.86
yso = [6.95, 5.25, 3.55]
max_no = max(n for _, n in OUTCOMES)
offo = np.linspace(0.5, -0.5, len(OUTCOMES)) * HH / 2
for (label, n), y, dy in zip(OUTCOMES, yso, offo):
    outcome_box(ax, XO, y, WO, HO, label, n)
    edge(ax, (XL + WH / 2, YL + dy), (XO - WO / 2, y), n, max_no)

# --- headers + legend ------------------------------------------------------
header(ax, XF, 10.7, "Contributing factors (learned from data)")
header(ax, XL, 10.7, "Failure mode")
header(ax, XO, 10.7, "Outcome layer (data)")
tier_legend(ax)

fig.tight_layout(pad=0.25)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
