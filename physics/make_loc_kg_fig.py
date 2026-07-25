"""Relation diagram: the data-learned contributing factors -> loss of control
-> outcomes, with the occurrence-model tier of each factor marked (physics /
surrogate vs data-only). Edge stats from the causation KG (Table 1 of the
paper). Output: paper/figs/fig_loc_kg.pdf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

plt.rcParams.update({"font.family": "serif", "font.size": 8.0,
                     "mathtext.fontset": "cm"})
OUT = "paper/figs/fig_loc_kg.pdf"

ORANGE_F, ORANGE_E = "#fbe3cf", "#d1710a"
GREY_F, GREY_E = "#ececec", "#8a8a8a"
DARK_F = "#c9c9c9"

# (label, model tag or None, n, q, tier) -- tier: phys | surr | data
FACTORS = [
    ("Stall", "JSBSim 6-DOF aero", 2428, 0.37, "phys"),
    ("Improper control input", None, 1430, 0.05, "data"),
    ("Engine failure", "hazard + icing path", 1016, 0.06, "surr"),
    ("Inappropriate decision", None, 697, 0.03, "data"),
    ("Control-surface anomaly", "reliability", 624, 0.35, "surr"),
    ("Airframe structural failure", "Pratt gust / V--n", 560, 0.21, "phys"),
    ("Spatial disorientation", None, 448, 0.48, "data"),
    ("Pilot incapacitation", None, 360, 0.27, "data"),
    ("Turbulence", "Dryden gust", 334, 0.23, "phys"),
    ("Wind shear / gust", "shear margin", 235, 0.11, "phys"),
]
OUTCOMES = [("Ground impact", 7355), ("Water impact", 457),
            ("Ground collision", 453)]

fig, ax = plt.subplots(figsize=(7.3, 5.6))
ax.set_xlim(0, 10.6)
ax.set_ylim(0.0, 10.6)
ax.axis("off")

def box(x, y, w, h, label, sub, fc, ec, lw, fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.1",
                 fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x, y + 0.14, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)
        ax.text(x, y - 0.21, sub, ha="center", va="center", fontsize=6.0,
                style="italic", color="#a85907", zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

# left: factors
XF, W, H = 1.55, 2.75, 0.82
ys = np.linspace(9.55, 0.85, len(FACTORS))
anchors = {}
for (label, sub, n, q, tier), y in zip(FACTORS, ys):
    fc, ec = (ORANGE_F, ORANGE_E) if tier in ("phys", "surr") else (GREY_F, GREY_E)
    lw = 1.1 if tier == "phys" else 0.7
    ls_box = "--" if tier == "surr" else "-"
    p = FancyBboxPatch((XF - W / 2, y - H / 2), W, H,
                       boxstyle="round,pad=0.03,rounding_size=0.1",
                       fc=fc, ec=ec, lw=lw, ls=ls_box, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(XF, y + 0.15, label, ha="center", va="center", fontsize=7.4, zorder=4)
        ax.text(XF, y - 0.2, sub, ha="center", va="center", fontsize=5.9,
                style="italic", color="#a85907", zorder=4)
    else:
        ax.text(XF, y, label, ha="center", va="center", fontsize=7.4, zorder=4)
    anchors[label] = (XF + W / 2, y)

# center: LOC node
XL, YL = 6.05, 5.2
box(XL, YL, 2.5, 1.25, "LOSS OF CONTROL\nIN-FLIGHT", None, DARK_F, "black",
    1.6, fs=8.6, bold=True)
ax.text(XL, YL - 1.0, r"$n{=}9{,}813$;  $P(\mathrm{serious/fatal}){=}0.59$",
        ha="center", fontsize=6.6, color="#444444")
ax.text(XL, YL + 1.05,
        r"$P(\mathrm{LOC}\mid O)=1-\prod_i\left(1-\pi_i(O)\,q_i\right)$",
        ha="center", fontsize=7.6, color="#222222")

# right: outcomes
XO = 9.35
yso = [6.9, 5.2, 3.5]
for (label, n), y in zip(OUTCOMES, yso):
    box(XO, y, 1.75, 0.8, label, None, "#e2e2e2", "#999999", 0.6, fs=7.2)
    ax.annotate("", xy=(XO - 1.75 / 2, y), xytext=(XL + 2.5 / 2, YL),
                zorder=2, arrowprops=dict(
                    arrowstyle="-|>", color="#8a8a8a",
                    lw=0.5 + 0.5 * np.log10(n) / 2, shrinkA=2, shrinkB=2))
    ax.text((XO - 0.9 + XL + 1.25) / 2, (y + YL) / 2 + 0.14, f"n={n}",
            ha="center", fontsize=5.9, color="#666666")

# factor -> LOC edges
for label, sub, n, q, tier in FACTORS:
    x0, y0 = anchors[label]
    lw = 0.4 + 0.62 * (np.log10(n) - 2.0)
    col = "black" if n > 2000 else "#7a7a7a"
    ax.annotate("", xy=(XL - 2.5 / 2, YL), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=max(lw, 0.5),
                                shrinkA=2, shrinkB=2))
    t = 0.42
    mx, my = x0 + t * (XL - 1.25 - x0), y0 + t * (YL - y0)
    ax.text(mx, my + 0.13, f"$n$={n}, $q$={q:.2f}", fontsize=5.7,
            ha="center", color=col,
            fontweight="bold" if n > 2000 else "normal")

# column captions + legend
for x, t in [(XF, "contributing factors (learned from data)"),
             (XL, "failure mode"), (XO, "outcome layer (data)")]:
    ax.text(x, 10.35, t, ha="center", fontsize=7.2, style="italic",
            color="#555555")
ax.legend(handles=[
    Patch(fc=ORANGE_F, ec=ORANGE_E, lw=1.1, label="physics occurrence model"),
    Patch(fc=ORANGE_F, ec=ORANGE_E, lw=0.9, ls="--", label="surrogate model"),
    Patch(fc=GREY_F, ec=GREY_E, lw=0.7, label="data-only (corpus rate)")],
    loc="lower right", fontsize=6.6, framealpha=0.95)

fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
