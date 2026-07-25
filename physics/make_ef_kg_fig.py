"""Relation diagram for Case Study 2: contributing factors -> ENGINE FAILURE
-> outcomes, with model tiers marked (mirror of make_loc_kg_fig.py).
Output: paper/figs/fig_ef_kg.pdf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

plt.rcParams.update({"font.family": "serif", "font.size": 8.0,
                     "mathtext.fontset": "cm"})
OUT = "paper/figs/fig_ef_kg.pdf"

ORANGE_F, ORANGE_E = "#fbe3cf", "#d1710a"
GREY_F, GREY_E = "#ececec", "#8a8a8a"
DARK_F = "#c9c9c9"

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

fig, ax = plt.subplots(figsize=(7.3, 4.9))
ax.set_xlim(0, 10.6)
ax.set_ylim(0.0, 9.4)
ax.axis("off")

def box(x, y, w, h, label, fc, ec, lw, fs=7.6, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.1",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)

XF, W, H = 1.62, 2.9, 0.86
ys = np.linspace(8.35, 0.85, len(FACTORS))
anchors = {}
for (label, sub, n, q, tier), y in zip(FACTORS, ys):
    fc, ec = (ORANGE_F, ORANGE_E) if tier in ("phys", "surr") else (GREY_F, GREY_E)
    lw = 1.1 if tier == "phys" else 0.7
    ls_box = "--" if tier == "surr" else "-"
    ax.add_patch(FancyBboxPatch((XF - W / 2, y - H / 2), W, H,
                 boxstyle="round,pad=0.03,rounding_size=0.1",
                 fc=fc, ec=ec, lw=lw, ls=ls_box, zorder=3))
    if sub:
        ax.text(XF, y + 0.16, label, ha="center", va="center", fontsize=7.3, zorder=4)
        ax.text(XF, y - 0.21, sub, ha="center", va="center", fontsize=5.9,
                style="italic", color="#a85907", zorder=4)
    else:
        ax.text(XF, y, label, ha="center", va="center", fontsize=7.3, zorder=4)
    anchors[label] = (XF + W / 2, y)

XL, YL = 6.05, 4.6
box(XL, YL, 2.35, 1.2, "ENGINE\nFAILURE", DARK_F, "black", 1.6, fs=8.6, bold=True)
ax.text(XL, YL - 0.98, r"$n{=}16{,}406$;  $P(\mathrm{serious/fatal}){=}0.24$",
        ha="center", fontsize=6.6, color="#444444")
ax.text(XL, YL + 1.0,
        r"$P(\mathrm{EF}\mid O)=1-\prod_i\left(1-\pi_i(O)\,q_i\right)$",
        ha="center", fontsize=7.6, color="#222222")

XO = 9.35
yso = [6.2, 4.6, 3.0]
for (label, n), y in zip(OUTCOMES, yso):
    box(XO, y, 1.8, 0.8, label, "#e2e2e2", "#999999", 0.6, fs=7.2)
    ax.annotate("", xy=(XO - 0.9, y), xytext=(XL + 1.175, YL), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="#8a8a8a",
                                lw=0.5 + 0.5 * np.log10(n) / 2,
                                shrinkA=2, shrinkB=2))
    ax.text((XO - 0.9 + XL + 1.175) / 2, (y + YL) / 2 + 0.14, f"n={n}",
            ha="center", fontsize=5.9, color="#666666")

for label, sub, n, q, tier in FACTORS:
    x0, y0 = anchors[label]
    lw = 0.4 + 0.62 * (np.log10(n) - 2.0)
    col = "black" if n > 3000 else "#7a7a7a"
    ax.annotate("", xy=(XL - 1.175, YL), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=max(lw, 0.5),
                                shrinkA=2, shrinkB=2))
    t = 0.42
    mx, my = x0 + t * (XL - 1.175 - x0), y0 + t * (YL - y0)
    ax.text(mx, my + 0.13, f"$n$={n}, $q$={q:.2f}", fontsize=5.7,
            ha="center", color=col,
            fontweight="bold" if n > 3000 else "normal")

for x, t in [(XF, "contributing factors (learned from data)"),
             (XL, "failure mode"), (XO, "outcome layer (data)")]:
    ax.text(x, 9.15, t, ha="center", fontsize=7.2, style="italic",
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
