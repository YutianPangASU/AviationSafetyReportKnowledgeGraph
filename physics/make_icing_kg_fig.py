"""Data-learned causal neighborhood of the carburetor-icing node, with the
physics occurrence model attached beneath it (the hybrid, drawn).
Edges/supports are read from the built causation KG. Output:
paper/figs/fig_icing_kg.pdf.  (graphviz 'dot' is absent, so we lay it out
directly in matplotlib.)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8.2,
                     "mathtext.fontset": "cm"})

OUT = "paper/figs/fig_icing_kg.pdf"

# --- nodes: (key, x, y, label, kind) ; kind styles the box --------------------
GRY, DARK, ACC, TERM = "#e8e8e8", "#c9c9c9", "#fbe3cf", "#dfdfdf"
nodes = {
    # antecedents (leading factors), left column
    "ICE_ENC":  (0.9, 8.4, "Icing / moisture\nconditions", "ante"),
    "PROC":     (0.9, 6.7, "Procedure not\nfollowed", "ante"),
    "DEC":      (0.9, 5.0, "Inappropriate\ndecision", "ante"),
    "IMC":      (0.9, 3.3, "Low visibility /\nIMC", "ante"),
    "MAINT":    (0.9, 1.6, "Inadequate\nmaintenance", "ante"),
    # the failure node
    "CARB":     (4.2, 5.0, "CARBURETOR /\nINDUCTION ICING", "carb"),
    # physics sub-model beneath the failure node
    "PHYS":     (4.2, 1.7, r"Physics model  $\pi_{\mathrm{ice}}(T,T_d,\varpi)$"
                           "\ndrivers: temperature, dewpoint, power", "phys"),
    # consequences
    "ENG":      (7.2, 6.6, "Engine failure /\npower loss", "cons"),
    "FUELSYS":  (7.2, 4.7, "Fuel system\nanomaly", "cons"),
    "STALL":    (7.2, 3.1, "Stall", "cons"),
    # outcomes
    "GI":       (9.7, 7.3, "Ground\nimpact", "out"),
    "EL":       (9.7, 5.6, "Emergency\nlanding", "out"),
    "WI":       (9.7, 4.0, "Water\nimpact", "out"),
}
STYLE = {"ante": (GRY, "#888888", 0.6), "carb": (DARK, "black", 1.6),
         "cons": (GRY, "#666666", 0.9), "out": (TERM, "#999999", 0.6),
         "phys": (ACC, "#d1710a", 1.2)}
BOX = {"ante": (1.55, 0.95), "carb": (1.95, 1.15), "cons": (1.6, 0.95),
       "out": (1.25, 0.95), "phys": (2.7, 1.05)}

# --- edges: (src, dst, support, label_or_None, style) ------------------------
edges = [
    ("ICE_ENC", "CARB", 22, "n=22, lift 1.9", "ante"),
    ("PROC", "CARB", 44, "n=44", "ante"),
    ("DEC", "CARB", 21, "n=21", "ante"),
    ("IMC", "CARB", 15, "n=15", "ante"),
    ("MAINT", "CARB", 11, "n=11", "ante"),
    ("CARB", "ENG", 971, "n=971 | P=0.67 | lift 2.27", "strong"),
    ("CARB", "FUELSYS", 30, "n=30", "weak"),
    ("CARB", "STALL", 22, "n=22", "weak"),
    ("ENG", "GI", 6343, "n=6343", "strong"),
    ("ENG", "EL", 630, "n=630", "weak"),
    ("ENG", "WI", 426, "n=426", "weak"),
    ("PHYS", "CARB", None, None, "phys"),
]

fig, ax = plt.subplots(figsize=(7.3, 4.05))
ax.set_xlim(0, 11.2)
ax.set_ylim(0.6, 9.5)
ax.axis("off")

boxdim = {}
for k, (x, y, label, kind) in nodes.items():
    w, h = BOX[kind]
    fc, ec, lw = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.03,rounding_size=0.12",
                 fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, label, ha="center", va="center", zorder=4,
            fontsize=8.0 if kind != "carb" else 8.4,
            fontweight="bold" if kind in ("carb", "phys") else "normal")
    boxdim[k] = (x, y, w, h)


def anchor(k, side):
    x, y, w, h = boxdim[k]
    return {"r": (x + w / 2, y), "l": (x - w / 2, y),
            "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]


for s, d, supp, lab, st in edges:
    if st == "phys":
        p0, p1 = anchor(s, "t"), anchor(d, "b")
        ax.annotate("", xy=p1, xytext=p0, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color="#d1710a",
                                    lw=1.6, ls="--", shrinkA=1, shrinkB=1))
        ax.text((p0[0] + p1[0]) / 2 + 0.15, (p0[1] + p1[1]) / 2,
                "supplies\noccurrence P", color="#a85907", fontsize=6.8,
                style="italic", ha="left", va="center")
        continue
    p0, p1 = anchor(s, "r"), anchor(d, "l")
    lw = 0.5 + 0.55 * np.log10(supp)
    col = "black" if st == "strong" else "#8a8a8a"
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                shrinkA=2, shrinkB=2))
    if lab:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + (0.22 if st == "strong" else 0.16), lab,
                ha="center", va="bottom",
                fontsize=6.8 if st == "strong" else 6.0,
                color=col, fontweight="bold" if st == "strong" else "normal")

# column captions
for x, t in [(0.9, "leading factors (data)"), (4.2, "failure mode"),
             (7.2, "consequences (data)"), (9.7, "outcomes")]:
    ax.text(x, 9.25, t, ha="center", va="center", fontsize=7.3,
            style="italic", color="#555555")

fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
