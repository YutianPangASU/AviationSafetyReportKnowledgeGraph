"""Shared publication style for the contributing-factor relation diagrams.

Conventions follow the scientific-plotting rules the manuscript adopts:
Times-family serif, at most two text sizes per figure, the Wong
colorblind-safe palette for categorical color, no on-canvas titles, vector
PDF at double-column width (180 mm). Edge statistics live inside the boxes
they describe, never on the arrows, so no label can collide with a line.

Used by make_loc_kg_fig.py and make_ef_kg_fig.py.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

# --- typography: Times family, two sizes only ------------------------------
FS_MAIN = 8.0        # labels, legend, equation
FS_SUB = 6.8         # secondary statistics lines
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral",
                   "Times New Roman", "DejaVu Serif"],
    "font.size": FS_MAIN,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

# --- Wong palette slots ----------------------------------------------------
WONG_ORANGE = "#E69F00"
ORANGE_FILL = "#FAF0DC"      # light tint of the orange for box fills
GREY_EDGE = "#8C8C8C"
GREY_FILL = "#F2F2F2"
HUB_FILL = "#D9D9D9"
INK = "#1A1A1A"
INK_SUB = "#595959"
EDGE_COL = "#4D4D4D"

WIDTH_IN = 7.087             # 180 mm double column


def rbox(ax, x, y, w, h, fc, ec, lw, ls="-", z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.07",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def factor_box(ax, x, y, w, h, label, model_tag, n, q, tier):
    """One contributing-factor node: label, optional model tag, and the edge
    statistics (n, q) as an in-box line instead of an arrow label."""
    if tier == "phys":
        fc, ec, lw, ls = ORANGE_FILL, WONG_ORANGE, 1.2, "-"
    elif tier == "surr":
        fc, ec, lw, ls = "white", WONG_ORANGE, 1.0, (0, (4, 2))
    else:
        fc, ec, lw, ls = GREY_FILL, GREY_EDGE, 0.7, "-"
    rbox(ax, x, y, w, h, fc, ec, lw, ls)
    stats = f"$n$ = {n:,}   $q$ = {q:.2f}"
    if model_tag:
        ax.text(x, y + h * 0.27, label, ha="center", va="center",
                fontsize=FS_MAIN, color=INK, zorder=4)
        ax.text(x, y - 0.02, model_tag, ha="center", va="center",
                fontsize=FS_SUB, style="italic", color="#8A6100", zorder=4)
        ax.text(x, y - h * 0.29, stats, ha="center", va="center",
                fontsize=FS_SUB, color=INK_SUB, zorder=4)
    else:
        ax.text(x, y + h * 0.18, label, ha="center", va="center",
                fontsize=FS_MAIN, color=INK, zorder=4)
        ax.text(x, y - h * 0.22, stats, ha="center", va="center",
                fontsize=FS_SUB, color=INK_SUB, zorder=4)


def hub_box(ax, x, y, w, h, title, substats):
    rbox(ax, x, y, w, h, HUB_FILL, "black", 1.3)
    ax.text(x, y + h * 0.16, title, ha="center", va="center",
            fontsize=FS_MAIN, fontweight="bold", color=INK, zorder=4)
    ax.text(x, y - h * 0.26, substats, ha="center", va="center",
            fontsize=FS_SUB, color=INK_SUB, zorder=4)


def outcome_box(ax, x, y, w, h, label, n):
    rbox(ax, x, y, w, h, GREY_FILL, GREY_EDGE, 0.7)
    ax.text(x, y + h * 0.18, label, ha="center", va="center",
            fontsize=FS_MAIN, color=INK, zorder=4)
    ax.text(x, y - h * 0.24, f"$n$ = {n:,}", ha="center", va="center",
            fontsize=FS_SUB, color=INK_SUB, zorder=4)


def edge(ax, p0, p1, support, max_support, rad=0.0):
    lw = 0.5 + 1.6 * np.sqrt(support / max_support)
    ax.annotate("", xy=p1, xytext=p0, zorder=2, arrowprops=dict(
        arrowstyle="-|>,head_width=0.14,head_length=0.28", color=EDGE_COL,
        lw=lw, shrinkA=1.5, shrinkB=1.5,
        connectionstyle=f"arc3,rad={rad:.3f}"))


def header(ax, x, y, text):
    ax.text(x, y, text.upper(), ha="center", va="center", fontsize=FS_SUB,
            color=INK_SUB, zorder=4)


def tier_legend(ax, loc="lower right"):
    ax.legend(handles=[
        Patch(fc=ORANGE_FILL, ec=WONG_ORANGE, lw=1.2,
              label="physics occurrence model"),
        Patch(fc="white", ec=WONG_ORANGE, lw=1.0, ls=(0, (4, 2)),
              label="surrogate model"),
        Patch(fc=GREY_FILL, ec=GREY_EDGE, lw=0.7,
              label="data only (corpus rate)")],
        loc=loc, fontsize=FS_SUB, frameon=False,
        handlelength=1.6, labelspacing=0.45, borderaxespad=0.2)
