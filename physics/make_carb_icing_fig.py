"""Figure for the carburetor-icing worked factor: physics risk surface with
the real icing accidents overlaid (single panel; the counterfactual numbers
are quoted in the text). Outputs paper/figs/fig_carb_icing.pdf."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing

FS = 8.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral",
                   "Times New Roman", "DejaVu Serif"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS, "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

ACC = "physics/out/carb_icing_accidents.csv"
OUT = "paper/figs/fig_carb_icing.pdf"

fig, ax1 = plt.subplots(figsize=(4.9, 3.6))

T = np.linspace(-12, 35, 190)
Td = np.linspace(-18, 32, 190)
TT, DD = np.meshgrid(T, Td)
Z = np.full_like(TT, np.nan)
for i in range(TT.shape[0]):
    for j in range(TT.shape[1]):
        if DD[i, j] <= TT[i, j]:
            Z[i, j] = p_carb_icing(TT[i, j], DD[i, j], "descent").p_ice
pcm = ax1.pcolormesh(TT, DD, Z, cmap="Greys", vmin=0, vmax=1.0, shading="auto")
cs = ax1.contour(TT, DD, Z, levels=[0.40, 0.80], colors="k",
                 linewidths=0.7, linestyles=["--", "-"])
ax1.clabel(cs, fmt={0.40: "Moderate", 0.80: "Serious"}, fontsize=FS)

# Non-icing accidents as density contours, so the figure shows separation
# and not only concentration (review 2026-09-14, comment 11). The contours
# enclose 50, 80 and 95 percent of the 38,565 other weather-reported accidents.
try:
    from physics.ceiling_carb_icing import load_joined
    joined = load_joined()
    other = joined[joined["CARBURETOR_OR_INDUCTION_ICING"] == 0]
    Hc, xe, ye = np.histogram2d(other["temp_c"], other["dew_c"], bins=[47, 50],
                                range=[[-12, 35], [-18, 32]])
    Hc = Hc.T
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(Hc, 2.5)
    flat = np.sort(Hs.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    levels = sorted({float(flat[np.searchsorted(cum, q)]) for q in (0.80, 0.50)})
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    cs2 = ax1.contour(xc, yc, Hs, levels=levels, colors="#0072B2",
                      linewidths=1.1, linestyles=["--", "-"])
    ax1.plot([], [], color="#0072B2", lw=1.1,
             label=f"Other accidents, n = {len(other):,} (50 and 80% contours)")
except Exception as exc:  # the join needs mdb-export and the NTSB databases
    print("non-icing contours skipped:", exc)

if os.path.exists(ACC):
    acc = pd.read_csv(ACC)
    jit = np.random.default_rng(0).normal(0, 0.5, (len(acc), 2))
    ax1.scatter(acc["temp_c"] + jit[:, 0], acc["dew_c"] + jit[:, 1], s=5,
                c="#e66101", alpha=0.55, edgecolors="none",
                label=f"Carburetor icing accidents, n = {len(acc):,}")
ax1.legend(loc="lower right", framealpha=0.9)
ax1.set_xlabel(r"Ambient temperature $T$ ($^\circ$C)")
ax1.set_ylabel(r"Dewpoint $T_d$ ($^\circ$C)")
ax1.set_xlim(-12, 35)
ax1.set_ylim(-18, 32)
cb = fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.03, ticks=[0, .25, .5, .75, 1])
cb.set_label(r"$\pi_{\mathrm{ice}}$", rotation=0, labelpad=8)

fig.tight_layout(pad=0.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
