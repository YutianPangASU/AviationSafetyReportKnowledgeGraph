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

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "mathtext.fontset": "cm",
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
pcm = ax1.pcolormesh(TT, DD, Z, cmap="Greys", vmin=0, vmax=1.15, shading="auto")
cs = ax1.contour(TT, DD, Z, levels=[0.40, 0.80], colors="k",
                 linewidths=0.7, linestyles=["--", "-"])
ax1.clabel(cs, fmt={0.40: "moderate", 0.80: "serious"}, fontsize=6.5)

if os.path.exists(ACC):
    acc = pd.read_csv(ACC)
    jit = np.random.default_rng(0).normal(0, 0.5, (len(acc), 2))
    ax1.scatter(acc["temp_c"] + jit[:, 0], acc["dew_c"] + jit[:, 1], s=5,
                c="#e66101", alpha=0.55, edgecolors="none",
                label=f"carb-icing accidents (n={len(acc)})")
    ax1.legend(loc="lower right", framealpha=0.9)
ax1.set_xlabel(r"ambient temperature $T$ ($^\circ$C)")
ax1.set_ylabel(r"dewpoint $T_d$ ($^\circ$C)")
ax1.set_title(r"$\pi_{\mathrm{ice}}(T,T_d)$ at descent power, with accidents")
ax1.set_xlim(-12, 35)
ax1.set_ylim(-18, 32)
cb = fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.03, ticks=[0, .25, .5, .75, 1])
cb.set_label(r"$\pi_{\mathrm{ice}}$", rotation=0, labelpad=8)

fig.tight_layout(pad=0.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
