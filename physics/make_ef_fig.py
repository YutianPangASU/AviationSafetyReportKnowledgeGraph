"""Engine-failure case-study figure: (a) per-factor contribution to P(EF) by
model tier; (b) icing-weather + thin-fuel scenario and its composing
counterfactuals. Reads physics/out/ef_risk.json.
Output: paper/figs/fig_ef.pdf."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "serif", "font.size": 8.4,
                     "axes.titlesize": 8.6, "mathtext.fontset": "cm"})

D = json.load(open("physics/out/ef_risk.json"))
OUT = "paper/figs/fig_ef.pdf"
PHYS, SURR, DATA = "#e08214", "#f5c48a", "#8a8a8a"

NICE = {"FUEL_EXHAUSTION_OR_STARVATION": "Fuel exhaustion / starvation",
        "LATENT_MECHANICAL_DEFECT": "Latent mechanical defect",
        "PROCEDURE_NOT_FOLLOWED": "Procedure not followed",
        "OTHER_SYSTEM_FAILURE": "Other system failure",
        "CARBURETOR_OR_INDUCTION_ICING": "Carburetor / induction icing",
        "FUEL_SYSTEM_ANOMALY": "Fuel system anomaly",
        "FUEL_CONTAMINATION": "Fuel contamination",
        "MAINTENANCE_INADEQUATE": "Inadequate maintenance"}
COL = {"physics": PHYS, "surrogate": SURR, "data": DATA}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                               gridspec_kw={"width_ratios": [1.45, 1]})

rows = sorted(D["factor_contributions"], key=lambda r: r["contribution"])
labels = [NICE.get(r["factor"], r["factor"]) for r in rows]
vals = [r["contribution"] for r in rows]
cols = [COL[r["tier"]] for r in rows]
ax1.barh(range(len(rows)), vals, color=cols, edgecolor="k", linewidth=0.4)
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(labels, fontsize=7.6)
ax1.set_xlabel(r"contribution to $P(\mathrm{EF})$  (drop if factor removed)")
ax1.set_title("(a) Contributing-factor decomposition")
ax1.legend(handles=[Patch(fc=PHYS, ec="k", lw=.4, label="physics"),
                    Patch(fc=SURR, ec="k", lw=.4, label="surrogate"),
                    Patch(fc=DATA, ec="k", lw=.4, label="data-only")],
           loc="lower right", fontsize=7)
ax1.grid(axis="x", ls=":", lw=0.5, alpha=0.6)

scn = D["physics_scenario"]
base = D["noisy_or_P_ef_baseline"]
cf = scn["counterfactuals"]
names = ["baseline", "icing wx +\nthin fuel", "do(carb heat)",
         "do(fuel res. 2 h)", "do(both)"]
vals2 = [base, scn["P_ef"], cf["do(carb heat)"], cf["do(fuel reserve 2 h)"],
         cf["do(both)"]]
cols2 = ["#bdbdbd", "#b30000", PHYS, PHYS, PHYS]
bars = ax2.bar(range(len(vals2)), vals2, color=cols2, edgecolor="k",
               linewidth=0.5, width=0.62)
for b, v in zip(bars, vals2):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
             ha="center", va="bottom", fontsize=7.3)
ax2.set_xticks(range(len(vals2)))
ax2.set_xticklabels(names, fontsize=6.6, rotation=25, ha="right")
ax2.set_ylabel(r"$P(\mathrm{engine\ failure})$")
ax2.set_ylim(0, 0.85)
ax2.set_title("(b) Operating-point scenario and counterfactuals")
ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

fig.tight_layout(w_pad=1.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
