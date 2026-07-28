"""Loss-of-control failure-mode figure: (a) per-factor contribution to P(LOC),
colored by model tier (physics vs data); (b) a physics operating-point scenario
and its counterfactuals. Reads physics/out/loc_risk.json.
Output: paper/figs/fig_loc.pdf."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 8.4,
                     "axes.titlesize": 8.6, "mathtext.fontset": "cm"})

D = json.load(open("physics/out/loc_risk.json"))
U = json.load(open("physics/out/risk_uncertainty.json"))["loss_of_control"]
OUT = "paper/figs/fig_loc.pdf"
PHYS, SURR, DATA = "#e08214", "#f6c58a", "#8a8a8a"
COL = {"physics": PHYS, "surrogate": SURR, "data": DATA}

NICE = {"STALL": "Stall", "CONTROL_INPUT_IMPROPER": "Improper control input",
        "ENGINE_FAILURE": "Engine failure", "DECISION_INAPPROPRIATE": "Inappropriate decision",
        "CONTROL_SURFACE_ANOMALY": "Control-surface anomaly",
        "AIRFRAME_STRUCTURAL_FAILURE": "Airframe structural failure",
        "SPATIAL_DISORIENTATION": "Spatial disorientation",
        "PILOT_INCAPACITATION_OR_IMPAIRMENT": "Pilot incapacitation",
        "TURBULENCE_ENCOUNTER": "Turbulence", "WIND_SHEAR_OR_GUST": "Wind shear / gust"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                               gridspec_kw={"width_ratios": [1.45, 1]})

# --- (a) per-factor contribution, colored by tier ----------------------------
rows = sorted(D["factor_contributions"], key=lambda r: r["contribution_to_P_loc"])
labels = [NICE.get(r["factor"], r["factor"]) for r in rows]
vals = [r["contribution_to_P_loc"] for r in rows]
cols = [COL[r["tier"]] for r in rows]
# 95 % bootstrap interval on each contribution, from risk_uncertainty.py
uc = {f["factor"]: f for f in U["factors"]}
err = [[v - uc[r["factor"]]["contribution_ci95"][0] for r, v in zip(rows, vals)],
       [uc[r["factor"]]["contribution_ci95"][1] - v for r, v in zip(rows, vals)]]
ax1.barh(range(len(rows)), vals, color=cols, edgecolor="k", linewidth=0.4,
         xerr=err, error_kw={"elinewidth": 0.7, "capsize": 1.8, "ecolor": "#333333"})
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(labels, fontsize=7.6)
ax1.set_xlabel(r"Contribution to $P(\mathrm{LOC})$")
ax1.text(0.02, 1.02, "(a)", transform=ax1.transAxes, fontsize=8.6, va="bottom")
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(fc=PHYS, ec="k", lw=.4, label="physics"),
                    Patch(fc=SURR, ec="k", lw=.4, label="surrogate"),
                    Patch(fc=DATA, ec="k", lw=.4, label="data-only")],
           loc="lower right", fontsize=7)
ax1.grid(axis="x", ls=":", lw=0.5, alpha=0.6)

# --- (b) physics scenario + counterfactuals ----------------------------------
scn = D["physics_scenario"]
base = D["noisy_or_P_loc_baseline"]
cf = scn["counterfactuals"]
names = ["baseline", "maneuver + icing", "do(stall margin)", "do(carb heat)"]
vals2 = [base, scn["P_loc"], cf["do(recover stall margin)"],
         cf["do(carb heat: remove icing path)"]]
cols2 = ["#bdbdbd", "#b30000", PHYS, PHYS]
ci2 = [U["combined_baseline"]["ci95"], U["scenario"]["combined"]["ci95"],
       U["scenario"]["counterfactuals"]["do(recover stall margin)"]["ci95"],
       U["scenario"]["counterfactuals"]["do(carburetor heat)"]["ci95"]]
err2 = [[v - c[0] for v, c in zip(vals2, ci2)],
        [c[1] - v for v, c in zip(vals2, ci2)]]
bars = ax2.bar(range(len(vals2)), vals2, color=cols2, edgecolor="k", linewidth=0.5,
               width=0.62, yerr=err2,
               error_kw={"elinewidth": 0.7, "capsize": 2.0, "ecolor": "#333333"})
for b, v in zip(bars, vals2):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.2f}",
             ha="center", va="bottom", fontsize=7.3)
ax2.set_xticks(range(len(vals2)))
ax2.set_xticklabels(names, fontsize=6.8, rotation=25, ha="right")
ax2.set_ylabel(r"$P(\mathrm{Loss\ of\ Control})$")
ax2.set_ylim(0, 0.45)
ax2.text(0.02, 1.02, "(b)", transform=ax2.transAxes, fontsize=8.6, va="bottom")
ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

fig.tight_layout(w_pad=1.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
