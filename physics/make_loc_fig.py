"""Loss-of-control failure-mode figure: (a) per-factor contribution to P(LOC)
under the fitted leaky noisy-OR, colored by model tier; (b) the maneuvering
and gusty-wind scenario with its physics-level interventions. Reads
physics/out/loc_risk.json and physics/out/risk_uncertainty.json.
Output: paper/figs/fig_loc.pdf."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FS = 8.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral",
                   "Times New Roman", "DejaVu Serif"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS, "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

D = json.load(open("physics/out/loc_risk.json"))
U = json.load(open("physics/out/risk_uncertainty.json"))["loss_of_control"]
OUT = "paper/figs/fig_loc.pdf"
PHYS, SURR, DATA = "#e08214", "#f6c58a", "#8a8a8a"
COL = {"physics": PHYS, "surrogate": SURR, "data": DATA}

NICE = {"STALL": "Stall", "CONTROL_INPUT_IMPROPER": "Improper control input",
        "ENGINE_FAILURE": "Engine failure", "DECISION_INAPPROPRIATE": "Inappropriate decision",
        "CONTROL_SURFACE_ANOMALY": "Control surface anomaly",
        "AIRFRAME_STRUCTURAL_FAILURE": "Structural failure",
        "SPATIAL_DISORIENTATION": "Spatial disorientation",
        "PILOT_INCAPACITATION_OR_IMPAIRMENT": "Pilot incapacitation",
        "TURBULENCE_ENCOUNTER": "Turbulence", "WIND_SHEAR_OR_GUST": "Wind shear or gust",
        "PROCEDURE_NOT_FOLLOWED": "Procedure not followed",
        "PERCEPTION_FAILURE": "Perception failure",
        "ADVERSE_WIND_CONDITION": "Adverse wind condition",
        "OTHER_SYSTEM_FAILURE": "Other system failure"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                               gridspec_kw={"width_ratios": [1.45, 1]})

# --- (a) per-factor contribution, colored by tier ----------------------------
rows = sorted(D["factors"], key=lambda r: r["contribution"])
labels = [NICE.get(r["factor"], r["factor"]) for r in rows]
vals = [r["contribution"] for r in rows]
cols = [COL[r["tier"]] for r in rows]
uc = {f["factor"]: f for f in U["factors"]}
err = [[max(0.0, v - uc[r["factor"]]["contribution_ci95"][0]) for r, v in zip(rows, vals)],
       [max(0.0, uc[r["factor"]]["contribution_ci95"][1] - v) for r, v in zip(rows, vals)]]
ax1.barh(range(len(rows)), vals, color=cols, edgecolor="k", linewidth=0.4,
         xerr=err, error_kw={"elinewidth": 0.7, "capsize": 1.8, "ecolor": "#333333"})
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(labels)
ax1.set_xlabel(r"Contribution $\Delta_i$ to $P(\mathrm{LOC})$")
ax1.text(0.02, 1.02, "(a)", transform=ax1.transAxes, va="bottom")
ax1.legend(handles=[Patch(fc=PHYS, ec="k", lw=.4, label="Physics"),
                    Patch(fc=SURR, ec="k", lw=.4, label="Surrogate"),
                    Patch(fc=DATA, ec="k", lw=.4, label="Data only")],
           loc="lower right")
ax1.grid(axis="x", ls=":", lw=0.5, alpha=0.6)

# --- (b) scenario + interventions -------------------------------------------
scn = D["scenario"]
cf = scn["counterfactuals"]
keys = list(cf.keys())
names = ["Baseline", "Scenario", "Coordinated\nturn", "Gust\nmargin", "Both"]
vals2 = [D["fitted_mean"], scn["P"]] + [cf[k]["P"] for k in keys]
cols2 = ["#bdbdbd", "#b30000", PHYS, PHYS, PHYS]
ci2 = [U["combined_baseline"]["ci95"], U["scenario"]["combined"]["ci95"]] + \
      [U["scenario"]["counterfactuals"][k]["ci95"] for k in keys]
err2 = [[max(0.0, v - c[0]) for v, c in zip(vals2, ci2)],
        [max(0.0, c[1] - v) for v, c in zip(vals2, ci2)]]
bars = ax2.bar(range(len(vals2)), vals2, color=cols2, edgecolor="k", linewidth=0.5,
               width=0.62, yerr=err2,
               error_kw={"elinewidth": 0.7, "capsize": 2.0, "ecolor": "#333333"})
for b, v in zip(bars, vals2):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom")
ax2.set_xticks(range(len(vals2)))
ax2.set_xticklabels(names, rotation=0, ha="center", fontsize=FS - 0.5)
ax2.set_ylabel(r"$P(\mathrm{Loss\ of\ Control})$")
ax2.set_ylim(0, max(vals2) * 1.35)
ax2.text(0.02, 1.02, "(b)", transform=ax2.transAxes, va="bottom")
ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

fig.tight_layout(w_pad=1.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
