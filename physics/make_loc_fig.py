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
OUT = "paper/figs/fig_loc.pdf"
PHYS, DATA = "#e08214", "#8a8a8a"

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
cols = [PHYS if r["tier"] == "physics" else DATA for r in rows]
ax1.barh(range(len(rows)), vals, color=cols, edgecolor="k", linewidth=0.4)
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(labels, fontsize=7.6)
ax1.set_xlabel(r"contribution to $P(\mathrm{LOC})$  (drop if factor removed)")
ax1.set_title("(a) Contributing-factor decomposition")
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(fc=PHYS, ec="k", lw=.4, label="physics-modeled"),
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
bars = ax2.bar(range(len(vals2)), vals2, color=cols2, edgecolor="k", linewidth=0.5, width=0.62)
for b, v in zip(bars, vals2):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
             ha="center", va="bottom", fontsize=7.3)
ax2.set_xticks(range(len(vals2)))
ax2.set_xticklabels(names, fontsize=6.8, rotation=25, ha="right")
ax2.set_ylabel(r"$P(\mathrm{loss\ of\ control})$")
ax2.set_ylim(0, 0.45)
ax2.set_title("(b) Operating-point scenario and counterfactuals")
ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

fig.tight_layout(w_pad=1.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
