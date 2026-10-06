"""Engine-failure failure-mode figure: (a) per-factor contribution to P(EF)
under the fitted leaky noisy-OR by model tier; (b) the icing-weather
scenario and the two physical interventions on it, all on the accident
population. The fuel-reserve arms were dropped on 2026-09-30 because the
fuel term is per exposure and would mix scales with the calibrated terms;
the manuscript reports the reserve effect as a ratio on the fuel term. The
endurance model is uncalibrated, so fuel exhaustion enters at its corpus base
rate and is hatched in (a). Panel (b) breaks its axis between 0 and the
plotted range (review 2026-10-06). Reads
physics/out/ef_risk.json and physics/out/risk_uncertainty.json.
Output: paper/figs/fig_ef.pdf."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
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

D = json.load(open("physics/out/ef_risk.json"))
U = json.load(open("physics/out/risk_uncertainty.json"))["engine_failure"]
OUT = "paper/figs/fig_ef.pdf"
PHYS, SURR, DATA = "#e08214", "#f5c48a", "#8a8a8a"
COL = {"physics": PHYS, "surrogate": SURR, "data": DATA}

NICE = {"FUEL_EXHAUSTION_OR_STARVATION": "Fuel exhaustion or starvation",
        "LATENT_MECHANICAL_DEFECT": "Latent mechanical defect",
        "PROCEDURE_NOT_FOLLOWED": "Procedure not followed",
        "OTHER_SYSTEM_FAILURE": "Other system failure",
        "CARBURETOR_OR_INDUCTION_ICING": "Carburetor or induction icing",
        "FUEL_SYSTEM_ANOMALY": "Fuel system anomaly",
        "FUEL_CONTAMINATION": "Fuel contamination",
        "MAINTENANCE_INADEQUATE": "Inadequate maintenance",
        "AIRFRAME_STRUCTURAL_FAILURE": "Structural failure",
        "CONTROL_INPUT_IMPROPER": "Improper control input",
        "DECISION_INAPPROPRIATE": "Inappropriate decision"}

def r3(v):
    """Round half up to three decimals, as the manuscript text does."""
    return str(Decimal(repr(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


fig = plt.figure(figsize=(7.2, 3.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[14, 1], hspace=0.06)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax2z = fig.add_subplot(gs[1, 1], sharex=ax2)   # stub of the axis near zero

rows = sorted(D["factors"], key=lambda r: r["contribution"])
labels = [NICE.get(r["factor"], r["factor"]) for r in rows]
vals = [r["contribution"] for r in rows]
# a physics model that is not calibrated enters the combination at its base rate
uncal = [r["tier"] == "physics" and r.get("model") == "endurance" for r in rows]
cols = [COL[r["tier"]] for r in rows]
uc = {f["factor"]: f for f in U["factors"]}
err = [[max(0.0, v - uc[r["factor"]]["contribution_ci95"][0]) for r, v in zip(rows, vals)],
       [max(0.0, uc[r["factor"]]["contribution_ci95"][1] - v) for r, v in zip(rows, vals)]]
bars = ax1.barh(range(len(rows)), vals, color=cols, edgecolor="k", linewidth=0.4,
                xerr=err, error_kw={"elinewidth": 0.7, "capsize": 1.8, "ecolor": "#333333"})
for b, u in zip(bars, uncal):
    if u:
        b.set_facecolor("white"); b.set_edgecolor(PHYS); b.set_hatch("////")
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(labels)
ax1.set_xlabel(r"Contribution $\Delta_i$ to $P(\mathrm{EF})$")
ax1.text(0.02, 1.02, "(a)", transform=ax1.transAxes, va="bottom")
ax1.legend(handles=[Patch(fc=PHYS, ec="k", lw=.4, label="Physics"),
                    Patch(fc="white", ec=PHYS, hatch="////", lw=.4,
                          label="Physics, uncalibrated"),
                    Patch(fc=SURR, ec="k", lw=.4, label="Surrogate"),
                    Patch(fc=DATA, ec="k", lw=.4, label="Data only")],
           loc="lower right")
ax1.grid(axis="x", ls=":", lw=0.5, alpha=0.6)

ice = D["icing_scenario"]
cf = ice["counterfactuals"]
UI = U["icing_scenario"]
keys = ["do(carburetor heat)", "do(drier air, dewpoint -5 C)"]
names = ["Population", "Icing weather", "do(heat)", "do(drier air)"]
vals2 = [D["fitted_mean"], ice["P"]] + [cf[k]["P"] for k in keys]
cols2 = ["#bdbdbd", "#d7301f", PHYS, PHYS]
ci2 = [U["combined_baseline"]["ci95"], UI["combined"]["ci95"]] + \
      [UI["counterfactuals"][k]["ci95"] for k in keys]
xs = [0, 1.25, 2.25, 3.25]
for x, v, c, col in zip(xs, vals2, ci2, cols2):
    ax2.errorbar([x], [v], yerr=[[v - c[0]], [c[1] - v]], fmt="o", ms=5.5, mfc=col,
                 mec="k", mew=0.5, ecolor="#333333", elinewidth=0.7, capsize=2.2)
    ax2.text(x + 0.13, v, r3(v), ha="left", va="center")
ax2.axhline(vals2[0], color="#8a8a8a", ls="--", lw=0.6, zorder=0)
# bracket grouping the two interventions under the scenario they modify
yb = min(c[0] for c in ci2) - 0.0035
ax2.plot([1.05, 1.05, 3.45, 3.45], [yb + 0.0012, yb, yb, yb + 0.0012], color="k", lw=0.5)
ax2.text(2.25, yb - 0.0007, "Icing weather scenario", ha="center", va="top", fontsize=FS - 0.5)
ax2.set_xlim(-0.5, 3.95)
ax2.set_ylim(yb - 0.0045, max(c[1] for c in ci2) + 0.004)
ax2.set_ylabel(r"$P(\mathrm{Engine\ Failure})$")
# broken axis: the stub below shows where zero lies
ax2.spines["bottom"].set_visible(False)
ax2.tick_params(axis="x", bottom=False, labelbottom=False)
ax2z.set_ylim(0, 0.002)
ax2z.set_yticks([0])
ax2z.spines["top"].set_visible(False)
ax2z.set_xticks(xs)
ax2z.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor", fontsize=FS - 0.5)
d = 0.012
kw = dict(color="k", clip_on=False, lw=0.6)
ax2.plot((-d, +d), (-d * 1.2, +d * 1.2), transform=ax2.transAxes, **kw)
ax2.plot((1 - d, 1 + d), (-d * 1.2, +d * 1.2), transform=ax2.transAxes, **kw)
ax2z.plot((-d, +d), (1 - d * 9, 1 + d * 9), transform=ax2z.transAxes, **kw)
ax2z.plot((1 - d, 1 + d), (1 - d * 9, 1 + d * 9), transform=ax2z.transAxes, **kw)
ax2.text(0.02, 1.02, "(b)", transform=ax2.transAxes, va="bottom")
ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

fig.subplots_adjust(wspace=0.32)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
