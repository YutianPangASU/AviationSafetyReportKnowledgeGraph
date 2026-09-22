"""Figure-1-style physics panels for the other physics-tier factors of the
loss-of-control case study: (a) JSBSim accelerated stall, (b) gust structural
overload (Pratt), (c) Dryden gust-induced stall vs recorded surface wind,
(d) wind-shear stall-margin erosion. Outputs paper/figs/fig_factor_physics.pdf."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.jsbsim_stall import p_stall, PHASES, Phase
from physics.factor_models import (C172, p_structural_overload,
                                   p_gust_stall_dryden, p_shear_stall)

FS = 8.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral",
                   "Times New Roman", "DejaVu Serif"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS, "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})
ORANGE, GREY, DARK = "#e08214", "#8a8a8a", "#4d4d4d"
OUT = "paper/figs/fig_factor_physics.pdf"

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
(axa, axb), (axc, axd) = axes

# --- (a) JSBSim accelerated stall: P(stall) vs mean load factor -------------
# Clean vs landing-flap config for the C172: JSBSim's flap lift increment
# (CLmax 1.47 -> 1.82) lowers the accelerated-stall boundary.
nbar_grid = np.linspace(1.0, 2.4, 15)
base = PHASES["maneuvering"]
for flaps, col, lab in [(False, ORANGE, "Clean, JSBSim $C_{L,\\max}$ = 1.47"),
                        (True, GREY, "Full flaps, $C_{L,\\max}$ = 1.82")]:
    ps = []
    for nbar in nbar_grid:
        PHASES["maneuvering"] = Phase(base.weight_frac, base.speed_ratio,
                                      (float(nbar), 0.15), flaps)
        ps.append(p_stall("c172x", 0.0, "maneuvering", n_samples=8000,
                          seed=3).p_stall)
    axa.plot(nbar_grid, ps, color=col, lw=1.6, label=lab)
PHASES["maneuvering"] = base
axa.axvspan(1.7, 1.9, color=ORANGE, alpha=0.12, lw=0)
axa.text(1.8, 0.02, "Aggressive\nbase to final", ha="center", fontsize=FS,
         color=DARK)
axa.set_xlabel(r"Mean maneuvering load factor $\bar n$")
axa.set_ylabel(r"$\widehat\pi_{\mathrm{stall}}$")
axa.text(0.02, 1.02, "(a)", transform=axa.transAxes, va="bottom")
axa.legend(loc="upper left")
axa.grid(ls=":", lw=0.5, alpha=0.6)

# --- (b) Structural overload: P(exceed) vs airspeed (thunderstorm) ----------
v_grid = np.linspace(70, 160, 60)
for ult, col, lab in [(False, GREY, "Limit load, $n$ = 3.8"),
                      (True, ORANGE, "Ultimate load, 1.5 times limit")]:
    axb.semilogy(v_grid, [p_structural_overload(C172, v, "thunderstorm", ult)
                          for v in v_grid], color=col, lw=1.6, label=lab)
ybot = axb.get_ylim()[0]
for v, lab in [(C172.va_kcas, "$V_A$"), (C172.vc_kcas, "$V_C$")]:
    axb.axvline(v, color=DARK, lw=0.7, ls="--")
    axb.text(v + 1.5, ybot * 1.5, lab, fontsize=FS, color=DARK)
axb.set_xlabel("Equivalent airspeed (KEAS)")
axb.set_ylabel("Exceedance probability per encounter")
axb.text(0.02, 1.02, "(b)", transform=axb.transAxes, va="bottom")
axb.legend(loc="upper left")
axb.grid(ls=":", lw=0.5, alpha=0.6, which="both")

# --- (c) Dryden gust-induced stall vs recorded surface wind -----------------
u20_grid = np.linspace(5, 40, 70)
for ratio, col, ls in [(1.15, ORANGE, "-"), (1.25, DARK, "-"),
                       (1.40, GREY, "--")]:
    axc.plot(u20_grid, [p_gust_stall_dryden(C172, ratio * C172.vs1g_kcas, u)
                        for u in u20_grid], color=col, lw=1.6, ls=ls,
             label=f"Approach at {ratio:.2f}$\\,V_s$")
axc.set_xlabel(r"Recorded surface wind $u_{20}$ (kt)")
axc.set_ylabel("Occurrence probability per approach")
axc.text(0.02, 1.02, "(c)", transform=axc.transAxes, va="bottom")
axc.legend(loc="upper left")
axc.grid(ls=":", lw=0.5, alpha=0.6)

# --- (d) Wind-shear stall-margin erosion ------------------------------------
v_app = np.linspace(55, 85, 60)
for scale, col, ls in [(5, GREY, "--"), (10, DARK, "-"), (20, ORANGE, "-")]:
    axd.plot(v_app, [p_shear_stall(C172, v, scale) for v in v_app],
             color=col, lw=1.6, ls=ls, label=f"Shear scale {scale} kt")
axd.axvline(C172.vs1g_kcas, color=DARK, lw=0.7, ls=":")
axd.text(C172.vs1g_kcas + 0.5, 0.05, "$V_{s,1g}$", fontsize=FS, color=DARK)
axd.set_xlabel("Approach speed (KCAS)")
axd.set_ylabel(r"$\pi_{\mathrm{shear}}$")
axd.text(0.02, 1.02, "(d)", transform=axd.transAxes, va="bottom")
axd.legend(loc="upper right")
axd.grid(ls=":", lw=0.5, alpha=0.6)

fig.tight_layout(w_pad=1.6, h_pad=1.6)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
