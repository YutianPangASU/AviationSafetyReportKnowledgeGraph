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

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.4, "axes.titlesize": 8.6,
    "axes.labelsize": 8.4, "legend.fontsize": 7.0, "mathtext.fontset": "cm",
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
for flaps, col, lab in [(False, ORANGE, "C172 clean (JSBSim $C_{L,\\max}$=1.47)"),
                        (True, GREY, "C172 full flaps ($C_{L,\\max}$=1.82)")]:
    ps = []
    for nbar in nbar_grid:
        PHASES["maneuvering"] = Phase(base.weight_frac, base.speed_ratio,
                                      (float(nbar), 0.15), flaps)
        ps.append(p_stall("c172x", 0.0, "maneuvering", n_samples=8000,
                          seed=3).p_stall)
    axa.plot(nbar_grid, ps, color=col, lw=1.6, label=lab)
PHASES["maneuvering"] = base
axa.axvspan(1.7, 1.9, color=ORANGE, alpha=0.12, lw=0)
axa.text(1.8, 0.02, "aggressive\nbase-to-final", ha="center", fontsize=6.4,
         color=DARK)
axa.set_xlabel(r"mean maneuvering load factor $\bar n$")
axa.set_ylabel(r"$\widehat\pi_{\mathrm{stall}}$")
axa.set_title("(a) Accelerated stall (JSBSim aero)")
axa.legend(loc="upper left")
axa.grid(ls=":", lw=0.5, alpha=0.6)

# --- (b) Structural overload: P(exceed) vs airspeed (thunderstorm) ----------
v_grid = np.linspace(70, 160, 60)
for ult, col, lab in [(False, GREY, "limit load ($n$=3.8)"),
                      (True, ORANGE, "ultimate load ($1.5\\times$)")]:
    axb.semilogy(v_grid, [p_structural_overload(C172, v, "thunderstorm", ult)
                          for v in v_grid], color=col, lw=1.6, label=lab)
for v, lab in [(C172.va_kcas, "$V_A$"), (C172.vc_kcas, "$V_C$")]:
    axb.axvline(v, color=DARK, lw=0.7, ls="--")
    axb.text(v + 1.5, 2e-4, lab, fontsize=7.5, color=DARK)
axb.set_xlabel("equivalent airspeed (KEAS)")
axb.set_ylabel("P(exceed) per encounter")
axb.set_title("(b) Gust structural overload, thunderstorm")
axb.legend(loc="lower right")
axb.grid(ls=":", lw=0.5, alpha=0.6, which="both")

# --- (c) Dryden gust-induced stall vs recorded surface wind -----------------
u20_grid = np.linspace(5, 40, 70)
for ratio, col, ls in [(1.15, ORANGE, "-"), (1.25, DARK, "-"),
                       (1.40, GREY, "--")]:
    axc.plot(u20_grid, [p_gust_stall_dryden(C172, ratio * C172.vs1g_kcas, u)
                        for u in u20_grid], color=col, lw=1.6, ls=ls,
             label=f"approach at {ratio:.2f}$\\,V_s$")
axc.set_xlabel(r"recorded surface wind $u_{20}$ (kt)")
axc.set_ylabel(r"$\pi$ per approach")
axc.set_title("(c) Gust-induced stall (Dryden, per surface wind)")
axc.legend(loc="upper left")
axc.grid(ls=":", lw=0.5, alpha=0.6)

# --- (d) Wind-shear stall-margin erosion ------------------------------------
v_app = np.linspace(55, 85, 60)
for scale, col, ls in [(5, GREY, "--"), (10, DARK, "-"), (20, ORANGE, "-")]:
    axd.plot(v_app, [p_shear_stall(C172, v, scale) for v in v_app],
             color=col, lw=1.6, ls=ls, label=f"shear scale {scale} kt")
axd.axvline(C172.vs1g_kcas, color=DARK, lw=0.7, ls=":")
axd.text(C172.vs1g_kcas + 0.5, 0.55, "$V_{s,1g}$", fontsize=7.5, color=DARK)
axd.set_xlabel("approach speed (KCAS)")
axd.set_ylabel(r"$\pi_{\mathrm{shear}}$")
axd.set_title("(d) Wind-shear stall-margin erosion")
axd.legend(loc="upper right")
axd.grid(ls=":", lw=0.5, alpha=0.6)

fig.tight_layout(w_pad=1.6, h_pad=1.6)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
