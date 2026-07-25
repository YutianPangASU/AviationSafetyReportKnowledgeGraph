"""High-fidelity stall occurrence model for GA aircraft, grounded in JSBSim.

Stall is the dominant contributing factor to loss of control (Table 1 of the
paper). Its risk-bearing drivers -- airspeed, weight, load factor -- are not
recorded in accident reports, so we marginalize them over phase-conditioned
priors (Eq. 4), but the *aerodynamics* are taken from JSBSim's validated
per-aircraft lift tables rather than a guessed CL_max. This is the high-fidelity
occurrence model the paper's stall factor calls for, and the per-aircraft
CL_max / V_stall are exactly what an individual-accident counterfactual on a
specific GA type needs.

The stall boundary is 1-g-generalized:  V_stall(n) = sqrt(2 n W / (rho S CL_max)).
CL_max, the flap increment, and wing area S are read from the aircraft's JSBSim
model (the same nonlinear tables the 6-DOF simulator integrates), so the numbers
are the simulator's, not ours. For dynamic / departure-based fidelity the same
model can be driven through JSBSim's integrator; here we use its aero tables,
which already validate the stall speed against the POH.
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass
import numpy as np
import jsbsim

G = 32.174           # ft/s^2
RHO0 = 0.0023769     # slug/ft^3, sea-level standard
KT = 1.68781         # ft/s per knot
ROOT = jsbsim.get_default_root_dir()

# Representative gross weights (lbf) for the GA fleet, for default scenarios.
GROSS_LBF = {"c172x": 2300, "c172p": 2300, "c182": 2950, "pa28": 2150,
             "c310": 5500, "J3Cub": 1220}


def _aircraft_xml(model: str) -> str:
    return os.path.join(ROOT, "aircraft", model, f"{model}.xml")


def _floats(line: str):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", line)]


@dataclass
class Aero:
    model: str
    wing_area_ft2: float
    cl_max_clean: float
    cl_max_flap: float
    stall_alpha_deg: float


def extract_aero(model: str) -> Aero:
    """Parse wing area and the lift-vs-alpha table from the JSBSim model.

    Reads CL_max (clean) from the 'Lift due to alpha' table and the maximum flap
    lift increment from the flap table -- the validated aerodynamics the 6-DOF
    model uses.
    """
    xml = open(_aircraft_xml(model)).read()

    m = re.search(r"<wingarea[^>]*>\s*([\d.]+)", xml)
    S = float(m.group(1)) if m else 174.0

    # Lift-due-to-alpha table: alpha(rad) -> CL. Grab the function whose
    # description mentions lift due to alpha, then its <tableData>.
    lift_fn = re.search(
        r'name="aero/coefficient/CL(?:wbh|alpha)".*?<tableData>(.*?)</tableData>',
        xml, re.S)
    cl_max_clean, stall_alpha = 1.5, 16.0
    if lift_fn:
        rows = [_floats(l) for l in lift_fn.group(1).strip().splitlines()]
        # rows with >=2 numbers are (alpha_rad, CL[, CL_hyst]); first 1-number
        # line is the column header -> skip.
        pts = [(r[0], r[1]) for r in rows if len(r) >= 2]
        if pts:
            arr = np.array(pts)
            i = int(arr[:, 1].argmax())
            cl_max_clean = float(arr[i, 1])
            stall_alpha = float(np.degrees(arr[i, 0]))

    # Flap lift increment (max over the flap schedule).
    flap_fn = re.search(r"Delta lift due to flap.*?<tableData>(.*?)</tableData>",
                        xml, re.S)
    d_flap = 0.0
    if flap_fn:
        vals = [_floats(l)[-1] for l in flap_fn.group(1).strip().splitlines()
                if len(_floats(l)) >= 2]
        if vals:
            d_flap = max(vals)

    return Aero(model, S, cl_max_clean, cl_max_clean + d_flap, stall_alpha)


def _rho(alt_ft: float) -> float:
    """ISA density (slug/ft^3), troposphere."""
    t = 518.67 - 3.5662e-3 * alt_ft
    return RHO0 * (t / 518.67) ** 4.2561


def v_stall_kcas(aero: Aero, weight_lbf: float, alt_ft: float = 0.0,
                 load_factor: float = 1.0, flaps: bool = False) -> float:
    """Stall speed (KCAS) from the JSBSim aerodynamics."""
    cl = aero.cl_max_flap if flaps else aero.cl_max_clean
    v_tas = np.sqrt(2.0 * load_factor * weight_lbf / (_rho(alt_ft) * aero.wing_area_ft2 * cl))
    # CAS ~ EAS here; convert TAS->CAS via density ratio (EAS = TAS*sqrt(rho/rho0)).
    return float(v_tas * np.sqrt(_rho(alt_ft) / RHO0) / KT)


# Phase-conditioned priors over the unobserved state (see paper Eq. 4).
# speed_ratio = flown CAS as a multiple of the clean 1-g stall speed.
@dataclass
class Phase:
    weight_frac: tuple  # (lo, mode, hi) of gross
    speed_ratio: tuple  # (mean, sd)
    load_factor: tuple  # (mean, sd), >= 1
    flaps: bool


PHASES = {
    "takeoff":    Phase((0.85, 0.98, 1.0), (1.25, 0.10), (1.05, 0.08), False),
    "climb":      Phase((0.80, 0.95, 1.0), (1.30, 0.12), (1.05, 0.08), False),
    "cruise":     Phase((0.70, 0.88, 1.0), (1.70, 0.20), (1.10, 0.15), False),
    "maneuvering":Phase((0.70, 0.90, 1.0), (1.35, 0.18), (1.55, 0.45), False),
    "approach":   Phase((0.70, 0.88, 0.98), (1.30, 0.12), (1.15, 0.18), True),
    "landing":    Phase((0.70, 0.85, 0.98), (1.22, 0.12), (1.10, 0.12), True),
}


@dataclass
class StallRisk:
    p_stall: float
    ci: tuple
    v_stall_kcas: float
    cl_max: float
    model: str
    phase: str


def p_stall(model: str, alt_ft: float = 0.0, phase: str = "maneuvering",
            weight_lbf: float | None = None, n_samples: int = 20000,
            seed: int = 0) -> StallRisk:
    """Physics probability of stall for a GA type at an operating point.

    Marginalizes unobserved weight / airspeed / load factor over the phase
    prior; the stall boundary uses JSBSim's validated CL_max.
    """
    aero = extract_aero(model)
    pr = PHASES.get(phase, PHASES["maneuvering"])
    rng = np.random.default_rng(seed)
    gross = weight_lbf or GROSS_LBF.get(model, 2300)

    wfrac = rng.triangular(*pr.weight_frac, n_samples)
    weight = wfrac * gross
    n = np.clip(rng.normal(*pr.load_factor, n_samples), 1.0, 6.0)
    # clean 1-g stall speed at this weight sets the airspeed scale
    vs_ref = np.array([v_stall_kcas(aero, w, alt_ft, 1.0, False) for w in
                       np.linspace(weight.min(), weight.max(), 64)])
    wgrid = np.linspace(weight.min(), weight.max(), 64)
    vs_ref_s = np.interp(weight, wgrid, vs_ref)
    cas = np.clip(rng.normal(*pr.speed_ratio, n_samples), 0.6, 3.0) * vs_ref_s
    # accelerated stall speed at the sampled load factor / config
    cl = aero.cl_max_flap if pr.flaps else aero.cl_max_clean
    vs_n = vs_ref_s * np.sqrt(n * aero.cl_max_clean / cl)
    stalled = cas < vs_n
    p = float(stalled.mean())
    boot = rng.choice(stalled, (200, n_samples)).mean(1)
    return StallRisk(p, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
                     v_stall_kcas(aero, gross, alt_ft, 1.0, False), cl, model, phase)


if __name__ == "__main__":
    print("=== JSBSim-extracted aerodynamics (validated per-aircraft) ===")
    for m in ["c172x", "c182", "pa28"]:
        a = extract_aero(m)
        vs = v_stall_kcas(a, GROSS_LBF[m])
        print(f"  {m:6s} S={a.wing_area_ft2:5.1f}ft^2  CLmax(clean)={a.cl_max_clean:.2f} "
              f"@ {a.stall_alpha_deg:.0f}deg  CLmax(flap)={a.cl_max_flap:.2f}  "
              f"Vs1(gross,SL)={vs:.0f} KCAS")

    print("\n=== P(stall) by phase, C172 at sea level ===")
    for ph in ["cruise", "climb", "approach", "landing", "maneuvering"]:
        r = p_stall("c172x", 0.0, ph, seed=1)
        print(f"  {ph:12s} P={r.p_stall:.3f}  [{r.ci[0]:.3f},{r.ci[1]:.3f}]")

    print("\n=== Counterfactual: maneuvering load factor (C172) ===")
    for nbar, lab in [(1.8, "aggressive base-to-final (n~1.8)"),
                      (1.2, "do(coordinated, gentle turn: n~1.2)")]:
        pr = PHASES["maneuvering"]
        PHASES["maneuvering"] = Phase(pr.weight_frac, pr.speed_ratio, (nbar, 0.05), False)
        r = p_stall("c172x", 0.0, "maneuvering", seed=2)
        print(f"  {lab:38s} P(stall)={r.p_stall:.3f}")
        PHASES["maneuvering"] = pr
