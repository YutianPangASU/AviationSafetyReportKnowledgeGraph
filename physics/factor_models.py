"""Physics occurrence models for the remaining physics-tier contributing
factors of loss of control: gust/structural overload, gust- and shear-induced
stall (turbulence, wind shear), and an engine-failure hazard surrogate.

These complement the two fully-worked models:
  - carb icing  : physics/carb_icing_model.py  (thermodynamics)
  - stall       : physics/jsbsim_stall.py      (JSBSim validated aero)

Gust mechanics follow the certification-basis Pratt gust-load formula
(FAR 23.341 / NACA Report 1206): a discrete gust of derived velocity U_de
produces an incremental load factor

    dn = K_g * U_de * V_e * a / (498 * (W/S))          [imperial units]
    K_g = 0.88 mu / (5.3 + mu),   mu = 2 (W/S) / (rho c_bar a g)

with V_e in KEAS, U_de in ft/s, lift-curve slope a per rad, wing loading W/S
in psf. Two distinct failures share this one mechanism:

  * STRUCTURAL OVERLOAD:   dn exceeds the V-n limit (n_lim - 1) or ultimate
    (1.5 x limit) increment;
  * GUST-INDUCED STALL:    the gust load factor pushes the wing past CL_max,
    i.e. 1 + dn > (V / V_s1g)^2  -- the same accelerated-stall boundary the
    JSBSim model uses, now driven by the atmosphere instead of the pilot.

Gust climate enters as an exponential tail on U_de (scale by encounter class),
the standard single-parameter approximation of measured gust exceedance
statistics (Hoblit 1988). Wind shear erodes airspeed directly: a tailwind
shear of dU knots moves the aircraft dU closer to the stall boundary before
thrust can respond.

Engine failure has no first-principles occurrence model; we use a transparent
hazard-rate surrogate: a base in-flight-shutdown rate lambda0 (order 1e-4 per
flight hour for GA piston engines) composed noisy-OR with the condition-driven
paths (carburetor icing via physics/carb_icing_model, fuel exhaustion via
endurance margin).

C172-class defaults are used for the demo; all parameters are per-aircraft.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, sqrt, sin, radians

G = 32.174
RHO0 = 0.0023769  # slug/ft^3


@dataclass(frozen=True)
class GustAircraft:
    name: str
    weight_lbf: float
    wing_area_ft2: float
    cl_alpha_per_rad: float   # lift-curve slope a
    chord_ft: float           # mean geometric chord
    n_limit: float            # positive limit load factor (normal category: 3.8)
    vs1g_kcas: float          # 1-g clean stall speed at weight
    va_kcas: float            # design maneuvering speed
    vc_kcas: float            # design cruise speed


C172 = GustAircraft("Cessna 172", 2300.0, 174.0, 4.6, 4.9, 3.8, 52.0, 99.0, 120.0)

# Exponential U_de scale (ft/s) by gust climate / encounter class.
GUST_SCALE_FPS = {"light": 5.0, "moderate": 10.0, "severe": 20.0,
                  "thunderstorm": 20.0}


def pratt_dn_per_ude(ac: GustAircraft, v_keas: float, rho: float = RHO0) -> float:
    """Incremental load factor per ft/s of derived gust velocity (Pratt)."""
    ws = ac.weight_lbf / ac.wing_area_ft2
    mu = 2.0 * ws / (rho * ac.chord_ft * ac.cl_alpha_per_rad * G)
    kg = 0.88 * mu / (5.3 + mu)
    return kg * v_keas * ac.cl_alpha_per_rad / (498.0 * ws)


# ---------------------------------------------------------------------------
# Guard margins (paper Table tbl:guards). Each is the signed quantity the
# occurrence probabilities below already threshold internally; a transition is
# enabled iff its margin is >= 0. The probabilities are the availabilities
# Lambda_e = P(margin >= 0) under the latent-gust prior.
# ---------------------------------------------------------------------------

def overload_margin_n(ac: GustAircraft, v_keas: float, u_de_fps: float,
                      ultimate: bool = True) -> float:
    """Structural-overload guard: n - n_ult (dimensionless).

    Gust load factor n = 1 + dn(U_de) against the ultimate (1.5 x limit) or
    limit envelope.
    """
    n = 1.0 + pratt_dn_per_ude(ac, v_keas) * u_de_fps
    n_env = 1.5 * ac.n_limit if ultimate else ac.n_limit
    return n - n_env


def gust_stall_margin_kts(ac: GustAircraft, v_kcas: float,
                          u_de_fps: float) -> float:
    """Gust-stall guard: V_s(n) - V (knots) at the gust load factor.

    The accelerated-stall speed V_s1g * sqrt(1 + dn) against the flown
    airspeed; nonnegative exactly when the gust erases the stall margin.
    """
    n = max(0.0, 1.0 + pratt_dn_per_ude(ac, v_kcas) * u_de_fps)
    return ac.vs1g_kcas * sqrt(n) - v_kcas


def shear_stall_margin_kts(ac: GustAircraft, v_kcas: float,
                           du_kts: float) -> float:
    """Shear-stall guard: V_s1g - (V - dU) (knots).

    Tailwind shear dU erodes airspeed before thrust responds; the guard is
    nonnegative when the eroded airspeed reaches the 1-g stall speed.
    """
    return ac.vs1g_kcas - (v_kcas - du_kts)


def p_structural_overload(ac: GustAircraft, v_keas: float,
                          climate: str = "thunderstorm",
                          ultimate: bool = True) -> float:
    """Availability of the overload transition: P(overload margin >= 0).

    The margin n - n_ult crosses zero at U_de = n_inc / dn1; the exponential
    gust tail turns that crossing point into an exceedance probability.
    """
    dn1 = pratt_dn_per_ude(ac, v_keas)
    n_inc = (1.5 * ac.n_limit - 1.0) if ultimate else (ac.n_limit - 1.0)
    u_crit = n_inc / dn1
    return exp(-u_crit / GUST_SCALE_FPS.get(climate, 10.0))


def p_gust_stall(ac: GustAircraft, v_kcas: float,
                 climate: str = "moderate") -> float:
    """Availability of the gust-stall transition: P(gust-stall margin >= 0).

    The margin V_s(n) - V crosses zero where 1 + dn = (V / V_s1g)^2, the
    accelerated-stall boundary with the load factor supplied by the gust
    instead of the pilot; the gust tail supplies the crossing probability.
    """
    n_margin = (v_kcas / ac.vs1g_kcas) ** 2 - 1.0
    if n_margin <= 0:
        return 1.0
    u_crit = n_margin / pratt_dn_per_ude(ac, v_kcas)
    return exp(-u_crit / GUST_SCALE_FPS.get(climate, 10.0))


def p_shear_stall(ac: GustAircraft, v_kcas: float, shear_scale_kts: float = 10.0
                  ) -> float:
    """Availability of the shear-stall transition: P(shear-stall margin >= 0).

    The margin V_s1g - (V - dU) crosses zero at dU = V - V_s1g;
    dU ~ Exp(shear_scale) supplies the crossing probability. Microburst
    cores reach 20-40 kt.
    """
    du_crit = -shear_stall_margin_kts(ac, v_kcas, 0.0)
    if du_crit <= 0:
        return 1.0
    return exp(-du_crit / shear_scale_kts)


def pi_engine_failure(t_hr: float = 1.5, lambda0_per_hr: float = 1e-4,
                      p_ice: float = 0.0, p_ef_given_ice: float = 0.667,
                      fuel_margin_hr: float = 2.0,
                      fuel_sigma_hr: float = 0.75) -> float:
    """Engine-failure hazard surrogate: base rate + condition-driven paths.

    Noisy-OR of (i) the random in-flight-shutdown hazard 1-exp(-lambda0 t),
    (ii) the carburetor-icing path pi_ice * P(EF|ice) from the icing physics,
    and (iii) fuel exhaustion P(endurance shortfall) with Gaussian margin.
    """
    p_base = 1.0 - exp(-lambda0_per_hr * t_hr)
    p_icing_path = p_ice * p_ef_given_ice
    # P(fuel margin < 0) via a normal margin model
    z = fuel_margin_hr / max(fuel_sigma_hr, 1e-6)
    p_fuel = 0.5 * (1.0 - _erf(z / sqrt(2.0)))
    return 1.0 - (1.0 - p_base) * (1.0 - p_icing_path) * (1.0 - p_fuel)


def _erf(x: float) -> float:
    # Abramowitz-Stegun 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return y if x >= 0 else -y


def crosswind_component(wind_kts: float, angle_deg: float) -> float:
    return wind_kts * abs(sin(radians(angle_deg)))


# ---------------------------------------------------------------------------
# Dryden continuous-turbulence upgrade (spectral; per MIL-F-8785C low-altitude
# parameterization, as implemented in the TASAT 6-DOF simulator's wind model).
# Links the RECORDED surface wind (NTSB events wind_vel_kts) to gust intensity:
#     sigma_w = 0.1 * u_20   (RMS vertical gust from the 20-ft wind)
# Exceedance over an encounter uses Rice level-crossing statistics
# (Hoblit 1988): crossings of level U occur at rate nu0*exp(-U^2/2 sigma^2).
# ---------------------------------------------------------------------------

def dryden_sigma_w_fps(u20_kts: float) -> float:
    """RMS vertical gust velocity (ft/s) from the 20-ft surface wind."""
    return 0.1 * u20_kts * 1.68781


def p_exceed_dryden(u_crit_fps: float, u20_kts: float, v_fps: float,
                    duration_s: float = 120.0, l_w_ft: float = 200.0) -> float:
    """P(vertical gust exceeds u_crit at least once during the encounter).

    Rice's formula: level crossings of a Gaussian process at rate
    nu0 * exp(-U^2 / 2 sigma^2), nu0 ~ V / (2 L_w) for the Dryden w-spectrum
    at low altitude (L_w ~ height AGL). P = 1 - exp(-nu*T).
    """
    sigma = dryden_sigma_w_fps(u20_kts)
    if sigma <= 0:
        return 0.0
    nu0 = v_fps / (2.0 * l_w_ft)
    nu = nu0 * exp(-u_crit_fps ** 2 / (2.0 * sigma ** 2))
    return 1.0 - exp(-nu * duration_s)


def p_gust_stall_dryden(ac: GustAircraft, v_kcas: float, u20_kts: float,
                        duration_s: float = 120.0) -> float:
    """Gust-stall availability with Dryden intensity set by recorded wind.

    Same margin zero-crossing as p_gust_stall; the crossing probability now
    comes from the Rice statistics of the recorded-wind gust spectrum.
    """
    n_margin = (v_kcas / ac.vs1g_kcas) ** 2 - 1.0
    if n_margin <= 0:
        return 1.0
    u_crit = n_margin / pratt_dn_per_ude(ac, v_kcas)
    return p_exceed_dryden(u_crit, u20_kts, v_kcas * 1.68781, duration_s)


if __name__ == "__main__":
    ac = C172
    dn = pratt_dn_per_ude(ac, ac.vc_kcas)
    print(f"=== Gust mechanics, {ac.name} (Pratt / FAR 23.341) ===")
    print(f"  dn per ft/s U_de at Vc={ac.vc_kcas:.0f} KEAS: {dn:.4f}"
          f"  (50-fps FAR gust -> dn={50*dn:.2f}, n={1+50*dn:.2f} vs limit {ac.n_limit})")

    print("\n=== Structural overload (thunderstorm penetration) ===")
    for v, lab in [(ac.vc_kcas, "at Vc (cruise speed)"),
                   (ac.va_kcas, "do(slow to Va)")]:
        pl = p_structural_overload(ac, v, "thunderstorm", ultimate=False)
        pu = p_structural_overload(ac, v, "thunderstorm", ultimate=True)
        print(f"  {lab:24s} P(exceed limit)={pl:.3f}  P(exceed ultimate)={pu:.4f}")

    print("\n=== Gust-induced stall (turbulence factor) ===")
    for v, cl, lab in [(1.25 * ac.vs1g_kcas, "moderate", "approach at 1.25 Vs"),
                       (1.44 * ac.vs1g_kcas, "moderate", "do(+10 kt gust margin)"),
                       (1.25 * ac.vs1g_kcas, "severe", "approach, severe turb")]:
        print(f"  {lab:24s} P={p_gust_stall(ac, v, cl):.3f}")

    print("\n=== Gust-induced stall, Dryden (driven by RECORDED surface wind) ===")
    for u20, lab in [(10, "calm day, wind 10 kt"), (25, "windy, 25 kt"),
                     (35, "strong/gusty, 35 kt")]:
        p = p_gust_stall_dryden(ac, 1.25 * ac.vs1g_kcas, u20)
        print(f"  approach 1.25 Vs, {lab:22s} P={p:.4f}")

    print("\n=== Shear-induced stall (wind-shear factor) ===")
    for v, s, lab in [(65, 10, "approach 65 kt, shear~10kt"),
                      (75, 10, "do(+10 kt approach speed)")]:
        print(f"  {lab:28s} P={p_shear_stall(ac, v, s):.3f}")

    print("\n=== Engine-failure hazard surrogate ===")
    from physics.carb_icing_model import p_carb_icing
    ice = p_carb_icing(13, 12, "descent").p_ice
    for lab, kw in [("benign (1.5 h, fuel margin 2 h)", {}),
                    ("icing conditions (13/12C descent)", {"p_ice": ice}),
                    ("do(carb heat)", {"p_ice": 0.0}),
                    ("thin fuel margin (0.5 h)", {"fuel_margin_hr": 0.5})]:
        print(f"  {lab:34s} pi_EF={pi_engine_failure(**kw):.4f}")
