"""Tier-1 physics model for the STALL node of the aviation causation KG.

This is the foundation module for the model-data hybrid framework (v3 design
doc, 4-tier probability system): it replaces the corpus-derived scalar risk of
the ``STALL`` factor with a physics-based probability *conditioned on the
observable operating point* (aircraft type, density altitude, flight phase).

Modeling stance
---------------
Per-accident we can observe the aircraft *type* (-> wing area, CL_max, reference
weights) and can recover *density altitude* (airport elevation + temperature).
We cannot observe the instantaneous state at the moment of stall (actual weight,
airspeed, load factor). So we do NOT reconstruct a single deterministic flight;
we compute

    P(stall | type, density_altitude, phase)
        = E_{W, V, n ~ phase-conditioned priors} [ 1{ V < V_stall(W, rho, n) } ]

by Monte Carlo. Density altitude and weight enter through V_stall: raising
either raises the stall speed, so for the same phase-typical airspeed
distribution more probability mass falls below V_stall. That is the physical
mechanism the corpus can only see as a marginal frequency.

What this module deliberately does NOT do
-----------------------------------------
* It does not fix the absolute base rate. It returns a *calibrated-shape*
  probability; the Tier-1 <-> Tier-3 base-rate calibration lives in
  ``calibrate_stall.py`` (Module 3). Treat the absolute numbers here as
  uncalibrated until then.
* The aircraft constants and phase priors below are first-draft engineering
  estimates meant to be tuned against POH / AFM data. Every one is a named
  parameter so it can be overridden.

Units are SI throughout (m, kg, s, N, m^2, kg/m^3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

G = 9.80665  # m/s^2

# ---------------------------------------------------------------------------
# Atmosphere (International Standard Atmosphere, troposphere)
# ---------------------------------------------------------------------------

_ISA_T0 = 288.15      # K, sea-level standard temperature
_ISA_P0 = 101325.0    # Pa, sea-level standard pressure
_ISA_LAPSE = 0.0065   # K/m
_R_AIR = 287.05287    # J/(kg K)
_ISA_RHO0 = _ISA_P0 / (_R_AIR * _ISA_T0)  # ~1.225 kg/m^3


def isa_density(pressure_alt_m: float, delta_isa_c: float = 0.0) -> float:
    """Air density (kg/m^3) at a given pressure altitude and ISA deviation.

    ``delta_isa_c`` is the temperature offset from ISA in degrees C (K).
    """
    t_isa = _ISA_T0 - _ISA_LAPSE * pressure_alt_m
    pressure = _ISA_P0 * (t_isa / _ISA_T0) ** (G / (_R_AIR * _ISA_LAPSE))
    temperature = t_isa + delta_isa_c
    return pressure / (_R_AIR * temperature)


def density_altitude_m(rho: float) -> float:
    """Invert ISA (standard temperature) to report a density altitude for rho."""
    ratio = rho / _ISA_RHO0
    exponent = (G / (_R_AIR * _ISA_LAPSE)) - 1.0
    t_ratio = ratio ** (1.0 / exponent)
    return (_ISA_T0 / _ISA_LAPSE) * (1.0 - t_ratio)


# ---------------------------------------------------------------------------
# Aerodynamics
# ---------------------------------------------------------------------------

def v_stall(weight_n: float, rho: float, wing_area_m2: float,
            cl_max: float, load_factor: float = 1.0) -> float:
    """1-g-generalized stall speed (m/s, true airspeed).

        V_s = sqrt( 2 n W / (rho S CL_max) )
    """
    return float(np.sqrt(2.0 * load_factor * weight_n /
                         (rho * wing_area_m2 * cl_max)))


# ---------------------------------------------------------------------------
# Aircraft parameter table (first-draft; tune against POH/AFM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Aircraft:
    name: str
    wing_area_m2: float
    mtow_kg: float
    empty_kg: float
    cl_max_clean: float
    cl_max_flap: float
    n_engines: int = 1


# Covers the bulk of GA fixed-wing LOC-I in the NTSB corpus. CL_max values are
# representative light-GA numbers (clean ~1.5, full-flap ~1.9-2.1) and are the
# single most important thing to calibrate against type POH data.
AIRCRAFT_TABLE: dict[str, Aircraft] = {
    "CESSNA 152":     Aircraft("Cessna 152", 15.0, 757, 490, 1.50, 2.00),
    "CESSNA 172":     Aircraft("Cessna 172", 16.2, 1111, 767, 1.55, 2.10),
    "CESSNA 182":     Aircraft("Cessna 182", 16.3, 1406, 880, 1.55, 2.05),
    "CESSNA 210":     Aircraft("Cessna 210", 16.2, 1723, 1050, 1.50, 1.95),
    "PIPER PA-28":    Aircraft("Piper PA-28 (Cherokee/Warrior)", 15.8, 1055, 640, 1.50, 1.95),
    "PIPER PA-28-235": Aircraft("Piper PA-28-235", 15.8, 1360, 730, 1.50, 1.95),
    "PIPER PA-18":    Aircraft("Piper PA-18 Super Cub", 16.6, 794, 460, 1.55, 2.05),
    "PIPER PA-32":    Aircraft("Piper PA-32 (Cherokee Six)", 16.6, 1633, 900, 1.50, 1.95),
    "BEECH A36":      Aircraft("Beechcraft A36 Bonanza", 16.8, 1656, 1080, 1.50, 1.95),
    "BEECH 35":       Aircraft("Beechcraft 35 Bonanza", 16.5, 1315, 860, 1.50, 1.95),
    "MOONEY M20":     Aircraft("Mooney M20", 15.5, 1157, 730, 1.45, 1.90),
    "CIRRUS SR22":    Aircraft("Cirrus SR22", 13.5, 1542, 1010, 1.45, 1.85),
    "CHAMPION 7AC":   Aircraft("Champion/Aeronca 7AC", 16.6, 544, 400, 1.55, 1.80),
    "GENERIC SEP":    Aircraft("Generic single-engine piston", 16.0, 1100, 700, 1.50, 1.95),
}


def lookup_aircraft(make: str, model: str) -> Aircraft:
    """Best-effort map from NTSB make/model strings to the parameter table.

    Deliberately conservative: returns GENERIC SEP when unmatched so the caller
    can flag low-confidence coverage rather than silently guess a specific type.
    """
    text = f"{make} {model}".upper()
    # Longest keys first so "PIPER PA-28-235" beats "PIPER PA-28".
    for key in sorted(AIRCRAFT_TABLE, key=len, reverse=True):
        if key == "GENERIC SEP":
            continue
        needle = key.replace("CESSNA ", "").replace("PIPER ", "") \
                    .replace("BEECH ", "").replace("MOONEY ", "") \
                    .replace("CHAMPION ", "").replace("CIRRUS ", "")
        if key in text or needle in text:
            return AIRCRAFT_TABLE[key]
    return AIRCRAFT_TABLE["GENERIC SEP"]


# ---------------------------------------------------------------------------
# Phase-conditioned priors over the unobserved state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhasePrior:
    """Priors over unobserved state, conditioned on flight phase.

    weight_frac_*   : loading as a fraction of MTOW (triangular lo/mode/hi).
    speed_ratio_*   : indicated airspeed as a multiple of the clean 1-g stall
                      speed (mean/sd of a truncated normal). The left tail is
                      what produces stalls; approach/maneuvering sit lower.
    load_factor_*   : |load factor| (mean/sd of a truncated normal, >=1). Pattern
                      and maneuvering turns pull this up (base-to-final skid).
    use_flap        : whether the flapped CL_max applies in this phase.
    """
    weight_frac_lo: float
    weight_frac_mode: float
    weight_frac_hi: float
    speed_ratio_mean: float
    speed_ratio_sd: float
    load_factor_mean: float
    load_factor_sd: float
    use_flap: bool


# Speed ratios are relative to clean V_s; approach/maneuvering carry the mass
# that dips toward stall. These are first-draft and meant to be calibrated.
PHASE_PRIORS: dict[str, PhasePrior] = {
    "takeoff":     PhasePrior(0.75, 0.95, 1.00, 1.25, 0.10, 1.05, 0.08, False),
    "initial_climb": PhasePrior(0.75, 0.95, 1.00, 1.22, 0.10, 1.08, 0.10, False),
    "climb":       PhasePrior(0.70, 0.90, 1.00, 1.30, 0.12, 1.05, 0.08, False),
    "cruise":      PhasePrior(0.65, 0.85, 1.00, 1.70, 0.20, 1.10, 0.15, False),
    "maneuvering": PhasePrior(0.65, 0.85, 1.00, 1.35, 0.18, 1.60, 0.45, False),
    "approach":    PhasePrior(0.65, 0.85, 0.98, 1.30, 0.12, 1.15, 0.18, True),
    "landing":     PhasePrior(0.65, 0.82, 0.98, 1.22, 0.12, 1.10, 0.12, True),
    "go_around":   PhasePrior(0.70, 0.88, 1.00, 1.20, 0.12, 1.12, 0.15, False),
}
DEFAULT_PHASE = "maneuvering"


def _truncated_normal(rng: np.random.Generator, mean: float, sd: float,
                      lo: float, hi: float, size: int) -> np.ndarray:
    out = rng.normal(mean, sd, size)
    # Reflect/clip rather than rejection-sample: adequate for a smooth prior.
    return np.clip(out, lo, hi)


def _triangular(rng: np.random.Generator, lo: float, mode: float, hi: float,
                size: int) -> np.ndarray:
    if hi <= lo:
        return np.full(size, lo)
    mode = min(max(mode, lo), hi)
    return rng.triangular(lo, mode, hi, size)


@dataclass
class StallRisk:
    p_stall: float
    ci_low: float
    ci_high: float
    density_altitude_m: float
    v_stall_nominal_ms: float  # at mode weight, n=1, clean
    aircraft: str
    phase: str
    n_samples: int


def p_stall(aircraft: Aircraft,
            pressure_alt_m: float,
            phase: str = DEFAULT_PHASE,
            delta_isa_c: float = 0.0,
            n_samples: int = 20000,
            seed: int = 0,
            prior: Optional[PhasePrior] = None) -> StallRisk:
    """Physics probability of stall given the observable operating point.

    Marginalizes unobserved weight, airspeed and load factor over the
    phase-conditioned priors. Returns the probability plus a bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    pr = prior or PHASE_PRIORS.get(phase, PHASE_PRIORS[DEFAULT_PHASE])
    rho = isa_density(pressure_alt_m, delta_isa_c)
    cl_max = aircraft.cl_max_flap if pr.use_flap else aircraft.cl_max_clean

    weight_kg = _triangular(rng, pr.weight_frac_lo, pr.weight_frac_mode,
                            pr.weight_frac_hi, n_samples) * aircraft.mtow_kg
    weight_kg = np.maximum(weight_kg, aircraft.empty_kg)
    weight_n = weight_kg * G
    load_factor = _truncated_normal(rng, pr.load_factor_mean, pr.load_factor_sd,
                                    1.0, 6.0, n_samples)

    # Clean 1-g reference stall speed at sea level sets the scale for the
    # airspeed prior (pilots fly a target IAS, roughly DA-independent).
    vs_ref_sl = np.sqrt(2.0 * weight_n /
                        (_ISA_RHO0 * aircraft.wing_area_m2 * aircraft.cl_max_clean))
    ias = _truncated_normal(rng, pr.speed_ratio_mean, pr.speed_ratio_sd,
                            0.6, 3.0, n_samples) * vs_ref_sl

    # Convert the flown IAS to TAS for the actual density before comparing to
    # the (true-airspeed) stall speed: TAS = IAS * sqrt(rho0 / rho).
    tas = ias * np.sqrt(_ISA_RHO0 / rho)
    vs = np.sqrt(2.0 * load_factor * weight_n /
                 (rho * aircraft.wing_area_m2 * cl_max))

    stalled = tas < vs
    p = float(stalled.mean())

    # Bootstrap CI over the sample.
    boot = rng.choice(stalled, size=(200, n_samples), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    w_mode_n = pr.weight_frac_mode * aircraft.mtow_kg * G
    vs_nominal = v_stall(w_mode_n, rho, aircraft.wing_area_m2,
                         aircraft.cl_max_clean, 1.0)

    return StallRisk(
        p_stall=p, ci_low=float(ci_low), ci_high=float(ci_high),
        density_altitude_m=density_altitude_m(rho),
        v_stall_nominal_ms=vs_nominal, aircraft=aircraft.name,
        phase=phase, n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Self-test / demo: the density-altitude risk surface for a Cessna 172
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ac = AIRCRAFT_TABLE["CESSNA 172"]
    print(f"Aircraft: {ac.name}  (S={ac.wing_area_m2} m^2, MTOW={ac.mtow_kg} kg)")
    print(f"Sea-level ISA density: {isa_density(0):.4f} kg/m^3\n")

    KT = 0.514444
    print("Sanity: clean 1-g V_stall at MTOW, sea level:")
    vs = v_stall(ac.mtow_kg * G, isa_density(0), ac.wing_area_m2, ac.cl_max_clean)
    print(f"  {vs:.1f} m/s = {vs / KT:.0f} kt  (POH Vs1 ~ 48 kt -> tune CL_max)\n")

    print("P(stall) risk surface over density altitude, maneuvering phase:")
    print(f"  {'press_alt_ft':>12} {'DA_ft':>8} {'P(stall)':>9} {'95% CI':>16}")
    for alt_ft in (0, 2000, 4000, 6000, 8000, 10000):
        alt_m = alt_ft * 0.3048
        r = p_stall(ac, alt_m, phase="maneuvering", delta_isa_c=15.0, seed=1)
        print(f"  {alt_ft:12d} {r.density_altitude_m/0.3048:8.0f} "
              f"{r.p_stall:9.3f}  [{r.ci_low:.3f}, {r.ci_high:.3f}]")

    print("\nCounterfactual illustration (what the binary KG cannot express):")
    lo = p_stall(ac, 0.0, phase="maneuvering", delta_isa_c=0.0, seed=2)
    hi = p_stall(ac, 8000 * 0.3048, phase="maneuvering", delta_isa_c=25.0, seed=2)
    print(f"  do(density_altitude: high hot -> sea level std):"
          f" P(stall) {hi.p_stall:.3f} -> {lo.p_stall:.3f}")
