"""Guard registry for the chain automaton (paper Sec. 3.4, Table tbl:guards).

Maps causation-graph transitions (src_factor, dst_factor) to the physics
guards that decide their per-case admissibility. Each guard exposes

    checkable(drivers)     -> the guard's drivers are recorded for this case
    availability(drivers)  -> Lambda_e in [0, 1], P(margin >= 0 | recorded);
                              0 means the transition is unavailable as
                              reconstructed (a violation if the chain uses it)

Drivers are the recorded observables of one case: temperature/dewpoint
(deg C), surface wind (kts), density altitude (ft), and the chain node's
phase_of_flight. Latent drivers (power setting, instantaneous speed/weight/
load factor, gust velocity) are marginalized inside the availability, per
Eq. (eq:avail): a guard reports zero only when no admissible latent state
enables the transition.

Guard semantics per transition:
  * -> CARBURETOR_OR_INDUCTION_ICING   w(T, Td, power) - w_thr  [g/m^3]
        power from the node phase when mapped, else the best case over
        power settings (availability zero only if no setting ices).
  * gust/turbulence -> STALL           V_s(n_gust) - V          [kts]
        Dryden gust intensity from the recorded surface wind; approach-
        regime airspeed prior; zero recorded wind gives availability 0.
  * wind shear -> STALL                V_s1g - (V - dU)         [kts]
        shear magnitude tied to the recorded surface wind.
  * gust/turbulence -> STRUCTURAL_*    n_gust - n_ult           [-]
        Dryden exceedance of the ultimate envelope from recorded wind.
  * -> STALL (any other parent)        V_s(W, n, rho) - V       [kts]
        phase-conditioned maneuvering prior (JSBSim aero); requires the
        jsbsim package, imported lazily; never zero, so it contributes
        coverage but not violations.
  * -> FUEL_EXHAUSTION_OR_STARVATION   t - t_end                [hours]
        drivers (fuel state, flight time) are not in the coded record;
        guarded but never checkable from recorded drivers.

The corrupted-guard control (paper Sec. 6) enters here: corrupt="negate"
flips every margin's sign before the availability is read, destroying the
physical signal while preserving the bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from physics.carb_icing_model import p_carb_icing
from physics.factor_models import (
    C172,
    dryden_sigma_w_fps,
    p_exceed_dryden,
    pratt_dn_per_ude,
)

# Factor types (v4 vocabulary) with a guard on their inbound transitions.
GUST_SOURCES = {"TURBULENCE_ENCOUNTER", "WIND_SHEAR_OR_GUST",
                "ADVERSE_WIND_CONDITION"}
STRUCT_DSTS = {"STRUCTURAL_OVERLOAD", "AIRFRAME_STRUCTURAL_FAILURE"}

# Phase -> carburetor power setting; None marginalizes over all settings.
PHASE_TO_POWER = {
    "taxi": "idle", "takeoff": "takeoff", "initial_climb": "climb",
    "climb": "climb", "cruise": "cruise", "descent": "descent",
    "approach": "descent", "landing": "idle", "go_around": "takeoff",
    "maneuvering": "cruise",
}
POWERS = ("takeoff", "climb", "cruise", "descent", "idle")

# Approach-regime airspeed for the wind-driven stall guards (kts CAS): the
# regime where surface wind is the relevant gust source.
V_APPROACH_KTS = 1.25 * C172.vs1g_kcas
ENCOUNTER_S = 120.0


@dataclass
class Drivers:
    """Recorded observables of one case (None = not recorded)."""
    temp_c: float | None = None
    dew_c: float | None = None
    wind_kts: float | None = None
    dens_alt_ft: float | None = None
    phase: str | None = None


def _sign(x: float, corrupt: str | None) -> float:
    return -x if corrupt == "negate" else x


@dataclass(frozen=True)
class Guard:
    name: str
    units: str
    checkable: Callable[[Drivers], bool]
    availability: Callable[[Drivers, str | None], float]


def _icing_checkable(d: Drivers) -> bool:
    return d.temp_c is not None and d.dew_c is not None


def _icing_availability(d: Drivers, corrupt: str | None) -> float:
    """Marginalizes the power setting: available iff any setting ices.

    The chain node's phase records where the power loss MANIFESTED, not
    where the ice accreted -- classically the ice forms at low power (taxi,
    descent) and bites at throttle-up -- so power is always latent here.
    """
    margins = [p_carb_icing(d.temp_c, d.dew_c, p).margin_gm3 for p in POWERS]
    return 1.0 if any(_sign(m, corrupt) >= 0 for m in margins) else 0.0


# Phases in which the recorded surface wind is a defensible gust source: the
# MIL-F-8785C relation sigma_w = 0.1 u20 is the low-altitude form, so the
# wind guards are checkable only on transitions the chain places near the
# ground (review 2026-09-14, comment 6).
LOW_ALTITUDE_PHASES = {"takeoff", "initial_climb", "go_around", "approach",
                       "final_approach", "landing"}


def _wind_checkable(d: Drivers) -> bool:
    return d.wind_kts is not None and d.phase in LOW_ALTITUDE_PHASES


def _gust_stall_availability(d: Drivers, corrupt: str | None) -> float:
    """Dryden crossing probability of the gust-stall margin zero."""
    n_margin = (V_APPROACH_KTS / C172.vs1g_kcas) ** 2 - 1.0
    u_crit = n_margin / pratt_dn_per_ude(C172, V_APPROACH_KTS)
    if corrupt == "negate":
        # sign-flipped margin is nonnegative exactly where the true one is
        # negative: below the crossing, probability 1 - Lambda
        return 1.0 - _gust_stall_availability(d, None)
    if dryden_sigma_w_fps(d.wind_kts) <= 0.0:
        return 0.0
    return p_exceed_dryden(u_crit, d.wind_kts, V_APPROACH_KTS * 1.68781,
                           ENCOUNTER_S)


def _shear_stall_availability(d: Drivers, corrupt: str | None) -> float:
    """Shear loss dU reaching V - V_s1g, dU bounded by the recorded wind."""
    du_crit = V_APPROACH_KTS - C172.vs1g_kcas
    margin = d.wind_kts - du_crit  # max shear loss vs required loss
    return 1.0 if _sign(margin, corrupt) >= 0 else 0.0


def _overload_availability(d: Drivers, corrupt: str | None) -> float:
    """Dryden crossing of the ultimate-envelope margin from recorded wind."""
    dn1 = pratt_dn_per_ude(C172, C172.vc_kcas)
    u_crit = (1.5 * C172.n_limit - 1.0) / dn1
    if corrupt == "negate":
        return 1.0 - _overload_availability(d, None)
    if dryden_sigma_w_fps(d.wind_kts) <= 0.0:
        return 0.0
    return p_exceed_dryden(u_crit, d.wind_kts, C172.vc_kcas * 1.68781,
                           ENCOUNTER_S)


@lru_cache(maxsize=512)
def _stall_prior_p(phase: str, alt_bucket_kft: int) -> float:
    from physics.jsbsim_stall import p_stall

    return p_stall("c172x", 1000.0 * alt_bucket_kft, phase).p_stall


def _stall_prior_availability(d: Drivers, corrupt: str | None) -> float:
    """Phase-prior stall availability (JSBSim aero, lazy import).

    Memoized on (phase, density altitude to the nearest 1000 ft); the Monte
    Carlo behind it is far too slow to rerun per chain transition.
    """
    phase = d.phase if d.phase in ("takeoff", "climb", "cruise",
                                   "maneuvering", "approach", "landing") \
        else "maneuvering"
    p = _stall_prior_p(phase, round((d.dens_alt_ft or 0.0) / 1000.0))
    return 1.0 - p if corrupt == "negate" else p


ICING_GUARD = Guard("carb_icing", "g/m^3", _icing_checkable,
                    _icing_availability)
GUST_STALL_GUARD = Guard("gust_stall", "kts", _wind_checkable,
                         _gust_stall_availability)
SHEAR_STALL_GUARD = Guard("shear_stall", "kts", _wind_checkable,
                          _shear_stall_availability)
OVERLOAD_GUARD = Guard("gust_overload", "n/n_ult", _wind_checkable,
                       _overload_availability)
STALL_PRIOR_GUARD = Guard("stall_prior", "kts",
                          lambda d: d.dens_alt_ft is not None,
                          _stall_prior_availability)
FUEL_GUARD = Guard("fuel_exhaustion", "hours", lambda d: False,
                   lambda d, c: 1.0)  # drivers never in the coded record


def guard_for_edge(src_type: str, dst_type: str) -> Guard | None:
    """The guard on transition src -> dst, or None if unguarded."""
    if dst_type == "CARBURETOR_OR_INDUCTION_ICING":
        return ICING_GUARD
    if dst_type == "STALL":
        if src_type == "WIND_SHEAR_OR_GUST":
            return SHEAR_STALL_GUARD
        if src_type in GUST_SOURCES:
            return GUST_STALL_GUARD
        return STALL_PRIOR_GUARD
    if dst_type in STRUCT_DSTS and src_type in GUST_SOURCES:
        return OVERLOAD_GUARD
    if dst_type == "FUEL_EXHAUSTION_OR_STARVATION":
        return FUEL_GUARD
    return None
