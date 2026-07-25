"""Tier-1 physics model for the CARBURETOR_OR_INDUCTION_ICING node, fused with
the corpus (Tier-3) edge to ENGINE_FAILURE.

This is the first end-to-end demonstration of the model-data hybrid goal: a risk
number backed by *both* physics and data, so the counterfactual is explainable.

  physics  ->  P(carb ice forms | temp, dewpoint, power)     [published chart]
  data     ->  P(engine failure | carb ice)  = 0.667  (corpus edge, n=971)
           ->  P(serious/fatal  | carb ice)  = 0.227  (outcome layer, n=1456)
  fused    ->  P(ice-induced power loss | weather) = physics x data
  do()     ->  carb heat / warmer-drier air moves the physics term, and the
               whole risk moves with a legible reason.

Why carb icing is the clean first node
--------------------------------------
Its two physics inputs -- ambient temperature and dewpoint -- are recorded in
the NTSB ``events`` table for 99.6% of records, and the failure is a published
function of them. No latent instantaneous flight state (unlike STALL).

The physics
-----------
A float carburetor cools the intake charge by (a) the static-pressure drop
across the venturi/throttle plate and (b) latent heat of fuel vaporization. The
total drop is largest at low power (throttle nearly closed -> big pressure drop),
which is why the published chart shows "serious icing at descent power" zones.
If the charge is cooled below both its dewpoint and 0 degC, moisture condenses
and freezes on the throttle plate -> induction blockage -> power loss.

The chart is reproduced here by a transparent parameterization with two knobs --
the power-dependent venturi cooling ``dT(power)`` and the exposure constant
``K`` -- set to recover the chart's named zones. That is the physics absorbing
the empirical chart; the knobs are documented and adjustable.

Units: temperature/dewpoint in degrees Celsius; vapor pressure in hPa.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

# ---------------------------------------------------------------------------
# Corpus (Tier-3) constants -- sourced from the built causation KG.
#   causation_edges.csv : CARBURETOR_OR_INDUCTION_ICING -> ENGINE_FAILURE
#   outcome_layer.csv   : CARBURETOR_OR_INDUCTION_ICING severity
# Kept as named constants so the fusion is auditable; can be re-read from the
# CSVs via load_corpus_terms() to stay in sync.
# ---------------------------------------------------------------------------
P_ENGINE_FAILURE_GIVEN_ICE = 0.667   # P(dst|src); support 971, lift 2.27
P_SERIOUS_FATAL_GIVEN_ICE = 0.227    # outcome layer; n = 1456
BASELINE_SERIOUS_FATAL = 0.262       # corpus baseline P(serious/fatal)


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------

def sat_vapor_pressure(temp_c: float) -> float:
    """Saturation vapor pressure over water (hPa), Magnus/WMO coefficients."""
    return 6.112 * exp(17.62 * temp_c / (243.12 + temp_c))


def relative_humidity(temp_c: float, dewpoint_c: float) -> float:
    """Relative humidity in [0, 1] from temperature and dewpoint."""
    return min(1.0, sat_vapor_pressure(dewpoint_c) / sat_vapor_pressure(temp_c))


# Power-dependent total venturi + vaporization cooling (deg C). Larger at low
# power because the near-closed throttle plate produces a bigger pressure drop.
# These reproduce the chart's "serious at descent/glide power" behavior.
VENTURI_COOLING_C = {
    "takeoff": 12.0,
    "climb": 18.0,
    "cruise": 25.0,
    "descent": 32.0,   # glide / low power -- worst case
    "idle": 35.0,
}
DEFAULT_POWER = "cruise"

# Exposure constant for P(ice) = 1 - exp(-K * ice_index). Tuned so a saturated
# mid-teens ambient at descent power reads "serious" (P ~ 0.9).
_K_EXPOSURE = 0.60


@dataclass
class IcingRisk:
    p_ice: float           # physics: P(carb ice forms | weather, power)
    zone: str              # chart zone: serious / moderate / light / nil
    throat_temp_c: float   # cooled charge temperature at the throttle plate
    ice_index_hpa: float   # condensable, subfreezing vapor (hPa)
    rh: float
    power: str


def carb_ice_zone(p_ice: float) -> str:
    if p_ice >= 0.80:
        return "serious"
    if p_ice >= 0.40:
        return "moderate"
    if p_ice >= 0.10:
        return "light"
    return "nil"


def p_carb_icing(temp_c: float, dewpoint_c: float,
                 power: str = DEFAULT_POWER) -> IcingRisk:
    """Physics probability that carburetor ice forms, given the weather.

    Ice forms where the charge, cooled to the throttle-plate temperature, drops
    below both its dewpoint and 0 degC; the freezing (sub-0) portion of the
    condensed moisture is the ice-forming water.
    """
    dewpoint_c = min(dewpoint_c, temp_c)  # dewpoint cannot exceed temperature
    dt = VENTURI_COOLING_C.get(power, VENTURI_COOLING_C[DEFAULT_POWER])
    throat = temp_c - dt

    # Condensable vapor that reaches the sub-freezing region and freezes:
    # es(min(dewpoint, 0)) - es(throat), counted only when the throat is < 0.
    if throat < 0.0:
        upper = min(dewpoint_c, 0.0)
        ice_index = max(0.0, sat_vapor_pressure(upper) - sat_vapor_pressure(throat))
    else:
        ice_index = 0.0

    p_ice = 1.0 - exp(-_K_EXPOSURE * ice_index)
    return IcingRisk(
        p_ice=p_ice, zone=carb_ice_zone(p_ice), throat_temp_c=throat,
        ice_index_hpa=ice_index, rh=relative_humidity(temp_c, dewpoint_c),
        power=power,
    )


# ---------------------------------------------------------------------------
# Physics x data fusion -- the explainable hybrid risk number
# ---------------------------------------------------------------------------

@dataclass
class HybridIcingRisk:
    p_ice: float                 # physics term
    p_ice_engine_failure: float  # physics x P(engine failure | ice)
    p_ice_serious_fatal: float   # physics x P(serious/fatal | ice)
    risk_ratio_vs_baseline: float
    detail: IcingRisk


def hybrid_icing_risk(temp_c: float, dewpoint_c: float,
                      power: str = DEFAULT_POWER,
                      p_engfail_given_ice: float = P_ENGINE_FAILURE_GIVEN_ICE,
                      p_serious_given_ice: float = P_SERIOUS_FATAL_GIVEN_ICE
                      ) -> HybridIcingRisk:
    """Fuse the physics P(ice) with the corpus consequence terms.

    P(ice-induced power loss | weather) = P(ice | weather) x P(engfail | ice).
    The physics answers "will ice form here"; the data answers "if it forms,
    how often does it down the aircraft and how badly".
    """
    r = p_carb_icing(temp_c, dewpoint_c, power)
    p_engfail = r.p_ice * p_engfail_given_ice
    p_serious = r.p_ice * p_serious_given_ice
    return HybridIcingRisk(
        p_ice=r.p_ice, p_ice_engine_failure=p_engfail,
        p_ice_serious_fatal=p_serious,
        risk_ratio_vs_baseline=p_serious / BASELINE_SERIOUS_FATAL,
        detail=r,
    )


def load_corpus_terms(kg_dir: str = "event_extraction/out/causation_kg"):
    """Re-read the two Tier-3 terms from the built KG CSVs (keeps fusion honest).

    Returns (p_engfail_given_ice, p_serious_given_ice). Falls back to the module
    constants if the files are unavailable.
    """
    import csv
    import os
    factor = "CARBURETOR_OR_INDUCTION_ICING"
    p_ef, p_sf = P_ENGINE_FAILURE_GIVEN_ICE, P_SERIOUS_FATAL_GIVEN_ICE
    try:
        with open(os.path.join(kg_dir, "causation_edges.csv")) as f:
            for row in csv.DictReader(f):
                if row["src"] == factor and row["dst"] == "ENGINE_FAILURE":
                    p_ef = float(row["p_dst_given_src"])
                    break
        with open(os.path.join(kg_dir, "outcome_layer.csv")) as f:
            for row in csv.DictReader(f):
                if row["factor"] == factor:
                    p_sf = float(row["p_serious_or_fatal_given_factor"])
                    break
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return p_ef, p_sf


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Carburetor icing chart, reproduced (descent power) ===")
    print("  rows = dewpoint, cols = temperature (deg C); "
          "S=serious M=moderate L=light .=nil\n")
    temps = list(range(-10, 36, 3))
    dewpts = list(range(30, -11, -3))
    header = "  Td\\T " + "".join(f"{t:4d}" for t in temps)
    print(header)
    for td in dewpts:
        cells = []
        for t in temps:
            if td > t:
                cells.append("   -")  # dewpoint > temp is unphysical
                continue
            z = p_carb_icing(t, td, power="descent").zone
            cells.append(f"   {'S' if z=='serious' else 'M' if z=='moderate' else 'L' if z=='light' else '.'}")
        print(f"  {td:4d} " + "".join(cells))

    print("\n=== Canonical points (temp / dewpoint) ===")
    for t, td, label in [(13, 13, "saturated mid-teens (classic serious)"),
                         (20, 15, "warm, humid"),
                         (25, 20, "hot, humid"),
                         (5, 0, "cool, moist"),
                         (0, -8, "cold, drier"),
                         (28, 2, "hot, dry (nil)")]:
        c = p_carb_icing(t, td, "cruise")
        d = p_carb_icing(t, td, "descent")
        print(f"  T={t:3d} Td={td:3d} RH={c.rh*100:3.0f}%  "
              f"cruise: P={c.p_ice:.2f} ({c.zone:8s})  "
              f"descent: P={d.p_ice:.2f} ({d.zone:8s})   {label}")

    print("\n=== Hybrid risk (physics x data) + counterfactual ===")
    p_ef, p_sf = load_corpus_terms()
    print(f"  corpus terms: P(engine failure|ice)={p_ef:.3f}  "
          f"P(serious/fatal|ice)={p_sf:.3f}\n")
    base = hybrid_icing_risk(13, 12, "descent", p_ef, p_sf)
    print(f"  Scenario: OAT 13C, dewpoint 12C, descent power")
    print(f"    physics  P(carb ice forms)        = {base.p_ice:.3f}  "
          f"[{base.detail.zone}]")
    print(f"    x data   P(engine failure | ice)  = {p_ef:.3f}")
    print(f"    = hybrid P(ice-induced power loss)= {base.p_ice_engine_failure:.3f}")
    print(f"             P(serious/fatal via ice) = {base.p_ice_serious_fatal:.3f}"
          f"  ({base.risk_ratio_vs_baseline:.1f}x baseline)")

    cf = hybrid_icing_risk(13, 12, "descent", p_ef, p_sf)
    # do(apply carb heat): heat restores ~+30C to the intake, lifting the throat
    # temperature above freezing -> physics term collapses.
    heated = p_carb_icing(13 + 30, 12, "descent")
    p_heated = heated.p_ice * p_ef
    print(f"\n  do(apply carb heat, +30C intake): "
          f"P(carb ice) {base.p_ice:.3f} -> {heated.p_ice:.3f}, "
          f"hybrid power-loss risk {base.p_ice_engine_failure:.3f} -> {p_heated:.3f}")
    dry = hybrid_icing_risk(13, -5, "descent", p_ef, p_sf)
    print(f"  do(drier air, dewpoint 12 -> -5): "
          f"P(carb ice) {base.p_ice:.3f} -> {dry.p_ice:.3f}, "
          f"hybrid power-loss risk {base.p_ice_engine_failure:.3f} -> "
          f"{dry.p_ice_engine_failure:.3f}")
