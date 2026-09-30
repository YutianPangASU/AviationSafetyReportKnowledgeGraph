"""A rung-three counterfactual on named accidents: the probability of
necessity of the carburetor-heat omission (review 2026-09-14, comment 4).

Everything else in the framework is interventional (rung two). Here the
three steps of a counterfactual are carried out on individual accidents:

  abduction   the icing mechanism is the structural equation
              ICE = 1{ U <= g(gamma(T, Td)) }, with U uniform on [0, 1] the
              exogenous noise of the calibrated link g (calibrate_physics.py)
              and gamma the condensable-water margin. Observing the record
              fixes (T, Td) and the outcome; the posterior of U is the part
              of [0, 1] consistent with what was observed.
  action      do(heat): the intake is warmed by 30 C, so the margin becomes
              gamma' = gamma(T + 30, Td) and the threshold g(gamma').
  prediction  the same U is pushed through the modified equation.

Under this monotone coupling the counterfactual icing probability given that
icing occurred is g(gamma') / g(gamma), so the probability of necessity of
the heat omission for the icing is

    PN_ice = 1 - g(gamma') / g(gamma).

For the power loss, the engine-failure node is the leaky noisy-OR of
ef_risk_model.py: EF = (ICE and V) or O, with V the ice-to-engine-failure
noise (strength p_ice) and O the union of the other fitted parents at their
base rates and the leak. Two evidence sets are reported:

    given the weather, the omission and the power loss:
        PN_EF = (1 - P_O) p_ice [g(gamma) - g(gamma')] / P(EF | gamma)
    given, in addition, that icing occurred (the chain names it):
        PN_EF|ice = (1 - P_O) p_ice [g(gamma) - g(gamma')]
                    / ( g(gamma) [1 - (1 - p_ice)(1 - P_O)] )

Bounds without the monotonicity assumption follow Tian and Pearl (2000):
    max(0, 1 - P(EF | do(heat)) / P(EF | no heat)) <= PN
        <= min(1, P(no EF | do(heat)) / P(EF | no heat)).

The margins are evaluated at descent power, the setting the calibrated link
was fitted on. Output: physics/out/probability_of_necessity.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing  # noqa: E402
from physics.violation_rate import build_driver_index  # noqa: E402

CAL = "physics/out/calibration.json"
EF = "physics/out/ef_risk.json"
OUT = "physics/out/probability_of_necessity.json"
ICE = "CARBURETOR_OR_INDUCTION_ICING"
HEAT_C = 30.0

# Named accidents whose chains carry carburetor icing, a heat omission or
# delay, and a power loss (see per_accident_chains.jsonl), and whose NTSB
# probable cause names carburetor icing or the late use of carburetor heat.
# 20190119X05628 was dropped on 2026-09-30: its probable cause is a power
# loss for undetermined reasons, and the NTSB could not place its weather
# on the icing chart.
CASES = {
    "20001212X19636": "delayed carburetor heat at glide power, partial power loss, "
                      "forced landing to a grass strip",
    "20141208X13517": "carburetor heat off at low power in humid subfreezing air, total "
                      "power loss on short final, collision with rising terrain "
                      "(the appendix example)",
    "20020225X00252": "delayed carburetor heat, power loss in cruise, forced landing",
    "20200427X62554": "carburetor heat turned off after run-up, ice accumulated on the "
                      "ground, power loss on initial climb",
}


def main() -> None:
    cal = json.load(open(CAL))["factors"][ICE]
    xs = np.array(cal["curve"]["score"])
    ps = np.array(cal["curve"]["p"])

    def g(margin: float) -> float:
        return float(np.interp(margin, xs, ps))

    ef = json.load(open(EF))
    leak = ef["leak"]
    p_ice = next(r["p"] for r in ef["factors"] if r["factor"] == ICE)
    others = [(r["base_rate"], r["p"]) for r in ef["factors"] if r["factor"] != ICE]
    p_not_o = (1.0 - leak) * float(np.prod([1.0 - b * p for b, p in others]))
    p_o = 1.0 - p_not_o

    drv = build_driver_index()
    out = {"heat_intake_warming_C": HEAT_C, "p_ice_to_ef": p_ice, "P_other_causes": round(p_o, 4),
           "cases": {}}
    for ev_id, desc in CASES.items():
        d = drv.get(ev_id)
        if d is None or d.temp_c is None:
            out["cases"][ev_id] = {"description": desc, "note": "temperature and dewpoint not recorded"}
            continue
        r0 = p_carb_icing(d.temp_c, d.dew_c, "descent")
        r1 = p_carb_icing(d.temp_c + HEAT_C, d.dew_c, "descent")
        g0, g1 = g(r0.margin_gm3), g(r1.margin_gm3)
        pn_ice = 1.0 - g1 / g0 if g0 > 0 else float("nan")
        p_ef_x = 1.0 - (1.0 - g0 * p_ice) * p_not_o           # no heat
        p_ef_x1 = 1.0 - (1.0 - g1 * p_ice) * p_not_o          # do(heat)
        pn_ef = p_not_o * p_ice * (g0 - g1) / p_ef_x
        pn_ef_ice = (p_not_o * p_ice * (g0 - g1)
                     / (g0 * (1.0 - (1.0 - p_ice) * p_not_o)))
        lower = max(0.0, 1.0 - p_ef_x1 / p_ef_x)
        upper = min(1.0, (1.0 - p_ef_x1) / p_ef_x)
        out["cases"][ev_id] = {
            "description": desc,
            "recorded": {"T_C": round(d.temp_c, 1), "Td_C": round(d.dew_c, 1)},
            "chart_pi_no_heat": round(r0.p_ice, 4), "chart_pi_heat": round(r1.p_ice, 4),
            "margin_no_heat_gm3": round(r0.margin_gm3, 4), "margin_heat_gm3": round(r1.margin_gm3, 4),
            "g_no_heat": round(g0, 4), "g_heat": round(g1, 4),
            "PN_icing_given_icing_observed": round(pn_ice, 3),
            "P_EF_no_heat": round(p_ef_x, 4), "P_EF_do_heat": round(p_ef_x1, 4),
            "PN_power_loss_given_weather_and_power_loss": round(pn_ef, 3),
            "PN_power_loss_given_icing_and_power_loss": round(pn_ef_ice, 3),
            "tian_pearl_bounds_power_loss": [round(lower, 3), round(upper, 3)],
        }
    os.makedirs("physics/out", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
