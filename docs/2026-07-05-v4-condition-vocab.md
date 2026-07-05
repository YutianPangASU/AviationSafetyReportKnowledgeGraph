# v4 Condition Vocabulary — Derivation Memo

**Date:** 2026-07-05
**Input:** 106,035 condition nodes in `full_corpus_v3.jsonl` (free-text `description` + 10 coarse `type` values), clustered by top-frequency descriptions.
**Output:** the 12 condition values in [`schema_v4.json`](../event_extraction/prompts/schema_v4.json)'s `factor_type` enum.

## Why conditions get closed-vocabulary treatment in v4

v3 aggregated conditions by their coarse `type` (`mechanical`, `environmental`, ...), which collapsed causally distinct factors (fuel exhaustion, carb ice, latent defects — all `mechanical`) into single hub nodes and destroyed chain structure in the aggregate KG. v4 promotes the ~12 conditions that actually recur in the corpus to first-class causal factors, at the same granularity as the 42-value event vocabulary.

## Cluster → enum mapping

| v4 condition value | v3 description cluster (count) |
|---|---|
| `FUEL_EXHAUSTION_OR_STARVATION` | "fuel exhaustion" (2,264), "fuel starvation" (745) |
| `CARBURETOR_OR_INDUCTION_ICING` | "carburetor ice accumulation" (541), "carburetor icing" (196), "conditions conducive to carburetor icing" (169), "carburetor icing conditions" (131) |
| `ADVERSE_WIND_CONDITION` | "crosswind" (318), "tailwind" (222), "gusty wind conditions" (205), "crosswind with gusts" (177), "gusty crosswind" (149+139), "gust of wind" (148), "quartering tailwind" (117), "downdraft" (115), "crosswind conditions" (123) |
| `UNSUITABLE_TERRAIN_FOR_FORCED_LANDING` | "lack of suitable terrain for forced landing" (358), "unsuitable terrain for forced landing" (256) |
| `HIGH_DENSITY_ALTITUDE` | "high density altitude" (307) |
| `MOUNTAINOUS_OR_RISING_TERRAIN` | "mountainous terrain" (285), "rising terrain" (113) |
| `LOW_ALTITUDE_OPERATION` | "low altitude" (272), "low altitude maneuvering" (177) |
| `DARK_NIGHT_OR_LOW_LIGHT` | "dark night conditions" (159), "dark night" (140), "night conditions" (129) |
| `FUEL_CONTAMINATION` | "water contamination in fuel system" (202) |
| `LATENT_MECHANICAL_DEFECT` | v3 `type` histogram: "latent_defect" (840); pre-failure wear/defect descriptions |
| `AIRCRAFT_WEIGHT_OR_BALANCE_OUT_OF_LIMITS` | v3 `type` histogram: "weight" (370); overweight/CG descriptions |
| `PILOT_FATIGUE_OR_PHYSIOLOGICAL_STATE` | v3 `type` histogram: "physiological"/"medical" (~1,200 combined); fatigue/illness/hypoxia descriptions |

Deliberately **not** added:
- "visual meteorological conditions" / "vmc" (~1,200 combined) — non-causal boilerplate; the v4 prompt instructs omission.
- "imc" / "night vmc" ceilings — covered by the existing `LOW_VISIBILITY_OR_IMC` factor (class=condition when ambient).
- "engine power loss" phrasings (~700) — those are `ENGINE_FAILURE` **events** mislabeled as conditions in v3; the v4 prompt's condition-vs-event rules address this.
- "spatial disorientation" (219) — the `SPATIAL_DISORIENTATION` event exists.
- "expired medical certificate" (113) — folded into `TRAINING_OR_CURRENCY_GAP` guidance (qualification lapse), too rare to earn a slot.

## Other vocabulary changes in v4 (drift-informed)

From the v3 OOV analysis (`event_extraction/out/validation_v3.summary.json`):
- **`OTHER_SYSTEM_FAILURE`** (event, aircraft-systems family) — the model repeatedly invented `HYDRAULIC_SYSTEM_FAILURE` (124), `BRAKE_FAILURE` (85), `ELECTRICAL_SYSTEM_FAILURE` (59), `BRAKE_ANOMALY` (49); v3 had no home for non-engine non-gear system failures.
- **`GROUND_COLLISION`** (outcome) — `GROUND_COLLISION` was the #1 invented event type (443); v3 only offered `GROUND_IMPACT` (terrain) and `MIDAIR_COLLISION`, leaving taxi/ground collisions with objects or other aircraft (CICTT GCOL) homeless.
- **`phase_of_flight`** gains `en_route` (9,354 OOV instances), `taxi` (8,077), `post_impact` (5,020 as "impact"/"post-impact") — the three phases the model most wanted and v3 lacked.

Final v4 `factor_type` count: 35 causal events + 12 conditions + 9 outcomes + `unknown` = **57**.
