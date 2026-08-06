# Extraction Task — Gold-496 Evaluation

You are performing causal chain extraction from ONE full NTSB investigation
report, for a controlled evaluation against the investigation's own findings.
Follow the extraction specification below exactly. The specification is the
same v4 schema used across this project; do not deviate from it.

## Procedure

1. Read the input file you were given (the report narrative; the
   investigation's conclusions have been withheld from it).
2. Build the causal chain per the specification below, based ONLY on the
   narrative you read. Do not use outside knowledge of this accident beyond
   general aviation domain knowledge; extract what the text supports.
3. Write the single JSON object to the output path you were given, using the
   Write tool. The file must contain ONLY the JSON object (no markdown fences,
   no commentary).
4. Reply with one line: `DONE <report_number> nodes=<n>` where `<n>` is the
   number of chain nodes you wrote.

If the input file is unreadable or empty, reply `FAIL <report_number> <reason>`
and write nothing.

## Extraction Specification (v4)

Read ONE accident or incident narrative and extract its CAUSAL CHAIN — the
ordered sequence of causally connected factors an investigator would cite to
answer "what caused this outcome, and through what sequence?"

The output is a single JSON object: `{"chain": [...], "outcome_severity": "..."}`.

### The chain

- One node per causally relevant factor, ordered by time / causal precedence:
  earliest factor first, terminal outcome last.
- Each node is an object:
  `{"idx": <int>, "class": "...", "factor_type": "...", "label": "<short free text>",
    "trigger": "<short quote/paraphrase from the narrative>",
    "cause_role": "...", "caused_by": [{"src": <int>, "strength": "...", "evidence": "<quote>"}]}`
- `idx` is the node's 0-based position in the array and MUST equal it.
- `caused_by` lists this node's causal parents. Every `src` MUST be STRICTLY
  LESS than the node's own `idx` — you may only point backward to earlier
  nodes, never forward. Root causes and ambient conditions have `caused_by: []`.
- `strength` is `direct` (the parent mechanistically produced this factor:
  "due to", "resulted in", "caused") or `contributing` (made it more likely or
  worse: "a factor was", "contributed to").
- A crew RESPONSE to a failure (switching tanks after power loss, a go-around
  after a bounced landing) is simply a later event whose `caused_by` points at
  the failure that prompted it.

### Node classes

- `condition` — a pre-existing or ambient STATE: fuel exhaustion, carburetor
  icing conditions, dark night, gusty crosswind, latent defect, high density
  altitude. Not an action.
- `event` — an ACTION or state change in the sequence: a decision, a control
  input, a system failure, a weather encounter, a stall.
- `outcome` — a terminal consequence: impact, collision, runway excursion,
  injury. An outcome node may only be cited as `src` by OTHER outcome nodes
  (impact -> injury is fine; outcome -> event is forbidden).

### The factor_type vocabulary (57 values)

Causal events, seven families:
- Pilot acts (8): DECISION_INAPPROPRIATE, CONTROL_INPUT_IMPROPER,
  PROCEDURE_NOT_FOLLOWED, PERCEPTION_FAILURE, SPATIAL_DISORIENTATION,
  PILOT_INCAPACITATION_OR_IMPAIRMENT, TRAINING_OR_CURRENCY_GAP,
  CREW_COORDINATION_FAILURE
- External operators (2): ATC_OR_DISPATCH_INADEQUATE, GROUND_PERSONNEL_ERROR
- Aircraft systems (9): ENGINE_FAILURE, FUEL_SYSTEM_ANOMALY,
  AIRFRAME_STRUCTURAL_FAILURE, CONTROL_SURFACE_ANOMALY,
  INSTRUMENT_OR_AVIONICS_FAILURE, LANDING_GEAR_ANOMALY, AUTOMATION_ANOMALY,
  INFLIGHT_FIRE_OR_SMOKE, OTHER_SYSTEM_FAILURE (hydraulic, electrical,
  brakes, pneumatic...)
- Environmental encounters (8): ICING_ENCOUNTER, TURBULENCE_ENCOUNTER,
  THUNDERSTORM_OR_CONVECTIVE, LOW_VISIBILITY_OR_IMC, WIND_SHEAR_OR_GUST,
  BIRD_OR_WILDLIFE_STRIKE, TERRAIN_OR_OBSTACLE_PROXIMITY,
  RUNWAY_CONDITION_HAZARD
- Organizational (3): MAINTENANCE_INADEQUATE, ORG_OR_REGULATORY_INADEQUATE,
  DESIGN_DEFECT_LATENT
- Aerodynamic states (5): STALL, LOSS_OF_CONTROL_INFLIGHT,
  LOSS_OF_CONTROL_GROUND, STRUCTURAL_OVERLOAD, ALTITUDE_DEVIATION_UNCONTROLLED

Conditions (12): FUEL_EXHAUSTION_OR_STARVATION, FUEL_CONTAMINATION,
CARBURETOR_OR_INDUCTION_ICING, ADVERSE_WIND_CONDITION, HIGH_DENSITY_ALTITUDE,
DARK_NIGHT_OR_LOW_LIGHT, MOUNTAINOUS_OR_RISING_TERRAIN,
UNSUITABLE_TERRAIN_FOR_FORCED_LANDING, AIRCRAFT_WEIGHT_OR_BALANCE_OUT_OF_LIMITS,
LATENT_MECHANICAL_DEFECT, LOW_ALTITUDE_OPERATION,
PILOT_FATIGUE_OR_PHYSIOLOGICAL_STATE

Outcomes (9): GROUND_IMPACT, WATER_IMPACT, MIDAIR_COLLISION, GROUND_COLLISION,
RUNWAY_EXCURSION_OR_OVERRUN, INFLIGHT_BREAKUP, EMERGENCY_LANDING,
SUCCESSFUL_RECOVERY, INJURY_OR_FATALITY

Use `unknown` only as a last resort.

### How to pick factor_type

Prefer the value closest to the causal MECHANISM, not the surface description.
"Pilot pulled up sharply" is CONTROL_INPUT_IMPROPER.

Condition vs event disambiguation:
- The engine quitting is an ENGINE_FAILURE event; the fuel state that caused
  it is a FUEL_EXHAUSTION_OR_STARVATION condition. Both nodes, linked.
- Airframe ice accumulating in flight is an ICING_ENCOUNTER event; carburetor
  ice is a CARBURETOR_OR_INDUCTION_ICING condition.
- A sudden gust/shear on short final is a WIND_SHEAR_OR_GUST event; a steady
  crosswind/tailwind is an ADVERSE_WIND_CONDITION condition.

DECISIONS are events, not conditions. The world-state is the condition; the
pilot's response to it is the event. A go/continue/proceed decision made
AGAINST available warning information is DECISION_INAPPROPRIATE with
cause_role primary.

### What NOT to extract

Do not emit boilerplate that plays no causal role: "visual meteorological
conditions prevailed", "a flight plan was filed", registration/certification
recitals, "the pilot was in radio contact". Every node should cause something,
be caused by something, or be a root cause / terminal outcome of the chain.
A chain of 3-15 well-connected nodes beats a padded one. These are long
transport-category investigation reports: focus on the accident sequence and
the investigation's causal threads, not on every procedural detail.

### cause_role

- `primary` — the narrative cites it as a cause ("the cause was..."). Usually
  1-3 nodes.
- `contributing` — cited as a factor, or a causal intermediate in the chain.
- `outcome` — consequences; use for all class=outcome nodes.
- `context` — rare; only when a factor must be recorded but is non-causal.

### outcome_severity

Highest consequence in the record: fatal > serious_injury > minor_injury >
aircraft_damage_only > no_damage_or_injury. Use `unknown` only if the
narrative truly does not say.

### Rules

1. 3-15 nodes typical; hard cap 24. Atomic factors only — split compound
   statements. EXCEPTION: never chain two nodes of the SAME factor_type.
   Merge repeated occurrences of the same factor into ONE node.
2. Strict back-reference ordering: no caused_by src >= its node's idx.
3. Outcome nodes may only be cited as src by other outcome nodes.
4. If the narrative has no causal content, write
   `{"chain": [], "outcome_severity": "unknown"}`.
