# ACE-Graph extraction schema v3

This document is the human-readable companion to [`schema_v3.json`](schema_v3.json) — the JSON Schema enforced via vLLM `guided_json` constrained generation. The two must stay in sync; if you change one, change the other.

v3 closes three root-cause gaps from v1:

1. **Drops the `OPERATIONAL` HAEM catch-all** that absorbed 53 % of extracted events in v1, defeating the H/A/E/M ontology.
2. **Adds a closed `event_type` field** drawn from a 42-value vocabulary aligned to HFACS Tier-3 + CICTT — the Stage-3 aggregation node.
3. **Adds an explicit `cause_role` field** (primary / contributing / outcome / context) so cause attribution is first-class and supervisable against NTSB `Findings.Cause_Factor`.

Constrained generation enforces every closed vocabulary mechanically, so the model cannot invent strings.

## Three node kinds

### `Event` — atomic action or state change

Required fields: `id`, `kind="event"`, `trigger`, `haem`, `event_type`, `cause_role`, `phase_of_flight`, `severity`.

```json
{
  "id": "e1",
  "kind": "event",
  "trigger": "engine 2 suffered a loss of power",
  "trigger_span": [368, 401],
  "haem": "A",
  "event_type": "ENGINE_FAILURE",
  "cause_role": "primary",
  "phase_of_flight": "climb",
  "sociotechnical_level": 1,
  "severity": "critical",
  "actor_id": "ac1",
  "object_id": null,
  "time_anchor": "approximately five seconds later",
  "uca_type": null,
  "controller": null,
  "controlled_process": null
}
```

### `Entity` — persistent thing

```json
{ "id": "ac1", "kind": "entity", "type": "aircraft", "name": "Dassault Falcon 20", "identifier": "F-GPAD", "operator": null, "sociotechnical_level": 1 }
```

`type` ∈ aircraft | person | component | instrument | location | organization | procedure | document | vehicle | wildlife.

### `Condition` — state-like factor that is not an action

```json
{ "id": "c1", "kind": "condition", "type": "weather", "description": "icing conditions", "scope": "environmental", "sociotechnical_level": 1 }
```

`type` and `scope` use the closed vocabularies in the JSON Schema.

## The 42-value `event_type` vocabulary (Stage-3 nodes)

Grouped into seven families. Each value is the **mechanism** it names — pick the one closest to the causal mechanism, not the surface description.

**Pilot acts (HFACS Level 1) — 8**
DECISION_INAPPROPRIATE · CONTROL_INPUT_IMPROPER · PROCEDURE_NOT_FOLLOWED · PERCEPTION_FAILURE · SPATIAL_DISORIENTATION · PILOT_INCAPACITATION_OR_IMPAIRMENT · TRAINING_OR_CURRENCY_GAP · CREW_COORDINATION_FAILURE

**External operators — 2**
ATC_OR_DISPATCH_INADEQUATE · GROUND_PERSONNEL_ERROR

**Aircraft systems (HAEM-A) — 8**
ENGINE_FAILURE · FUEL_SYSTEM_ANOMALY · AIRFRAME_STRUCTURAL_FAILURE · CONTROL_SURFACE_ANOMALY · INSTRUMENT_OR_AVIONICS_FAILURE · LANDING_GEAR_ANOMALY · AUTOMATION_ANOMALY · INFLIGHT_FIRE_OR_SMOKE

**Environmental (HAEM-E) — 8**
ICING_ENCOUNTER · TURBULENCE_ENCOUNTER · THUNDERSTORM_OR_CONVECTIVE · LOW_VISIBILITY_OR_IMC · WIND_SHEAR_OR_GUST · BIRD_OR_WILDLIFE_STRIKE · TERRAIN_OR_OBSTACLE_PROXIMITY · RUNWAY_CONDITION_HAZARD

**Organizational (HAEM-M) — 3**
MAINTENANCE_INADEQUATE · ORG_OR_REGULATORY_INADEQUATE · DESIGN_DEFECT_LATENT

**Aerodynamic / flight-path states — 5**
STALL · LOSS_OF_CONTROL_INFLIGHT · LOSS_OF_CONTROL_GROUND · STRUCTURAL_OVERLOAD · ALTITUDE_DEVIATION_UNCONTROLLED

**Outcomes / terminal events — 8**
GROUND_IMPACT · WATER_IMPACT · MIDAIR_COLLISION · RUNWAY_EXCURSION_OR_OVERRUN · INFLIGHT_BREAKUP · SUCCESSFUL_RECOVERY · EMERGENCY_LANDING · INJURY_OR_FATALITY

Use `unknown` only as a last resort. The vocabulary is intentionally finite — Stage 3 aggregates causal edges across accidents at this level, so an explosion of subtypes would defeat aggregation.

### How to pick `event_type` — guidance for the model

When more than one fits, prefer the one closer to the causal **mechanism**, not the surface description:

- "Pilot pulled up sharply" → `CONTROL_INPUT_IMPROPER`, not a generic "maneuver".
- "Wing sheared off after exceeding Va" → two events: `STRUCTURAL_OVERLOAD` (cause) → `AIRFRAME_STRUCTURAL_FAILURE` (consequence).
- "Aircraft entered icing while continuing IFR climb" → `ICING_ENCOUNTER` (the environment event), not `ENGINE_FAILURE` (which is downstream).

### Decisions are events, not conditions

A common v1 failure was encoding a pilot's CHOICE as a *condition* instead of as an *event*. The world-state is the condition; the pilot's response to that state is the event.

| Narrative | Right | Wrong |
|---|---|---|
| "Pilot was warned of severe icing but continued the flight" | event `DECISION_INAPPROPRIATE` (cause_role: primary) + condition `icing` | weather condition only, no decision event |
| "Pilot pressed on with insufficient fuel reserves" | event `DECISION_INAPPROPRIATE` | condition only |
| "Pilot descended below MDA in IMC" | event `PROCEDURE_NOT_FOLLOWED` | condition only |

If the narrative cites a **go / continue / proceed** decision made AGAINST available warning information, it is `DECISION_INAPPROPRIATE` with `cause_role: primary`. The condition (icing/IMC/fatigue/defect) goes alongside it, not in its place.

## `cause_role` — explicit cause attribution

| Value | Meaning |
|---|---|
| `primary` | Narrative or NTSB Findings explicitly cite this as a cause of the accident ("the pilot's failure to...", "due to...", "the cause was..."). Usually 1–2 events per accident. |
| `contributing` | Listed as a contributing factor or "also contributed". |
| `outcome` | Consequence / effect (collision, injury, structural separation following overload). |
| `context` | Happened during the accident sequence but neither caused nor was caused by it ("the crew was in radio contact with ATC"). |

When NTSB structured `Findings` are available (joined into [`corpus_enriched.jsonl`](../../data/corpus/corpus_enriched.jsonl) via [Stage 0 root-1 fix](../../docs/2026-04-27-v3-schema-redesign.md)), they are the supervision target: anything flagged with `Cause_Factor='C'` should be `cause_role: primary` if the model extracts it.

## Edges — closed type set

Use only these 17 types. Participation (who performed/affected) is encoded via `actor_id` / `object_id` on the event, **not** as edges.

| Family | Types |
|---|---|
| event ↔ event causal/temporal | CAUSES · CONTRIBUTES_TO · ENABLES · PREVENTS · MITIGATES · TRIGGERS · DETECTS · RESPONDS_TO · PRECEDES · CONCURRENT · CO_OCCURS · IF_THEN · INHIBITS |
| event ↔ condition context | UNDER_CONDITION · INDUCED_CONDITION · MASKED_BY_CONDITION |
| entity ↔ entity | SUPERVISED_BY |

`PREVENTS` only when prevention SUCCEEDED. If a crew action was attempted but the worse outcome still occurred, that is NOT prevention — it is `RESPONDS_TO` plus a continuing `CAUSES` chain.

## STAMP UCA layer (optional)

For events with `haem="UCA"`, also fill `uca_type`, `controller`, `controlled_process`. See the JSON Schema for allowed values.

## AcciMap sociotechnical levels

| Level | Scope |
|---|---|
| 1 | Equipment & physical process (aircraft systems, weather, terrain) |
| 2 | Operational activities (flight crew, ATC controllers — front-line) |
| 3 | Technical / operational management (maintenance org, dispatch, training) |
| 4 | Company management (airline policy, scheduling, safety culture) |
| 5 | Regulators & industry bodies (FAA, EASA, ICAO, manufacturer SBs) |
| 6 | Government policy |

## Output shape

```json
{
  "nodes": [ {"id": "e1", "kind": "event", ...}, ... ],
  "edges": [ {"src": "e1", "dst": "e2", "type": "CAUSES", "evidence": "due to"} ]
}
```

## Invariants

- Every id referenced by an edge must exist in `nodes`.
- All closed-vocabulary fields are enforced via `guided_json`. Never invent a value.
- Aim for 4–15 events per narrative; cap at 20. Atomic events only.
- Edges are sparse (~0.7–1.5 per event), mostly event→event causal/temporal.
- Empty extraction is allowed: `{"nodes": [], "edges": []}` for narratives with no causal/temporal content.

## v1 vs v3 calibration delta

Measured on 2,000 NTSB records with structured supervision, scored by an LLM-as-judge against NTSB `Findings.Cause_Factor='C/F'` items:

| Metric | v1 | v3 |
|---|---:|---:|
| Recall (factors with ≥1 matching event) | 73.86 % | **80.52 %** |
| Precision (events matching a factor) | 21.31 % | **29.87 %** |
| Events / record | 7.45 | 4.84 |
| Schema violations (HAEM `OPERATIONAL`) | 53 % | **0 %** |

See [docs/2026-04-27-v3-schema-redesign.md](../../docs/2026-04-27-v3-schema-redesign.md) for the full delta and design history.
