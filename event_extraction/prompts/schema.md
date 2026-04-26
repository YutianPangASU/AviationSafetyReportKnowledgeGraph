# Aviation Causality Knowledge Graph Schema (v2)

This schema replaces v1 (`schema.md`). The redesign synthesizes:

- **HAEM** (Human-Aircraft-Environment-Management) ontology — Chen et al. 2024
- **HFACS** (Human Factors Analysis and Classification System) — Wiegmann & Shappell, used by NTSB
- **STAMP** Unsafe Control Action types — Leveson, for automation / software failures
- **AcciMap** sociotechnical levels — Rasmussen, for cross-system-layer causality
- **NTSB / ASRS** phase-of-flight and occurrence vocabularies for compatibility

## Design principles

1. **Three node kinds, not just events.** A causal narrative needs *events* (actions, state changes), *entities* (aircraft, people, components), and *conditions* (background states like "icing", "fatigue", "IMC") that aren't events but causally matter.
2. **Closed vocabularies for type fields.** The model picks from a fixed list per field — keeps extraction consistent and queryable.
3. **HAEM as the event-type backbone.** Every event is tagged with one of four HAEM categories (H/A/E/M) plus a finer HFACS-aligned subtype.
4. **Edges typed by relation kind.** Event↔event causal/temporal edges use a 13-type set; event↔entity and event↔condition edges use participation roles.
5. **Sociotechnical level as metadata.** Each event/entity gets an AcciMap level (1–6), enabling queries like "which regulatory failures preceded this front-line event?"

## Three node kinds

### `Event`
Atomic action or state change explicitly mentioned in the narrative.

```json
{
  "id": "e1",
  "kind": "event",
  "trigger": "engine 2 suffered a loss of power",
  "trigger_span": [368, 401],

  "haem": "A",                          // H | A | E | M | OPERATIONAL | UCA
  "subtype": "engine_failure",          // see taxonomies_v2.json -> event_subtypes
  "phase_of_flight": "climb",           // see taxonomies_v2.json -> phases
  "sociotechnical_level": 1,            // 1..6 per AcciMap
  "severity": "serious",                // minor | serious | critical | catastrophic

  "actor": "engine 2",                  // entity that performs/undergoes the event
  "object": null,                       // entity affected (optional)
  "time_anchor": "approximately five seconds later",
  "location": null
}
```

### `Entity`
Persistent things — aircraft, people, components, organizations, locations.

```json
{
  "id": "ac1",
  "kind": "entity",
  "type": "aircraft",                   // see taxonomies_v2.json -> entity_types
  "name": "Dassault Falcon 20",
  "identifier": "F-GPAD",               // tail number, person role, part number
  "operator": "Aviation Défense Service",
  "sociotechnical_level": 1
}
```

### `Condition`
State-like factors that aren't events but causally matter (icing conditions, pilot fatigue, latent design flaw, ambiguous procedure).

```json
{
  "id": "c1",
  "kind": "condition",
  "type": "weather",                    // see taxonomies_v2.json -> condition_types
  "description": "icing conditions",
  "scope": "environmental",             // physiological | environmental | mechanical | procedural | cognitive | regulatory
  "sociotechnical_level": 1
}
```

## Edge kinds

### Event ↔ Event (causal / temporal — the core of the KG)

| Type | Meaning |
|---|---|
| `CAUSES` | Direct causation explicitly asserted |
| `CONTRIBUTES_TO` | Partial / contributing factor (one of several causes) |
| `ENABLES` | Necessary precondition; without it, the next event couldn't occur |
| `PREVENTS` | Successfully stopped the next event |
| `MITIGATES` | Reduced severity but didn't fully prevent |
| `TRIGGERS` | Sudden-onset cause (distinguishes proximate from gradual) |
| `DETECTS` | Observation/perception event that drives a response |
| `RESPONDS_TO` | Reactive crew/system action against a preceding event |
| `PRECEDES` | Temporal order only, no causal claim |
| `CONCURRENT` | Overlap in time |
| `CO_OCCURS` | Happen together, no precedence claim |
| `IF_THEN` | Conditional dependency (counterfactual) |
| `INHIBITS` | Reduces likelihood of next event without fully preventing |

### Event ↔ Entity (participation roles)

| Type | Meaning |
|---|---|
| `PERFORMED_BY` | Who/what initiated the event |
| `AFFECTED_BY` | Who/what underwent the event |
| `INVOLVES` | Generic participation |
| `AT_LOCATION` | Where it occurred |
| `USING_INSTRUMENT` | Tool/equipment used |
| `OBSERVED_BY` | Detector role (often crew, ATC, sensor) |

### Event ↔ Condition (context)

| Type | Meaning |
|---|---|
| `UNDER_CONDITION` | Event happened in the presence of this condition |
| `INDUCED_CONDITION` | Event created or exposed this condition |
| `MASKED_BY_CONDITION` | Condition prevented detection of the event |

### Entity ↔ Entity (structural)

| Type | Meaning |
|---|---|
| `OPERATES` | Operator → aircraft, pilot → aircraft |
| `PART_OF` | Engine → aircraft, runway → airport |
| `LOCATED_AT` | Aircraft → airport |
| `CERTIFIED_BY` | Aircraft → manufacturer / regulator |
| `SUPERVISED_BY` | Person → org |

## STAMP layer (optional, only when applicable)

For events of `haem = UCA` (Unsafe Control Action), add:

```json
{
  "uca_type": "control_not_provided",   // not_provided | provided_unnecessarily | wrong_timing | wrong_duration
  "controller": "pilot",                 // who/what should have controlled
  "controlled_process": "engine power",
  "safety_constraint": "engine power must remain above stall threshold during climb",
  "process_model_gap": "pilot did not realize anti-ice activation degraded thrust"
}
```

This gives you a clean systems-thinking layer on top of the causal KG without forcing every event through the STAMP framework.

## AcciMap sociotechnical levels

| Level | Scope | Examples |
|---|---|---|
| 1 | Equipment & physical process | Aircraft systems, weather, terrain |
| 2 | Operational activities | Flight crew acts, ATC clearances |
| 3 | Technical / operational management | Maintenance org, dispatch, training depts |
| 4 | Company management | Airline policy, scheduling, safety culture |
| 5 | Regulators & industry bodies | FAA, EASA, ICAO, manufacturer SBs |
| 6 | Government policy | Funding, deregulation, treaty changes |

## Invariants

- Every id referenced by an edge MUST exist in `nodes`.
- `trigger_span` offsets MUST be valid character indices into the input narrative.
- `haem`, `subtype`, `phase_of_flight`, `severity`, `type`, `scope` MUST come from the controlled vocabularies in `taxonomies_v2.json`. If no value fits, use `unknown`.
- Use closed-vocab `unknown` sparingly — prefer the closest-fitting category.
- Keep event count modest: 4–15 per narrative is the sweet spot. Cap at 20.
- Empty extraction is allowed: `{"nodes": [], "edges": []}` if narrative carries no causal/temporal content.

## Output shape

```json
{
  "nodes": [
    {"id": "e1", "kind": "event", ...},
    {"id": "ac1", "kind": "entity", ...},
    {"id": "c1", "kind": "condition", ...}
  ],
  "edges": [
    {"src": "e1", "dst": "e2", "type": "CAUSES", "evidence": "due to"},
    {"src": "e1", "dst": "ac1", "type": "AFFECTED_BY", "evidence": null},
    {"src": "e1", "dst": "c1", "type": "UNDER_CONDITION", "evidence": "in icing conditions"}
  ]
}
```
