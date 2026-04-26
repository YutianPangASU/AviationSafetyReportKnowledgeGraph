# Event Extraction Schema

The pilot extractor asks the model to read one aviation safety narrative and
return a JSON object with two arrays: `events` (atomic actions / state
changes mentioned in the text) and `relations` (temporal and causal edges
between them).

This schema is deliberately minimal. Revise after seeing pilot output.

## Event

```json
{
  "id": "e1",                          // short local id, unique within the narrative
  "trigger": "engine lost power",      // verbatim verb phrase from the text
  "trigger_span": [start, end],        // character offsets in the narrative
  "actor": "engine 2",                 // entity that performs or undergoes the event
  "object": null,                      // entity affected (optional)
  "phase_of_flight": "climb",          // one of: taxi, takeoff, climb, cruise, descent, approach, landing, rollout, ground, unknown
  "time_anchor": "after selecting engine 2 anti-ice ON",  // free-text temporal reference if any
  "location": null                     // free-text location if explicit
}
```

## Relation

```json
{
  "src": "e1",                         // source event id
  "dst": "e2",                         // target event id
  "type": "PRECEDES",                  // PRECEDES | CAUSES | CONTRIBUTES_TO | ENABLES | PREVENTS | CONCURRENT
  "evidence": "shortly afterwards"     // verbatim connective from the text if any
}
```

## Constraints

- All event ids referenced in `relations` MUST exist in `events`.
- `trigger_span` offsets MUST correspond to characters in the input.
- Use at most ~15 events per narrative; prefer higher-level summary events over
  sentence-by-sentence transcription.
- If the narrative carries no meaningful causal chain, return
  `{"events": [], "relations": []}`.

## Phase of flight

Canonical values (lowercase): `taxi`, `takeoff`, `climb`, `cruise`, `descent`,
`approach`, `landing`, `rollout`, `ground`, `unknown`.

## Relation semantics

| Type | Meaning |
|---|---|
| `PRECEDES` | `src` happened before `dst` in time, no causal claim |
| `CAUSES` | `src` is asserted as the direct cause of `dst` |
| `CONTRIBUTES_TO` | `src` is a contributing factor to `dst` (not sole cause) |
| `ENABLES` | `src` created the conditions that allowed `dst` |
| `PREVENTS` | `src` stopped or would have stopped `dst` |
| `CONCURRENT` | `src` and `dst` overlap in time |
