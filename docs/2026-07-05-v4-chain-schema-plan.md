# v4 Chain-Based Extraction → Causation KG — Execution Plan

**Date:** 2026-07-05
**Objective:** Re-target Stage-1 extraction from free-form graphs to **ordered causal chains**, so the aggregate causation KG is clean, acyclic, and directly consumable by downstream counterfactual analysis and risk calculation.
**Hardware:** existing Qwen3.6-35B-A3B via vLLM (`event_extraction/scripts/serve_qwen.sh`), same GPUs as the v3 run.

---

## Design changes, v3 → v4

| | v3 (current) | v4 (proposed) |
|---|---|---|
| Output shape | `nodes[]` + `edges[]` free graph | single ordered `chain[]`; causality via per-node `caused_by` back-references |
| Edge vocabulary | 17 types (73 % `CAUSES`, several drift/inverted) | 2 strengths: `direct` / `contributing` (maps to NTSB `Cause_Factor` C/F). Temporal order = list order |
| Cycles / inverted edges | possible (`RESPONDS_TO` inverted ~16 k times) | impossible by construction — `caused_by` may only reference **earlier** indices; validated post-hoc with retry |
| Conditions | 10 coarse types (`mechanical` hub destroys chains) | first-class causal factors with closed vocab (~14 values), same granularity as events |
| Outcomes | ordinary nodes → `GROUND_IMPACT` hub (20.5 % of events) | `class: "outcome"` nodes, terminal-only; plus record-level `outcome_severity`. Excluded from the causal factor set downstream |
| Structured supervision | never in prompt (despite design intent) | grounded track: `Findings` + `seq_of_events` in prompt as skeleton |
| Entities / STAMP / AcciMap / haem / per-event severity | required fields, mostly unused downstream | dropped (haem + family derivable from `factor_type` via lookup; per-event severity replaced by record-level outcome) |

## v4 schema sketch

One record produces:

```json
{
  "chain": [
    {"idx": 0, "class": "condition", "factor_type": "MAINTENANCE_INADEQUATE",
     "trigger": "inadequate maintenance of fuel selector", "phase_of_flight": "preflight",
     "cause_role": "contributing", "caused_by": []},
    {"idx": 1, "class": "condition", "factor_type": "FUEL_STARVATION",
     "trigger": "fuel selector positioned between tanks", "phase_of_flight": "climb",
     "cause_role": "primary", "caused_by": [{"src": 0, "strength": "contributing"}]},
    {"idx": 2, "class": "event", "factor_type": "ENGINE_FAILURE",
     "trigger": "engine lost power", "phase_of_flight": "climb",
     "cause_role": "primary", "caused_by": [{"src": 1, "strength": "direct"}]},
    {"idx": 3, "class": "outcome", "factor_type": "WATER_IMPACT",
     "trigger": "forced landing in a lake", "phase_of_flight": "emergency_descent",
     "cause_role": "outcome", "caused_by": [{"src": 2, "strength": "direct"}]}
  ],
  "outcome_severity": "substantial_damage_no_injury"
}
```

Notes:

- **`factor_type` vocabulary ≈ 56 values**: the 34 non-outcome v3 event types + ~14 new condition types (`FUEL_STARVATION`, `FUEL_CONTAMINATION`, `WEIGHT_OR_BALANCE_OUT_OF_LIMITS`, `DENSITY_ALTITUDE_HIGH`, `SURFACE_CONTAMINATED`, `LATENT_MECHANICAL_DEFECT`, `PILOT_FATIGUE`, `PILOT_MEDICAL_CONDITION`, `VISIBILITY_RESTRICTED`, `AIRCRAFT_OVERWEIGHT`, `TERRAIN_HOSTILE`, `EQUIPMENT_NOT_INSTALLED`, `PLANNING_INADEQUATE`, `unknown_condition`) + the 8 outcome types. Exact list finalized in Phase 1 by mining the v3 condition `description` free text for the natural clusters.
- Chains are **ordered DAGs, not strict lines** — an event may have multiple parents (`caused_by` is a list) and multiple children; concurrency is expressible (two items, neither referencing the other).
- `guided_json` cannot enforce "`src` < own `idx`" — a post-validator checks monotonicity + enum membership and re-prompts once with the specific violation appended. Target: < 0.5 % second-pass failures.

## Two extraction tracks (circularity control)

1. **Grounded track (production).** Prompt = narrative + supervision block (`Findings` rows with C/F flags, `seq_of_events` occurrence codes, from `corpus_enriched.jsonl`). Instructions: coded sequence is the ordering skeleton; every `Cause_Factor='C'` finding supported by the narrative must appear with `cause_role: primary`; LLM adds what codes miss (decisions, conditions). This is what builds the KG — maximal accuracy is the goal, benchmark purity is not.
2. **Narrative-only track (honest eval).** Same schema, no supervision block. Run on the 2 k calibration set only. This measures extraction skill and keeps the v1/v3/v4 comparison meaningful; also gives an unleaked Kendall's-τ ordering score against `seq_of_events`.

---

## Phases

### Phase 0 — Baseline fixes, no GPU (~0.5 day)

1. Fix [stage3/build_precedence_matrices.py](../event_extraction/scripts/stage3/build_precedence_matrices.py): remove `RESPONDS_TO` and `DETECTS` from `PRECEDENCE_EDGE_TYPES` (their src→dst direction is temporally inverted). Re-run pieces 1→6 on existing v3 data → corrected v3 baseline DAGs for later comparison.
2. Add `event_extraction/scripts/validate_extraction.py`: enum-membership + structural checks over any extraction JSONL (reusable for v4's retry loop). Report OOV rates on v3 output as the baseline number.

### Phase 1 — Schema, prompts, runner (~1 day)

Deliverables in `event_extraction/prompts/`:
- `schema_v4.json` — guided_json contract per the sketch above.
- `system_v4.txt` — chain-first instructions; grounded-mode section; keep the v3 "decisions are events" table (it worked).
- `few_shot_v4.json` — 2 worked examples: one grounded (with supervision block), one narrative-only. Reuse the two records spot-checked in the v3 review (`20001208X06675_1`, `20001211X14564_1`) — both have known-good manual chains.
- Condition-vocabulary memo: top-50 v3 condition descriptions clustered → final ~14 enum values.

Runner changes to [extract_vllm.py](../event_extraction/scripts/extract_vllm.py):
- `--supervision {grounded,none}` flag; grounded mode joins `corpus_enriched.jsonl` and renders the supervision block into the user turn.
- Post-validation with single retry (violation text appended to a repair turn).
- Emit both native v4 chain and a derived `nodes/edges` view for backward compatibility with existing tooling.

### Phase 2 — Calibration on the 2 k set (~1–2 days, few GPU-hours)

Run both tracks on `calibration_2k_input.jsonl`. Adapt [semantic_eval.py](../event_extraction/scripts/semantic_eval.py) to the chain format.

Acceptance gates (vs. v3's 80.5 % R / 29.9 % P):

| Metric | Gate |
|---|---|
| Recall vs. `Findings` C/F (narrative-only track) | ≥ 80 % (no regression) |
| Precision (narrative-only) | ≥ 45 % |
| Recall (grounded track) | ≥ 92 % |
| Kendall's τ, chain order vs. `seq_of_events` (narrative-only) | ≥ 0.75 |
| Cycles / forward references after retry | 0 |
| OOV enum values | 0 |
| Manual spot-check (20 records + 10 LOC-I feasibility cases) | chains judged faithful |

Iterate prompt until gates pass; keep every iteration's summary JSON in `out/` as before.

### Phase 3 — Full re-extraction (~2–3 days GPU, unattended)

Same 56 k-record input as v3 (`prep_full_v3_input.py` reused). Grounded track. Expect throughput ≥ v3 (output is leaner — no entities, no edge list). `--resume`-safe as before. Output: `out/full_corpus_v4.jsonl`.

### Phase 4 — Causation KG build (~1–2 days)

New `build_kg_layer2.py` (v4-native), producing three decoupled artifacts:

1. **Per-accident chain DAGs** — instance level, evidence-linked; the provenance layer.
2. **Factor-vector table** (parquet): records × ~48 binary causal factors + `outcome_severity` + category. *This is the causal-discovery and risk-calculation input.*
3. **Aggregate causation KG** — the "clear graph" deliverable:
   - nodes = causal factor vocab only (outcomes in a separate outcome layer, never as sources);
   - edge i→j counted once per record when any `caused_by` links type-i → type-j, i ≠ j (no self-loops by construction);
   - edge attributes: `support`, `P(j|i)`, `lift`, `direct_share`, sample `record_id`s for audit;
   - default rendered view: min support ≥ 30, top-k out-edges per node by lift — plus the unfiltered CSV for analysis.

Also: category harmonization pass — proper CICTT mapping for avall HFACS-category and FAA AIDS buckets (crosswalk table committed), or restrict per-category graphs to records with genuine occurrence codes.

### Phase 5 — Stage-3 rerun + downstream-readiness gates (~1–2 days)

Re-run the Stage-3 pipeline on v4 factor vectors (precedence now trivial from chain order; PC + LLM order + Laplacian + bootstrap unchanged, reorient with corrected precedence).

**Readiness gates** (definition of "ready for counterfactual / risk work"):
1. Per-category DAG acyclic, ≤ 40 nodes, edges reported with bootstrap stability.
2. ≥ 90 % of DAG edge directions consistent with aggregate chain-order precedence.
3. **BN smoke test on LOC-I**: fit CPTs (pgmpy) on the factor-vector table restricted to LOC-I, compute `P(outcome_severity ≥ serious | do(X = absent))` for the four case-9 factors, confirm the ranking reproduces the manual walkthrough (strong / strong / moderate / moderate).
4. Every aggregate edge traceable to source records.

Gate 3 is the proof the KG is actually consumable for the counterfactual objective, not just visually clean.

---

## Timeline & compute

| Phase | Calendar | GPU |
|---|---|---|
| 0 — baseline fixes | 0.5 d | none |
| 1 — schema + runner | 1 d | none |
| 2 — 2 k calibration | 1–2 d | ~2–4 h |
| 3 — full re-extraction | 2–3 d | ~2–3 d unattended |
| 4 — KG build | 1–2 d | none |
| 5 — Stage-3 + gates | 1–2 d | ~1 h (LLM order) |
| **Total** | **~1.5–2 weeks** | ~3 d |

## Risks

| Risk | Mitigation |
|---|---|
| guided_json can't enforce index monotonicity | post-validate + one repair turn; measured in Phase 2 |
| Grounding leaks supervision into eval | two-track design; narrative-only numbers are the honest ones |
| Chain format under-expresses branching | `caused_by` is a list → ordered DAG, not a line; spot-check branching cases in Phase 2 |
| New condition vocab mis-scoped | derived empirically from v3 condition free-text clusters before locking |
| Qwen drift reappears | zero-OOV gate + validator retry |
