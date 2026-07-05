"""Build few_shot_v4.json from two hand-annotated corpus records.

Example 1 (grounded): NTSB 20001214X35670_1 — NPA flight 1802, Beech 65-A80,
Soldotna AK, 9 fatal. Icing + in-flight decision + unauthorized circling ->
collision with trees. Demonstrates the STRUCTURED FINDINGS block, [C]/[F] ->
cause_role mapping, condition-vs-event split, and outcome_severity from the
injury data.

Example 2 (narrative-only): NTSB 20001208X06675_1 — Piper PA-18 glider tow,
fuel selector between tanks -> fuel starvation -> engine failure -> ditching.
Demonstrates chain re-ordering so causes precede effects (supervision failure
first), crew-response-as-later-event, and aircraft_damage_only severity.

The user-turn content is composed with extract_vllm.build_user_content so the
few-shot format is byte-identical to what the runner sends at extraction time.

Run:  python event_extraction/scripts/build_fewshot_v4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_vllm import build_user_content  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
ENRICHED = REPO / "data/corpus/corpus_enriched.jsonl"
OUT = REPO / "event_extraction/prompts/few_shot_v4.json"

GROUNDED_ID = "20001214X35670_1"
NARRATIVE_ONLY_ID = "20001208X06675_1"

EXTRACTION_GROUNDED = {
    "chain": [
        {"idx": 0, "class": "event", "factor_type": "MAINTENANCE_INADEQUATE",
         "trigger": "recurring problems with the anti-ice system; 'single' mode inoperative; two de-ice boots missing from propeller blades",
         "phase_of_flight": "preflight", "cause_role": "contributing", "caused_by": []},
        {"idx": 1, "class": "event", "factor_type": "ORG_OR_REGULATORY_INADEQUATE",
         "trigger": "no FAA inspection of the weather station in 2 years; inadequate surveillance of the operation; ceilometer inoperative",
         "phase_of_flight": "preflight", "cause_role": "contributing", "caused_by": []},
        {"idx": 2, "class": "condition", "factor_type": "LOW_VISIBILITY_OR_IMC",
         "trigger": "ceiling 600 to 800 ft deteriorating to below minimums, fog and rain",
         "phase_of_flight": "approach", "cause_role": "contributing", "caused_by": []},
        {"idx": 3, "class": "condition", "factor_type": "MOUNTAINOUS_OR_RISING_TERRAIN",
         "trigger": "high terrain approximately 1.5 mi SE of the airport",
         "phase_of_flight": "approach", "cause_role": "contributing", "caused_by": []},
        {"idx": 4, "class": "event", "factor_type": "DECISION_INAPPROPRIATE",
         "trigger": "flight into known adverse weather with known deficiencies in equipment",
         "phase_of_flight": "en_route", "cause_role": "contributing",
         "caused_by": [
             {"src": 0, "strength": "contributing",
              "evidence": "recurring prblms with the anti-ice sys, its 'single' mode was inop"}]},
        {"idx": 5, "class": "event", "factor_type": "ICING_ENCOUNTER",
         "trigger": "the aircraft accumulated a heavy load of ice",
         "phase_of_flight": "approach", "cause_role": "contributing",
         "caused_by": [
             {"src": 2, "strength": "direct", "evidence": "icg forcasted"},
             {"src": 4, "strength": "contributing",
              "evidence": "continued into forecast icing with an inadequate anti-ice system"}]},
        {"idx": 6, "class": "event", "factor_type": "DECISION_INAPPROPRIATE",
         "trigger": "elected a VOR approach back to Soldotna instead of diverting to the Kenai ILS; did not acknowledge the divert recommendation",
         "phase_of_flight": "approach", "cause_role": "primary",
         "caused_by": [
             {"src": 5, "strength": "contributing",
              "evidence": "the crew reported the acft had accumulated a hvy load of ice"},
             {"src": 2, "strength": "contributing",
              "evidence": "wx observer advised the wx had deteriorated to below mins & recommended diverting, but the crew did not acknowledge"}]},
        {"idx": 7, "class": "event", "factor_type": "PROCEDURE_NOT_FOLLOWED",
         "trigger": "circling south of runway 7/25 where circling was not authorized; minimum descent altitude not maintained",
         "phase_of_flight": "approach", "cause_role": "primary",
         "caused_by": [
             {"src": 6, "strength": "direct",
              "evidence": "elected to make a VOR apch back to Soldotna"}]},
        {"idx": 8, "class": "outcome", "factor_type": "GROUND_IMPACT",
         "trigger": "collided with trees on high terrain while circling",
         "phase_of_flight": "approach", "cause_role": "outcome",
         "caused_by": [
             {"src": 7, "strength": "direct",
              "evidence": "there was evidence the acft was circling when it crashed"},
             {"src": 2, "strength": "contributing", "evidence": "wx had deteriorated to below mins"},
             {"src": 3, "strength": "contributing", "evidence": "trees on hi terrain aprx 1.5 mi se of the arpt"}]},
        {"idx": 9, "class": "outcome", "factor_type": "INJURY_OR_FATALITY",
         "trigger": "nine occupants fatally injured",
         "phase_of_flight": "post_impact", "cause_role": "outcome",
         "caused_by": [
             {"src": 8, "strength": "direct", "evidence": "the acft collided with trees on hi terrain"}]}
    ],
    "outcome_severity": "fatal",
}

EXTRACTION_NARRATIVE_ONLY = {
    "chain": [
        {"idx": 0, "class": "event", "factor_type": "CREW_COORDINATION_FAILURE",
         "trigger": "inadequate supervision by the flight instructor (CFI)",
         "phase_of_flight": "ground", "cause_role": "primary", "caused_by": []},
        {"idx": 1, "class": "event", "factor_type": "PROCEDURE_NOT_FOLLOWED",
         "trigger": "fuel selector positioned between the left and right fuel tank positions",
         "phase_of_flight": "ground", "cause_role": "primary",
         "caused_by": [
             {"src": 0, "strength": "contributing",
              "evidence": "inadequate supervision by the flight instructor"}]},
        {"idx": 2, "class": "condition", "factor_type": "FUEL_EXHAUSTION_OR_STARVATION",
         "trigger": "fuel starvation",
         "phase_of_flight": "initial_climb", "cause_role": "primary",
         "caused_by": [
             {"src": 1, "strength": "direct",
              "evidence": "fuel starvation, which resulted from improper positioning of the fuel tank selector"}]},
        {"idx": 3, "class": "event", "factor_type": "ENGINE_FAILURE",
         "trigger": "the engine lost power at 200 to 300 feet above ground level on the fourth glider tow",
         "phase_of_flight": "initial_climb", "cause_role": "contributing",
         "caused_by": [
             {"src": 2, "strength": "direct",
              "evidence": "loss of engine power due to fuel starvation"}]},
        {"idx": 4, "class": "event", "factor_type": "PROCEDURE_NOT_FOLLOWED",
         "trigger": "the crew did not attempt to reselect a fuel tank after the power loss",
         "phase_of_flight": "emergency_descent", "cause_role": "primary",
         "caused_by": [
             {"src": 3, "strength": "contributing",
              "evidence": "asked directly if they had attempted to reselect a fuel tank and he stated 'no'"}]},
        {"idx": 5, "class": "outcome", "factor_type": "WATER_IMPACT",
         "trigger": "missed the muskeg field and landed in a small lake; the airplane sank",
         "phase_of_flight": "emergency_descent", "cause_role": "outcome",
         "caused_by": [
             {"src": 3, "strength": "direct",
              "evidence": "the engine lost power ... they missed the field and had to land in the lake"},
             {"src": 4, "strength": "contributing",
              "evidence": "the flight crew's improper emergency procedure"}]}
    ],
    "outcome_severity": "aircraft_damage_only",
}


def main() -> None:
    wanted = {GROUNDED_ID, NARRATIVE_ONLY_ID}
    recs: dict[str, dict] = {}
    with ENRICHED.open() as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in wanted:
                recs[r["record_id"]] = r
                if len(recs) == len(wanted):
                    break
    missing = wanted - set(recs)
    if missing:
        raise SystemExit(f"records not found in {ENRICHED}: {missing}")

    fewshot = [
        {
            "record_id": GROUNDED_ID,
            "user": build_user_content(
                recs[GROUNDED_ID]["text"],
                recs[GROUNDED_ID].get("structured"),
                supervision="grounded"),
            "extraction": EXTRACTION_GROUNDED,
        },
        {
            "record_id": NARRATIVE_ONLY_ID,
            "user": build_user_content(
                recs[NARRATIVE_ONLY_ID]["text"], None, supervision="none"),
            "extraction": EXTRACTION_NARRATIVE_ONLY,
        },
    ]
    OUT.write_text(json.dumps(fewshot, indent=2, ensure_ascii=False))
    print(f"wrote {OUT} ({len(fewshot)} examples)")


if __name__ == "__main__":
    main()
