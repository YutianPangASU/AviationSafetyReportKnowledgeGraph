# Aggregate Knowledge Graph (Layer 1)

Generated from `full_corpus_v3.jsonl`. 2,452 records contributed.

## Topline

- Records (ok): **2,452** of 2,456 total
- Aggregate nodes: **189**
- Aggregate edges (distinct (src,type,dst) triples): **2,446**
- Total event-instance count across narratives: 11,704
- Total edge-instance count across narratives: 14,444

## Top 25 event_types by frequency

| Rank | event_type | Count | Top cause_role | Top phase | Top HAEM |
|---|---|---:|---|---|---|
| 1 | `GROUND_IMPACT` | 2,505 | outcome (2250) | landing (1232) | A (2277) |
| 2 | `CONTROL_INPUT_IMPROPER` | 1,393 | primary (839) | landing (507) | A (775) |
| 3 | `DECISION_INAPPROPRIATE` | 1,139 | primary (652) | takeoff (216) | H (897) |
| 4 | `PROCEDURE_NOT_FOLLOWED` | 935 | contributing (471) | takeoff (119) | H (768) |
| 5 | `ENGINE_FAILURE` | 758 | primary (635) | climb (203) | A (756) |
| 6 | `RUNWAY_EXCURSION_OR_OVERRUN` | 504 | outcome (182) | landing (343) | A (437) |
| 7 | `INJURY_OR_FATALITY` | 479 | outcome (479) | landing (95) | A (397) |
| 8 | `PERCEPTION_FAILURE` | 430 | primary (228) | approach (79) | H (320) |
| 9 | `LOSS_OF_CONTROL_GROUND` | 406 | primary (238) | landing (203) | A (375) |
| 10 | `LOSS_OF_CONTROL_INFLIGHT` | 403 | primary (243) | takeoff (81) | A (352) |
| 11 | `STALL` | 283 | primary (208) | takeoff (60) | A (270) |
| 12 | `LANDING_GEAR_ANOMALY` | 206 | primary (87) | landing (122) | A (206) |
| 13 | `EMERGENCY_LANDING` | 187 | outcome (113) | landing (176) | H (187) |
| 14 | `AIRFRAME_STRUCTURAL_FAILURE` | 181 | outcome (94) | landing (76) | A (181) |
| 15 | `TERRAIN_OR_OBSTACLE_PROXIMITY` | 162 | primary (82) | takeoff (50) | A (117) |
| 16 | `WIND_SHEAR_OR_GUST` | 159 | contributing (110) | landing (60) | E (159) |
| 17 | `MAINTENANCE_INADEQUATE` | 156 | primary (100) | pre_flight (50) | M (88) |
| 18 | `STRUCTURAL_OVERLOAD` | 122 | outcome (85) | landing (66) | A (119) |
| 19 | `INFLIGHT_FIRE_OR_SMOKE` | 115 | outcome (68) | post-impact (25) | A (111) |
| 20 | `FUEL_SYSTEM_ANOMALY` | 113 | primary (85) | cruise (32) | A (104) |
| 21 | `ALTITUDE_DEVIATION_UNCONTROLLED` | 111 | contributing (82) | takeoff (25) | A (87) |
| 22 | `CONTROL_SURFACE_ANOMALY` | 110 | primary (84) | climb (27) | A (110) |
| 23 | `WATER_IMPACT` | 107 | outcome (102) | landing (57) | A (80) |
| 24 | `RUNWAY_CONDITION_HAZARD` | 80 | contributing (53) | landing (35) | E (64) |
| 25 | `LOW_VISIBILITY_OR_IMC` | 79 | context (42) | en_route (38) | E (71) |

## Top 30 edges by frequency (event ↔ event)

| Rank | src | edge | dst | Count |
|---|---|---|---|---:|
| 1 | `GROUND_IMPACT` | CAUSES | `INJURY_OR_FATALITY` | 389 |
| 2 | `CONTROL_INPUT_IMPROPER` | CAUSES | `GROUND_IMPACT` | 342 |
| 3 | `LOSS_OF_CONTROL_INFLIGHT` | CAUSES | `GROUND_IMPACT` | 248 |
| 4 | `RUNWAY_EXCURSION_OR_OVERRUN` | CAUSES | `GROUND_IMPACT` | 234 |
| 5 | `DECISION_INAPPROPRIATE` | CAUSES | `GROUND_IMPACT` | 211 |
| 6 | `LOSS_OF_CONTROL_GROUND` | CAUSES | `GROUND_IMPACT` | 210 |
| 7 | `STALL` | CAUSES | `GROUND_IMPACT` | 201 |
| 8 | `GROUND_IMPACT` | CAUSES | `GROUND_IMPACT` | 187 |
| 9 | `CONTROL_INPUT_IMPROPER` | CAUSES | `LOSS_OF_CONTROL_GROUND` | 170 |
| 10 | `ENGINE_FAILURE` | CAUSES | `GROUND_IMPACT` | 142 |
| 11 | `CONTROL_INPUT_IMPROPER` | CAUSES | `STALL` | 132 |
| 12 | `ENGINE_FAILURE` | RESPONDS_TO | `PROCEDURE_NOT_FOLLOWED` | 130 |
| 13 | `EMERGENCY_LANDING` | CAUSES | `GROUND_IMPACT` | 118 |
| 14 | `CONTROL_INPUT_IMPROPER` | CAUSES | `RUNWAY_EXCURSION_OR_OVERRUN` | 116 |
| 15 | `PROCEDURE_NOT_FOLLOWED` | CAUSES | `GROUND_IMPACT` | 111 |
| 16 | `DECISION_INAPPROPRIATE` | CAUSES | `CONTROL_INPUT_IMPROPER` | 91 |
| 17 | `ENGINE_FAILURE` | CAUSES | `EMERGENCY_LANDING` | 91 |
| 18 | `LANDING_GEAR_ANOMALY` | CAUSES | `GROUND_IMPACT` | 85 |
| 19 | `ENGINE_FAILURE` | TRIGGERS | `DECISION_INAPPROPRIATE` | 83 |
| 20 | `TERRAIN_OR_OBSTACLE_PROXIMITY` | CAUSES | `GROUND_IMPACT` | 82 |
| 21 | `DECISION_INAPPROPRIATE` | CAUSES | `RUNWAY_EXCURSION_OR_OVERRUN` | 81 |
| 22 | `LOSS_OF_CONTROL_GROUND` | CAUSES | `RUNWAY_EXCURSION_OR_OVERRUN` | 78 |
| 23 | `ENGINE_FAILURE` | RESPONDS_TO | `DECISION_INAPPROPRIATE` | 73 |
| 24 | `CONTROL_INPUT_IMPROPER` | CAUSES | `LOSS_OF_CONTROL_INFLIGHT` | 68 |
| 25 | `GROUND_IMPACT` | CAUSES | `AIRFRAME_STRUCTURAL_FAILURE` | 64 |
| 26 | `GROUND_IMPACT` | CAUSES | `STRUCTURAL_OVERLOAD` | 64 |
| 27 | `PERCEPTION_FAILURE` | CAUSES | `CONTROL_INPUT_IMPROPER` | 62 |
| 28 | `ALTITUDE_DEVIATION_UNCONTROLLED` | CAUSES | `GROUND_IMPACT` | 59 |
| 29 | `ENGINE_FAILURE` | TRIGGERS | `PROCEDURE_NOT_FOLLOWED` | 57 |
| 30 | `GROUND_IMPACT` | CAUSES | `INFLIGHT_FIRE_OR_SMOKE` | 55 |

## Records per CICTT category

| Category | Records |
|---|---:|
| `USOS` | 369 |
| `LOC-I` | 323 |
| `LOC-G` | 287 |
| `ARC` | 231 |
| `CFIT` | 185 |
| `WSTRW` | 181 |
| `SCF-PP` | 148 |
| `WX` | 137 |
| `MAC` | 120 |
| `ICE` | 102 |
| `GCOL` | 85 |
| `RE` | 57 |
| `RI` | 42 |
| `BIRD` | 37 |
| `FUEL` | 24 |
| `CTOL` | 18 |
| `OTHER:NTSB-190` | 14 |
| `OTHER:NTSB-271` | 14 |
| `OTHER:NTSB-192` | 14 |
| `OTHER:NTSB-171` | 11 |
| `OTHER:NTSB-232` | 9 |
| `OTHER:NTSB-191` | 9 |
| `OTHER:NTSB-353` | 6 |
| `HFACS_CAT_01` | 5 |
| `HFACS_CAT_02` | 5 |
| `EVAC` | 3 |
| `OTHER:NTSB-351` | 3 |
| `HFACS_CAT_03` | 2 |
| `ATM` | 2 |
| `OTHER:NTSB-196` | 1 |
| `OTHER:NTSB-198` | 1 |
| `OTHER:NTSB-194` | 1 |
| `OTHER:NTSB-132` | 1 |
| `OTHER:NTSB-195` | 1 |
| `OTHER:NTSB-150` | 1 |
| `OTHER:NTSB-352` | 1 |
| `OTHER:NTSB-231` | 1 |
| `F-NI` | 1 |

## Per-category subgraph sizes

| Category | Nodes | Edges | Top edge |
|---|---:|---:|---|
| `LOC-I` | 99 | 638 | `LOSS_OF_CONTROL_INFLIGHT` --CAUSES--> `GROUND_IMPACT` (112) |
| `USOS` | 66 | 491 | `mechanical` --CAUSES--> `ENGINE_FAILURE` (260) |
| `LOC-G` | 80 | 468 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `LOSS_OF_CONTROL_GROUND` (109) |
| `CFIT` | 80 | 466 | `GROUND_IMPACT` --CAUSES--> `INJURY_OR_FATALITY` (48) |
| `WSTRW` | 81 | 460 | `GROUND_IMPACT` --CAUSES--> `INJURY_OR_FATALITY` (44) |
| `ARC` | 76 | 437 | `mechanical` --CAUSES--> `ENGINE_FAILURE` (130) |
| `SCF-PP` | 58 | 423 | `mechanical` --CAUSES--> `CONTROL_SURFACE_ANOMALY` (42) |
| `WX` | 66 | 367 | `GROUND_IMPACT` --CAUSES--> `INJURY_OR_FATALITY` (30) |
| `MAC` | 64 | 332 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `GROUND_IMPACT` (32) |
| `ICE` | 52 | 266 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `GROUND_IMPACT` (50) |
| `GCOL` | 59 | 259 | `LOSS_OF_CONTROL_GROUND` --CAUSES--> `GROUND_IMPACT` (20) |
| `RI` | 55 | 210 | `mechanical` --CAUSES--> `CONTROL_SURFACE_ANOMALY` (5) |
| `RE` | 44 | 196 | `RUNWAY_EXCURSION_OR_OVERRUN` --CAUSES--> `GROUND_IMPACT` (38) |
| `BIRD` | 39 | 142 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `GROUND_IMPACT` (7) |
| `FUEL` | 35 | 101 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `GROUND_IMPACT` (8) |
| `OTHER:NTSB-192` | 24 | 62 | `LANDING_GEAR_ANOMALY` --CAUSES--> `GROUND_IMPACT` (7) |
| `OTHER:NTSB-271` | 22 | 61 | `PERCEPTION_FAILURE` --CAUSES--> `MIDAIR_COLLISION` (6) |
| `CTOL` | 29 | 59 | `PERCEPTION_FAILURE` --CAUSES--> `MIDAIR_COLLISION` (13) |
| `OTHER:NTSB-171` | 24 | 50 | `mechanical` --CAUSES--> `INFLIGHT_FIRE_OR_SMOKE` (7) |
| `OTHER:NTSB-190` | 18 | 46 | `LANDING_GEAR_ANOMALY` --CAUSES--> `GROUND_IMPACT` (8) |
| `OTHER:NTSB-191` | 18 | 41 | `mechanical` --CAUSES--> `LANDING_GEAR_ANOMALY` (5) |
| `OTHER:NTSB-353` | 22 | 38 | `mechanical` --CAUSES--> `ENGINE_FAILURE` (2) |
| `OTHER:NTSB-232` | 20 | 36 | `PROCEDURE_NOT_FOLLOWED` --CAUSES--> `GROUND_IMPACT` (5) |
| `HFACS_CAT_02` | 15 | 26 | `LOSS_OF_CONTROL_GROUND` --CAUSES--> `GROUND_IMPACT` (3) |
| `HFACS_CAT_01` | 18 | 24 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `STALL` (2) |
| `EVAC` | 13 | 17 | `weather` --UNDER_CONDITION--> `TURBULENCE_ENCOUNTER` (2) |
| `OTHER:NTSB-351` | 11 | 16 | `mechanical` --CAUSES--> `ENGINE_FAILURE` (3) |
| `HFACS_CAT_03` | 9 | 9 | `CONTROL_INPUT_IMPROPER` --CAUSES--> `LOSS_OF_CONTROL_GROUND` (1) |
| `ATM` | 8 | 8 | `LOW_VISIBILITY_OR_IMC` --UNDER_CONDITION--> `weather` (1) |
| `OTHER:NTSB-194` | 7 | 7 | `human_factors` --CAUSES--> `CONTROL_INPUT_IMPROPER` (1) |
| `OTHER:NTSB-198` | 6 | 6 | `PILOT_INCAPACITATION_OR_IMPAIRMENT` --INDUCED_CONDITION--> `physiological` (1) |
| `OTHER:NTSB-132` | 6 | 6 | `DESIGN_DEFECT_LATENT` --CAUSES--> `mechanical` (1) |
| `OTHER:NTSB-352` | 4 | 6 | `mechanical` --CAUSES--> `mechanical` (1) |
| `OTHER:NTSB-196` | 5 | 5 | `PERCEPTION_FAILURE` --CAUSES--> `mechanical` (1) |
| `OTHER:NTSB-195` | 6 | 5 | `MAINTENANCE_INADEQUATE` --CAUSES--> `mechanical` (1) |
| `F-NI` | 6 | 5 | `mechanical` --CAUSES--> `INFLIGHT_FIRE_OR_SMOKE` (1) |
| `OTHER:NTSB-150` | 5 | 4 | `psychological` --CAUSES--> `LOSS_OF_CONTROL_INFLIGHT` (1) |
| `OTHER:NTSB-231` | 4 | 3 | `habit` --CAUSES--> `CONTROL_INPUT_IMPROPER` (1) |