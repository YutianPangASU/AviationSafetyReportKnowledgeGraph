# ACE-Graph v3: Model-Data Hybrid Aviation Risk Framework — Design Document

**Status:** Draft v3 — Framework concept with LOC-I demo scope.
**Created:** 2026-04-10 (v1) | **Revised:** 2026-04-11 (v2) | **Revised:** 2026-04-13 (v3)
**Working title:** Aviation Causal Event Graph (ACE-Graph)
**One-line summary:** A causal knowledge graph extracted from aviation safety reports where each node embeds hierarchical failure models (fault trees, HRA, bow-tie) with tiered quantitative risk values, enabling counterfactual intervention analysis with measurable risk reduction.

**Prior versions:** [v1](2026-04-10-ace-graph-design.md) | [v2](2026-04-11-ace-graph-design-v2.md) | [v1→v2 changelog](2026-04-11-ace-graph-changelog.md) | [expert review](../review/2026-04-11-design-critique.md)

---

## Key Change from v2 to v3

v2 produced a **qualitative** causal graph with LLM-assisted counterfactual text answers ("Factor X contributed to the accident"). v3 produces a **quantitative** risk framework where every node carries a probability, fault trees and human reliability models are embedded inside KG nodes, and counterfactual analysis yields numerical risk reduction with confidence intervals ("Removing factor X reduces LOC-I risk by 23%, CI: 15-31%").

| Aspect | v2 | v3 |
|---|---|---|
| Node content | Flat event label | Hierarchical container (fault tree / HRA / bow-tie) |
| Risk quantification | None (qualitative causal graph) | 4-tier quantitative probability at every node |
| Counterfactual method | LLM-assisted QA (text answer) | Bayesian propagation (numerical risk reduction) |
| Evaluation | Agreement with investigator judgment | Risk reduction accuracy + investigator agreement |
| Output | "Factor X contributed to accident" | "Removing factor X reduces LOC-I risk by 23% (CI: 15-31%)" |
| Paper scope | Full pipeline across categories | Framework paper + LOC-I demo with 3 deep-dive nodes |

**What carries over from v2:** Stage 0 corpus harmonization (UASC), NLP extraction pipeline, CICTT taxonomy anchoring, cross-source deduplication, honest framing on NTSB dominance and ground-truth limitations, LLM circularity mitigation policy, cross-stage consistency checks.

---

## 1. Motivation and Scope

### 1.1 Goal

Build a model-data hybrid framework that:

1. **Extracts causal structure** from unstructured aviation accident reports using NLP
2. **Embeds quantitative failure models** (fault trees, human reliability analysis, bow-tie) at each node of the causal knowledge graph
3. **Propagates risk** through the graph to produce quantitative risk estimates at every node
4. **Supports counterfactual intervention analysis** — "If we apply intervention X, how much does the overall risk decrease?" — with numerical answers and confidence intervals

This targets the gap between qualitative causal analysis from accident reports (what caused what) and quantitative risk assessment from engineering models (how likely is each failure). No existing system bridges these automatically.

### 1.2 Primary Research Contribution

A **framework paper** that proposes and demonstrates the model-data hybrid architecture:

- The **causal KG** (from NLP extraction) provides the backbone structure
- **Hierarchical failure models** (fault trees, HRA, bow-tie) are embedded inside KG nodes
- A **4-tier probability system** handles the reality that some nodes have physics models and others only have corpus statistics
- **Bayesian risk propagation** through the KG yields quantitative risk at every node
- **Counterfactual intervention analysis** produces measurable risk reduction with uncertainty

The demonstration is on **LOC-I (Loss of Control In-flight)** with 3 showcase nodes (mechanical, human factors, environmental) that exercise all four probability tiers.

### 1.3 Relationship to Prior Art

| System | What it does | What ACE-Graph adds |
|---|---|---|
| **CATS** (NLR/TU Delft) | Manually built fault trees + event trees for aviation accident categories | Automates causal structure extraction from text; embeds quantitative models inside NLP-derived KG |
| **SAE ARP 4761** | Prescribes FTA/ETA/CCA for aircraft certification | Uses ARP 4761 methods (FTA, bow-tie) as sub-models within KG nodes; extends to corpus-derived probabilities |
| **HABERT** (Zhao et al.) | Hierarchical event extraction from NTSB reports | Builds on HABERT for extraction; adds quantitative risk layer and counterfactual engine |
| **HFACS** (Shappell & Wiegmann) | Human factors taxonomy for accident classification | Informs the human-factors node decomposition; HRA models (HEART/CREAM) quantify HFACS categories |
| **CAST Safety Enhancements** | Expert-estimated risk reduction percentages for interventions | ACE-Graph derives risk reduction from causal graph propagation rather than expert judgment |
| **Bow-tie models** (ARMS, airline risk management) | Threats → barriers → hazard → barriers → consequences | Bow-tie is a view on the KG; each threat chain maps to a KG path with embedded fault trees |

### 1.4 Data Sources

Same four sources as v2, treated as a single unified corpus:

- **NTSB** (US) — `avall.mdb` + published reports (496 PDFs scraped)
- **TSB Canada** — public CSVs (124,250 event records)
- **BEA** (France) — scraped notified events (~6,500)
- **FAA AIDS** (US) — tab-delimited accident/incident records

Additionally, for Workstream 2 (risk formulation):
- **MIL-HDBK-217** — electronic component failure rates
- **NPRD/EPRD** — non-electronic part reliability data
- **HEART/CREAM** — human error probability tables
- **SAFTE-FAST** or similar — biomathematical fatigue models
- **Published CATS model failure rates** (Ale et al., 2006, 2009)
- **FAA/NASA technical reports** — component-specific reliability data (e.g., DARWIN for engine rotors)

### 1.5 Scope: Demo-First

The first paper demonstrates the **framework concept** on LOC-I with selected deep-dive nodes. Full implementation across all categories and all nodes is future work.

| In scope (demo) | Future work |
|---|---|
| LOC-I category end-to-end | All CICTT categories |
| 3 showcase nodes with hierarchical models | All nodes with embedded models |
| Tier 1-2 physics/data for showcase nodes | Comprehensive reliability database integration |
| 2-3 counterfactual case studies | Systematic intervention analysis |
| Interactive visualization prototype | Production tool |
| Framework paper | Full system paper |

### 1.6 Out of Scope

- Real-time or operational decision support
- Safety-of-life decisions based on system output
- Full parametric SCM (functional form estimation for all edges)
- Fleet-level risk dashboards
- Runway/airspace simulation

---

## 2. High-Level Architecture: Three Workstreams

```
═══════════════════════════════════════════════════════════════════
 WORKSTREAM 1: Build the Causal Knowledge Graph
═══════════════════════════════════════════════════════════════════

  Raw Data Sources              NLP / Extraction            Causal KG
  ┌──────────┐
  │ NTSB     │──┐             ┌──────────────┐          ┌──────────────┐
  │ avall.mdb│  │ ┌────────┐  │ Hierarchical │          │  Per-Category │
  ├──────────┤  ├→│Stage 0 │─→│ Event        │─→ DAGs ─→│  Causal KG   │
  │ NTSB     │  │ │Corpus  │  │ Extraction   │          │  (LOC-I      │
  │ Reports  │──┤ │Harmoniz│  │ (HABERT+LLM) │          │   first)     │
  ├──────────┤  │ └────────┘  └──────────────┘          └──────┬───────┘
  │ TSB/BEA/ │  │      ↓              ↓                        │
  │ FAA AIDS │──┘   UASC-Rich   Temporal DAGs                  │
  └──────────┘                  (per accident)                  ▼

═══════════════════════════════════════════════════════════════════
 WORKSTREAM 2: Risk Formulation (Quantitative)
═══════════════════════════════════════════════════════════════════

  For each KG node, embed a hierarchical failure model:

  ┌─────────────────────────┐     ┌─────────────────────────┐
  │ KG Node: Engine Failure │     │ KG Node: Pilot Error    │
  │ ┌─────────────────────┐ │     │ ┌─────────────────────┐ │
  │ │     Fault Tree      │ │     │ │     HRA Model       │ │
  │ │  ┌── OR ──┐         │ │     │ │  ┌── OR ──┐        │ │
  │ │ [Turbine] [Fuel]    │ │     │ │ [Fatigue] [Train]   │ │
  │ │  T1:2e-6   T2:5e-5  │ │     │ │  T2:1e-3   T3:4e-3 │ │
  │ │ P(node) = 5.2e-5/fh │ │     │ │ P(node) = 5.1e-3   │ │
  │ │ CI:[3.1e-5, 8.4e-5] │ │     │ │ CI:[2.8e-3, 9.2e-3]│ │
  │ └─────────────────────┘ │     │ └─────────────────────┘ │
  └─────────────────────────┘     └─────────────────────────┘

  Probability Sources (4 Tiers):
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Tier 1   │ │ Tier 2   │ │ Tier 3   │ │ Tier 4   │
  │ Physics  │ │ Published│ │ Corpus   │ │ Expert/  │
  │ models   │ │ rates    │ │ Bayesian │ │ LLM      │
  │ Narrow CI│ │ Mod CI   │ │ Wide CI  │ │Widest CI │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
        ↓            ↓            ↓            ↓
        └────────────┴────────────┴────────────┘
                         ↓
           Bayesian risk propagation through KG
                         ↓
           Quantitative risk at every node
                         ▼

═══════════════════════════════════════════════════════════════════
 WORKSTREAM 3: Counterfactual Analysis & Intervention
═══════════════════════════════════════════════════════════════════

  ┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
  │ Select         │    │ Intervene       │    │ Report         │
  │ intervention   │───→│ do(X=absent)    │───→│ risk delta     │
  │                │    │ or modify P(X)  │    │                │
  │ - Add TAWS     │    │ Re-propagate    │    │ "Adding TAWS   │
  │ - Fatigue      │    │ through KG      │    │  reduces CFIT  │
  │   monitoring   │    │                 │    │  risk by 62%   │
  │ - Redundant    │    │ Compare:        │    │  CI:[48-71%]"  │
  │   hydraulics   │    │ P_before vs     │    │                │
  │                │    │ P_after         │    │ Visualize:     │
  │                │    │                 │    │ before/after   │
  └────────────────┘    └─────────────────┘    └────────────────┘
```

### Workstream Summary

| Workstream | Input | Method | Output |
|---|---|---|---|
| **1. Build KG** | Raw accident reports | NLP extraction → temporal DAGs → causal discovery | Per-category causal KG |
| **2. Risk Formulation** | KG + reliability data + physics models + corpus stats | Fault trees / HRA / bow-tie embedded per node, 4-tier probability | Quantitative P(failure) at every node with CI |
| **3. Counterfactual** | Quantified KG + proposed intervention | do-calculus intervention → Bayesian re-propagation | Risk reduction estimate with CI |

---

## 3. Workstream 1: Build the Causal Knowledge Graph

This workstream is largely the same as v2 Stages 0-3. See [v2 design doc](2026-04-11-ace-graph-design-v2.md) for full details. Key components:

### 3.1 Stage 0 — Corpus Harmonization

Unchanged from v2. The UASC (Unified Aviation Safety Corpus) construction pipeline:
- Four-source loader (NTSB, TSB, BEA, FAA AIDS)
- Canonical UASC schema with multi-label CICTT categories
- Richness scoring with lognorm + 5-gate filter
- Cross-source deduplication
- Output: UASC parquet + UASC-Rich subset

Implementation plan: [2026-04-12-stage0-corpus-harmonization.md](../plans/2026-04-12-stage0-corpus-harmonization.md)

### 3.2 Stage 1 — Hybrid Event Extraction

Extract structured events from UASC-Rich narratives:
- HABERT-style coarse labels (Phase / Occurrence)
- LLM free-form subevent role extraction (actor, action, object, time)
- Reconciliation against NTSB structured `Findings` / `seq_of_events`
- Output: Event table per record

**v3 addition:** Extraction must identify **component-level entities** (not just event types) so that Workstream 2 can attach failure models. For example, "engine" is not sufficient — we need "left engine turbine first-stage disk" when the narrative provides that level of detail.

### 3.3 Stage 2 — Per-Accident Temporal Graph

Build temporal DAGs from event sequences:
- Topological ordering within each accident
- Merge structured `seq_of_events` with narrative-derived ordering
- Confidence-weighted temporal edges
- Output: Per-accident temporal DAGs

### 3.4 Stage 3 — Per-Category Causal Graph

Aggregate temporal DAGs into category-level causal KGs:
- LLM triplet-prompted causal order (Vashishtha et al.)
- Laplacian similarity prior (Zhang et al.)
- Constraint-based discovery with priors
- Multi-label category handling with cross-category bridge edges
- Output: **One causal KG per CICTT category** — this is the backbone that Workstream 2 builds upon

**v3 addition:** KG nodes must be typed for Workstream 2 compatibility:

| Node Type | Example | Embedded Model Type |
|---|---|---|
| `mechanical_component` | Engine failure, hydraulic system, landing gear | Fault tree |
| `human_factor` | Pilot error, fatigue, spatial disorientation | HRA model (HEART/CREAM) |
| `environmental` | Icing, turbulence, low visibility | Probabilistic weather model / corpus frequency |
| `organizational` | Maintenance program, training program | Corpus frequency / expert estimate |
| `operational` | ATC instruction, flight planning | Corpus frequency |
| `outcome` | LOC-I, CFIT, ground impact, fatality | Consequence node (not a failure model) |

---

## 4. Workstream 2: Risk Formulation

This is the primary new contribution in v3. Each node in the causal KG becomes a **container** that holds:

1. **Causal edges** — inherited from Workstream 1 (what causes it, what it causes)
2. **Internal hierarchical decomposition** — a fault tree, HRA model, or bow-tie that breaks the node into sub-components
3. **Quantitative risk value** — P(node failure) propagated up from the internal model
4. **Tier label** — which data source(s) the probability comes from
5. **Uncertainty bounds** — confidence interval reflecting the tier quality

### 4.1 The 4-Tier Probability System

Every node gets a quantitative probability. The tier indicates the source and confidence:

| Tier | Source | Method | Uncertainty | Example |
|---|---|---|---|---|
| **Tier 1** | Physics / reliability model | Fracture mechanics (DARWIN), fatigue life (S-N curves), aerodynamic simulation | Narrowest — model-calibrated CI | Turbine disk burst: P=2.1e-8/fh, CI from Monte Carlo on material properties |
| **Tier 2** | Published failure rate data | MIL-HDBK-217, NPRD/EPRD, HEART/CREAM HRA tables, CATS model rates, airline reliability data | Moderate — literature-sourced with known validity bounds | Hydraulic pump failure: λ=3.2e-5/fh from NPRD. Pilot error under fatigue: P=0.09 from HEART |
| **Tier 3** | Corpus-derived Bayesian | Count factor occurrences in NTSB/UASC corpus, normalize by fleet exposure (BTS flight-hour data), apply Bayesian estimation with conjugate priors | Wide — dependent on corpus size and reporting bias | "Spatial disorientation" in LOC-I: 47/892 LOC-I accidents = P≈0.053, Beta posterior CI:[0.039, 0.069] |
| **Tier 4** | Expert / LLM-elicited estimate | Structured expert elicitation or LLM-prompted probability estimation with calibration | Widest — acknowledged as informed estimate | "Inadequate safety culture": LLM-elicited P=0.15 with CI:[0.05, 0.30] |

**Key design principle:** Tier 3 (corpus-derived) is the **backbone** — it provides a quantitative probability for every node that appears in the accident corpus. Tiers 1-2 sharpen specific nodes where better data exists. Tier 4 fills in nodes too rare for corpus statistics.

**Reporting bias correction for Tier 3:** Accident reports only capture failures that led to reportable events, not all failures. This creates an upward bias in corpus-derived P(failure | accident). We address this by:
- Normalizing by fleet exposure hours (from BTS T-100 or FAA OPSNET) to get P(accident involving factor X)
- Clearly labeling this as "involvement rate" not "failure rate"
- Comparing against Tier 2 published rates where available to calibrate the corpus-derived estimates

### 4.2 Hierarchical Node Decomposition

Each KG node can be expanded into a sub-model. The sub-model type depends on the node type:

#### 4.2.1 Fault Tree (for mechanical_component nodes)

Standard FTA per SAE ARP 4761. Top event = the KG node's failure. Basic events = sub-components. Gates = AND/OR logic.

Example for "Engine Failure" node:

```
Engine Failure (OR)
├── Uncontained failure (OR)
│   ├── Turbine disk burst (Tier 1: DARWIN fracture mechanics)
│   ├── Compressor blade release (Tier 2: published MTBF)
│   └── Fan blade failure (Tier 2: published MTBF)
├── Contained failure — loss of thrust (OR)
│   ├── Fuel system malfunction (Tier 2: NPRD)
│   ├── FADEC failure (Tier 2: MIL-HDBK-217)
│   └── Oil system failure (Tier 3: corpus frequency)
└── Foreign object damage (OR)
    ├── Bird strike (Tier 3: corpus + FAA wildlife strike database)
    └── Volcanic ash (Tier 3: corpus frequency)
```

P(Engine Failure) is computed bottom-up through the fault tree using standard FTA cut-set analysis.

#### 4.2.2 HRA Model (for human_factor nodes)

Human Reliability Analysis using HEART (Human Error Assessment and Reduction Technique) or CREAM (Cognitive Reliability and Error Analysis Method).

Example for "Pilot Error" node:

```
Pilot Error (OR)
├── Fatigue-induced error
│   ├── Base error probability (Tier 2: HEART generic task type)
│   └── Fatigue multiplier (Tier 2: SAFTE-FAST or biomathematical model)
├── Skill-based error
│   ├── Spatial disorientation (Tier 3: corpus frequency)
│   ├── Incorrect control input (Tier 3: corpus frequency)
│   └── Automation mode confusion (Tier 3: corpus frequency)
├── Training deficit
│   ├── Upset recovery training gap (Tier 3: corpus + NTSB safety recs)
│   └── Type-rating inadequacy (Tier 4: expert estimate)
└── Decision error
    ├── Continued VFR into IMC (Tier 3: corpus frequency)
    └── Improper go-around decision (Tier 3: corpus frequency)
```

HEART provides base human error probabilities for generic task types (e.g., "fairly simple task performed rapidly" = 0.09) and Error Producing Conditions (EPCs) that multiply the base rate (e.g., "operator inexperienced" = 3x multiplier). The framework maps NTSB findings to HEART task types and EPCs.

#### 4.2.3 Bow-Tie (as a view on the KG)

The bow-tie model emerges naturally from the KG structure:
- **Left side (threats):** KG paths leading into the hazardous event node, each backed by a fault tree
- **Center:** The hazardous event (e.g., LOC-I)
- **Right side (consequences):** KG paths from hazardous event to outcomes (recovery, incident, accident, fatal)
- **Barriers:** Nodes on the KG paths that represent safety barriers (e.g., stick shaker, autopilot disconnect, TAWS)

The bow-tie is not a separate data structure — it's a **visualization and analysis view** on the KG with its embedded fault trees.

### 4.3 Risk Propagation Through the KG

Once every node has a P(failure) from its embedded model, risk propagates through the KG edges:

1. **Edge weights** represent conditional probabilities: P(child_event | parent_event), estimated from:
   - Co-occurrence frequency in accident corpus (Tier 3)
   - Published causal strength data where available (Tier 2)
   - Expert judgment (Tier 4)

2. **Propagation method:** The KG is treated as a Bayesian network. Given the DAG structure from Workstream 1 and the node/edge probabilities from Workstream 2, standard BN inference (variable elimination or junction tree algorithm) computes:
   - P(outcome node) = overall risk for the accident category
   - P(any intermediate node) = risk contribution of that factor

3. **Uncertainty propagation:** Monte Carlo simulation over the probability distributions at each node (shaped by tier-specific confidence intervals) yields uncertainty bounds on all propagated quantities.

### 4.4 Demo Scope: Three Showcase Nodes

For the framework paper, we build full hierarchical decompositions for three nodes in the LOC-I KG:

| Showcase Node | Type | Embedded Model | Tier Mix | Why This Node |
|---|---|---|---|---|
| **Engine/powerplant failure** | mechanical_component | Fault tree with 2-3 levels | Tier 1 (turbine disk) + Tier 2 (NPRD rates) | Demonstrates physics-model integration; well-characterized failure modes |
| **Pilot error / fatigue** | human_factor | HEART-based HRA with fatigue multiplier | Tier 2 (HEART tables) + Tier 3 (corpus) | Demonstrates HRA integration; highly relevant to LOC-I; Colgan 3407 case study |
| **Icing / weather** | environmental | Corpus-derived + meteorological encounter rates | Tier 2 (icing encounter data) + Tier 3 (corpus) | Demonstrates environmental factor quantification; AF447 case study relevance |

Remaining LOC-I nodes get Tier 3 (corpus-derived Bayesian) probabilities by default.

---

## 5. Workstream 3: Counterfactual Analysis & Intervention

### 5.1 Intervention Types

An intervention modifies the quantified KG in one of three ways:

| Intervention Type | Graph Operation | Example |
|---|---|---|
| **Remove factor** | Set P(node) = 0, re-propagate | "What if pilot fatigue were eliminated?" |
| **Reduce probability** | Multiply P(node) by reduction factor, re-propagate | "Fatigue monitoring reduces P(fatigue error) by 60%" |
| **Add barrier** | Insert new node on a KG edge with barrier effectiveness | "Adding TAWS with P(detection) = 0.95" |

### 5.2 Counterfactual Computation

For an intervention on node X:

1. **Baseline risk:** Compute P(outcome) with current KG probabilities using BN inference
2. **Intervened risk:** Apply the intervention (modify P(X) or add barrier), re-compute P(outcome)
3. **Risk reduction:** ΔR = P_baseline - P_intervened. Relative reduction = ΔR / P_baseline
4. **Uncertainty:** Monte Carlo over tier-specific distributions for both baseline and intervened states. Report CI on the risk reduction.

Formally, this is:

```
P(Y=accident | do(X=absent)) vs P(Y=accident | current_state)

Risk reduction = 1 - P(Y|do(X=absent)) / P(Y|current_state)
```

where `do(X=absent)` is Pearl's do-operator applied to the Bayesian network.

### 5.3 Demo Case Studies

Two well-known LOC-I accidents where the causal chain is well-established and interventions have been implemented:

#### Case Study 1: Colgan Air 3407 (2009)

- **Accident:** Bombardier Q400, approach to Buffalo, 50 fatalities
- **Key factors:** Pilot fatigue (commuting), inadequate stall training, improper stall recovery (pulled back instead of pushing forward)
- **Interventions implemented post-accident:** FAA fatigue rules (Part 117), upset prevention and recovery training (UPRT) mandate
- **ACE-Graph analysis:**
  - Quantify P(fatigue error) using HEART + SAFTE-FAST with the actual crew rest data
  - Quantify P(improper stall recovery | no UPRT) from corpus
  - Apply intervention: P(fatigue error) reduced by Part 117 compliance, P(improper recovery) reduced by UPRT
  - Report risk reduction and compare against post-intervention accident rate trends

#### Case Study 2: Air France 447 (2009)

- **Accident:** A330, Atlantic Ocean, 228 fatalities
- **Key factors:** Pitot tube icing → unreliable airspeed → autopilot disconnect → pilot spatial disorientation → stall → LOC-I
- **Interventions:** Improved pitot probes (Thales), revised training for unreliable airspeed
- **ACE-Graph analysis:**
  - Quantify P(pitot icing) from environmental encounter data + component reliability
  - Quantify P(LOC | unreliable airspeed, insufficient training) from corpus + HRA
  - Apply intervention: upgraded pitot tubes reduce P(icing) by X%; revised training reduces P(incorrect response) by Y%
  - Report total risk reduction for the LOC-I chain

### 5.4 Validation

- **Internal consistency:** Risk reduction estimates should be monotonic (adding a barrier should never increase risk) and bounded (removing one factor should not eliminate all risk if other paths exist)
- **Historical calibration:** Compare predicted risk reduction from post-accident interventions against actual observed accident rate changes in the years following implementation (using NTSB/FAA statistics)
- **CAST comparison:** Compare ACE-Graph's computed risk reduction for specific interventions against CAST Safety Enhancement estimates (which are expert-derived)
- **Investigator agreement:** For the case studies, verify that ACE-Graph identifies the same key contributing factors as the NTSB investigation

---

## 6. Visualization

### 6.1 Interactive KG Viewer

The visualization is a core deliverable — the framework must be intuitive to explore.

**Top-level view:** Force-directed or hierarchical layout of the per-category causal KG.
- Nodes colored by **risk level** (heatmap: green → yellow → red)
- Node size proportional to **risk contribution** to the outcome
- Edge thickness proportional to **conditional probability strength**
- Tier indicator badge on each node (T1/T2/T3/T4 with color coding)

**Click-to-expand:** Clicking a node reveals its internal hierarchical decomposition:
- Fault tree view for mechanical nodes
- HRA breakdown for human factor nodes
- Sub-component probabilities with tier labels and CIs

**Intervention mode:** User selects an intervention:
- Affected nodes highlighted
- Before/after risk values shown side-by-side
- Risk reduction percentage displayed at the outcome node
- Confidence interval visualized as an error bar or range

**Bow-tie view:** Alternative layout showing threats → barriers → hazard → barriers → consequences for a selected accident category, with quantitative probabilities on each path.

### 6.2 Technology (preliminary)

- **Graph rendering:** D3.js or Cytoscape.js for interactive web-based KG
- **Fault tree rendering:** Custom SVG or adapting existing FTA visualization libraries
- **Dashboard:** Streamlit or Panel for rapid prototyping; React for production
- Technology choices are preliminary and will be finalized during implementation planning.

---

## 7. KG Node Schema

Each node in the causal KG carries:

```
KGNode
├── identity
│   ├── node_id                    # unique identifier
│   ├── label                      # human-readable name
│   ├── node_type                  # {mechanical_component, human_factor,
│   │                              #  environmental, organizational,
│   │                              #  operational, outcome}
│   └── cictt_category             # which accident category KG this belongs to
├── causal_edges
│   ├── parents[]                  # nodes that cause this node
│   ├── children[]                 # nodes this causes
│   └── edge_weights{}             # conditional probabilities on edges
├── hierarchical_model
│   ├── model_type                 # {fault_tree, hra, bowtie, none}
│   ├── sub_nodes[]                # internal decomposition tree
│   ├── gate_logic[]               # AND/OR gates (for fault trees)
│   └── model_parameters{}         # model-specific params (HEART task type, etc.)
├── quantitative
│   ├── probability                # P(failure) — propagated from internal model
│   ├── tier                       # {1, 2, 3, 4} — primary data source
│   ├── tier_mix                   # e.g., {1: 0.3, 2: 0.5, 3: 0.2} for mixed sources
│   ├── confidence_interval        # [lower, upper]
│   ├── exposure_basis             # "per_flight_hour" | "per_departure" | "conditional"
│   └── last_updated               # timestamp
└── provenance
    ├── corpus_support             # number of UASC records mentioning this factor
    ├── data_sources[]             # which databases/papers provide the probability
    └── harmonization_flags{}      # inferred vs. original data
```

---

## 8. Evaluation Plan

### 8.1 Per-Workstream Evaluation

| Workstream | Metric | Method |
|---|---|---|
| **1. KG construction** | Causal edge P/R/F1 vs. NTSB Findings | Same as v2 Section 8 |
| **2. Risk formulation** | Probability calibration | Compare Tier 3 corpus-derived rates against Tier 2 published rates for nodes where both exist; report calibration plot |
| **2. Risk formulation** | Tier coverage | % of nodes at each tier; target: showcase nodes at Tier 1-2, >80% of nodes at Tier 3+ |
| **3. Counterfactual** | Risk reduction accuracy | Compare predicted ΔR against historical accident rate changes post-intervention |
| **3. Counterfactual** | CAST agreement | Compare computed risk reduction against CAST Safety Enhancement estimates |
| **3. Counterfactual** | Investigator agreement | Same as v2 — agreement with NTSB probable cause on contributing factors |

### 8.2 Baselines

| Baseline | What it tests |
|---|---|
| **CATS model published results** | Does our automated approach match manually constructed CATS risk estimates? |
| **Single-stage LLM** | Give GPT-4/Claude the narrative + ask for risk quantification directly |
| **KG without quantitative layer** | Qualitative causal graph only (v2 approach) — how much does quantification add? |
| **Flat corpus frequency (no hierarchy)** | Skip fault tree decomposition, assign corpus frequency directly to top-level nodes |
| **Expert-only estimates** | Human safety analysts estimate risk reduction for the case studies — does the automated system match? |

---

## 9. Limitations and Honest Framing

### 9.1 Inherited from v2

- NTSB dominance in the corpus (80-90% of reasoning-ready records)
- Ground truth is investigator judgment, not objective causal truth
- LLM circularity risk in extraction stages

### 9.2 New in v3

- **Tier 3 reporting bias:** Corpus-derived probabilities reflect accident-conditional frequencies, not true failure rates. We clearly distinguish "involvement rate" from "failure rate."
- **Sparse data for rare nodes:** Some nodes have very few corpus occurrences, leading to wide uncertainty. The framework is honest about this via confidence intervals.
- **Human factors model limitations:** HEART/CREAM were developed for industrial settings; aviation-specific calibration is limited. We note this and compare against aviation-specific HRA literature where available.
- **Physics model scope:** Only 1-2 nodes get Tier 1 physics models in the demo. The framework claims extensibility but the paper only demonstrates it for a narrow set of failure modes.
- **Edge probability estimation:** Conditional probabilities on KG edges (P(child | parent)) are harder to estimate than node probabilities. Co-occurrence in accident reports gives an upper bound but conflates correlation with causation. The BN structure from Workstream 1 partially addresses this, but edge parameterization remains a weakness.

---

## 10. Computational Budget and Reproducibility

Same protocol as v2 Section 11: LLM call caching, temperature=0, 3-run variance reporting, version pinning.

Additional for v3:
- Fault tree computations are deterministic given input probabilities
- Monte Carlo uncertainty propagation: 10,000 samples per analysis, seed fixed for reproducibility
- BN inference using exact methods (junction tree) where tractable, approximate (loopy belief propagation) for large graphs

---

## 11. Ethical Considerations

Same as v2 Section 12, with one addition:

- **Quantitative risk estimates must not be misinterpreted as safety certification.** The system produces research-grade risk estimates, not certification-grade safety assessments per SAE ARP 4761. Any use of these numbers outside a research context would require extensive independent validation. This caveat must be prominent in any publication or tool release.

---

## 12. References

### Core pipeline ingredients (from v2)
1. Vashishtha et al. *Causal Order.* ICLR 2025.
2. Jin et al. *CLadder.* NeurIPS 2023.
3. Zhang et al. *Laplacian similarity prior.* NeurIPS CaLM 2024.
4. Zhao et al. *HABERT.* INFORMS JDS 2025.
5. Zhao. *Hierarchical sequential event prediction.* ASU Dissertation, 2022.

### Quantitative risk frameworks (new in v3)
6. Ale, Bellamy, et al. *Causal model for air transport safety (CATS).* Safety Science, 2006.
7. Roelen et al. *CATS — Final report.* NLR, 2011.
8. SAE ARP 4761. *Guidelines for Conducting Safety Assessments of Civil Airborne Systems.* Rev A, 2023.
9. SAE ARP 4754A. *Development Assurance for Airborne Systems.* 2010.
10. FAA AC 25.1309-1A. *System Design and Analysis.* 2002.

### Human reliability analysis
11. Williams, J.C. *HEART — A proposed method for assessing and reducing human error.* 9th Advances in Reliability Technology Symposium, 1986.
12. Hollnagel, E. *Cognitive Reliability and Error Analysis Method (CREAM).* Elsevier, 1998.
13. Dawson et al. *SAFTE-FAST: Fatigue Avoidance Scheduling Tool.* Institutes for Behavior Resources, 2011.

### Aviation safety frameworks (from v2)
14. Shappell & Wiegmann. *HFACS.* DOT/FAA/AM-00/7, 2000.
15. Leveson. *STAMP/STPA.* Safety Science, 2004.
16. Rasmussen. *AcciMap.* Safety Science, 1997.
17. Pearl. *Causality.* 2nd ed., 2009.

### Component reliability
18. MIL-HDBK-217F. *Reliability Prediction of Electronic Equipment.* DoD, 1991.
19. NPRD-2016. *Nonelectronic Parts Reliability Data.* RIAC, 2016.
20. Leverant et al. *DARWIN — Turbine rotor probabilistic risk assessment.* ASME J Eng Gas Turbines Power, 2004.

### Causal discovery and inference (from v2)
21. Spirtes, Glymour & Scheines. *Causation, Prediction, and Search.* MIT Press, 2000.
22. Peters, Janzing & Scholkopf. *Elements of Causal Inference.* MIT Press, 2017.
23. Luxhoj et al. *Bayesian network model for aviation safety risk.* FAA/Rutgers, 2003.
24. Ancel et al. *Bayesian network for system-level aviation safety.* NASA, 2015.

### Bow-tie and airline risk management
25. Reason, J. *Managing the Risks of Organizational Accidents.* Ashgate, 1997.
26. ARMS Working Group. *Airline Risk Management Solutions: Event Risk Classification.* 2010.

---

## Appendix A: v2 → v3 Change Summary

| Change | Rationale |
|---|---|
| Added Workstream 2 (Risk Formulation) | Core v3 contribution: quantitative risk at every node |
| Added 4-tier probability system | Handles reality that data quality varies across nodes |
| Nodes become hierarchical containers | Fault trees / HRA models embedded inside KG nodes |
| Counterfactual method changed from LLM QA to Bayesian propagation | Produces numerical risk reduction instead of text answers |
| Paper scope narrowed to LOC-I demo | Framework paper with 3 showcase nodes is more publishable than incomplete full implementation |
| Added KG node schema (Section 7) | Formal definition of what each node carries |
| Added visualization requirements (Section 6) | Interactive KG with click-to-expand hierarchy is a core deliverable |
| Added CATS, ARP 4761, HEART/CREAM to references | Key prior art for the quantitative layer |
| Added new limitations (Section 9.2) | Reporting bias, edge parameterization, HRA model validity |
| Retained all v2 content for Workstream 1 | KG construction pipeline unchanged |

## Appendix B: Open Questions

1. **HEART vs. CREAM for HRA:** Which HRA method is better suited? HEART is simpler and more widely used; CREAM is more systematic. Decision deferred to implementation.
2. **BN inference scalability:** For large KGs (hundreds of nodes), exact BN inference may be intractable. May need approximate methods. Assess during LOC-I demo.
3. **Edge probability estimation:** Best method for estimating P(child | parent) on KG edges — corpus co-occurrence, LLM-elicited, or learned from data? Critical open question.
4. **Exposure normalization:** Which exposure metric (flight hours, departures, aircraft-years) is most appropriate for Tier 3 corpus-derived rates? May vary by factor type.
5. **Visualization library:** D3.js vs. Cytoscape.js vs. other. Decision deferred to implementation planning.
