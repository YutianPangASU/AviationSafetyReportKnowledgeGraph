# Hybrid Fusion and Risk Validation: Assessment and Direction

2026-09-05. Full-repository review (code, outputs, manuscript, both
reviewer documents) with a feasibility check of NTSB injury/damage fields as
risk ground truth. Numbers quoted below come from
`physics/severity_groundtruth_check.py` and are written to
`physics/out/severity_groundtruth_check.json`.

## 1. What the work is, restated

Five stages, all implemented:

| Stage | Object | Status |
|---|---|---|
| 0 | 178k-record corpus from NTSB (Pre2008 + avall), FAA AIDS, TSB, BEA; 56,202 kept after length filter | done |
| 1 | v4 ordered causal chains, 57-term vocabulary, constrained decoding, `caused_by` back-references | done; recall 0.91/0.93, zero schema violations, gold496 multi-model |
| 2/3 | Causation KG (47 factors, 434 edges), factor vectors (55,940 x 47), per-category PC DAGs with LLM order + Laplacian priors, backdoor do() rankings | done |
| Physics | Icing (thermodynamic chart), stall (JSBSim), fuel endurance, Pratt gust/shear/overload; signed guard margins; `guards.py` registry; corpus violation sweep | done; only icing tested against data |
| Fusion | R = pi_F x P(Y given F); noisy-OR over parents; chain automaton with guards; two do() forms | written; Eq. (5) calibration never executed |

Manuscript claim: separate *which factors lead to a failure* (corpus) from
*how likely that failure is at an operating point* (physics). Two reviewer
documents agree the extraction is the strongest part and the causal/risk
layer claims more than the estimates support (Yifang M1 to M6), and that the
defensible novelty is a per-case physics margin that judges chain
admissibility (Xinyu).

## 2. Where the innovation actually is

Revised 2026-09-05 after discussion. The user's point stands: measured
against computer-science extraction papers, the extraction *method* is not
novel. A 35B model with constrained decoding against a JSON grammar is
standard practice. What the paper can legitimately claim, in the order it
should list them:

1. **The hybrid formulation (method).** The causal chain as a path over the
   modes of a guarded transition system, with per-case physics margins as
   guards, and a physics-informed probabilistic model as its quantitative
   completion (Section 4.1). Nobody rejects or re-weights a
   text-reconstructed causal account on physical infeasibility. This is the
   contribution both reviewers isolate, and it is the one to build the paper
   around. Evidence today is thin (0.5% of transitions checkable, three of
   four guards fail on model adequacy) and Section 6 says how to thicken it.
2. **The domain schema and the corpus-scale causation graph (resource).**
   The schema is the domain-specific part of the extraction and is not in
   any prior paper: an ordered chain with backward-only `caused_by`
   references (acyclic by construction), condition / event / outcome
   classes carrying the latent-versus-active distinction of Reason and
   Rasmussen, a 57-value vocabulary organized along HFACS layers and aligned
   with CICTT categories, link strengths that mirror the NTSB cause and
   factor flags so the output is scoreable against investigator coding,
   evidence spans on every link, and grounded prompting with the coded
   findings in context. The twelve first-class conditions were derived from
   a drift analysis of 106k free-text condition nodes, which is itself a
   defensible design step (docs/2026-07-05-v4-condition-vocab.md). The
   artifact that results, 56k chains and a 47-node, 434-edge causation graph
   with per-category backdoor tables, is a resource contribution in its own
   right. State it as a resource, not as a method.
3. **The layer-wise evaluation protocol (evaluation).** Recall against
   investigator cause flags with a cross-family judge, ordering agreement
   against coded sequences with the coding-convention finding, three model
   tiers, cross-jurisdiction transfer, the corrupted-guard control, and, if
   Section 5 is executed, per-layer validation against realized injury and
   damage. Reviewers in this field have not seen a system evaluated this
   way.
4. **The failure studies (application).** Loss of control and engine power
   loss end to end, with counterfactuals attributed to named physical
   drivers.

The fused risk number as it stands today is not on this list. It becomes the
quantitative half of item 1 once rebuilt (Section 4.1), with the current
product and noisy-OR retained as the baseline it must beat.

### 2.1 Show the schema as code in the paper

Recommendation, settled 2026-09-05: yes, as an appendix. The field table
(Table tbl:schema) stays in Section 3.2 as the compact summary, and
Appendix A grows from the vocabulary table into the full schema. A table
cannot show the two things that make the schema a contribution: that
causality is carried by backward references inside the nodes, and that a
chain is a readable object a domain expert can audit. A listing shows both
in twenty lines. Concretely:

- One figure in Appendix A, after the vocabulary table, in two panels:
  (a) the record type in a compact typed notation (21 lines), (b) one real
  extracted chain from the corpus containing a condition, an event, an
  outcome, and both link strengths, with the evidence quotes kept. NTSB
  record 20020225X00252 is the chosen instance: carburetor icing conditions
  at the recorded temperature and dewpoint, delayed carburetor heat, power
  loss in cruise, forced landing on unsuitable terrain. The reader meets the
  worked icing factor of Section 4.3 again in the appendix.
- A lead-in paragraph stating the four constraints the grammar cannot
  express and the runner enforces (index equals position, every parent
  index is smaller, outcome nodes parent only outcomes, repeats of a type
  merge), and naming the released files (schema, system prompt, few-shot
  example) as matching the listing verbatim.
- One pointer sentence at the end of the "Schema" paragraph in Section 3.2.
- Preamble: `\usepackage{listings}` plus a style definition, and a figure
  counter reset for the appendix, since the appendix currently renumbers
  tables only. All of this is in docs/2026-09-05-schema-listing.tex, which
  compiles as a drop-in block with Figure A.1 numbering.

Why this helps beyond presentation: a safety-journal reviewer sees the
object being aggregated; a CS reviewer sees the grammar and can judge the
constrained-decoding claim; and the "resource" framing of item 2 becomes
concrete instead of a pointer to a repository.

## 3. Why the current fusion is naive, with evidence

**3.1 The physics probability is not a probability of anything measurable.**
Decile reliability of pi_ice (descent power) against the realized icing
factor across 39,770 weather-recorded NTSB accidents:

| pi_ice decile mean | 0.00 | 0.06 | 0.42 | 0.62 | 0.74 | 0.83 | 0.87 | 0.91 | 0.93 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| realized icing rate | 0.009 | 0.014 | 0.028 | 0.028 | 0.028 | 0.034 | 0.032 | 0.031 | 0.043 | 0.055 |

The ordering is right (six-fold lift bottom to top, AUC 0.612) and the level
is off by a factor of 20 to 100. Brier score 0.50 against a base-rate Brier of
0.029. This is Yifang's M1 in one table: pi_ice is a chart-zone label, not
P(icing | weather) on any population, and everything multiplied by it
inherits the wrong level.

**3.2 The physics term carries no severity information.** pi_ice against
serious-or-fatal has AUC 0.556; within the 1,205 icing accidents it is 0.546.
Adding pi_ice and wind to a logistic model of the extracted factors moves
severity AUC from 0.818 to 0.820. The physics layer, as built, informs
*occurrence* of a few factors and nothing downstream. Any claim that the
hybrid risk is "more accurate" than the corpus alone would fail a per-record
test on severity today.

**3.3 The combination rule is structurally wrong.** Noisy-OR link strengths
must be each cause acting alone; the code plugs in marginal conditionals
P(Y|X_i) over parents that cause one another, so it double counts, and the
independence assumption makes it undershoot (0.137 vs 0.175, 0.212 vs
0.293). Icing enters LOC twice (directly and through engine failure). The
LOC code is additive where the paper is multiplicative, and the scenario
weather differs between code (5, 3) and text (13, 12). A further
definitional mix: the edge strength P(EF | ice) read from explicit
`caused_by` links is 0.667, while the same-record co-presence conditional is
0.950; the noisy-OR uses the former as if it were the latter.

**3.4 Physics coverage is asserted more than realized.** Density altitude is
never recorded (0% fill in both databases), so the phase-prior stall guard is
never checkable. Gust is recorded 12%. Fuel on board is recorded 31% in
avall only. The chain-level activation model of Eq. (pathrisk) has never been
run on an actual extracted chain per record; the failure studies use the
depth-two special case at population base rates.

**3.5 The counterfactuals are interventions.** Everything computed is rung 2.
The physics guards are the one place a rung-3 quantity is cheaply available
(Section 4.1, step 5).

## 4. Better ways to do the hybrid calculation

Four options, ranked. Option A is the recommendation; B is its ablation; C
is the follow-up that fixes the population; D is the minimal patch.

### 4.1 Option A: physics-informed Bayesian network on the causation graph (recommended)

Turn the causation graph into an actual generative model instead of a
bookkeeping product. Concretely:

1. **Nodes.** The 47 causal factors plus the outcome, *and the recorded
   continuous drivers as exogenous nodes*: temperature, dewpoint, surface
   wind, light condition, phase of flight, aircraft class. Adding the drivers
   as nodes also answers M3 (causal sufficiency), since the hidden common
   causes the icing analysis exposed become explicit.
2. **Guarded-node CPDs from physics through a calibrated link.** For a node
   with a guard, P(F | drivers, parents) = link(gamma_F(theta)) with the
   *shape* fixed by the margin and the *level* fixed by a one- or
   two-parameter monotone link fit on the corpus (isotonic or Platt). The
   feasibility run shows this works out of fold for icing: isotonic on the
   icing margin gives Brier 0.0292 (base rate 0.0294, raw pi_ice 0.50),
   mean 0.0303 against prevalence 0.0303, AUC 0.605. The fused depth-two
   quantity P_cal(ice|wx) x P(EF|ice) then has mean 0.0288 against a
   realized joint rate of 0.0288. Eq. (5) becomes executable today as a
   per-record link rather than a scalar rescale, with the population
   stated honestly as "given a reportable accident."
3. **Data-node CPDs fit by maximum likelihood.** Leaky noisy-OR per node with
   parents from the learned DAG, fit on the factor-presence table (Heckerman
   / Onisko). This removes the marginal-conditional double count (M5) and
   the undershoot, and keeps the per-parent decomposition the paper values.
4. **Outcome node.** Ordinal five-level severity (or the two-axis damage x
   injury, Section 5.2) with parents = the terminal event nodes, fit on the
   corpus. This is what makes the whole thing scorable against NTSB fields.
5. **Inference and counterfactuals.** Fifty nodes with at most three parents
   is exact or cheap-approximate inference. A(v | O) for every node per
   record replaces the never-run Eq. (pathrisk). The guarded nodes have
   structural equations F = 1{gamma_F(theta, xi) >= 0} with an explicit
   noise term, so abduction of xi from the observed record gives a genuine
   rung-3 quantity: for a specific icing accident, the probability of
   necessity of the carburetor-heat omission,
   PN = P(no power loss | do(heat), observed power loss without heat).
   Monotonicity of the icing mechanism identifies PN (Tian and Pearl). One
   worked accident with bounds makes the title true (M4).

What this buys: one model, one population, one likelihood, per-record
outputs that can be validated, the existing decomposition preserved, and the
physics entering exactly where the formalism says (guarded nodes) with a
countable coverage number. The chain automaton stays as the admissibility
layer; the BN is its probabilistic completion, which is what Section 3.4
already claims Eq. (pathrisk) is.

**The current fusion is the baseline.** Agreed 2026-09-05: the product rule
and noisy-OR of the manuscript are kept as arm B0 and every alternative is
scored against them on the same records, same split, same metrics. The
comparison table the paper needs:

| Arm | What it is | Isolates |
|---|---|---|
| B0 | current fusion: chart-zone pi_F x edge conditional, noisy-OR at corpus base rates (manuscript Eqs. 6 and 7) | the starting point |
| B1 | B0 with leaky noisy-OR fit by maximum likelihood and the icing double path removed (Option D) | the combination-rule defect alone |
| B2 | Option A with the physics links replaced by corpus CPDs (no physics anywhere) | what the physics adds |
| B3 | monotone-covariate held-out model on margins + factors + pre-outcome covariates (Option B) | the pure prediction ceiling |
| A | physics-informed BN with calibrated links on guarded nodes | the proposal |
| A-corrupt | A with sign-inverted or permuted margins | that A used the physics |

Scored per layer as in Section 5.3: occurrence AUC and reliability on the
guarded factors, consequence calibration, fused per-record risk against
realized (factor and severe) and ordinal severity, all on a 2015-onward
test split with a TSB Canada transfer row where the labels exist. B0 is
expected to lose on calibration by a wide margin (Section 3.1) and to tie
or lose on discrimination; A must beat B2 on the guarded factors or the
physics claim is not earned.

Effort: two to three weeks. pgmpy or a hand-rolled noisy-OR BN is enough.

### 4.2 Option B: margins as monotone covariates in a held-out predictive model (the ablation)

Xinyu's standard: a computer-science claim judged on held-out prediction.
Severity (ordinal) or next-event ~ physics margins with monotone constraints
+ categorical factors + pre-outcome covariates, gradient boosting with
monotone constraints or an ordinal GLM. Physics contribution measured by
ablation (arms: no physics, retrieved constants, per-case margins,
corrupted margins). Cheap, and it is the benchmark row Option A must beat.
On today's data the honest result is that physics adds nothing to severity
prediction (0.818 to 0.820); it adds to *occurrence* prediction. Report that
rather than hide it, and use it to fix the claim: the physics layer conditions
occurrence on the operating point, the corpus carries consequence.

### 4.3 Option C: exposure-based hazard model (fixes the population)

The only route to per-flight or per-flight-hour risk. Hazard
lambda(theta) = lambda_0 x LR(theta), with LR from the physics link, the
climatology of theta from NOAA ISD/METAR at GA-representative airports
weighted by FAA GA Survey hours, and lambda_0 from accident counts over
those hours. This is Yifang's R2 and the paper's own first limitation. It
also upgrades the icing validation from accidents-versus-accidents to
hazardous-weather-versus-routine-weather. Two to three weeks, all public
data.

### 4.4 Option D: minimal patch

Leaky noisy-OR by MLE, drop the icing double path, align code and text,
report risk ratios only. One week. Fixes M5 and the numeric inconsistencies
without changing the story. Do this regardless if A is out of scope.

## 5. Validating risk with NTSB injury and damage fields

### 5.1 What the data already provide

Severity ground truth is already joined: `factor_vectors.parquet` carries a
resolved five-level `outcome_severity` for 55,940 records (structured injury
and damage override the model's guess). The NTSB join adds, for 43,507 NTSB
records (39,770 with weather):

| Field | Table | Values | Fill |
|---|---|---|---|
| `ev_highest_injury` | events | FATL / SERS / MINR / NONE | 99.8% |
| `inj_tot_f/s/m/n` | events | counts aboard and on ground | high |
| `damage` | aircraft | DEST / SUBS / MINR / NONE (49 CFR 830.2) | 97% |
| `injury_level` x `inj_person_category` | injury | per person category | high |
| `crew_inj_level` | Flight_Crew | per crew member | high |
| `ev_type` | events | ACC / INC | 1,139 INC only |
| survivability, injuries | TSB Canada CSVs | external, per occurrence | for the 1,500 TSB summaries |

Damage is coarse and skewed (SUBS is 81%); injury is the informative axis.
The two are only loosely coupled: DEST pairs with FATL 4,012 times and with
NONE 626 times; SUBS pairs with NONE 22,463 times.

### 5.2 Ground-truth constructions worth building

1. **Ordinal five-level severity** (exists). Score with the ranked
   probability score and Somers' D, not only binary AUC.
2. **Two-axis severity class**: damage x highest injury, mapped to an
   ICAO/ARMS-style severity matrix (catastrophic / hazardous / major / minor
   / negligible). Predict the joint and score each axis; the framework's
   physics should move the damage axis (energy at impact) differently from
   the injury axis (survivability).
3. **Fatality fraction aboard**: inj_tot_f / (f + s + m + n). Continuous,
   less coarse than the class, directly comparable with survivability
   literature.
4. **Occurrence labels**: the extracted factor presence (already the target
   of the icing AUC), and, from the coded `Findings` / `seq_of_events`, the
   investigator-flagged factor presence as an extraction-independent version.
5. **Accident versus incident**: 1,139 NTSB incidents is too few, and the
   icing physics scores them the wrong way (AUC 0.43 for ACC vs INC), because
   incidents are a different operation mix. The FAA AIDS portion of the
   corpus (12,585 records, mostly incidents) is a better partial control but
   lacks joined weather. NASA ASRS would be the true non-accident population.

### 5.3 The validation design: score each layer against the thing it predicts

The single most important design decision: **do not validate the physics
occurrence term against severity.** It carries none (Section 3.2), and it
should not: icing decides whether power is lost, not how the forced landing
ends. The framework has three layers and each has its own ground truth.

| Layer | Quantity | Ground truth | Metrics | Status |
|---|---|---|---|---|
| Occurrence (physics) | P(F | drivers) | factor label (extracted, and coded Findings) | AUC vs ceiling; reliability by decile; pairwise weather-bin ordering | AUC done; reliability and pairwise added here (0.78 of 464 bin pairs agree) |
| Consequence (corpus) | P(severity | F, parents) | realized injury/damage | held-out calibration per factor, RPS on ordinal, time-blocked | not done |
| Fused per-record risk | P(F and severe | O), A(Y | O) | realized (F and severe), ordinal severity | AUC, PR-AUC, Brier, reliability, RPS, risk-ratio agreement per bin | not done; depth-two icing case checked here |
| System | answers to templated questions | corpus-derived (Yifang Q1 to Q5) | leaderboard vs simpler systems | not done |

Splits: by time (test on 2015 onward; coding conventions changed in 2008),
by jurisdiction (TSB injuries CSV), by category (LOC-I, SCF-PP first).

### 5.4 Pitfalls the design has to state

1. **Narrative leakage.** The extracted factors are post-hoc reconstructions
   written by investigators who know the outcome. Pre-outcome covariates
   alone (weather, light, phase, aircraft class, FAR part) predict
   serious-or-fatal at AUC 0.763; extracted factors alone 0.829; both 0.861.
   Part of the 0.829 is the narrative describing the outcome (spatial
   disorientation is inferred from a fatal impact). Every validation must
   name its information set: pre-outcome only (what a risk model could know
   before the flight), or post-hoc (what an analyst knows after). The hybrid
   framework's honest per-flight claim lives in the first set.
2. **Selection on the accident.** All severity is conditional on S = 1. State
   it in an estimand box; use Option C to leave the accident population.
3. **Era and coding.** Pre2008 and avall code causes in different tables;
   temperature units and sentinel (0, 0) pairs need the hygiene the
   violation script already applies.
4. **Multi-aircraft events.** Join on (ev_id, Aircraft_Key), never ev_id
   alone; damage is per aircraft, highest injury is per event.
5. **Coarse damage.** Substantial is the regulatory threshold for an
   accident, so 81% of records sit there by construction; treat damage as a
   three-level variable (minor-or-none / substantial / destroyed).

### 5.5 Concrete plan

1. `physics/severity_groundtruth_check.py` (written today) builds the join
   and the layer-wise baselines. Extend it into an evaluation module that
   writes one table per layer.
2. Fit the calibrated icing link on the corpus and replace `pi_ice` in the
   fused number; re-report the icing worked example with the population
   stated. One day; the feasibility run is the prototype.
3. Build Option A for LOC-I and SCF-PP; score A(Y | O) per record against
   ordinal severity on a 2015-onward test split, with the Option B rows as
   ablation. Two to three weeks.
4. One rung-3 probability of necessity for a specific icing accident. Three
   days once Option A exists.
5. Option C exposure study with ISD climatology. Follow-up paper or
   appendix.

## 6. What to improve, ranked by value per effort

1. **Execute the calibration as a per-record link and rebuild the fusion
   (Sections 3.4 to 3.6 of the manuscript).** This is the make-or-break item
   for both reviewers and the feasibility run shows the fix works. Option A
   in full if time allows; Option D at minimum.
2. **Evaluate the fused number per record against realized severity.**
   Section 6 today validates components, never the fused output. One table
   with the layer-wise design of Section 5.3, including the honest result
   that physics moves occurrence and not severity.
3. **Name the information set and the population in every number.** An
   estimand box; pre-outcome versus post-hoc; "given a reportable accident."
4. **Fix or retire the wind guards.** They reject 68 to 98% of what they
   check and permutation does not change that, so they measure the surface
   wind, not the extraction. Either source turbulence aloft (HRRR/RAP
   reanalysis at the event time and location, both recorded) or drop them
   from the headline table and keep them as the model-adequacy diagnostic.
5. **Add guards whose drivers are recorded.** Glide reach (apt_dist is
   recorded 99.5%; altitude at power loss often in the narrative), fuel
   endurance for the 31% of avall records with fuel on board, takeoff and
   climb performance from field elevation and temperature (density altitude
   can be computed from apt_elev, wx_temp and altimeter even though
   wx_dens_alt is blank). Coverage would rise from 0.5% of transitions to
   something worth reporting.
6. **One genuine counterfactual** (PN for carburetor heat). Cheapest way to
   justify the word in the title.
7. **Drivers as nodes in discovery** (M3), with the backdoor table
   re-reported. Falls out of Option A.
8. **Manuscript hygiene** from Yifang P1 to P8: placeholder references, dead
   link, highlight lengths, code-text mismatches (additive vs multiplicative,
   (5, 3) vs (13, 12)), repository cleanup before the link goes live.

## 7. Numbers from today's feasibility run

| Check | Result |
|---|---|
| NTSB records with resolved severity, weather | 43,507; 39,770 |
| pi_ice AUC vs icing label / EF / serious-fatal / fatal | 0.612 / 0.493 / 0.556 / 0.573 |
| pi_ice AUC vs serious-fatal within icing accidents | 0.546 |
| pi_ice Brier vs icing label; base rate | 0.501; 0.029 |
| isotonic link on icing margin, out of fold: AUC / Brier / mean vs prevalence | 0.605 / 0.029 / 0.030 vs 0.030 |
| fused P_cal(ice) x P(EF given ice): mean vs realized joint | 0.0288 vs 0.0288 |
| pairwise weather-bin ordering agreement | 362 of 464 (0.78) |
| severity AUC, logistic: factors / physics drivers / both | 0.818 / 0.555 / 0.820 |
| severity AUC, time-blocked (train < 2010, test >= 2010, n = 13,831) | 0.812 |
| severity AUC, GBM: pre-outcome covariates / extracted factors / both | 0.763 / 0.829 / 0.861 |
| fatal AUC, logistic: factors / physics / both | 0.886 / 0.571 / 0.887 |
| density altitude fill; gust fill; fuel-on-board fill (avall) | 0% ; 12% ; 31% |
