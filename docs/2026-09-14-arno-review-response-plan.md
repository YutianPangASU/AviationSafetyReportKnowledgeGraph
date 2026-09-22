# Response Plan for Arno's Review (received 2026-09-14)

Investigation of the Overleaf update against the code, the data, and the
manuscript, with a revision plan per comment. Numbers marked "computed
today" come from the four scripts in `physics/review_checks/` (run with the
repo-root `python3`; `wind_by_phase.py` and `icing_perm_p.py` need
`mdb-export` and the two NTSB `.mdb` files; `icing_delong_bayes.py` needs
scikit-learn).

## 0. What arrived from Overleaf

Commit `cd7b528` on `overleaf/main`, fast-forwarded into `paper/main`
(`paper/` is its own git repository with the Overleaf remote):

- `arno_aviation_comments.txt`: venue recommendation and 12 conceptual
  comments. Arno asks for a written reply stating what was done per point
  and a revised manuscript.
- `Aviation_Safety_Causation_Knowledge_Graph_annotated.pdf`: 70 distinct
  editorial annotations (142 raw objects, each highlight paired with a
  note). The deduplicated list, numbered as in Section 3 below, is written
  to `paper/arno_annotations.md`.
- Two edits to `main.tex` made in the Overleaf editor (the commit carries
  the Overleaf identity, so the author is whoever edited the project):
  - Highlight 1: "per-case physics margins acting as guards" was changed to
    "acting as safety assurance". Decide whether to keep it. "Guards" is the
    term the formalism uses; "safety assurance" is vaguer and does not match
    Section 3.4.
  - The Acknowledgment section was commented out. Presumably pending the
    funding statement for the non-NASA authors (comment 12).

Equation and float numbers Arno cites map to labels as follows: Eq. (2)
`eq:leading`, (9) `eq:calib`, (10) `eq:fusion`, (11) `eq:combine`, (13)
`eq:backdoor`, (14) `eq:phys_do`, (16) `eq:engfail`, (18) `eq:icewater`;
Table 4 `tbl:locfactors`, 6 `tbl:iceedges`, 7 `tbl:effactors`, 8
`tbl:extract`, 9 `tbl:violation`, 10 `tbl:ceiling`, 11 `tbl:backdoor`;
Fig. 2 `fig:lockg`, 3 `fig:locdag`, 4 `fig:factorphys`, 5 `fig:icing`, 6
`fig:loc`, 7 `fig:efkg`, 8 `fig:efdag`, 9 `fig:ef`.

## 1. Verdict in one page

Arno's two lead concerns are correct and the code confirms both. The
calibration of Eq. (9) is not executed anywhere (`loc_risk_model.py`,
`ef_risk_model.py`, and `carb_icing_model.py` multiply the chart
probability by an accident-conditioned edge directly), and no experiment
scores the fused number against either layer alone. The 2026-09-05
assessment (`docs/2026-09-05-hybrid-fusion-and-risk-validation-assessment.md`)
reached the same two conclusions and already prototyped the fixes; Arno's
review is the external confirmation that they must be done before
submission, not after.

Where the investigation adds something beyond agreeing:

| Comment | Verdict | Key evidence computed today |
|---|---|---|
| 1 Reference class | Correct. | Eq. (9) never executed; icing constants come from the chart; isotonic link on the icing margin executes it out of fold (mean 0.0303 vs prevalence 0.0303). |
| 2 Prevalence gap | Correct, and the leaky noisy-OR closes it exactly. | Leak 0.110 (LOC) and 0.084 (EF); fitted mean 0.1752 vs 0.1754 and 0.2922 vs 0.2933. Chain-level parent coverage 79.0% (LOC) and 74.9% (EF). |
| 3 Berkson bias | Correct, and stronger than he states. | The MLE assigns weight exactly zero to the three LOC parents with lift below 1. The LOC leading-factor set is hand-picked, not threshold-defined. FCI is available in the same library. |
| 4 Rung two | Correct. | No abduction in code or text. The title carries "Counterfactual Analysis". One named-accident probability of necessity is 2 to 3 days of work. |
| 5 Extraction evaluation | Mostly correct; one premise wrong. | There is no crosswalk table because matching is done by an LLM judge (Qwen, same family as the extractor). Per-role precision is computable offline from stored judge outputs. 104 of the 151 gold accidents have their database narrative in the corpus, 2 of 157 documents are in it verbatim. |
| 6 Physics priors | Correct on every item. | All constants located (table in 2.6). The stall-speed offset does not move pi_stall as marginalized (prior is in multiples of the model's own Vs) but shifts the absolute margin by 4 to 5.4 kt. Phase-restricted wind guards still reject 67 to 100%. |
| 7 Icing evaluation | Correct on leakage and CIs; the ceiling is confirmed. | DeLong: physics 0.612 (0.597, 0.628), logistic 0.642 (0.628, 0.656), paired z = 4.8 so the gap is real but 0.03. Density-ratio (KDE, out of fold) 0.640 (0.626, 0.655). |
| 8 Admissibility monitor | Correct. | Coverage 0.50%. Fuel guard has no coded drivers. Permutation null: p about 3e-7 against the paper's own 5.6%, about 1e-56 against the population never-ices rate of 16.0%. |
| 9 Hybrid vs layers | Correct. | Physics adds nothing to severity (0.818 to 0.820); it adds to occurrence (AUC 0.605 vs 0.5 at matched Brier). The experiment must be framed on occurrence. |
| 10 Positioning | Correct. | HCL, CATS, TK1951, dynamic PRA, ISAM, and two physics-informed BN papers verified with DOIs (2.10). Zhang and Mahadevan mischaracterized. |
| 11 Figures and tables | Correct. | Implied base rates match the actual presence rates exactly (0.501, 0.392). Full DAG CSVs exist for release. |
| 12 Front and back matter | Correct. | Model and decoding configuration collected (2.12). |

Effort, in order of value: leaky noisy-OR and reference-class rewrite (3
days, changes every fused number), FCI and collider paragraph (2 days),
held-out occurrence experiment (1 week), one probability of necessity (3
days), human adjudication of 100 reports (1 week of annotator time),
METAR control set (1 week), self-consistency run (2 GPU hours), editorial
pass (2 days), bibliography (1 day).

## 2. The twelve conceptual comments

### 2.1 Reference class of the fused numbers

What the code does. `carb_icing_model.py` sets Delta T(power) to
12/18/25/32/35 C and kappa = 0.60 per hPa so that Eq. (19) reproduces the
chart zones. Nothing anchors these to P(F). `loc_risk_model.py` and
`ef_risk_model.py` multiply the chart pi by the edge conditional q read from
`causation_edges.csv`. Eq. (16) combines lambda0 t = 1.5e-4 (per flight
hour), pi_ice (chart), P(EF | ice) = 0.667 (accident-conditioned), and a
Gaussian fuel margin. The lambda0 term is numerically irrelevant (the benign
0.004 is the fuel term Phi(-2/0.75) = 0.0038).

What the 2026-09-05 run showed. pi_ice is a chart-zone label, not a
probability on any population: decile means 0.00 to 0.95 against realized
icing rates 0.9% to 5.5% (Brier 0.50 vs 0.029 for the base rate). An
isotonic link fitted out of fold on the icing margin gives Brier 0.0292,
mean 0.0303 against prevalence 0.0303, AUC 0.605, and the depth-two product
with P(EF | ice) then has mean 0.0288 against a realized joint rate of
0.0288. That is Eq. (9) executed for icing, with D = the 39,770
weather-recorded NTSB accidents and the population stated as "given a
reportable accident".

Revision.

1. Add a short subsection after 3.5 (or an estimand box inside it) that
   states the reference class of every quantity: corpus base rates and edge
   conditionals are conditional on a reportable accident; the guard verdicts
   and rankings are population-free; the chart pi is a per-exposure
   probability with no population; the calibrated link is conditional on a
   reportable accident; the leaky noisy-OR output is conditional on a
   reportable accident.
2. Report the failure-study scenarios as risk ratios against the population
   baseline (LOC 0.240/0.137 = 1.75; EF 0.749/0.212 = 3.5; interventions as
   ratios likewise) and keep absolute values only where the link is
   calibrated. The manuscript already says ratios are the fallback (end of
   3.4); make the failure studies obey it.
3. State explicitly that Eq. (9) is executed for carburetor icing as a
   per-record isotonic link, give its shape (a short table of margin bins
   and calibrated probabilities), and state that stall, gust, shear, and
   fuel are not calibrated and enter only through ratios and guards.
   Annotations 22 and 35 ask for exactly this cross-reference fix.
4. Rewrite Eq. (16). Either drop lambda0 (it does nothing at 1.5e-4) and
   present the icing and fuel paths as the two guarded parents of engine
   failure inside the leaky noisy-OR of 2.2, or keep it and simplify per
   annotation 31 with the reference class of each factor named.
5. Delete the sentence in Section 7 claiming reporting bias weighs less
   because pi is unconditional. After calibration it is not.

Reply line for Arno: reference class stated per quantity; scenarios as risk
ratios; Eq. (9) executed for icing with the operating-point distribution and
resulting link reported; uncalibrated factors named as such.

### 2.2 The prevalence gap is coverage, not independence

Computed today (`coverage_leaky.py`), all on the 55,940 presence vectors.

| Quantity | LOC (10 parents) | EF (8 parents) |
|---|---|---|
| Empirical prevalence | 0.1754 | 0.2933 |
| Paper noisy-OR at base rates | 0.1368 | 0.2115 |
| Records with the mode present that have at least one listed parent present | 88.8% | 86.3% |
| Same fraction among records without the mode | 89.8% | 36.2% |
| Chain-level: mode node has an explicit parent in the set | 79.0% | 74.9% |
| Chain-level: parents present but all outside the set | 18.6% | 8.2% |
| Chain-level: mode node is a root | 2.4% | 16.9% |
| Leaky noisy-OR MLE leak | 0.110 | 0.084 |
| Leaky noisy-OR mean prediction | 0.1752 | 0.2922 |
| Non-leaky MLE mean prediction (nll per record) | 0.1536 (0.79) | 0.2511 (1.07) |
| Leaky MLE nll per record | 0.42 | 0.38 |

Fitted per-factor strengths p_i (the probability the factor alone produces
the mode), LOC: stall 0.31, spatial disorientation 0.43, control-surface
anomaly 0.29, airframe failure 0.15, incapacitation 0.33, turbulence 0.19,
wind shear 0.04, improper control input 0.00, engine failure 0.00,
inappropriate decision 0.00. EF: fuel exhaustion 0.96, icing 0.93,
contamination 0.92, fuel system 0.63, maintenance 0.35, other system 0.20,
latent defect 0.19, procedure not followed 0.11.

Two readings follow. For LOC the presence of a listed parent is
uninformative (88.8% vs 89.8%), because improper control input and
inappropriate decision appear in half and two fifths of all chains; the
mode is identified by which parent, not whether one is present. For EF the
set is genuinely diagnostic (86.3% vs 36.2%).

Revision.

1. Replace Eq. (11) with the leaky noisy-OR, fitted by maximum likelihood
   on the presence table, and state that the leak absorbs unmodeled and
   root-onset causes. The per-factor decomposition Delta_i survives.
2. Report the coverage fractions above in 4.5 and 5.3 (one sentence each)
   and the leak values.
3. Regenerate every downstream number: Tables 4 and 7 (add p_i and base
   rate columns), Figs. 6 and 9, the scenario values 0.240 and 0.749 and
   their interventions, and the bootstrap in `risk_uncertainty.py`, which
   must now resample the fit rather than the raw q_i. Section 7's third
   limitation becomes a statement of what the leak absorbs rather than an
   apology for the gap.
4. Note in the text that q_i as used today is a marginal conditional,
   whereas noisy-OR semantics require the cause acting alone, and that the
   MLE supplies the latter. This is also point 3.3 of the 2026-09-05
   assessment.

### 2.3 Selection bias threatens the causal layer

Evidence. Lifts below one into LOC: improper control input 0.29,
inappropriate decision 0.18, engine failure 0.35, procedure not followed
0.14; into icing: procedure not followed 0.10. The leaky MLE of 2.2 assigns
these parents zero weight, which is the negative dependence of a collider
expressed in a fitted model. The thresholds (s0, lambda0) of Eq. (2) are
not defined in any script. For EF the eight parents are exactly the eight
largest in-edges by support (next: airframe failure, 333). For LOC the ten
are not the top ten: procedure not followed (387), perception failure
(318), adverse wind (307), and other system failure (292) are omitted while
wind shear (235) is included because it carries a physics model. So Table 4
is not the leading-factor set of Eq. (2) under any threshold. The
icing-to-stall edge in Table 6 has support 22 and lift 0.13. PC runs
through causal-learn with chi-square tests at alpha 0.05
(`run_pc_per_category.py`); `causallearn.search.ConstraintBased.FCI` is
importable in the `qwen-vllm` environment.

Revision.

1. Add a collider paragraph (3.2 or 7) with the negative lifts and the MLE
   zeros as evidence, naming Berkson's bias and stating that within-accident
   contrasts measure competition among causes for the same accident.
2. Define L(F) honestly. Recommended: support alone with s0 = 200 (LOC
   gains procedure not followed, perception failure, adverse wind, other
   system failure; EF unchanged), lambda0 dropped, and lift kept as a
   reported diagnostic. Alternative: state the actual rule (top parents by
   support plus every parent with a physics model). Annotation 19 asks for
   the values either way.
3. Run FCI on LOC-I and SCF-PP with the same tests and priors, report how
   many PC-derived adjustment sets FCI marks as possibly confounded
   (bidirected or circle marks), and recompute Table 11 where the effect
   remains identifiable. One afternoon of compute plus interpretation.
4. Attach a sensitivity analysis to the two headline adjusted effects
   (spatial disorientation 0.264, airframe failure 0.088): an E-value or a
   bounded unmeasured-confounder analysis.
5. Drop the icing-to-stall path from 4.3 (annotation 33 as well) and
   qualify the engine-failure-to-LOC edge used for composition in Section 5
   as a low-lift edge whose value lies in exercising the mechanics.

### 2.4 The counterfactual claim is rung two

Evidence. Section 3.7 is backdoor adjustment plus re-evaluation at a
transformed input. No abduction anywhere. The title, keywords, Section 3.7,
Section 6.4, and the abstract all say "counterfactual". The guards already
carry the latent variables abduction needs: the power setting for icing
(marginalized over five settings in `guards.py`), and weight, airspeed, and
load factor for stall.

Revision (do both).

1. Rename to "interventional" wherever the computation is rung two: 3.7,
   6.4, the do-operator sentences, the abstract's "counterfactual analysis
   ranks interventions". Keep "counterfactual" only for item 2.
2. Compute one probability of necessity on a named icing accident. NTSB
   20001212X19636 (delayed carburetor heat at glide power, serious icing
   conditions, forced landing) or 20190119X05628 (carburetor heat off,
   power loss in cruise) are clean candidates from `per_accident_chains.jsonl`.
   Abduce the latent power setting and exposure noise from the observed
   power loss without heat at the recorded temperature and dewpoint, then
   evaluate under do(heat). Under the monotone mechanism the quantity is
   identified (Tian and Pearl 2000) and the calibrated link of 2.1 supplies
   the noise distribution. Report PN with bounds. Then the title is true for
   one worked case and the text says so.
3. If item 2 is out of scope, change the title's last clause to "and
   Interventional Analysis" and the keyword accordingly.

### 2.5 Extraction evaluation

Evidence. The main evaluation (`semantic_eval.py`) has no crosswalk table.
An LLM judge (Qwen3.6-35B-A3B on vLLM, guided JSON, thinking disabled)
matches each cause-flagged NTSB finding to extracted nodes; recall is the
fraction of findings with a match and precision the fraction of nodes
matched to any finding (3,309 of 11,105 narrative-only). The stored outputs
(`semantic_eval_v4_narrative.jsonl`) hold the per-record match lists and
the chains hold each node's `cause_role`, so precision by role is
computable offline without new model calls. `eval_chain_order.py` already
matches coded occurrence entries to nodes, so event-node precision against
the coded sequence can be read from its outputs. No second-seed run exists,
so self-consistency is untested. The gold check used Claude Opus 5, Sonnet
5, and Haiku 4.5 (`claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`)
as extractors, run 2026-08-09 to 2026-08-10, with the same Qwen as judge.
Overlap with the corpus: 2 of the 157 documents (AIR-23-02, AIR-24-05) are
among the corpus's published-report records, and 104 of the 151 accidents
have their NTSB database narrative in the corpus as an ordinary record. The
inputs differ (long-form report body versus database narrative) and nothing
is fitted, so the check is not circular, but the paper must say this.

Revision.

1. State the judge, its configuration, and the matching rule in 6.1; put
   the judge prompt in an appendix; release the judge outputs. Replace any
   wording that implies a code crosswalk.
2. Human adjudication: 100 reports from the 2,000, two annotators, Cohen's
   kappa against the judge and between annotators. Highest-value cheap
   addition; needs annotator time, not compute.
3. Split precision: cause-role nodes against cause flags; event nodes
   against coded occurrences. Both computable from existing outputs.
4. Self-consistency: rerun the 2,000 narrative-only extractions at three
   seeds (temperature 0.2 as in production, and once at 0.7), report
   node-set Jaccard, recall spread, and edge stability. About two GPU hours
   on the vLLM replica.
5. Per-group recall by the vocabulary group of the matched node (crew,
   system, environment, organizational, aerodynamic, condition), from the
   stored matches; the organizational group is where reference 15 found
   experts lead.
6. State the overlap facts for the gold check (annotation 45) and give the
   model identifiers and dates (annotation 46).

### 2.6 Physics layer

Every prior and constant, with its source, for the appendix table Arno
asks for:

| Item | Value | Where | Source |
|---|---|---|---|
| Maneuvering prior: weight fraction of gross | triangular (0.70, 0.90, 1.00) | `jsbsim_stall.py` PHASES | authors' judgment |
| Maneuvering prior: airspeed ratio to model Vs | Normal (1.35, 0.18) | same | authors' judgment |
| Maneuvering prior: load factor | Normal (1.55, 0.45), clipped to [1, 6] | same | authors' judgment |
| Other phases (takeoff, climb, cruise, approach, landing) | listed in PHASES | same | authors' judgment |
| Aircraft aero | JSBSim c172x: CLmax 1.47 clean at 16 deg, 1.82 with flap, S 174 ft2 | `extract_aero` | JSBSim |
| Aircraft constants | W 2,300 lbf, a 4.6 per rad, chord 4.9 ft, n_lim 3.8, Vs 52, VA 99, VC 120 KCAS | `factor_models.py` C172 | POH class values |
| Discrete gust scale | 5 / 10 / 20 ft/s (light / moderate / severe, thunderstorm) | GUST_SCALE_FPS | Hoblit 1988, single-parameter fit |
| Shear scale | 10 kt exponential | `p_shear_stall` | authors' judgment; microburst cores 20 to 40 kt |
| Fuel margin sigma | 0.75 h, fixed | `p_fuel_exhaustion` | authors' judgment |
| Dryden | sigma_w = 0.1 u20, L_w 200 ft, encounter 120 s, approach speed 1.25 Vs | `dryden_sigma_w_fps`, `p_exceed_dryden` | MIL-F-8785C low-altitude form |
| Icing cooling Delta T by power | 12 / 18 / 25 / 32 / 35 C (takeoff, climb, cruise, descent, idle) | VENTURI_COOLING_C | set to reproduce FAA CE-09-35 chart zones |
| Icing exposure kappa | 0.60 per hPa; light-zone onset w_thr = 0.176 hPa | _K_EXPOSURE, ICE_INDEX_THR_HPA | same |
| Engine base hazard lambda0 | 1e-4 per flight hour | `pi_engine_failure` | order-of-magnitude literature value, uncited |

Specific items.

- Stall speed. The maneuvering prior samples airspeed as a multiple of the
  model's own stall speed, so rescaling CLmax to give 48 KCAS leaves
  pi_stall unchanged (0.335 either way; computed today). The offset matters
  only when airspeed is anchored absolutely: at 70 KCAS and n = 1.8 the
  accelerated stall speed is 69.8 kt with the JSBSim value and 64.4 kt with
  the published one, a 5.4 kt shift in the guard margin (4 kt at 1 g).
  Replace "validates the implementation" with this quantified statement.
- Dryden aloft. The surface-wind relation is the MIL-F-8785C low-altitude
  form and the guards apply it at every chain node. Computed today by node
  phase (`wind_by_phase.py`): restricted to low-altitude phases the guards
  still reject 85% (gust stall, n = 224), 100% (overload, n = 36), and 67%
  (shear, n = 93). Phase gating is the right restriction to state and
  implement, but it does not rescue the guards; the 0.1 u20 intensity is
  too small for the transitions the chains record. Present the three wind
  guards as model-adequacy diagnostics, restrict them to low-altitude
  phases, and name the fix (turbulence aloft from reanalysis at the event
  time and location).
- Eq. (18) units. The code computes w in hPa and converts the margin to
  g/m3 with 216.68 e / (T_thr + 273.15). Write the conversion into Eq. (18),
  state that kappa absorbs induction mass flow and exposure time, and make
  Table 3's units consistent (annotation 21).
- Fuel sigma. Make it multiplicative, sigma = c t_flight; with c = 0.25 the
  worked numbers hold for a 3 h flight, so state the flight length and
  re-derive the three values in Eq. (20).
- Cite the appendix table from 3.3, 4.2, and 5.2.

### 2.7 Icing evaluation

Computed today (`icing_delong_bayes.py`), 1,205 positives among 39,770.

| Score | AUC | DeLong 95% CI |
|---|---|---|
| Physics, descent power | 0.612 | (0.597, 0.628) |
| Logistic on (T, Td), out of fold | 0.642 | (0.628, 0.656) |
| KDE density ratio, bandwidth 3 C, out of fold | 0.640 | (0.626, 0.655) |
| KDE density ratio, bandwidth 2 C, out of fold | 0.638 | (0.623, 0.653) |

Paired DeLong test physics versus logistic: z = 4.8, p below 1e-5. The two
are distinguishable, so the ceiling claim should be stated as "the physics
reaches 0.61 against a Bayes-optimal estimate of 0.64; the gap is
statistically real and 0.03 wide". In-sample KDE reaches 0.67 to 0.70,
which is overfitting and must not be quoted.

Revision.

1. Label leakage paragraph in 6.3 and Section 7: the label is investigator
   attribution, investigators consult the same chart, so the AUC partly
   measures investigator practice. Mitigation to report: score against the
   coded finding for carburetor icing (an extraction-independent label) and
   state that the ceiling is then the ceiling of investigator practice.
2. Replace "a ceiling of 0.64 that bounds any model" with the density-ratio
   estimate and its interval; add DeLong intervals to Table 10 and the
   positive rate (3.0%).
3. METAR control set (Arno's highest-value experiment): NOAA ISD hourly
   observations at the accident airports over the accident years,
   exposure-weighted by month; score icing accidents against routine hours.
   About one week including download; no local data exist today. This also
   answers the paper's first limitation.

### 2.8 Admissibility monitor

Evidence. `violation_rate.json`: 334,611 transitions, 1,685 checkable
(0.50%), 14,236 guarded with drivers unrecorded (4.3%), 318,690 unguarded
(95.2%). The fuel guard is defined (`FUEL_GUARD` in `guards.py`) with
`checkable` always false because fuel on board and flight time are not
coded fields (fuel on board is filled for 31% of avall records only). The
sign-inverted arm returns 0.000 for the two gust guards because 1 minus
0.98 rounds to nothing informative.

Permutation null, computed today (`icing_perm_p.py`). Permuting driver
tuples among the icing-checkable transitions alone cannot change the
count, because the icing guard depends only on the tuple; the paper's 5.6%
comes from permuting across the mixed guarded work list, most of which is
itself icing chains, so that null is conservative. Against the paper's own
null the binomial tail is P(X at most 28 | n = 1,222, p = 0.056) about 3e-7.
Against the population of all 74,624 weather-recorded events, 16.0% of
tuples never ice at any power setting, the expected count under random
pairing is 196, and the tail is about 1e-56.

Revision.

1. State coverage up front in 6.2 and in 3.4 with the three fractions.
2. Table 9 caption: fuel guard absent because its drivers are not coded;
   add per-arm checkable columns (annotation 49).
3. Redefine the permuted arm as a population-weather null and report the
   rate (16.0%) and the p-value; keep the sign-inverted arm but say it is
   uninformative for guards that almost always fail.
4. Justify the framing in one paragraph placed before Eq. (5): the paper
   uses the hybrid automaton as a guarded transition system, with no flows,
   reachability, or composition, and the formalism buys three things: a
   per-case rejection that no reweighting reproduces, a countable coverage
   statistic, and one object whose guards also drive the occurrence
   probabilities. The disclaimer paragraph at the end of 3.4 already says
   what is not done; move the "what it buys" sentence ahead of the
   definition. If that reads thin, retitle 3.4 "The Guarded Transition
   System" and cite hybrid automata as the origin.

### 2.9 Missing evaluation of the hybrid against either layer

Evidence from 2026-09-05. Physics drivers add nothing to severity (AUC
0.818 to 0.820 on top of extracted factors; pi_ice alone 0.556). The
calibrated icing link beats the base rate on occurrence (AUC 0.605, Brier
0.0292 vs 0.0294) and the fused depth-two product matches the realized
joint rate. Arno's proposed severity ranking would therefore show the
hybrid losing, and rightly so: the physics term decides whether power is
lost, not how the forced landing ends.

Revision: one held-out experiment on occurrence, per record, test split
2015 onward.

| Arm | Content |
|---|---|
| B0 | current fusion: chart pi times edge conditional, noisy-OR at base rates |
| Corpus only | base rates conditioned on recorded covariates (phase, month, aircraft class, category) |
| Physics only | chart pi and the guard margin |
| Hybrid | calibrated link on the margin times the corpus consequence conditional |
| Hybrid, corrupted | same with sign-inverted or permuted margins |

Targets: icing occurrence; icing and engine failure jointly; fuel
exhaustion on the 31% of avall records with fuel on board (the one place
the fuel guard is checkable). Metrics: AUC, PR-AUC, Brier, reliability by
decile. Report the severity table alongside with the honest result that
physics does not move it. About one week. This is arms B2 versus A of the
2026-09-05 assessment, reduced to the guarded factors.

### 2.10 Positioning

Verified references to add (DOIs checked through Crossref or the
publisher today):

- Groth, Wang, Mosleh. Hybrid causal methodology and software platform for
  probabilistic risk assessment and safety monitoring of socio-technical
  systems. RESS 95(12):1276–1285, 2010. doi 10.1016/j.ress.2010.06.005.
- Borener, Trajkov, Balakrishna. Design and development of an Integrated
  Safety Assessment Model for NextGen. Int. Annual Conf. of the American
  Society for Engineering Management, 2012. (The FAA ISAM; ESD, fault
  trees, and BBN, built on CATS and HCL. Verify the proceedings details.)
- Ale, Bellamy, Cooke, Goossens, Hale, Roelen, Smith. Towards a causal
  model for air transport safety: an ongoing research project. Safety
  Science 44(8):657–673, 2006. doi 10.1016/j.ssci.2006.02.002.
- Ale et al. Further development of a Causal model for Air Transport Safety
  (CATS): building the mathematical heart. RESS 94(9):1433–1441, 2009.
  doi 10.1016/j.ress.2009.02.024.
- Ale, Bellamy, Cooper, Ababei, Kurowicka, Morales, Spouge. Analysis of the
  crash of TK 1951 using CATS. RESS 95(5):469–477, 2010.
  doi 10.1016/j.ress.2009.11.014.
- Aldemir. A survey of dynamic methodologies for probabilistic safety
  assessment of nuclear power plants. Annals of Nuclear Energy 52:113–124,
  2013. doi 10.1016/j.anucene.2012.08.001.
- Meng, An, Xing. A data-driven Bayesian network model integrating physical
  knowledge for prioritization of risk influencing factors. Process Safety
  and Environmental Protection 160:434–449, 2022. doi 10.1016/j.psep.2022.02.010.
- Xing, Qian, Peng, Zio. Physics-informed data-driven Bayesian network for
  the risk analysis of hydrogen refueling stations. Int. J. Hydrogen Energy
  110:371–385, 2025. doi 10.1016/j.ijhydene.2025.02.110.

Revision.

1. New paragraph in 2.4 (or a fifth group "Hybrid causal risk models"):
   HCL and ISAM combine ESDs, fault trees, and BBNs with expert-elicited
   structure and exposure-normalized frequencies; CATS did the same for air
   transport with a per-case analysis of TK 1951; dynamic PRA adds
   simulated physics to event trees; physics-informed BNs constrain
   structure learning with physical knowledge. None learns structure from
   narratives at corpus scale or checks a reconstructed chain against
   per-case physics. Rewrite the gap statement in 2.5 accordingly.
2. Correct the Zhang and Mahadevan sentence in 2.2: their priors come from
   BTS departure data, so they already use an exposure denominator from
   outside the archive. Add a sentence in Section 7 that FAA GA Survey
   flight hours would supply the same denominator here (Option C of the
   2026-09-05 assessment).
3. Replace "compact and faithful" in the abstract (faithfulness is the PC
   assumption) with "compact and consistent with investigator coding".

### 2.11 Figures and tables

- Figs. 2 and 7: scale arrow width by the contribution Delta_i, or by the
  fitted p_i after 2.2, and change the caption's "record support n" to
  "edge support". Scripts: `make_loc_kg_fig.py`, `make_ef_kg_fig.py`.
- Figs. 3 and 8: release the full networks. The CSVs exist
  (`event_extraction/out/aggregate_kg/per_category_dag/LOC-I.csv` and
  `SCF-PP.csv`, with stability and orientation columns); add them to the
  supplement and say so in the captions.
- Fig. 5: overlay the 38,565 non-icing accidents as density contours
  (points would not be legible); `validate_carb_icing.py` must export the
  non-icing (T, Td) pairs, which it currently drops.
- Figs. 6b and 9b captions: intervals cover the corpus terms only.
- Tables 4 and 7: add a base-rate column (LOC: 0.119, 0.501, 0.293, 0.392,
  0.032, 0.049, 0.017, 0.024, 0.026, 0.039; EF: 0.097, 0.149, 0.293, 0.065,
  0.026, 0.030, 0.020, 0.101) and label the support column "edge support
  s_iF". The implied base rates Arno derived (0.51, 0.42) equal the actual
  presence rates (0.501, 0.392), so they are consistent with the NTSB
  finding distribution; say so in one sentence.
- Table 10: DeLong intervals and the positive rate (2.7).
- Table 11: drop the rows with an empty adjustment set into the supplement
  (four per block), and remove the loss-of-control row from its own block
  (annotation 51).

### 2.12 Front and back matter

- Competing interests, CRediT (elsarticle takes a plain starred section),
  generative AI declaration (state that the extraction, the judge, and the
  ordering priors are model outputs, and whether any prose was
  model-assisted), data availability with the repository URL and an
  archived DOI, funding for every author, and the acknowledgment restored.
- Reproducibility paragraph: Qwen3.6-35B-A3B (mixture of experts, 35B
  total, 3B active), served with vLLM 0.19.1, bf16, tensor parallel 2,
  context 32,768, prefix caching, thinking disabled; decoding temperature
  0.2, max 6,144 output tokens, guided JSON against `schema_v4.json`, one
  repair turn; judge identical with guided JSON. Gold check extractors
  `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`, run 2026-08-09 to
  2026-08-10.

## 3. The 70 editorial annotations

### 3.1 Apply as written (wording, grammar, consistency)

1 (real URL, not "Link"), 2 (recall on the evaluation sample, not all
chains), 3 (split the abstract sentence; drop "We discover"), 4 (keyword
case), 5, 6, 7 (choose one serial-comma convention; the paper mostly omits
it), 8, 10, 11, 12, 14, 20, 23 ("interval" rather than "confidence
interval" for the Beta draws), 24, 26, 27, 32, 34, 36, 37, 39 ("warm, humid
air": the worked point is 13 C and 12 C), 43, 44, 47, 48, 53, 55, 56.

### 3.2 Need a fact or a number, resolved here

- 15: the checkpoint is Qwen3.6-35B-A3B, a mixture of experts with 35B
  total and 3B active parameters. Name it and cite the Qwen3.6 release
  rather than the Qwen3 report if a separate report exists.
- 16: the 262-record gap. `build_kg_v4.py` (lines 148 to 150) skips records
  whose extraction is not `ok` or whose chain is empty; 56,154 of the
  58,064 extracted records are ok overall. Verify the split of the 262
  between "no valid chain after the repair turn" and "empty chain" before
  writing the sentence, and note that 235 ok chains carry only outcome or
  unknown nodes.
- 17: rename the outcome set (script O) or the observables; use script Y
  for outcomes.
- 18 and 38: Table 6 reads the unfiltered edge list (1,309 edges); the
  support-30 filter applies to the aggregated graph and the figures. State
  it in the Table 6 caption.
- 19: resolved by 2.3 item 2.
- 21: resolved by 2.6 (Eq. 18 units).
- 22 and 35: resolved by 2.1 item 3.
- 25: "44 binary parents" has no source; the vocabulary holds 47
  non-outcome factors and the LOC-I candidate set holds 39. Use 47.
- 28: the 0.35 endpoint is control-surface anomaly, not structural overload;
  the structural-overload edge into LOC has support 46 and q = 0.03. Write
  "(q = 0.21)" for airframe failure and mention overload as below the table
  threshold.
- 29: cite 14 CFR 25.341 (unchanged in form) or Amendment 23-64 with the
  ASTM consensus standard; the Pratt formula itself is cited already.
- 30: the 0.15 figure uses the exponential gust tail at the moderate scale
  (10 ft/s); the Dryden paragraph replaces it. Say which turbulence
  intensity 0.15 assumes, or drop it.
- 31: simplify Eq. (16) (see 2.1 item 4).
- 33: resolved by 2.3 item 5 (drop the second path; the first reaches LOC
  through engine failure per Table 4 and the outcome layer through the
  forced landing).
- 40 and 42: rounding mismatches vanish when the numbers are regenerated
  after 2.2; regenerate figure labels and text from one JSON.
- 41: the 0.34 scenario uses the maneuvering prior with load factor Normal
  (1.55, 0.45), not a single n; say so and add the mean to Fig. 4a's band.
- 45 and 46: resolved by 2.5 item 6.
- 49: resolved by 2.8 item 2.
- 50 and 52: 39 and 27 are the candidate factors per category (support at
  least the minimum); 15 and 11 are those with a non-empty adjustment set.
  Define both once in 6.4 and reuse.
- 51: remove the loss-of-control row from the LOC-I block.
- 54: cite Woods, Dekker, Cook, Johannesen, Sarter, Behind Human Error, 2nd
  ed., Ashgate 2010, for hindsight bias in attribution.
- 13: the uncited sentence is "Potential-outcome methods appear in traffic
  and construction safety under the counterfactual label while remaining
  interventional in substance." Cite Karwa, Slavković, Donnell, Causal
  inference in transportation safety studies: comparison of potential
  outcomes and causal diagrams, Annals of Applied Statistics 5(2B), 2011,
  doi 10.1214/10-AOAS440, and one construction-safety paper, or soften to
  "appear in adjacent safety fields".
- 57: the listing comment "empty for roots and conditions" is a prompt
  instruction, not a runner-enforced rule (the prompt says root causes and
  ambient conditions have an empty parent list; some extracted conditions
  do carry parents). Change the comment to "empty for roots".
- 58: replace the example chain. Candidates with internally consistent
  evidence spans, from today's scan: NTSB 20190119X05628 (five nodes,
  carburetor temperature gauge, heat off, power loss in cruise, gear-up
  forced landing, minor injury) or 20001212X19636 (six nodes, delayed heat
  at glide power, partial power loss, forced landing to a grass strip). The
  current example's last evidence span ("forced landing on road")
  contradicts nodes 3 and 4, as Arno says.

### 3.3 Bibliography

- 59: every entry prints "doi:doi:". `main.bbl` defines `\DOIprefix` as
  "doi:" and the `doi` package makes `\doi` print the prefix again. Remove
  `\usepackage{doi}` from the preamble (or load it with `nolink` and
  redefine `\DOIprefix` to empty) and rebuild.
- 9 and 60: Aviation-BERT is Chandra, Jing, Bendarkar, Sawant, Elias, Kirby,
  Mavris, AIAA 2023-3436, doi 10.2514/6.2023-3436. Reference 3 (Jing et
  al., AIAA 2023-3438) is a different paper. Cite Chandra et al. for the
  model name, or reword.
- 61: `safetyscience2026kg` (Farzadnia, Merkert, Beck) is volume 200,
  article 107230, 2026 in the .bib; the ScienceDirect record
  S0925753526001219 exists. Confirm volume and article number on the page.
- 62: cite ARP4761A (2023) or state that the 1996 original is intended.
- 63: `almachot2025naspt` lists "Al Machot, Fadi and Al Machot, Fidaa";
  verify on arXiv 2510.05451 whether these are two people.
- 64: normalize arXiv entries to one pattern (`howpublished` with the
  identifier and no URL field, or the reverse) across 12, 13, 21, 25, 28,
  29, 32, 46, 54; write "arXiv" consistently.
- 65 and 69: `aei2026review` (Yiu et al.) is volume 71, article 104378,
  2026; verify against the published record.
- 66: `kiciman2023causal` should be an `@article` in Transactions on
  Machine Learning Research, 2024, with the arXiv identifier in `note`.
- 67: `recite2025` (Saklad et al.) is ACL 2026 Long Papers, pages
  21957–21989, anthology 2026.acl-long.1003; verified today.
- 68: `jin2023cladder`: replace the Proceedings.com DOI with the NeurIPS 36
  proceedings page or arXiv 2312.04350.
- 70: `cictt2017`: replace the ICAO APAC workshop URL with the
  intlaviationstandards.org copy already used for `cictt_phase2013`.

## 4. Suggested order of work

1. Leaky noisy-OR, leading-factor rule, reference-class subsection, risk
   ratios, Eq. (16) rewrite, regenerate Tables 4 and 7, Figs. 6 and 9, and
   every scenario number (2.1, 2.2, 2.3 items 1, 2, 5). Three days. Every
   later item reads its numbers from this.
2. Calibrated icing link written into the manuscript with its operating
   point distribution; DeLong and density-ratio ceiling; Table 10 (2.1 item
   3, 2.7 items 1 and 2). One day; code exists.
3. Coverage sentence, Table 9 columns, population permutation null,
   automaton framing paragraph, phase-restricted wind guards (2.8, 2.6
   Dryden item). Two days.
4. FCI on the two categories, collider paragraph, sensitivity on the two
   adjusted effects (2.3 items 3 and 4). Two days.
5. Interventional renaming plus one probability of necessity (2.4). Three
   days.
6. Held-out occurrence experiment (2.9). One week.
7. Extraction evaluation additions: judge statement, precision split,
   per-group recall, self-consistency, gold overlap statement (2.5). Two
   days of compute and writing; human adjudication runs in parallel.
8. Physics appendix table, Eq. (18) units, fuel sigma, stall statement (2.6).
   One day.
9. Related work paragraph and gap rewrite, Zhang and Mahadevan correction,
   abstract wording (2.10). Half a day.
10. Figures and tables (2.11), editorial pass (3.1, 3.2), bibliography
    (3.3), front matter (2.12). Three days.
11. METAR control set (2.7 item 3). One week; can follow the resubmission
    draft if time is short, but Arno rates it the highest-value experiment.

## 5. Venue

Arno recommends IEEE Transactions on Systems, Man, and Cybernetics:
Systems, then IEEE Transactions on Intelligent Transportation Systems, and
names IEEE Transactions on Reliability and RESS as the best substantive
fits. The manuscript is in the Elsevier `elsarticle` class with a 3p
layout; an IEEE submission means reformatting to IEEEtran, a page budget
with supplementary material for the appendices, the full networks, the
priors table, and the judge prompt, and the reviewer pool he warns about
(human and organizational factors are the thinnest tier, and the
reference-class question is what a reliability reviewer leads with). The
choice is the authors' call; the revision plan above is the same for any of
the four venues.

## 6. Draft skeleton of the reply to Arno

To be completed after the revisions land; one line per point.

1. Reference class stated per quantity; failure studies reported as risk
   ratios; Eq. (9) executed for icing (link, operating-point distribution,
   and parameters given); uncalibrated factors named.
2. Coverage fractions reported; leaky noisy-OR fitted by maximum
   likelihood; leak values and per-factor strengths given; prevalence
   reproduced.
3. Collider paragraph with the negative lifts; leading-factor rule stated
   with its threshold; FCI run alongside PC; sensitivity attached to the
   adjusted effects; the two weak paths dropped or qualified.
4. Interventional wording where the computation is rung two; one
   probability of necessity computed by abduction on a named accident.
5. Judge and matching rule stated; human adjudication with kappa;
   precision split by node role; self-consistency across seeds; per-group
   recall; overlap with the corpus stated.
6. Appendix table of every prior with its source; stall-speed bias
   quantified; Dryden restricted to low-altitude phases; Eq. (18) in
   absolute humidity; multiplicative fuel sigma.
7. Label leakage discussed and mitigated with the coded finding; DeLong
   intervals; density-ratio ceiling; METAR control set (done or stated as
   the next step).
8. Coverage stated up front; fuel guard explained; sign-inverted arm
   qualified; permutation p-value; framing justified.
9. Held-out occurrence comparison of hybrid, corpus-only, and physics-only
   arms; severity result reported as it is.
10. HCL, ISAM, CATS, TK 1951, dynamic PRA, and physics-informed BN cited
    and the gap restated; Zhang and Mahadevan corrected; "faithful"
    removed.
11. Figures and tables revised as listed.
12. Declarations, funding, data availability, and the model configuration
    added.

## 7. Status after execution (2026-09-14, evening)

Everything below is implemented in code and written into `paper/main.tex`,
which builds clean (50 pages, preprint layout, no undefined references or
citations). Nothing is committed or pushed. The paper repository is dirty on
`main.tex`, `refs.bib`, `figs/`, and `supplement/`; the main repository is
dirty on `physics/`, `event_extraction/scripts/`, `docs/`, and new outputs.

Decisions taken as defaults (all reversible, see the reply to the user):
- Leading-factor rule: relative threshold, edge support at least 2 percent of
  the records carrying the mode (`physics/leaky_noisy_or.py`, `RHO`).
- Rung two renamed "interventional" throughout; "counterfactual" kept for the
  probability of necessity on four named accidents; title unchanged.
- Arno's Highlight 1 wording ("safety assurance") left as he wrote it.
- Venue untouched; the manuscript stays in the Elsevier class.

| Item | Where | Result |
|---|---|---|
| Leaky noisy-OR by MLE | `physics/leaky_noisy_or.py`, `loc_risk_model.py`, `ef_risk_model.py`, `risk_uncertainty.py` | LOC leak 0.110, 14 parents, mean 0.175; EF leak 0.084, 9 parents, mean 0.300 |
| Calibration of Eq. (9) | `physics/calibrate_physics.py`, Section 6.3, Table 11 | isotonic links for icing (AUC 0.608), turbulence (0.583), shear (0.667); stall stratified by phase; fuel not executed |
| Reference classes | Section 3.5, Section 7 | stated per quantity; scenarios reported on the accident population with ratios |
| Collider paragraph, FCI, E-values | Section 3.2, Section 6.6, `stage3/fci_check.py` | 47 of 71 and 27 of 45 adjacencies bidirected; 11 of 15 and 9 of 11 adjusted factors confounded on the FCI reading |
| Probability of necessity | `physics/probability_of_necessity.py`, Section 4.3.5, Table 7 | PN of the heat omission 0.55 to 0.82 for icing, 0.39 to 0.58 for the power loss, four named accidents |
| Extraction evaluation | `eval_extraction_extras.py`, `run_selfconsistency.sh`, `score_selfconsistency.py`, Section 6.1, Table 9 | judge stated; precision by role 0.41 (cause-role) and 0.63 (event nodes vs coded occurrences); per-group recall; four repeated runs, recall 0.906 to 0.911, Jaccard 0.80; gold overlap stated; 100-report adjudication sample exported |
| Admissibility monitor | `physics/guards.py`, `violation_rate.py`, Section 6.2, Table 10 | coverage 0.46 percent stated first; wind guards phase-restricted; population null and p-values; fuel guard explained |
| Icing ceiling | `physics/review_checks/icing_delong_bayes.py`, Section 6.4, Table 12 | DeLong intervals; density-ratio ceiling 0.640 (0.626, 0.655); label-leakage caveat |
| Held-out comparison | `physics/heldout_occurrence.py`, Section 6.5, Table 13 | hybrid 0.584 vs corpus 0.553 vs physics 0.575; corrupted arms 0.551 to 0.553; raw-weather fit 0.597 |
| Positioning | Section 2.4, 2.5, refs.bib | HCL, ISAM, CATS (three papers), dynamic PRA, two physics-informed BNs cited; Zhang and Mahadevan corrected; "faithful" removed |
| Figures and tables | `make_*_fig.py`, Figs. 2, 5, 6, 7, 9; Tables 4, 8, 10, 12, 13 | arrow width by contribution; non-icing contours; base-rate and p columns; DeLong CIs; FCI and E-value columns; full networks in `paper/supplement/` |
| Front and back matter | end of `main.tex`, Appendix B | declarations, CRediT draft, AI-use declaration, data availability, funding, decoding configuration, priors table |
| Editorial annotations | throughout | all 70 addressed; `\usepackage{doi}` removed to fix "doi:doi:" |

Left open, on purpose:
- METAR control set (comment 7): needs an NOAA ISD download; about a week.
- Human adjudication with kappa (comment 5): the sample is exported to
  `event_extraction/out/adjudication/`; needs two annotators.
- CRediT roles for the co-authors, the writing-assistance sentence of the AI
  declaration, and funding for the non-NASA authors: placeholders are marked
  with LaTeX comments.
- Borener et al. 2012 (ISAM) carries no DOI; the proceedings details come
  from a citing paper and should be confirmed.
- The IEEE reformatting, if the venue changes.

## 8. Decisions taken by the user (2026-09-17) and follow-up

- Title and keyword now say "Interventional Analysis"; the named-accident
  probability of necessity stays as Section 4.3.5.
- Wind shear remains a LOC parent under the 2 percent rule (edge support
  235 against a threshold of 196), so no change to the rule.
- Highlight 1 keeps "safety assurance". The body keeps "guard" as the
  defined term of Section 3.4 and Table 3, since it is the hybrid-automaton
  vocabulary the section cites.
- IEEE version: `paper/main_ieee.tex` (IEEEtran journal, natbib numbers,
  IEEEtranN). The manuscript is split into `body.tex` and `abstract.tex`,
  shared by `main.tex` (Elsevier) and `main_ieee.tex`. Format hooks in each
  front end: `\tabw`, `\fwsingle`, `\fwsmall`, `\appref`,
  `\startappendices`, and the `tbl`, `tblwide`, `lstfig` environments. Both
  build clean: Elsevier 50 pages, IEEE 30 two-column pages. Four display
  equations were split across two lines for the column width. The
  pre-split file is kept as `main_presplit.bak`.
- Reply to Arno appended to `paper/arno_aviation_comments.txt`.
- Borener et al. 2012 confirmed online as a widely cited ASEM 2012
  conference paper without a DOI; entry unchanged.
- `event_extraction/scripts/score_adjudication.py` scores the human
  adjudication once two annotators return copies of
  `event_extraction/out/adjudication/sample_100.csv`.
- Open by the user's instruction: CRediT roles, AI-writing sentence and
  co-author funding (item 3); METAR control set (explained in the reply).

## 9. Second round (2026-09-22): METAR control, adjudication dropped, technical report and IEEE trim

User decisions: no human or LLM-judge adjudication in this paper; run the
METAR control experiment Arno asked for; put each reply directly after the
point it answers; keep the Elsevier document as it is; publish the full
document as a technical report and trim the IEEE paper to one failure study
at fewer than 16 pages. Order of work was Elsevier edits, then a
humanizer and academic-writing pass, then the IEEE adaptation.

### METAR control experiment (comment 7)

- Script `physics/metar_control.py`; data in `data/isd_lite/` (NOAA
  ISD-Lite hourly records plus `isd-history.csv`); output
  `physics/out/metar_control.json`.
- Design: each weather-recorded NTSB icing accident (1,033) and a sample of
  2,500 other accidents (2,190 with a station) is paired with the nearest
  ISD station within 60 km; the control is 100 routine hours drawn from the
  same station and calendar month, all hours or within 3 h of the accident
  hour.
- Results: icing accidents against routine hours, ROC area 0.431 (any hour)
  and 0.473 (hour matched); other accidents against routine hours 0.351 and
  0.401; icing against other accidents 0.618. About 68 percent of routine
  hours fall in the serious-icing zone of the chart. Icing-specific
  relative risk by chart zone rises from 0.54 (nil) to 1.20 (serious).
- Reading: routine station hours are not the flight exposure, since flying
  concentrates in fair weather, so the chart cannot separate any accident
  from the ambient climate; the accident-control design of Section 6 is the
  right comparison, and a per-flight rate needs activity data by weather.
- Written into `body.tex` as Section 6.5 "Separation Against Routine
  Weather" (`sec:metar`, table `tbl:metar`), the ceiling subsection, the
  first discussion limitation, and Arno's reply to comment 7.

### Adjudication removed

- All mentions of the 100-report human adjudication are out of the
  manuscript and of the data-availability statement; reply item 5 says no
  human adjudication is added. The exported sample under
  `event_extraction/out/adjudication/` and `score_adjudication.py` remain in
  the repository unused.

### Reply file

- `paper/arno_aviation_comments.txt` now has a preamble line and a `REPLY:`
  paragraph after the venue block and after each of the twelve numbered
  comments, in Arno's tone. The venue reply names the IEEE version and the
  technical report.

### Three front ends on one body

- `main.tex` (Elsevier, 50 pages), `main_techreport.tex` (article class
  with authblk, plainnat, title suffix "Technical Report", 51 pages) and
  `main_ieee.tex` (IEEEtran journal, 15 pages, references from page 14).
  All three `\input` `abstract.tex` and `body.tex`.
- `\newif\ifshort` is true only in the IEEE front end. Short mode drops the
  loss-of-control study, the appendices, the motivation and roadmap
  paragraphs of the introduction, the full related-work subsections
  (replaced by three condensed paragraphs that cite the technical report),
  Tables 1, 2, 6, 8, 9 and 12, Figs. 7 and 8, the consequence-guard and
  chain-activation equations, the TSB Canada and gold-report paragraphs,
  Sections 6.7 to 6.9, discussion limitation 2, and the backdoor rows with
  an adjustment shift below 0.05. It also carries condensed variants of the
  introduction, research gap, automaton, fusion, intervention, fidelity,
  admissibility, graph-intervention, discussion and conclusion paragraphs,
  and a 241-word abstract. `\appref` resolves to "the technical report
  \citep{pang2026techreport}" in IEEE and to the appendix elsewhere.
- Full-mode text is unchanged by the trimming (checked by resolving the
  conditionals both ways and diffing). Conditionals balance at 78 each.
- `bstctl.bib` holds the IEEEtran control entry that suppresses URLs in
  the reference list. The technical report uses `xurl` for the repository
  URL. The METAR table is set in `\small` with shortened labels so it fits
  the Elsevier text width.
- The IEEE build contains three figures (framework, icing surface, engine
  failure risk); the learned SCF-PP network is cited from the technical
  report.

### State

- Nothing committed or pushed in either repository.
- Still open by the user's instruction: CRediT roles, the AI-writing
  sentence and co-author funding placeholders (item 3).
- A 4.5 pt overfull line remains in the Dryden paragraph of the
  loss-of-control study (Elsevier and technical report only).
