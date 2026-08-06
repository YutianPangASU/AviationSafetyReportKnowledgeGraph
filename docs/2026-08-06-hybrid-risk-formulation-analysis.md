# Hybrid Risk Formulation: Which of Xinyu's Candidate Foundations Fits

2026-08-06. Analysis for the revision. Xinyu's note (paper/xinyu_feedback.tex,
Section 3) lists six candidate foundations for making the physics layer
rigorous, plus an overarching proposal: compute a signed margin with units per
case and use it to judge whether a chain transition is admissible, instead of
multiplying a probability into Eq. (6). This document evaluates each candidate
against what the paper actually has, and recommends a combined design.

## What we have, restated as objects

1. Per-report causal chain: discrete, ordered, extracted from text (v4 schema).
2. Corpus causation graph G_c: aggregated chains, edge conditionals
   q_i = P(Y|X_i), Tier-3 base rates.
3. Per-case physics models pi_F(theta): carb icing (T, Td, power), stall
   (V, W, n, DA via JSBSim tables), fuel exhaustion (endurance vs time),
   shear exceedance, structural overload.
4. Fusion Eq. (6): R = pi_F(O) * P(Y|F); combination Eq. (7): noisy-OR over
   parents; counterfactuals by do() at graph level and driver level.
5. The known defects (Yifang M1, M5): the calibration Eq. (5) that was to fix
   the population mismatch never runs, and the noisy-OR double counts.

Every physics model in item 3 already computes a signed quantity internally
before converting to probability: V minus V_s (stall), endurance minus planned
time (fuel), condensable water minus the chart-zone threshold (icing), V minus
Delta U minus V_s1g (shear), n_ult minus n (structural). The margin layer is a
refactor, not new physics.

## The six candidates

### 1. Hybrid automata (Tomlin 2003; Yang et al. 2022)

What it is: discrete modes with continuous dynamics per mode; transitions
carry guard conditions on the continuous state.

Instantiation here: chain nodes are modes (normal ops, power loss, forced
landing, impact); caused_by edges are transitions; each physics model supplies
a guard, the signed margin g_F(theta) evaluated at that flight's recorded
state. A transition is admissible when its guard is satisfied. The corpus
supplies which transitions exist and how often they fire; physics supplies
whether they were reachable for this case.

Fit: strong. The chain already is a path over discrete modes, which is
Xinyu's own observation. The two named slots (reset maps aside, we need only
modes and guards) give the "countable coverage" criterion he asks for: for
each edge in G_c one can state whether a guard exists, giving a coverage
number (k of 57 factor types carry a physics guard) instead of an asserted
"physics where available."

Shortfall he flags: the formalism does not choose the equations, and guards
are hand-built. True, but that is exactly our Tier structure: the corpus
selects the mechanism, domain judgment selects g_F. We state this boundary.

Cost: notation and one subsection; guards extracted from existing models;
one worked example. No new solver, no learned dynamics, no reachability
computation (we check single transitions, not reach sets, and should say so
honestly rather than invoking the full machinery).

### 2. Bias placement (Karniadakis et al. 2021)

What it is: taxonomy of where physics enters a learning system —
observational, inductive, or learning bias.

Fit as a formalism: weak. It names which stage carries the bias but is
silent on the object constrained, per Xinyu's own table. Nothing in our
pipeline is trained against a physics loss; there is no stage for an
inductive or learning bias to live in. Its value is one sentence of
positioning vocabulary in Related Work: our injection is neither a loss term
nor an architecture prior but an inference-time constraint on a reconstructed
object.

Verdict: cite, do not build on.

### 3. Source taxonomy (Willard et al. 2022)

Same class: names what is injected (mechanistic model, constants, simulation
data) but not where it enters. Useful for one Related Work sentence
classifying our Tier-1/2/3/4 sources. Not a formulation.

Verdict: cite, do not build on.

### 4. Exact output projection (KKT-hardNet; HardNet; PINN-Proj)

What it is: append a differentiable projection to a network so outputs
satisfy constraints to machine precision.

Fit: does not apply. Our output space is a discrete labeled graph; there is
no differentiable output to project, and we do not train the extractor
against constraints. The nearest analogue would be projecting the extracted
chain onto the set of physics-admissible chains, but with discrete objects
that is not a projection, it is a reject-or-repair check — which is candidate
5. Also worth keeping Xinyu's caveat: even in its home domain, projection can
land near-exact rather than exact (PINN-Proj), so the machine-precision
promise does not transfer.

Verdict: not applicable; one sentence in Related Work explaining why (the
contrast is actually useful for positioning: rigor for text-reconstructed
discrete objects has to come from checking, not construction).

### 5. Solver or monitor gating (ARc; Winston et al.; Alamdari et al.)

What it is: a checker that can reject — autoformalized policies checked by a
prover, SMT preconditions on tool calls, LTL monitors on action streams.
Deployed at scale; the constraint is policy, not dynamics.

Fit: this is the operational half of the recommendation. The guard from
candidate 1 needs a mechanism, and the mechanism is a monitor: after
extraction, each transition in the chain that touches a physics-modeled mode
is checked against its guard margin; violations are flagged (chain
physically infeasible as reconstructed — either the extraction is wrong or a
mode is missing, Xinyu's glide example). What these systems lack, and we
supply, is dynamics as the constraint: their checkers enforce policy. Our
checker enforces flight physics. That is the narrow novelty claim Xinyu's
second table isolates: nobody rejects a reconstructed dynamical account of a
past event on physical infeasibility.

The gaming/leniency literature he cites (Helff; Verus-SpecGym) argues for
reject semantics with a stated direction of failure: too-permissive checkers
get gamed, LLM judges miss 26 percent of what executable checkers catch. A
margin with units is an executable checker.

Cost: one checking pass over extracted chains (a script over gold496 /
corpus chains evaluating guards where drivers are recorded), reported as a
physical-violation rate. Moderate.

### 6. Rejection against scoring (Kim et al. 2025)

What it is: guaranteed satisfaction by rejection sampling from a proposal;
the result that a filter is not substitutable by tuning the generator.

Fit: we do not need the sampling machinery (we check one reconstructed chain,
not a distribution of generations), but the theorem earns its citation as
the argument for why the guard layer must reject rather than blend a physics
score into the probability — blending is exactly the dot product we are
moving away from. Xinyu's calibration argument is the same point from the
statistics side: judging needs no calibration, scoring does.

Verdict: cite as justification for reject semantics; no machinery to adopt.

## Comparison

| Candidate | Gives us | Blocks | Adopt? |
|---|---|---|---|
| Hybrid automata | the formal object: modes, guards, countable coverage | equations still domain judgment | yes — formalism |
| Bias placement | positioning vocabulary | no object | cite only |
| Source taxonomy | positioning vocabulary | no locus | cite only |
| Output projection | exactness, wrong space | discrete, untrained output | no; contrast |
| Monitor gating | the mechanism: a checker that rejects | their constraints are policy, ours dynamics | yes — mechanism |
| Rejection vs scoring | why reject beats blend | no filter definition | cite as justification |

## Recommended design

Hybrid automaton as the formalism, monitor gating as the mechanism, the
existing calibrated product retained as the quantitative layer inside it.
Three layers, cleanly separated by what they claim:

1. **Structure (corpus).** G_c supplies modes, transitions, and conditional
   strengths. Unchanged.

2. **Admissibility (physics, calibration-free).** For each transition e into
   a physics-modeled mode F, a guard margin g_F(theta_case) with units,
   evaluated from the case's recorded drivers. g > 0: transition physically
   available; g <= 0: not available as reconstructed. Reported per case and,
   aggregated, as a physical-violation rate of extracted chains. This layer
   ranks and judges; it needs no level calibration, which retires the M1
   objection for everything this layer claims. Coverage is countable: state
   which factor types carry guards (stall, icing, fuel, shear, structural
   overload) and which do not.

3. **Quantification (physics through a calibrated link, corpus
   conditionals).** Where absolute risk is wanted, pi_F = link(g_F) with the
   level fixed by Eq. (5) — executable once the exposure study supplies a
   non-accident denominator — then the existing R = pi_F * P(Y|F) and a
   corrected combine (leaky noisy-OR fit by maximum likelihood, icing
   double-count removed). Until Eq. (5) runs, this layer reports risk ratios
   only, never absolute per-flight levels.

The controls that make the formulation testable, both from Xinyu: a
corrupted-margin arm (shuffle or negate margins; admissibility judgments and
scenario rankings should collapse toward chance) and the countable-coverage
statement replacing "physics where available."

Contribution sentence for the paper: the causal chain is modeled as a path
over the modes of a hybrid automaton in which corpus data supply the
transition structure and conditional strengths while per-case physics
margins act as guards that decide transition admissibility and, through a
calibrated link, occurrence probability.

## What changes in the manuscript

- Section 3.4/3.5 (fusion, combine): reframed around the three layers;
  Eq. (6) survives as layer 3; guards defined once with a table mapping
  factor types to margins and units.
- Section 4: each physics subsection states its margin explicitly (the
  quantities already exist inside the models); one worked admissibility
  example (forced-landing glide reach or carburetor heat).
- Section 6: add physical-violation rate of extracted chains where drivers
  are recorded, plus the corrupted-margin control.
- Related Work: one paragraph placing the six candidates, with the
  projection contrast and the reject-not-score justification.
- Not adopted, stated as boundary: no reachability analysis, no learned
  automaton identification, no claim that the formalism chooses g_F.

Estimated effort: notation and text ~2 days; margin refactor of
physics/*.py ~1 day; violation-rate script over extracted chains ~1 day;
corrupted-margin control ~0.5 day. Independent of the gold496 evaluation
now running.
