# ACE-Graph Design Critique — Expert Review

**Reviewer perspective:** AI / causal inference / LLM systems  
**Date:** 2026-04-11  
**Document reviewed:** README.md (Sections 1–9) and linked design spec

---

## Overall Assessment

The project is ambitious and well-motivated. Combining hierarchical event extraction, LLM-generated causal priors, and counterfactual evaluation into an aviation-safety pipeline addresses a real gap. The scope decisions (per-category sub-graphs, LOC-I first, unified corpus) are sensible. However, the design has several methodological weaknesses that, if unaddressed, will undermine the credibility of the contribution at a top venue. The most critical issues are: (1) unjustified design choices presented as principled (especially the richness score), (2) circularity risks from heavy LLM dependence across multiple stages, (3) a conflation between investigator judgment and causal ground truth in the evaluation, and (4) insufficient treatment of error propagation through a four-stage pipeline.

---

## 1. Richness Score (Section 3.4) — Major Concern

### 1.1 Arbitrary weights with no justification

The richness score formula:

```
richness = 0.25 * norm(len(narrative_factual), 150, 800)
         + 0.20 * norm(len(narrative_cause),  50, 400)
         + 0.20 * min(1, n_structured_events / 5)
         + 0.15 * min(1, n_findings / 4)
         + 0.10 * n_temporal_markers_capped
         + 0.10 * (all_context_present ? 1 : 0)
```

The weights (0.25, 0.20, 0.20, 0.15, 0.10, 0.10) are not derived from any principled method — no sensitivity analysis, no empirical tuning, no information-theoretic justification. Why is `narrative_factual` worth 25% but `temporal_markers` only 10%? If temporal structure is critical to Stage 2 (per-accident DAG construction), arguably temporal markers should be weighted *more* heavily, not less.

**Recommendations:**
- Run a sensitivity analysis: vary weights across a grid and measure downstream impact on Stage 1 extraction quality. Report how sensitive the "reasoning-ready" subset is to weight perturbation.
- Consider learning the weights. A simple logistic regression predicting whether a record yields a valid causal chain (using a small annotated set) would be far more defensible than hand-tuned coefficients.
- At minimum, present the weights as a "default configuration" and report the Pareto front of weight choices vs. corpus size vs. downstream quality.

### 1.2 Additive functional form is unjustified

The score is a weighted sum, which assumes the components are linearly substitutable. But are they? A record with a 2000-token narrative and zero structured events is very different from one with a 200-token narrative and 10 structured events — yet the additive form treats them as roughly equivalent. In reality, some components are likely *complementary* (narrative + structured events together are worth more than the sum) while others may be *redundant* (narrative_cause and findings often encode the same information).

**Recommendation:** At least test a multiplicative or min-based gate alongside the additive score. Even better, frame richness as a multi-criteria decision and use dominance-based filtering (a record is "rich enough" if it exceeds thresholds on k-of-n criteria) rather than collapsing to a single scalar.

### 1.3 The `norm()` function is undefined

The document uses `norm(len, min, max)` without specifying it. Is it clipped linear interpolation? Sigmoid? Log-scaled? For narrative length, a log transform is more natural — the marginal information gain from 800 vs. 600 tokens is much less than from 200 vs. 150 tokens. Linear normalization over-rewards verbose reports.

**Recommendation:** Define `norm()` explicitly and justify the choice. Consider `norm(x, lo, hi) = clip((log(x) - log(lo)) / (log(hi) - log(lo)), 0, 1)` for token-count features.

### 1.4 Token-count thresholds (150, 50) are model-dependent

"150 tokens" depends on the tokenizer. GPT-4 tokens differ from LLaMA tokens differ from word-split tokens. The document doesn't specify which tokenizer. If these thresholds are meant to be stable across model choices, use word counts or character counts instead.

### 1.5 Tier thresholds lack justification

The tier boundaries (score >= 0.5 for reasoning_ready, >= 0.3 for partial) are arbitrary. Why 0.5? The document acknowledges that "cutoff will be tuned on the NTSB distribution during Stage 0 build" for the 150-token gate, but no similar commitment exists for the tier thresholds themselves.

**Recommendation:** Plot the richness score distribution per source and choose thresholds at natural break points (e.g., via Jenks natural breaks or KDE valley detection), rather than round numbers.

---

## 2. LLM Circularity Risk — Major Concern

The pipeline uses LLMs in at least four distinct roles:

1. **Stage 0:** LLM-assisted CICTT category classification for FAA AIDS records
2. **Stage 1:** LLM free-form subevent role extraction
3. **Stage 3:** LLM triplet-prompted causal order (Vashishtha et al.)
4. **Stage 3:** LLM-curated Laplacian similarity prior (Zhang et al.)
5. **Stage 4:** CausalCoT counterfactual reasoning

This creates a serious circularity risk: the same LLM's biases and knowledge gaps propagate through every stage. If the LLM has a systematic misconception about aviation causality (e.g., over-weighting pilot error, under-weighting maintenance failures — which is a known bias in public discourse that LLMs absorb), this bias will:
- Distort category assignment (Stage 0)
- Shape event extraction (Stage 1)
- Influence the causal prior (Stage 3)
- Determine counterfactual answers (Stage 4)

The evaluation would then measure the LLM's consistency with itself, not the quality of causal reasoning.

**Recommendations:**
- Use *different* LLMs (or different model families) for extraction vs. causal prior vs. counterfactual reasoning, and report the cross-model agreement.
- Where structured data exists (NTSB `Findings`, `seq_of_events`), use it as the *primary* signal and LLM output as supplementary. The current design says "reconciled against structured Findings where available" but doesn't specify what happens when LLM and structured data disagree. Define the conflict-resolution policy explicitly.
- Add an ablation: Stage 3 causal graph built *without* LLM priors (pure constraint-based discovery from data) as a baseline. This isolates the LLM's contribution from the structured data's contribution.

---

## 3. "Ground Truth" for Causal Evaluation — Major Concern

### 3.1 NTSB probable cause is not causal ground truth

The evaluation (Section 8) treats NTSB `Findings` + `probable_cause` as ground truth for counterfactual reasoning. This is problematic:

- **NTSB probable cause is a regulatory/legal judgment**, not a scientific causal determination. Investigators operate under time pressure, institutional norms, and legal frameworks that shape their conclusions. "Pilot failure to maintain airspeed" may be the proximate cause cited, but the counterfactual question "would the accident have occurred without this factor?" may have a different answer when considering systemic factors (training, cockpit design, ATC instructions).
- **NTSB findings are not independent of the narrative.** The same investigators who wrote the narrative also wrote the findings. A system trained on narratives and evaluated against findings from the same document is partly measuring textual consistency, not causal reasoning.
- **Selection bias in cause attribution.** NTSB systematically attributes causes to factors that are *actionable* within FAA jurisdiction. Systemic, organizational, or regulatory causes are under-represented in `probable_cause` relative to their true causal role.

**Recommendations:**
- Frame the evaluation honestly: "agreement with NTSB investigator judgment" rather than "causal accuracy." This is not just a wording issue — it changes what the metrics mean.
- Add a secondary evaluation against an independent source: HFACS classifications by trained human factors analysts, or published re-analyses of high-profile accidents where the causal story is well-established.
- Consider inter-rater reliability: for a subset, have domain experts independently assess the counterfactual claims and measure agreement with both the system and NTSB.

### 3.2 Leave-one-out protocol is underspecified

"Exactly how a factor is 'removed' and how the counterfactual probability is measured" is listed as TBD (Section 7). This is the *core* evaluation mechanism and the *primary metric*. The entire pipeline is designed to support this evaluation, yet the evaluation itself is unspecified. This is a significant design risk — if the leave-one-out protocol turns out to be ill-defined or trivial, the paper loses its evaluation backbone.

**Recommendation:** Prioritize formalizing the leave-one-out protocol *before* building Stages 1-3. Specifically:
- Define "removing a factor" in graph terms (node deletion? edge intervention? do-calculus `do(X=absent)`?).
- Define the counterfactual probability estimand precisely.
- Run a feasibility check: take 10 NTSB LOC-I accidents, manually construct the causal graph and counterfactual, and verify that the leave-one-out protocol produces meaningful distinctions.

---

## 4. Per-Category Sub-Graph Assumptions — Moderate Concern

### 4.1 Category boundaries are porous

The design assumes causal mechanisms are category-specific (Section 2.1). But many accidents involve multiple CICTT categories simultaneously (e.g., LOC-I triggered by SCF-PP, or CFIT following navigation error that also involves MAC risk). Assigning each accident to *one* category and building separate graphs loses these cross-category causal pathways.

**Recommendation:** Allow multi-label category assignment. Build per-category sub-graphs but also construct a small set of "bridge edges" between categories representing known cross-category causal pathways. Alternatively, build both per-category and a merged graph and compare.

### 4.2 Sample size concerns for rare categories

The design acknowledges LOC-I is chosen for sample size, but the paper claims to build per-category graphs. For rare categories (e.g., WSTRW, TURB, BIRD), the sample size in UASC-Rich may be too small for any causal discovery algorithm to produce stable results, even with LLM priors.

**Recommendation:** Report the category distribution in UASC-Rich early and define a minimum sample-size threshold below which a category is excluded or merged into an "Other" bucket. Be explicit about how many categories the paper will actually cover.

---

## 5. Pipeline Error Propagation — Moderate Concern

The four-stage pipeline has no error-correction mechanism. Errors in Stage 0 (wrong category assignment) propagate to Stage 3 (wrong sub-graph). Errors in Stage 1 (missed events) propagate to Stage 2 (incomplete DAGs) and Stage 3 (biased causal structure). The document does not discuss:

- What is the expected error rate at each stage?
- How sensitive is Stage N to errors in Stage N-1?
- Are there feedback loops or consistency checks between stages?

**Recommendations:**
- Design at least one cross-stage consistency check. For example: after Stage 2 produces temporal DAGs, verify that the event vocabulary is consistent with Stage 1's extraction. After Stage 3 produces causal graphs, verify that high-confidence causal edges are supported by temporal ordering in Stage 2.
- Report per-stage error rates in the paper, not just final evaluation metrics. A strong final number built on a shaky intermediate stage is a house of cards.
- Consider a joint or iterative approach for Stages 1-2 at minimum, where temporal structure informs event extraction and vice versa, rather than a strict feed-forward pipeline.

---

## 6. Corpus Harmonization (Section 3) — Moderate Concerns

### 6.1 NTSB dominance undermines the "unified corpus" claim

The document projects that NTSB will contribute ~70-80% of reasoning-ready records, BEA ~20-30% (conditional on full-report scraping, which is "optional"), FAA AIDS ~10%, and TSB ~40%. In practice, the reasoning-ready corpus will be overwhelmingly NTSB, especially if BEA scraping fails.

This is fine for a first paper, but the framing of "unified cross-jurisdiction corpus" oversells the cross-jurisdiction aspect. If 85%+ of reasoning-ready records are NTSB, the paper is primarily an NTSB pipeline with a cross-jurisdiction test set.

**Recommendation:** Be honest about this in framing. The corpus construction is a contribution regardless; don't undermine it with overclaiming.

### 6.2 Deduplication strategy is incomplete

The deduplication strategy (Section 3.6) uses blocking on `(date ± 1 day, country, aircraft_category)`. This will miss:
- Records where the date is recorded differently (occurrence date vs. report date vs. notification date)
- Records where aircraft_category is coded differently across sources
- Near-duplicates where one source has partial information

Also, "TF-IDF similarity >= 0.7" on location is a suspiciously specific threshold with no justification.

**Recommendation:** Use a more robust blocking scheme (e.g., LSH on concatenated text fields) and validate the deduplication against a manually reviewed sample of known cross-source duplicates.

### 6.3 Schema mapping for FAA AIDS is high-risk

FAA AIDS uses older cause codes that predate CICTT. The document acknowledges this requires "rule-based + LLM-assisted mapping" but doesn't discuss the expected accuracy. If this mapping is noisy, FAA AIDS records will pollute per-category sub-graphs.

**Recommendation:** Validate the FAA AIDS -> CICTT mapping on a sample of dual-coded records (accidents that appear in both NTSB and FAA AIDS with known CICTT codes).

---

## 7. Methodological Integration Concerns — Moderate

### 7.1 Compatibility of the four "ingredients"

The paper claims to integrate four methodological ingredients (Section 1.2). But these methods were developed for different settings:

- **HABERT** (Zhao et al.): designed for NTSB specifically, with NTSB's ontology. Applying it to non-NTSB text is a transfer-learning problem, not a direct application.
- **Vashishtha et al. (ICLR 2025):** causal order from LLM experts, designed for general causal inference benchmarks. Adapting to domain-specific aviation event vocabularies may require significant prompt engineering — the method's effectiveness is not guaranteed to transfer.
- **Zhang et al. (NeurIPS CaLM 2024):** Laplacian similarity prior for concurrent causes. This assumes a specific SCM framework. Is the aviation causal structure compatible with their assumptions (e.g., linearity, additive noise)?
- **CLadder (Jin et al., NeurIPS 2023):** evaluation framework for causal reasoning in LLMs. This is an eval tool, not a reasoning method. Using it as a "reasoning layer" (as Section 2 implies) conflates evaluation with inference.

**Recommendation:** For each ingredient, explicitly state: (a) what assumptions it makes, (b) whether those assumptions hold in the aviation domain, (c) what adaptations are needed. A "unified pipeline" paper must demonstrate that the components are compatible, not just that they can be arranged in sequence.

### 7.2 Pearl's ladder framing may be overreaching

Mapping the pipeline stages to Pearl's causal hierarchy (Rung 1: association, Rung 2: intervention, Rung 3: counterfactual) is elegant but potentially misleading. True Rung-3 counterfactual reasoning requires a fully specified structural causal model with known functional relationships. What the pipeline actually produces is:

- A causal *graph* (structure only, not functional form)
- LLM-generated counterfactual *judgments* (not formal counterfactual inference via abduction-action-prediction)

Claiming Rung-3 capability when the system relies on LLM text generation for counterfactuals — rather than on formal SCM-based counterfactual computation — will draw reviewer criticism.

**Recommendation:** Either (a) implement formal counterfactual inference (which requires parameterizing the SCM, not just learning the graph), or (b) frame the contribution as "LLM-assisted counterfactual question answering grounded in a learned causal graph" rather than "Rung-3 counterfactual reasoning." The latter is more honest and still a strong contribution.

---

## 8. Evaluation Design (Section 8) — Moderate Concerns

### 8.1 Missing baselines

The listed baselines (retrieval, raw LLM QA, random, graph-without-prior) are a start but miss important comparisons:

- **NTSB structured data alone** (no NLP, just Findings + seq_of_events): How much does the NLP pipeline add over what's already in the database?
- **Single-stage LLM** (give GPT-4/Claude the full narrative, ask for counterfactual directly): This is the baseline reviewers will immediately think of. If a single prompted LLM matches the four-stage pipeline, the pipeline's complexity is unjustified.
- **Non-LLM causal discovery** (e.g., PC algorithm on structured event co-occurrence, no LLM prior): Isolates the LLM prior contribution.

### 8.2 Metrics for graph quality are absent

The evaluation focuses on end-to-end counterfactual accuracy but says nothing about intermediate artifact quality:
- How good are the extracted events? (precision/recall against NTSB structured events)
- How good are the temporal DAGs? (agreement with `seq_of_events` ordering)
- How good are the causal graphs? (edge precision/recall against domain expert judgment)

Without intermediate metrics, a bad final number gives no diagnostic information about *where* the pipeline fails.

**Recommendation:** Define evaluation metrics for each stage, not just Stage 4.

---

## 9. Minor Issues

1. **Section 3.3, Gate 3 — temporal marker detection:** Counting keywords like "then" and "shortly after" is fragile. "Then" appears in non-temporal contexts frequently. Consider using a temporal relation classifier (e.g., based on TimeML/TempEval) rather than keyword matching.

2. **Section 3.5 — projection of non-NTSB events into NTSB Subject codes:** This is a significant modeling decision buried in a subsection. If the projection is noisy, it contaminates the fine-grained event vocabulary. Consider treating projected codes as a separate, lower-confidence tier in Stage 3.

3. **Missing discussion of computational cost.** The pipeline involves multiple LLM calls per record across multiple stages. For a corpus of potentially tens of thousands of records, this is expensive. Budget and throughput estimates should inform design choices (e.g., which LLM to use at each stage).

4. **No discussion of reproducibility.** LLM outputs are non-deterministic. How will you ensure reproducibility? Cache all LLM responses? Use temperature=0? Report variance across runs?

5. **Reference list is thin.** Four papers + one dissertation. A design document for a venue submission should engage more deeply with related work: HFACS, Rasmussen's AcciMap, STAMP/STPA, existing aviation NLP work (e.g., ASRS topic modeling literature), and the broader causal discovery literature (Spirtes, Peters, etc.).

6. **BEA scraping fragility.** The design treats BEA full-report scraping as "optional" but BEA records are metadata_only without it. If 20-30% pass rate is conditional on full-report text, and scraping is optional, the realistic BEA pass rate is ~5%. Be explicit about this.

7. **No ethical considerations section.** Aviation accident data involves fatalities. Even though this is public data, a paper should discuss ethical use, especially around named individuals in narratives.

---

## Summary of Recommendations by Priority

### Must-fix before building
1. Formalize the leave-one-out counterfactual evaluation protocol and validate feasibility on a small manual sample
2. Define conflict-resolution policy for LLM vs. structured data disagreements
3. Add the "single-stage LLM baseline" — if it wins, the pipeline is not justified
4. Address LLM circularity with cross-model or LLM-vs-structured-data ablations

### Should-fix before paper submission
5. Replace or rigorously justify the richness score weights (sensitivity analysis at minimum)
6. Define per-stage evaluation metrics, not just end-to-end
7. Temper the Pearl's ladder / Rung-3 framing to match what the system actually computes
8. Expand related work engagement significantly
9. Be honest about NTSB dominance in the "unified corpus"

### Nice-to-have improvements
10. Multi-label category assignment for cross-category accidents
11. Cross-stage consistency checks
12. Computational cost and reproducibility protocols
13. Ethics section
