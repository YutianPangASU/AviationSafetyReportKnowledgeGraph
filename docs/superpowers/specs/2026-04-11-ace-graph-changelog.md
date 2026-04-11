# ACE-Graph v1 -> v2 Change Log

**Date:** 2026-04-11
**Source:** [2026-04-10-ace-graph-design.md](2026-04-10-ace-graph-design.md) (v1)
**Target:** [2026-04-11-ace-graph-design-v2.md](2026-04-11-ace-graph-design-v2.md) (v2)
**Critique:** [../review/2026-04-11-design-critique.md](../review/2026-04-11-design-critique.md)

---

## Summary of Changes

v2 addresses all 13 critique points from the expert review. The reference list expanded from 5 to 25 papers. Four new sections were added (Limitations, Computational Budget, Ethics, Related Work). The richness score was overhauled with explicit definitions, sensitivity analysis protocol, and alternative filtering methods. The leave-one-out evaluation protocol was formalized. LLM circularity mitigation was added as a first-class design concern.

---

## Critique-Point Mapping

| # | Priority | Critique Summary | What Changed in v2 | Sections Affected |
|---|---|---|---|---|
| 1 | Must-fix | Richness score weights arbitrary, norm() undefined, thresholds unjustified | Defined `lognorm()` explicitly with log transform; switched to word counts; reframed weights as "default config pending sensitivity analysis"; added k-of-n dominance filter alternative; Jenks natural breaks for tier thresholds | 3.4, Appendix B |
| 2 | Must-fix | LLM circularity — same biases propagate through all stages | New Section 2.5: LLM Usage Policy table (5 roles), model family diversification policy, structured data primacy rule, conflict-resolution rules, mandatory no-LLM-prior ablation | 2.5, 4, 6, 7, 8.3 |
| 3 | Must-fix | NTSB probable cause is not causal ground truth | Reframed throughout as "agreement with investigator judgment"; documented 3 known biases in Section 7.2; added secondary eval sources (HFACS, inter-rater reliability); Limitations Section 10.2 | 1.1, 1.4, 7.2, 8.2, 10.2 |
| 4 | Must-fix | Leave-one-out protocol unspecified | New Section 7.1: formalized with do-calculus notation, counterfactual estimand, mandatory feasibility check on 10 LOC-I accidents before building pipeline | 7.1 |
| 5 | Should-fix | Richness weights need sensitivity analysis | Appendix B: full protocol with grid search, Pareto front, logistic regression alternative, k-of-n comparison | 3.4, 3.7, 3.8, Appendix B |
| 6 | Should-fix | No per-stage evaluation metrics | New Section 8.1: per-stage metrics table (Stage 0-4) with gold standard and measurement method for each; per-stage eval bullets added to Sections 4-6 stubs | 4, 5, 6, 8.1 |
| 7 | Should-fix | Pearl's Rung-3 framing overreaches | Tempered throughout: "LLM-assisted counterfactual QA" not "Rung-3 reasoning"; CLadder clarified as eval framework; out-of-scope updated; new Limitation 10.4 | 1.1, 1.2, 1.5, 2 (pipeline diagram), 10.4 |
| 8 | Should-fix | Related work thin (5 refs) | New Section 9: 25 references in 6 subsections covering aviation safety frameworks, aviation NLP, causal discovery, temporal extraction, event graphs, LLM reasoning | 9, 13 |
| 9 | Should-fix | NTSB dominance understated | Honest framing added in Section 1.3; BEA realistic projection (5% without scraping) in 3.1; cross-jurisdiction eval framing in 8.4; Limitation 10.1 | 1.3, 3.1, 8.4, 10.1 |
| 10 | Nice-to-have | Multi-label category assignment | Schema updated: `accident_category` split into `primary` + `secondary`; multi-label handling in 2.1 and 3.5; cross-category bridge edges in Stage 3 | 2.1, 3.2, 3.5, 6 |
| 11 | Nice-to-have | No cross-stage consistency checks | New Section 2.3: three explicit checks (event vocab alignment 1->2, temporal support for causal edges 2->3, counterfactual graph consistency 3->4) | 2.3, 5 |
| 12 | Nice-to-have | Computational cost/reproducibility missing | New Section 11: LLM call budget table, reproducibility protocol (caching, temperature=0, 3-run variance), hardware estimates | 11 |
| 13 | Nice-to-have | No ethics section | New Section 12: fatality data handling, named individuals, no safety-of-life decisions, misuse risk, data sourcing compliance | 12 |

---

## Additional Changes (not directly from critique but arising during revision)

| Change | Rationale | Sections |
|---|---|---|
| One-line summary revised to say "investigator judgments" not "probable-cause" | Consistency with Critique #3 reframing | Document header |
| "Formal SCM parameterization" added to out-of-scope | Honesty about what the system does not do (Critique #7) | 1.5 |
| FAA AIDS mapping validation on dual-coded records | Raised in critique Section 6.3 as high-risk | 3.5, 3.8 |
| Projected non-NTSB Subject codes treated as lower-confidence tier | Raised in critique Section 9, point 2 | 3.5, 3.8 |
| Deduplication: LSH added as complementary blocking | Raised in critique Section 6.2 | 3.6 |
| Deduplication: TF-IDF threshold replaced with "validated on sample" | Raised in critique Section 6.2 (unjustified 0.7 threshold) | 3.6 |
| Richness score deliverable #7 added (analysis report) | Supports Appendix B protocol | 3.7 |
| Expanded baselines in Section 8.3 | Single-stage LLM and NTSB-structured-only baselines (critique Section 8.1) | 8.3 |
| Assumption audits added to Stage 3 planned topics | Critique Section 7.1 on ingredient compatibility | 6 |
| Appendix A open questions updated with resolution status | Tracking which brainstorming questions are now resolved | Appendix A |
| Section numbering shifted: References now Section 13 | Accommodate new sections 10-12 | All |

---

## Structural Comparison

| Aspect | v1 | v2 |
|---|---|---|
| Sections | 9 + 1 appendix | 13 + 2 appendices |
| References | 5 | 25 |
| New sections | — | 2.3 (Consistency Checks), 2.4 (Error Propagation), 2.5 (LLM Policy), 10 (Limitations), 11 (Computational Budget), 12 (Ethics), Appendix B (Sensitivity Analysis) |
| Expanded sections | — | 3.4 (Richness Score), 7 (Stage 4), 8 (Evaluation), 9 (Related Work) |
| Baselines | 4 | 9 |
| Evaluation scope | End-to-end only | Per-stage + end-to-end |
| Richness score `norm()` | Undefined | Explicitly defined as `lognorm()` with log transform |
| Token vs word counts | Token counts (tokenizer-dependent) | Word counts (tokenizer-independent) |
| Ground truth framing | "probable-cause accuracy" | "agreement with investigator judgment" + 3 documented biases |
| LLM circularity | Not discussed | Full mitigation policy (Section 2.5) |
| Leave-one-out protocol | "TBD" | Formalized with do-calculus + feasibility check |
