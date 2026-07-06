# Related-Work Notes — Literature for the ACE-Graph Paper

**Date compiled:** 2026-07-05
**Provenance:** Collected via a web/Google-Scholar-style sweep (deep-research workflow + literature agents). Details for Elsevier-paywalled items (marked ⚠) come from abstracts/search snippets, not full texts — verify numbers against the full paper before citing. The adversarial verification pass of the sweep did not complete (rate limits), so treat every quantitative claim below as "as reported by the source," to be re-checked at writing time.

**How to use:** Each entry has a one/two-sentence summary and a "Relevance" line saying how it should be discussed relative to ACE-Graph (baseline / prior art to differentiate / gap evidence / methodology to borrow). Section 7 collects the positioning arguments the related-work section should make.

PDFs already in this folder are marked 📁.

---

## 1. LLM / NLP extraction and classification from aviation safety narratives

### Domain-adapted language models (pre-LLM-reasoning era)

- **SafeAeroBERT: Towards a Safety-Informed Aerospace-Specific Language Model** — Andrade, S.R. & Walsh, H.S. (NASA Ames), AIAA AVIATION Forum 2023 (AIAA 2023-3437). BERT further-pretrained on >400k ASRS + NTSB reports; safety-document classification; beats BERT/SciBERT on weather, procedure, human-factors, aircraft categories. https://arc.aiaa.org/doi/abs/10.2514/6.2023-3437
  - *Relevance:* establishes the domain-pretraining baseline lineage; classification only, no structured event/cause extraction.
- **Aviation-BERT / BERT for Aviation Text Classification** — Jing, X. et al. (Georgia Tech), AIAA AVIATION Forum 2023 (AIAA 2023-3438; model paper AIAA 2023-3436). Multi-label ASRS anomaly classification mapped to ICAO occurrence categories; domain-pretrained BERT superior to BERT-base. https://arc.aiaa.org/doi/10.2514/6.2023-3438
  - *Relevance:* same lineage; note it targets ICAO/CICTT-style occurrence categories, which our Stage-3 aggregation also uses — but at document level, not event level.
- **AviationGPT: A Large Language Model for the Aviation Domain** — Wang, L. et al. (MITRE), arXiv:2311.17686 / AIAA 2024-4250. LLaMA-2/Mistral domain-adapted with RAG for aviation IE, report querying, summarization; >40% gain over base models. https://arxiv.org/abs/2311.17686
  - *Relevance:* domain-LLM baseline; free-form IE/QA rather than schema-constrained causal-event extraction.

### Fine-grained event extraction (the closest extraction prior art)

- 📁 **Hierarchical Multilabel Classification for Fine-Level Event Extraction from Aviation Accident Reports** — Zhao, X. et al., 2024 (INFORMS J. on Data Science; arXiv:2403.17914; PDF in this folder). First attempt at fine-level (1,432 subject-code) event extraction from NTSB narratives; argues prior NLP work only identified coarse event categories. Cites Rao & Marais (2020): NTSB-coded occurrence chains are inconsistent and short — 152 distinct chains, >82% of GA accidents with average chain length ≤2 — so narrative text must be mined to fill in the missing causal logic. https://arxiv.org/pdf/2403.17914
  - *Relevance:* **load-bearing motivation citation.** (a) Confirms fine-grained schema-constrained extraction was open immediately pre-LLM; (b) the short/inconsistent NTSB event-chain statistic is our core argument for recovering causal structure from narratives rather than reading it off `seq_of_events`. Differentiator: they do multilabel classification onto codes; we extract typed event instances + edges + cause roles.

### HFACS × LLM cluster (human-factors classification)

- ⚠ **Accident investigation via LLMs reasoning: HFACS-guided Chain-of-Thoughts enhance general aviation safety** — Liu, Q. et al., Expert Systems with Applications, 2025 (S0957417425000442). HFACS-CoT prompting guides GPT-4o to infer unsafe acts/preconditions (HFACS 8.0) from NTSB GA witness narratives; outperforms zero-shot/few-shot/auto-CoT/plan-and-solve; reported to match or exceed human experts on some HFACS categories. https://www.sciencedirect.com/science/article/abs/pii/S0957417425000442
  - *Relevance:* the flagship "LLM ≈ expert on human-factors coding" citation; but single-taxonomy document-level labeling, no event graph, no causality.
- **Automated HFACS Classification Using Reinforcement Learning with GRPO** — Ahmadi, A., 2025, arXiv:2508.21201. Llama-3.1-8B GRPO-fine-tuned for multi-label HFACS classification of accident narratives; beats GPT-5-mini and Gemini-2.5-flash; exact-match 0.04→0.18, partial-match 0.88. https://arxiv.org/abs/2508.21201
  - *Relevance:* shows small open models + RL can compete with frontier APIs on this coding task — supports our local-OSS-model (Qwen/vLLM) pipeline choice.
- **UAV HFACS with LLMs** — Drones 9(10):704, 2025. 200 expert-annotated UAV ASRS reports (3,600 coded instances); HFACS-guided structured prompting across seven LLMs; macro-F1 0.58–0.76 over 18 HFACS 8.0 categories. Human experts still beat the best LLM on L1/L2 (F1 0.728 vs 0.669 for GPT-5-mini), though the LLM wins on some categories (e.g., AE200 Decision-Making Error). Quantified faithfulness analysis of GPT-4o: 8.57% hallucinated labels vs 18.44% omissions (2.15:1 omission:hallucination — conservative bias); organizational factors most missed (37.5% of omissions), psychological preconditions most hallucinated. Claims its directed-graph HFACS encoding "enables do-calculus," but performs **no** counterfactual or interventional evaluation ("counterfactual" never appears). https://doi.org/10.3390/drones9100704
  - *Relevance:* triple duty — (a) methodology template for our own hallucination/omission error-pattern analysis of Stage-1 extraction; (b) evidence experts still lead on latent/organizational factors (motivates supervision against NTSB Findings); (c) **gap evidence**: even papers that gesture at do-calculus never execute it — our Stage 3/4 is the unclaimed step.

### Other extraction / classification work

- **Uncovering Resilient Behavior in the ASRS Using Large Language Models** — Matthews, B. (NASA), 44th IEEE/AIAA DASC, 2025. Llama-3.1-Instruct extracts pilot resilient behaviors from >250k ASRS narratives; trend analysis across anomaly categories. https://ntrs.nasa.gov/citations/20250004265
  - *Relevance:* one of the few LLM-over-ASRS efforts at corpus scale (250k) — comparable scale to ours (178k corpus / 56k extracted); but single-concept extraction, not typed event graphs.
- **Towards Enhancing Aviation Safety Through Advanced Incident Analysis Using LLMs** — Siddeshwar, V. et al., CASCON 2024. Locally deployed LLaMA extracts structured fields (aircraft model, primary problem, contributing factors, human factors, summary) from 1,107 ASRS reports. https://www.researchgate.net/publication/388119000
  - *Relevance:* structured extraction prior art, but 1.1k reports and flat fields — 50× smaller and no graph/causality; useful scale contrast.
- **NASP-T: A Fuzzy Neuro-Symbolic Transformer for Logic-Constrained Aviation Safety Report Classification** — Al Machot, F. et al., 2025, arXiv:2510.05451. ASP domain rules + transformer fine-tuning on ASRS multi-label classification; micro/macro-F1 gains and up to 86% fewer rule violations. https://arxiv.org/abs/2510.05451
  - *Relevance:* neurosymbolic constraint enforcement in this exact domain — analogous in spirit to our `guided_json` closed-vocabulary enforcement; cite when arguing schema constraints reduce invalid outputs.
- **Deep Learning Approaches for Classifying Aviation Safety Incidents: Evidence from Australian Data** — AI (MDPI) 6(10):251, 2025 (UNSW group). BERT vs CNN vs LSTM on 53,273 ATSB records (2013–2023) for injury-severity classification; BERT ≈1.00 P/R/F1. https://www.mdpi.com/2673-2688/6/10/251
  - *Relevance:* non-US corpus (Australia) exists and is used — but monolithic classification; supports the cross-jurisdiction gap argument.
- ⚠ **Domain-adapted deep learning for aviation incident classification with multiple labels and risk assessment** — Engineering Applications of AI, 2026 (S0952197626007359). RoBERTa + instruction-based LLM data augmentation on ASRS. https://www.sciencedirect.com/science/article/abs/pii/S0952197626007359
  - *Relevance:* recent classification SOTA on ASRS; background.
- **Application of LLMs for Automatic Classification of Work Accident Text Data** — medRxiv 2025.10.02.25337141. LLM vs human coding agreement on occupational accident narratives. https://www.medrxiv.org/content/10.1101/2025.10.02.25337141.full.pdf
  - *Relevance:* cross-domain evidence on LLM-vs-human-annotator agreement for safety-report coding; cite in the evaluation-methodology discussion.

---

## 2. Knowledge graphs / causal KGs from accident reports (closest overall prior art)

- 📁 **Information Extraction of Aviation Accident Causation Knowledge Graph: An LLM-Based Approach** — Chen, L. et al., Electronics 13(19):3936, 2024 (PDF in this folder). "Claude-prompt" (prompt engineering + few-shot + self-judgment) with Claude 3.5 extracts cause entities/relations to build an aviation accident causation KG (~14,768 causal factors reported); beats ChatGLM-6B, GPT-3.5, GPT-4 on extraction P/R/F1. https://www.mdpi.com/2079-9292/13/19/3936
  - *Relevance:* **the closest single prior paper.** Differentiators to state explicitly: open-vocabulary entity/relation triples vs our closed 42-type event vocabulary with cause_role supervision; no structured-field supervision (NTSB Findings/Cause_Factor); no counterfactual layer; no cross-database corpus.
- ⚠ **LLM-based entity extraction + KG construction on aviation safety investigation reports** — Safety Science, 2026 (S0925753526001219). Peer-reviewed LLM+KG pipeline over investigation-report narratives; explicitly positioned as *diagnostic* — mapping safety-factor interdependencies and risk concentrations — **not** a causal model supporting interventional/counterfactual queries. https://www.sciencedirect.com/science/article/pii/S0925753526001219
  - *Relevance:* proves the LLM→KG idea has reached a top safety journal (differentiation needed), and simultaneously documents in-print that the causal/counterfactual layer is missing — quote its diagnostic framing as gap evidence.
- **AI4AirSafe** — J. of Automation & Intelligence, 2026. Quintuple-based KG + LLM causal reasoning on **63** aviation reports; extraction F1 ≈0.89; no counterfactual inference. (From sweep interim findings — pin down full citation before use.)
  - *Relevance:* recent KG+LLM+causal-flavored aviation paper, but ~3 orders of magnitude below our corpus scale and stops short of Rung 2/3; ideal scale-and-depth contrast.
- ⚠ **Retrieval-Augmented Generation-aided causal identification of aviation accidents** — Expert Systems with Applications, 2025 (S0957417425009285). RAG + LLM pipeline for causal-factor identification from accident reports. https://www.sciencedirect.com/science/article/abs/pii/S0957417425009285
  - *Relevance:* "causal identification" here = retrieving/naming causal factors, not building a causal model; keep the distinction sharp in related work.
- **Bayesian network modeling of accident investigation reports for aviation safety assessment** — Zhang, X. & Mahadevan, S., Reliability Engineering & System Safety, 2021 (S0951832020308607). Learns a BN over causes/consequences encoded from NTSB reports; probabilistic what-if/sensitivity assessment. https://www.sciencedirect.com/science/article/abs/pii/S0951832020308607
  - *Relevance:* the pre-LLM version of our Stage 3 — BN from NTSB reports, but manual/coded encoding, associational conditioning (Rung 1–2), no do-operator or abduction. Natural "we automate and upgrade this" citation, and RESS venue signal.
- ⚠ **Enhanced human factor and causation analysis in maritime accidents using LLMs** — Ocean Engineering, 2026 (S0029801826009595). LLM-assisted human-factor extraction from 682 maritime investigation reports + Bayesian-network causation modeling. (Related: DeepSeek HFACS labeling of 302 NTSB maritime reports, HS-MACs dataset, https://www.researchgate.net/publication/397148830.) https://www.sciencedirect.com/science/article/abs/pii/S0029801826009595
  - *Relevance:* the same LLM→causal-model recipe is emerging in maritime — cite to show the pattern generalizes across transport modes and that aviation-side Rung-3 remains open.
- **Automated knowledge extraction from marine accident reports using LLMs: graph construction and evaluation** — 2025/2026. https://www.researchgate.net/publication/399309927
  - *Relevance:* adjacent-domain KG construction + evaluation; background.

---

## 3. Counterfactual / Pearl-ladder causal inference on accident & incident data

**Headline from the sweep: true Rung-3 (abduction–action–prediction) on accident data is rare.** Formal: Ruiz-Tagle 2022, Zibaei 2024, Ibrahim 2020. Simulation/qualitative: Scanlon 2021, Hutchinson 2024. Everything else in the safety-BN literature is Rung 1–2 conditioning, often mislabeled "counterfactual." No paper found doing do-calculus or counterfactual evaluation over event graphs *extracted from aviation report text at scale*.

### Genuine Rung 3

- ⚠ **A novel probabilistic approach to counterfactual reasoning in system safety** — Ruiz-Tagle, A., Lopez-Droguett, E., Groth, K.M., Reliability Engineering & System Safety, 2022 (S0951832022003751). "Possible-worlds" method fusing a BN risk model with accident evidence to evaluate counterfactual hypotheses in accident investigation; case study: 2018 Sun Prairie WI gas explosion. Single-event analysis on an expert-built BN. https://www.sciencedirect.com/science/article/abs/pii/S0951832022003751
  - *Relevance:* **the strongest Rung-3 safety-domain precedent** — borrow its formal counterfactual-query semantics; differentiate on (a) learned-from-text graphs vs expert BN, (b) corpus scale vs single event, (c) aviation.
- **Building causal models for finding actual causes of unmanned aerial vehicle failures** — Zibaei, E. & Borth, R., Frontiers in Robotics and AI, 2024. NLP pipeline auto-builds causal graphs from ArduPilot text corpora (935k sentences, 2,238 cause-effect pairs), then runs Halpern-Pearl actual-causality checks (HP2SAT) on 8 real UAV crash logs. https://pmc.ncbi.nlm.nih.gov/articles/PMC10880731/
  - *Relevance:* **closest text→causal-graph→counterfactual pipeline in aviation-adjacent work.** Differentiators: forum/log text vs official investigation reports; binary HP models on 8 logs vs probabilistic queries over 56k accidents; no supervision against investigator findings.
- **From checking to inference: Actual causality computations as optimization problems** (line incl. "Efficiently checking actual causality with SAT solving") — Ibrahim, A. & Pretschner, A., ATVA 2020 (LNCS). HP actual-cause inference as SAT/ILP, scaling to 4,000+ variable models; aircraft/UAV-flavored case studies. https://link.springer.com/chapter/10.1007/978-3-030-59152-6_19
  - *Relevance:* computational machinery if our Stage-4 queries need formal actual-causality at scale.

### Rung 3 in spirit (simulation / qualitative)

- ⚠ **Waymo simulated driving behavior in reconstructed fatal crashes** — Scanlon, J.M. et al., Accident Analysis & Prevention 163:106454, 2021. Counterfactual replacement of human drivers with the ADS in 72 reconstructed fatal crashes; ADS avoided 100% as initiator, ~82% as responder. https://www.sciencedirect.com/science/article/abs/pii/S0001457521004851
  - *Relevance:* industrially consequential counterfactual accident analysis via reconstruction+simulation; frames why model-grounded counterfactuals matter for safety decisions.
- **How audits fail according to accident investigations: A counterfactual logic analysis** — Hutchinson, B., Process Safety Progress, 2024. Mines "could have / should have" counterfactual statements from 44 major accident investigation reports. https://aiche.onlinelibrary.wiley.com/doi/10.1002/prs.12579
  - *Relevance:* investigators already reason counterfactually in prose — our system formalizes exactly this latent practice. Good motivating citation.

### Rung 2 (interventional) and below — the "mislabeled counterfactual" genre

- **Exploiting the capabilities of Bayesian networks for engineering risk assessment: Causal reasoning through interventions** — Ruiz-Tagle, A. et al., Risk Analysis, 2022. Upgrades BN risk models to do-operator intervention queries (gas-pipeline BaNTERA). Rung 2, no abduction. https://onlinelibrary.wiley.com/doi/abs/10.1111/risa.13711
- **Causal Bayesian networks for data-driven safety analysis of complex systems** — Gansch, R. et al., 2025, arXiv:2505.19860. Pearl CBN (do-calculus, identifiability) as successor to fault trees; automated-driving perception case; explicitly stops at Rung 2. https://arxiv.org/abs/2505.19860
- **MSCT: marginal structural causal transformer for counterfactual post-crash traffic prediction** — Li, S. et al., 2024, arXiv:2407.14065. Potential-outcomes "counterfactual" prediction under treatment sequences; Rung 2. https://arxiv.org/abs/2407.14065
- **BN scenario deduction for inland intelligent-ship collisions** — Zhang, J. et al., Reliability Engineering & System Safety 243, 2024. Evidence-setting scenario comparison; Rung 1–2, representative of the maritime BN what-if genre. https://www.sciencedirect.com/science/article/abs/pii/S0951832023007305
- ⚠ **Causal inference of construction safety management measures towards workers' safety behaviors** — Safety Science, 2024 (S0925753524000225). SCM/do-operator effect estimation of management measures; Rung 2. https://www.sciencedirect.com/science/article/abs/pii/S0925753524000225
- **Road traffic accident severity prediction using causal inference and ML** — ACM CODS-COMAD, 2025. ATE/ITE estimation on accident records; Rung 2. https://dl.acm.org/doi/10.1145/3703323.3703749
- *Context:* **Oberst & Sontag, ICML 2019** (arXiv:1905.05824) — Gumbel-max SCM twin-network counterfactuals for sepsis management; the canonical healthcare Rung-3 reference. Mining/healthcare incident literature otherwise has no strong 2020+ Pearl-style work.
  - *Relevance of this block:* lets the related-work section draw the rung taxonomy crisply and place ACE-Graph as the first Rung-3 system over *text-extracted, corpus-scale aviation* causal graphs. Be careful and generous: several of these self-describe as counterfactual; critique the usage precisely, citing Pearl's ladder.

---

## 4. LLMs and causal reasoning — capability evidence, benchmarks, critiques

- **Causal Reasoning and Large Language Models: Opening a New Frontier for Causality** — Kıcıman, E. et al., arXiv:2305.00050 (TMLR 2024). GPT-3.5/4: 97% on pairwise causal-direction discovery (+13 pts over covariance methods), 92% on counterfactual reasoning benchmark (+20 pts), 86% on necessary/sufficient actual-causality vignettes. Cautions: unpredictable failure modes; LLMs use variable-name metadata, not data; names LLM+statistical-causal-method hybrids as the open direction. https://arxiv.org/abs/2305.00050
  - *Relevance:* the standard capability + caveat citation; its "combine LLMs with formal causal inference" future-work line is precisely ACE-Graph's design.
- **ReCITE: benchmark for full causal-graph construction from realistic text** — arXiv:2505.18931 (2025/26). 292 academic papers with ground-truth causal loop diagrams (graphs 5–140 nodes, 6–205 edges). Best SOTA LLM (Claude Opus 4.5) averages F1 0.535; F1 drops ~50% from most- to least-explicit causality; giving ground-truth node names yields only marginal gains (edge F1 +0.04) — the bottleneck is causal reasoning, not entity recognition; most hallucinated edges are textually plausible but wrong, while direction reversals are rare (<1.1%). https://arxiv.org/pdf/2505.18931
  - *Relevance:* **the sharpest critique to engage with.** It implies schema-constrained extraction alone won't solve edge inference — our answer: per-narrative extraction supervised by NTSB Findings + cross-accident statistical aggregation, rather than one-shot whole-graph generation. Also borrow its hallucination taxonomy for our faithfulness eval.
- **Spurious position heuristic in causal/temporal fine-tuning** — arXiv:2406.12158. LLMs fine-tuned on temporal-relation data infer "X causes Y" from mention order: 92.59% predicted causal when mention position matched training vs 1.85% when it did not. https://arxiv.org/pdf/2406.12158
  - *Relevance:* direct threat to narrative-order-based edge extraction — accident narratives are roughly chronological. Motivates an explicit robustness check (e.g., permuted-narrative probe) in our evaluation.
- 📁 **CLadder: Assessing Causal Reasoning in Language Models** (cladder.pdf, in this folder) — formal Pearl-ladder QA benchmark (associational/interventional/counterfactual); LLMs struggle without formal machinery.
  - *Relevance:* motivates outsourcing Rung-2/3 computation to an explicit causal model rather than prompting the LLM for counterfactual answers.
- 📁 **Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference** (PDF in this folder) — argues eliciting causal *order* from imperfect experts (incl. LLMs) is more robust than eliciting full graphs.
  - *Relevance:* supports our design where the LLM supplies local event ordering/typed edges and the global graph comes from aggregation.
- 📁 **Leveraging LLM-Generated Structural Priors for Causal Inference with Concurrent Causes** (PDF in this folder) — LLM output as structural prior for downstream statistical causal inference.
  - *Relevance:* same hybrid philosophy as ACE-Graph Stage 3; cite as concurrent methodological support.

---

## 5. Benchmarks, datasets, and QA resources on these corpora

- **Georgia Tech/FAA extractive QA dataset** — AIAA 2025-3248. SQuAD-style dataset: 351,662 QA pairs over 13,658 passages from NTSB + ASRS narratives (80,749 NTSB and 218,285 ASRS reports considered), organized by the FAA Integrated Safety Assessment Model's 37 Event Sequence Diagram categories; on Hugging Face. Notably: even with strict schema-constrained verbatim-span prompting, Llama-3.3-70B frequently produced non-verbatim and hallucinated outputs (including self-invented questions), requiring algorithmic re-matching/discarding. https://arc.aiaa.org/doi/10.2514/6.2025-3248
  - *Relevance:* (a) candidate external evaluation/pretraining resource on our exact corpora; (b) independent, in-domain evidence that LLM faithfulness under schema constraints is unsolved — supports our guided-decoding + supervision design; (c) the ISAM/ESD 37-category structure is an alternative aggregation ontology to compare with our 42-type vocabulary.
- **ATC-QA** — Shi, H. & Yu, Z., ICASSE 2025 (Springer). 47,151 QA pairs from 43,264 ASRS reports, 7 formats, 4 difficulty levels; multi-LLM evaluation. https://link.springer.com/chapter/10.1007/978-981-95-6366-1_34
  - *Relevance:* ASRS-derived benchmark; background for the evaluation-resources paragraph.

---

## 6. Field reviews and gap statements (quotable motivation)

- ⚠ **Review of AI for accident analysis** — Advanced Engineering Informatics, 2026 (S1474034626000704). Documents the field's transition from ML/DL extraction and causal-factor classification toward LLM reasoning; reports LLM+KG and RAG-aided causal identification improve accident analysis on NTSB reports. Names as limitations/directions: (a) **over-reliance on NTSB data** — lack of heterogeneous multi-jurisdiction datasets impedes robustness assessment; (b) **causal explainability beyond feature attribution (SHAP)** as a promising direction. https://www.sciencedirect.com/science/article/pii/S1474034626000704
  - *Relevance:* a review in print asking for exactly (a) our multi-source corpus (NTSB + FAA AIDS + BEA + TSB Canada) and (b) our causal layer. Quote both gap statements in the introduction.

---

## 7. Positioning: gaps no found paper claims (as of 2026-07)

Use these as the novelty skeleton of the related-work section. Each was probed directly in the sweep; "no hits" means no paper found, not proof of absence — re-run these searches at submission time.

1. **Rung-3 counterfactual queries over causal graphs extracted from official aviation accident reports at corpus scale.** Closest: Zibaei 2024 (forum text, 8 UAV logs, binary HP); Ruiz-Tagle 2022 (expert BN, one event). The Drones 2025 paper *mentions* do-calculus without performing it; Safety Science 2026 KG is explicitly diagnostic-only. Nothing combines learned-from-text graphs + probabilistic counterfactuals + tens of thousands of accidents.
2. **Counterfactual/causal-model evaluation against investigator ground truth** — validating `do(X = absent)` answers against NTSB probable-cause / Findings.Cause_Factor annotations. No hits anywhere; would be a first-of-kind evaluation methodology contribution on its own.
3. **Multi-jurisdiction harmonized extraction** (NTSB + FAA AIDS + BEA + TSB Canada) with cross-jurisdiction transfer testing — BEA/TSB as narrative-only held-out sets. No hits; AEI 2026 review explicitly names the NTSB-monoculture problem.
4. **Structured-supervision-grounded extraction**: using NTSB Findings/seq_of_events/Cause_Factor and FAA AIDS cause codes as supervision/calibration for LLM event extraction (rather than small hand-annotated sets). Found papers either hand-annotate small corpora (Drones 2025: 200 reports) or go unsupervised (Chen 2024).
5. **Faithfulness/hallucination evaluation for causal-edge extraction in this domain.** ReCITE provides the general benchmark and taxonomy; Drones 2025 quantifies label-level error patterns; AIAA 2025-3248 documents span-level hallucination. Nobody has done edge-level faithfulness analysis for aviation causal graphs — we can, at 335k-edge scale.
6. **Closed event vocabulary as causal-node ontology** bridging extraction and aggregation (42 types → per-CICTT-category subgraphs). Prior KG work uses open vocabularies that cannot aggregate across accidents (Chen 2024: free entities; Zhao 2024: 1,432 codes but classification-only).

**One-sentence positioning:** prior work has (i) classified aviation narratives with domain LMs and HFACS-guided LLMs, (ii) built diagnostic KGs from reports with open-vocabulary LLM extraction, and (iii) run formal counterfactuals on small expert-built or log-derived causal models — ACE-Graph is, to our knowledge, the first to connect all three: schema-constrained, supervision-grounded extraction over a harmonized multi-agency corpus, aggregated into per-category causal graphs that support model-grounded Rung-3 counterfactual queries evaluated against investigator cause attributions.

---

## 8. To-do before citing

- [ ] Pull full texts of the ⚠ Elsevier items (ESWA HFACS-CoT, Safety Science 2026 KG, AEI 2026 review, ESWA RAG-causal, Ocean Eng 2026, RESS Ruiz-Tagle 2022) and re-verify every number above.
- [ ] Pin down full citation for AI4AirSafe (J. Automation & Intelligence 2026) and GA-ACAPS (HFACS on 2,250 NTSB GA reports) — both from sweep interim notes only.
- [ ] Fetch AIAA 2025-3248 and locate its Hugging Face dataset; assess as external eval set.
- [ ] Check ReCITE's arXiv version/venue status at submission time (2505.18931).
- [ ] Re-run the section-7 gap probes ("do-calculus aviation reports," "counterfactual evaluation probable cause," "multi-jurisdiction LLM accident harmonization") near submission to confirm the niches are still open.
