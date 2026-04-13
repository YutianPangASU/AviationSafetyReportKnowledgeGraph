# ACE-Graph Design Note — Counterfactual Framing Update

## Updated wording

The system should support counterfactual queries of the form:

> *If contributing factor X had been absent, would the accident still have been likely to occur under the learned causal model?*

This wording is preferable to a simpler “would the accident still occur?” formulation because it makes clear that the answer is produced under an explicit learned causal model rather than as an unconstrained narrative judgment.

## Why this change helps

- It makes the counterfactual claim more defensible in a paper or proposal.
- It clarifies that the system is not asserting ground-truth alternate history, but model-based counterfactual inference.
- It aligns better with Pearl’s ladder, where counterfactual answers depend on structural assumptions.
- It reduces the chance that reviewers interpret the task as informal speculation.

## Suggested insertion in the Motivation section

Build an end-to-end, reproducible pipeline that transforms unstructured aviation accident reports into a causal reasoning system. The system should support counterfactual queries of the form: *"If contributing factor X had been absent, would the accident still have been likely to occur under the learned causal model?"* This targets Rung 3 counterfactual reasoning in Pearl's ladder while making the modeling assumptions explicit.

## Optional tighter version for a paper

Build an end-to-end, reproducible pipeline that transforms aviation accident reports into a causal reasoning framework capable of answering model-grounded counterfactual queries about candidate contributing factors.

## Optional evaluation note

If this framing is kept, the evaluation section should define clearly:
1. what it means to remove a factor,
2. whether the query is asked on a case-level or category-level model,
3. how the counterfactual probability is computed, and
4. how the answer is compared against probable-cause or contributing-factor annotations.
