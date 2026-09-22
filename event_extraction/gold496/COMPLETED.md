# Gold-496 Extraction Evaluation - Completed

Extraction finished: 2026-08-09T20:37 (12 cron windows) plus one final
retry window 2026-08-10 for the tasks still under the 3-attempt cap.
Final totals: {'done': 446, 'invalid': 24, 'pending': 1}

- opus: 152/157 valid (invalid: type-level self-loops)
- sonnet: 156/157 valid
- haiku: 138/157 valid (self-loops dominate; one out-of-vocabulary type)

Scored 2026-08-10 with the local Qwen3.6-35B judge (cross-family), thinking
disabled, gold decompositions cached in judge_cache/. Results in scores.json:

| model  | judged | cause recall | node support | severity agree | chain len (med) |
|--------|--------|--------------|--------------|----------------|-----------------|
| opus   | 149    | 0.944        | 0.938        | 0.954          | 13              |
| sonnet | 149    | 0.898        | 0.935        | 0.954          | 10              |
| haiku  | 148    | 0.787        | 0.931        | 0.953          | 8               |

Reading: capability moves recall while node support and severity agreement
stay flat near 93-95%, so weaker tiers omit causal factors rather than
invent them. Written into the manuscript (Section 6, Fidelity of the
Extracted Chains).
