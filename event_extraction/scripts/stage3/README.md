# Stage 3 — Per-Category Causal DAG Pipeline

End-to-end pipeline that turns the Layer-1 aggregate KG into per-category
causal DAGs with bootstrap stability, ready for Stage-4 `do(X = absent)` queries.

Run order (each piece is a standalone script and the outputs of one feed the next):

```
build_precedence_matrices.py       (no LLM, ~30s on 28k records)
   ↓
llm_causal_order.py                (LLM, ~2-5 min per category at concurrency 4)
   ↓
laplacian_prior.py                 (no LLM, ~5s per category)
   ↓
run_pc_per_category.py             (no LLM, ~30s-2min per category)
   ↓
reorient_dag.py                    (no LLM, ~5s per category)
   ↓
bootstrap_stability.py             (no LLM, ~5-10 min per category × 30 bootstraps)
   ↓
viz_dag.py                         (no LLM, ~10s for all categories)
```

## Outputs

```
event_extraction/out/aggregate_kg/
├── per_category_precedence/       piece 1: presence + cooccur + precede matrices
├── per_category_llm_order/        piece 2: pairwise LLM judgments
├── per_category_laplacian/        piece 3: similarity, Laplacian embedding, prior
├── per_category_dag/              piece 4-5: PC output, reoriented + bootstrapped
├── per_category_dag_noprior/      piece 4 ablation: PC with no priors (baseline)
├── html_dag/                      piece 6: HTML viewer for the with-prior DAGs
└── html_dag_noprior/              piece 6: HTML viewer for the no-prior baselines
```

Each per-category DAG CSV (`per_category_dag/{cat}.csv`) carries:

| Column | Meaning |
|---|---|
| `src`, `dst` | event_type endpoints |
| `direction_score` | precedence asymmetry; > 0.5 favours src→dst |
| `support_count` | accidents in category contributing the triple |
| `llm_prior` | LLM confidence on undirected-CPDAG resolution (only when applicable) |
| `laplacian_prior` | similarity prior from Zhang-style Laplacian (∈ [0, 1]) |
| `bootstrap_stability` | fraction of 30 bootstraps preserving the edge |
| `pc_directed` | 1 if PC orientation produced this direction |
| `pc_oriented_by_llm` | 1 if originally CPDAG-undirected, oriented by LLM |
| `from_cpdag_undirected` | 1 if PC left it undirected |
| `reoriented` | 1 if the reorient pass flipped the direction |
| `llm_evidence` | LLM relation string at reorient time |

## How to rerun on the full data tomorrow

After full v3 extraction completes (~2026-04-28 ~16:00):

```bash
# 1. rebuild Layer-1 aggregate KG
python event_extraction/scripts/build_kg_layer1.py

# 2. rebuild precedence matrices (uses fresh extraction)
python event_extraction/scripts/stage3/build_precedence_matrices.py

# 3. LLM causal-order on every category with >=50 records
python event_extraction/scripts/stage3/llm_causal_order.py --concurrency 8

# 4. Laplacian prior
python event_extraction/scripts/stage3/laplacian_prior.py

# 5. PC + priors (for all categories with both LLM and Laplacian)
python event_extraction/scripts/stage3/run_pc_per_category.py

# 6. Reorient
python event_extraction/scripts/stage3/reorient_dag.py

# 7. Bootstrap (slow — only for top-N categories, or all if you have time)
python event_extraction/scripts/stage3/bootstrap_stability.py --bootstraps 50

# 8. Render
python event_extraction/scripts/stage3/viz_dag.py

# 9. Rebuild Layer-1 HTML if you want a fresh aggregate view
python event_extraction/scripts/viz_kg.py
```

Each script accepts `--categories <name1> <name2>` to limit to a subset.

## Key implementation choices

### `_common.py` — frozen 42-event vocabulary
The `EVENT_VOCAB` list is the canonical row/column ordering for every matrix
in the pipeline. It mirrors `event_extraction/prompts/schema_v3.json`. **Do
not reorder.**

### Precedence edge filter (piece 1)
A precedence edge `i → j` is counted only when an extracted edge of one of
{CAUSES, CONTRIBUTES_TO, TRIGGERS, ENABLES, DETECTS, RESPONDS_TO, PRECEDES}
runs from an event of type `i` to one of type `j`, deduped within each
narrative. Self-loops are dropped.

### LLM pairwise (piece 2)
Pairwise rather than full triplets — covers a 42-event-type space tractably
(~600-1000 calls per category at the cooccur ≥ 10 threshold). The `relation`
field is one of `{a_before_b, b_before_a, concurrent, unrelated}` with a
confidence in [0, 1]. Triplet-level cycle resolution is deferred — pairwise
already produced sensible orderings at LOC-I (191 a_before_b, 33 b_before_a,
7 concurrent, 29 unrelated out of 260 pairs).

### Laplacian prior (piece 3)
Builds a 42×42 similarity matrix combining LLM pairwise judgments (heavy
weight) with PMI on cooccurrence (light weight). Computes 8 bottom
eigenvectors of L = D − S as a low-dim event embedding, then turns squared
distances in that embedding into a prior matrix in [0, 1] via Gaussian
kernel. The prior is used as a hard threshold in piece 4 to drop candidate
edges between events that don't sit in a coherent sub-cluster.

### PC orientation (piece 4)
PC alone cannot reliably orient edges from binary co-occurrence data without
v-structures. The pipeline therefore runs a **reorient pass** (piece 4.5)
after PC that flips directed edges where:

- LLM order says the opposite direction AND `direction_score < 0.5`, OR
- no LLM evidence AND `direction_score < 0.3` (precedence is strongly opposite).

Skeleton is preserved; only direction is swapped. Reoriented edges carry
`reoriented=1` for transparency.

### Bootstrap stability (piece 5)
80% subsample × 30 bootstraps × per category. Edge `(u, v)` gets
`bootstrap_stability` = fraction of bootstraps that recovered the edge in
either direction. Skeleton stability (not direction stability) is what's
measured — direction confidence comes from the reorient pass.

### No-prior ablation (mandatory baseline per the design critique)
`run_pc_per_category.py --no-laplacian --no-llm-order --out-dir
per_category_dag_noprior` produces PC-only DAGs over the same data. The paper
delta between with-prior and no-prior is what isolates the LLM contribution.

## Sanity check — top stable LOC-I edges (after reorient)

| Edge | bootstrap | direction | support |
|---|---:|---:|---:|
| CONTROL_INPUT_IMPROPER → STALL | 1.00 | 0.91 | 1442 |
| CONTROL_INPUT_IMPROPER → LOSS_OF_CONTROL_INFLIGHT | 1.00 | 0.68 | 1272 |
| PILOT_INCAPACITATION → CONTROL_INPUT_IMPROPER | 1.00 | 0.74 | 55 |
| STALL → LOSS_OF_CONTROL_INFLIGHT | 1.00 | 0.84 | 583 |
| STALL → STRUCTURAL_OVERLOAD | 1.00 | 0.83 | 28 |
| CREW_COORDINATION_FAILURE → DECISION_INAPPROPRIATE | 0.97 | 0.83 | 11 |
| MAINTENANCE_INADEQUATE → ENGINE_FAILURE | 0.97 | 0.50 | 26 |

This is the textbook LOC-I causal pattern. The chain
`DECISION_INAPPROPRIATE → CONTROL_INPUT_IMPROPER → STALL → LOSS_OF_CONTROL_INFLIGHT → ground impact`
matches the case-9 walkthrough in the v3 schema redesign doc.

## Open issues / tomorrow's improvements

1. **Validation against published HFACS.** Hand-transcribe HFACS chains for 3-5
   high-profile accidents (Colgan Air 3407, AF447) and check the LOC-I/CFIT
   DAG covers them. Open-ended; ~3-5 days of work.

2. **Cross-model ablation on LLM order** (root-cause-2 mitigation from the v3
   redesign). Re-run `llm_causal_order.py` on a 50-pair sample with a
   different model family (OpenAI API), report Kendall's τ. Critical for the
   paper's circularity-mitigation story.

3. **Triplet-level cycle resolution.** The current pairwise approach can
   produce intransitive orderings. Vashishtha's full triplet method explicitly
   detects and resolves these. Worth adding once we see a cycle in practice.

4. **Per-category run scaling.** Right now `bootstrap_stability.py` rebuilds
   the presence matrix per category by streaming the entire extraction file.
   For all categories that's O(N_categories × N_records). Cache the matrix
   per category once and reuse across bootstraps for a >10× speedup if needed.
