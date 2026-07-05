"""Causation KG builder for the v4 chain extraction (replaces build_kg_layer1.py).

Produces three decoupled artifacts under --out-dir (default
event_extraction/out/causation_kg/), per the v4 plan
(docs/2026-07-05-v4-chain-schema-plan.md, Phase 4):

1. per_accident_chains.jsonl — provenance layer. One row per record:
   {record_id, source, category, outcome_severity, chain}. outcome_severity is
   resolved structured-first (NTSB injury/damage beats the model's guess).

2. factor_vectors.parquet — the causal-discovery / risk-calculation input.
   One row per record: binary presence of each of the 57 factor_types +
   outcome_severity (+ ordinal), category, source.

3. The aggregate causation KG:
   - causation_edges.csv       causal-factor -> causal-factor edges. One count
                               per record carrying >= 1 caused_by link between
                               the two factor types (deduped within record).
                               Attributes: support, p_dst_given_src, lift,
                               direct_share, sample_records.
   - outcome_layer.csv         causal factor -> outcome coupling: per-factor
                               P(severity >= serious | factor) vs corpus
                               baseline (risk ratio), plus factor -> outcome
                               node edge counts. Kept OUT of the main graph so
                               outcome hubs cannot clutter it.
   - causation_kg.graphml      filtered view (>= --min-support), causal
                               factors only, for Cytoscape/Gephi.
   - per_category/*.csv        causation_edges format, one file per category.
   - causation_kg.summary.md / .json

Usage:
    python event_extraction/scripts/build_kg_v4.py \\
        --extraction event_extraction/out/full_corpus_v4.jsonl \\
        --enriched   data/corpus/corpus_enriched.jsonl \\
        --out-dir    event_extraction/out/causation_kg
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent / "stage3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import categorize  # noqa: E402
from validate_extraction import load_def_enums  # noqa: E402

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

OUTCOME_FACTORS = {
    "GROUND_IMPACT", "WATER_IMPACT", "MIDAIR_COLLISION", "GROUND_COLLISION",
    "RUNWAY_EXCURSION_OR_OVERRUN", "INFLIGHT_BREAKUP", "EMERGENCY_LANDING",
    "SUCCESSFUL_RECOVERY", "INJURY_OR_FATALITY",
}

SEVERITY_ORDER = ["no_damage_or_injury", "aircraft_damage_only",
                  "minor_injury", "serious_injury", "fatal"]
SEVERITY_ORDINAL = {s: i for i, s in enumerate(SEVERITY_ORDER)}
SERIOUS_THRESHOLD = SEVERITY_ORDINAL["serious_injury"]


def factor_vocab() -> list[str]:
    enums = load_def_enums(PROMPTS_DIR / "schema_v4.json")
    vocab = [v for v in enums["ChainNode"]["factor_type"] if v != "unknown"]
    # enum order in JSON Schema is meaningful; load_def_enums returns a set,
    # so re-read the raw schema to preserve ordering.
    raw = json.loads((PROMPTS_DIR / "schema_v4.json").read_text())
    ordered = [v for v in raw["$defs"]["ChainNode"]["properties"]["factor_type"]["enum"]
               if v != "unknown"]
    assert set(ordered) == set(vocab)
    return ordered


def resolve_outcome_severity(model_value: str | None, structured: dict | None) -> str:
    """Structured injury/damage data is primary; the model's record-level
    guess is the fallback (narrative-only sources)."""
    if structured:
        inj = structured.get("injury") or {}
        if (inj.get("fatal") or 0) > 0:
            return "fatal"
        if (inj.get("serious") or 0) > 0:
            return "serious_injury"
        if (inj.get("minor") or 0) > 0:
            return "minor_injury"
        dmg = ((structured.get("aircraft") or {}).get("damage") or "").strip().upper()
        if dmg in ("DEST", "SUBS", "MINR", "MIN"):
            return "aircraft_damage_only"
        if inj:  # injury table present, all zero, no damage code
            return "no_damage_or_injury"
    if model_value in SEVERITY_ORDINAL:
        return model_value
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", default="event_extraction/out/full_corpus_v4.jsonl", type=Path)
    ap.add_argument("--enriched", default="data/corpus/corpus_enriched.jsonl", type=Path)
    ap.add_argument("--out-dir", default="event_extraction/out/causation_kg", type=Path)
    ap.add_argument("--min-support", default=30, type=int,
                    help="minimum record support for an edge to enter the "
                    "GraphML/summary views (the CSV keeps everything)")
    ap.add_argument("--sample-records-per-edge", default=5, type=int)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_category").mkdir(exist_ok=True)

    vocab = factor_vocab()
    causal_vocab = [v for v in vocab if v not in OUTCOME_FACTORS]

    print(f"Indexing {args.enriched.name} (category + structured outcome)…")
    cat_by_id: dict[str, str] = {}
    structured_by_id: dict[str, dict] = {}
    with args.enriched.open() as f:
        for line in f:
            r = json.loads(line)
            cat_by_id[r["record_id"]] = categorize(r)
            s = r.get("structured")
            if s:
                structured_by_id[r["record_id"]] = {
                    "injury": s.get("injury"), "aircraft": s.get("aircraft")}
    print(f"  {len(cat_by_id):,} records indexed")

    # accumulators
    n_records = 0
    factor_presence: Counter = Counter()                 # factor -> n records
    edge_support: Counter = Counter()                    # (src,dst) -> n records
    edge_direct: Counter = Counter()                     # (src,dst) -> n records w/ direct link
    edge_samples: dict[tuple, list[str]] = defaultdict(list)
    per_cat_edges: dict[str, Counter] = defaultdict(Counter)
    per_cat_records: Counter = Counter()
    outcome_edge_support: Counter = Counter()            # (causal_factor, outcome_factor)
    sev_by_factor: dict[str, Counter] = defaultdict(Counter)  # factor -> Counter(sev)
    sev_overall: Counter = Counter()
    fv_rows: list[dict] = []

    print(f"Streaming {args.extraction.name}…")
    chains_out = (args.out_dir / "per_accident_chains.jsonl").open("w")
    with args.extraction.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok") or not r.get("chain"):
                continue
            rid = r["record_id"]
            chain = r["chain"]
            cat = cat_by_id.get(rid, "UNCATEGORIZED:?")
            sev = resolve_outcome_severity(r.get("outcome_severity"),
                                           structured_by_id.get(rid))
            n_records += 1
            per_cat_records[cat] += 1
            sev_overall[sev] += 1

            chains_out.write(json.dumps({
                "record_id": rid, "source": r.get("source"), "category": cat,
                "outcome_severity": sev, "chain": chain},
                ensure_ascii=False) + "\n")

            types_present = {n.get("factor_type") for n in chain
                             if n.get("factor_type") in set(vocab)}
            for t in types_present:
                factor_presence[t] += 1
                sev_by_factor[t][sev] += 1

            fv = {"record_id": rid, "source": r.get("source"), "category": cat,
                  "outcome_severity": sev,
                  "sev_ordinal": SEVERITY_ORDINAL.get(sev, -1)}
            for t in vocab:
                fv[t] = 1 if t in types_present else 0
            fv_rows.append(fv)

            # type-level causal links, deduped within record
            seen_pairs: set[tuple[str, str]] = set()
            by_idx = {n.get("idx"): n for n in chain if isinstance(n, dict)}
            for n in chain:
                dst_t = n.get("factor_type")
                if dst_t not in set(vocab):
                    continue
                for link in (n.get("caused_by") or []):
                    src_n = by_idx.get(link.get("src"))
                    if not src_n:
                        continue
                    src_t = src_n.get("factor_type")
                    if src_t not in set(vocab) or src_t == dst_t:
                        continue
                    if src_t in OUTCOME_FACTORS:
                        continue  # outcome may only feed the outcome layer
                    pair = (src_t, dst_t)
                    if dst_t in OUTCOME_FACTORS:
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            outcome_edge_support[pair] += 1
                        continue
                    first_time = pair not in seen_pairs
                    if first_time:
                        seen_pairs.add(pair)
                        edge_support[pair] += 1
                        per_cat_edges[cat][pair] += 1
                        if len(edge_samples[pair]) < args.sample_records_per_edge:
                            edge_samples[pair].append(rid)
                    if link.get("strength") == "direct" and first_time:
                        edge_direct[pair] += 1
    chains_out.close()
    print(f"  {n_records:,} records -> {len(edge_support):,} distinct causal edges")

    # ---- factor_vectors.parquet -------------------------------------------
    import pandas as pd
    df = pd.DataFrame(fv_rows)
    pq_path = args.out_dir / "factor_vectors.parquet"
    try:
        df.to_parquet(pq_path, index=False)
        print(f"  factor vectors -> {pq_path} ({len(df):,} rows)")
    except Exception as e:
        csv_path = pq_path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False)
        print(f"  parquet unavailable ({e}); wrote {csv_path}")

    # ---- causation_edges.csv ----------------------------------------------
    def lift(pair, sup):
        src, dst = pair
        p_dst_given_src = sup / factor_presence[src] if factor_presence[src] else 0
        p_dst = factor_presence[dst] / n_records if n_records else 0
        return p_dst_given_src, (p_dst_given_src / p_dst if p_dst else 0)

    edges_csv = args.out_dir / "causation_edges.csv"
    with edges_csv.open("w") as f:
        f.write("src,dst,support,p_dst_given_src,lift,direct_share,sample_records\n")
        for pair, sup in sorted(edge_support.items(), key=lambda kv: -kv[1]):
            p, lf = lift(pair, sup)
            ds = edge_direct[pair] / sup if sup else 0
            f.write(f"{pair[0]},{pair[1]},{sup},{p:.4f},{lf:.3f},{ds:.3f},"
                    f"\"{';'.join(edge_samples[pair])}\"\n")
    print(f"  causal edges -> {edges_csv}")

    # ---- outcome_layer.csv -------------------------------------------------
    n_serious_overall = sum(c for s, c in sev_overall.items()
                            if SEVERITY_ORDINAL.get(s, -1) >= SERIOUS_THRESHOLD)
    baseline = n_serious_overall / n_records if n_records else 0
    out_csv = args.out_dir / "outcome_layer.csv"
    with out_csv.open("w") as f:
        f.write("factor,n_records,p_serious_or_fatal_given_factor,baseline,risk_ratio,"
                "top_outcome_edges\n")
        for t in causal_vocab:
            n_f = factor_presence[t]
            if not n_f:
                continue
            n_serious = sum(c for s, c in sev_by_factor[t].items()
                            if SEVERITY_ORDINAL.get(s, -1) >= SERIOUS_THRESHOLD)
            p = n_serious / n_f
            rr = p / baseline if baseline else 0
            tops = sorted(((o, c) for (s, o), c in outcome_edge_support.items()
                           if s == t), key=lambda kv: -kv[1])[:3]
            tops_s = ";".join(f"{o}:{c}" for o, c in tops)
            f.write(f"{t},{n_f},{p:.4f},{baseline:.4f},{rr:.3f},\"{tops_s}\"\n")
    print(f"  outcome layer -> {out_csv}")

    # ---- GraphML (filtered causal view) ------------------------------------
    G = nx.DiGraph()
    for t in causal_vocab:
        if factor_presence[t]:
            G.add_node(t, count=factor_presence[t])
    for pair, sup in edge_support.items():
        if sup < args.min_support:
            continue
        p, lf = lift(pair, sup)
        G.add_edge(*pair, support=sup, p_dst_given_src=round(p, 4),
                   lift=round(lf, 3),
                   direct_share=round(edge_direct[pair] / sup, 3))
    nx.write_graphml(G, str(args.out_dir / "causation_kg.graphml"))
    print(f"  graphml ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"at support >= {args.min_support}) -> causation_kg.graphml")

    # ---- per-category edges -------------------------------------------------
    for cat, ctr in per_cat_edges.items():
        safe = cat.replace(":", "_").replace("/", "_").replace("\\", "_")
        with (args.out_dir / "per_category" / f"{safe}.csv").open("w") as f:
            f.write("src,dst,support\n")
            for (s, d), c in sorted(ctr.items(), key=lambda kv: -kv[1]):
                f.write(f"{s},{d},{c}\n")

    # ---- summary -------------------------------------------------------------
    summary = {
        "n_records": n_records,
        "n_causal_factors_present": sum(1 for t in causal_vocab if factor_presence[t]),
        "n_causal_edges_distinct": len(edge_support),
        "n_causal_edges_min_support": G.number_of_edges(),
        "severity_distribution": dict(sev_overall.most_common()),
        "baseline_p_serious_or_fatal": round(baseline, 4),
        "records_per_category": dict(per_cat_records.most_common(40)),
        "top_factors": dict(factor_presence.most_common(20)),
        "top_edges": [
            {"src": s, "dst": d, "support": c}
            for (s, d), c in sorted(edge_support.items(), key=lambda kv: -kv[1])[:25]],
    }
    (args.out_dir / "causation_kg.summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Causation KG (v4)\n",
             f"Records: **{n_records:,}** | causal factors: "
             f"{summary['n_causal_factors_present']} | distinct causal edges: "
             f"{len(edge_support):,} (>= {args.min_support} support: "
             f"{G.number_of_edges()})\n",
             f"Baseline P(serious/fatal) = {baseline:.3f}\n",
             "## Top 25 causal edges\n",
             "| src | dst | support | P(dst|src) | lift | direct share |",
             "|---|---|---:|---:|---:|---:|"]
    for (s, d), c in sorted(edge_support.items(), key=lambda kv: -kv[1])[:25]:
        p, lf = lift((s, d), c)
        lines.append(f"| `{s}` | `{d}` | {c:,} | {p:.3f} | {lf:.2f} | "
                     f"{edge_direct[(s, d)] / c:.2f} |")
    lines.append("\n## Top factors by record support\n")
    lines.append("| factor | records | P(serious/fatal | factor) | risk ratio |")
    lines.append("|---|---:|---:|---:|")
    for t, c in factor_presence.most_common(20):
        if t in OUTCOME_FACTORS:
            continue
        ns = sum(cc for s2, cc in sev_by_factor[t].items()
                 if SEVERITY_ORDINAL.get(s2, -1) >= SERIOUS_THRESHOLD)
        p = ns / c if c else 0
        lines.append(f"| `{t}` | {c:,} | {p:.3f} | {p / baseline if baseline else 0:.2f} |")
    (args.out_dir / "causation_kg.summary.md").write_text("\n".join(lines))
    print(f"  summary -> causation_kg.summary.md / .json\nDone.")


if __name__ == "__main__":
    main()
