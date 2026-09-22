"""FCI alongside PC (review 2026-09-14, comment 3).

The per-category networks were learned with PC, which assumes causal
sufficiency. The corpus is selected on the accident, so that assumption
fails by construction. This script runs FCI, which allows latent
confounders and selection, on the same presence data with the same
chi-square tests and significance level, and asks what happens to the
adjustment sets the backdoor table used:

  * for every factor X whose PC adjustment set Z is non-empty, each z in Z is
    classified by the FCI edge between them: z --> X (definite parent),
    z o-> X (possible parent), z <-> X (latent confounding, not a parent),
    z o-o X (undetermined), X --> z (reversed), or absent;
  * the adjusted effect is recomputed on the definite parents alone and on
    the definite-or-possible parents, and a factor with a bidirected edge
    into it is marked as not identifiable by adjustment on measured factors;
  * an E-value is attached to every adjusted effect as the sensitivity
    analysis: the minimum strength of association an unmeasured confounder
    would need with both the factor and the outcome to explain the effect
    away (VanderWeele and Ding 2017), computed on the risk ratio
    P(Y | do(X=1)) / P(Y | do(X=0)).

Run with the environment that carries causal-learn:
  ~/miniconda3/envs/qwen-vllm/bin/python event_extraction/scripts/stage3/fci_check.py
Output: event_extraction/out/causation_kg/fci_check_{cat}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bn_do_smoke_test import OUTCOME_FACTORS, SEV_SERIOUS_ORDINAL, p_y_do_x  # noqa: E402


def e_value(rr: float) -> float:
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def classify(G: np.ndarray, i: int, j: int) -> str:
    """Edge between nodes i (=z) and j (=X) in the FCI PAG.

    causal-learn stores the endpoint mark at node a of edge {a, b} in
    G[a, b]: -1 tail, 1 arrowhead, 2 circle, 0 no edge.
    """
    mi, mj = G[i, j], G[j, i]
    if mi == 0 and mj == 0:
        return "absent"
    if mi == -1 and mj == 1:
        return "z->X"
    if mi == 2 and mj == 1:
        return "z o->X"
    if mi == 1 and mj == 1:
        return "z<->X"
    if mi == 2 and mj == 2:
        return "z o-o X"
    if mi == 1 and mj == -1:
        return "X->z"
    if mi == 1 and mj == 2:
        return "X o->z"
    return f"other({mi},{mj})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-vectors",
                    default="event_extraction/out/causation_kg/factor_vectors.parquet")
    ap.add_argument("--categories", nargs="+", default=["LOC-I", "SCF-PP"])
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--min-support", default=30, type=int)
    ap.add_argument("--depth", default=-1, type=int)
    args = ap.parse_args()

    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import chisq

    df = pd.read_parquet(args.factor_vectors)
    for cat in args.categories:
        bd = json.load(open(f"event_extraction/out/causation_kg/backdoor_table_{cat}.json"))
        sub = df[df["category"] == cat].copy()
        sub["Y"] = (sub["sev_ordinal"] >= SEV_SERIOUS_ORDINAL).astype(int)
        factor_cols = [c for c in df.columns if c.isupper() and c != "Y"
                       and c not in OUTCOME_FACTORS]
        cands = [c for c in factor_cols if int(sub[c].sum()) >= args.min_support]
        X = sub[cands].to_numpy(dtype=int)
        print(f"[{cat}] {len(sub)} records, {len(cands)} candidate factors; running FCI "
              f"(alpha {args.alpha}, chisq) ...", flush=True)
        G, _ = fci(X, chisq, alpha=args.alpha, depth=args.depth, verbose=False,
                   show_progress=False)
        A = np.asarray(G.graph)
        idx = {c: k for k, c in enumerate(cands)}

        # PAG edge-type census
        census = {}
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                t = classify(A, i, j)
                if t == "absent":
                    continue
                key = {"z->X": "directed", "X->z": "directed", "z o->X": "partially directed",
                       "X o->z": "partially directed", "z<->X": "bidirected",
                       "z o-o X": "undetermined"}.get(t, "other")
                census[key] = census.get(key, 0) + 1

        rows = []
        for iv in bd["interventions"]:
            x = iv["factor"]
            Z = iv.get("adjustment_set") or []
            if not Z or x not in idx:
                continue
            marks = {z: classify(A, idx[z], idx[x]) if z in idx else "not in FCI set" for z in Z}
            definite = [z for z, m in marks.items() if m == "z->X"]
            possible = [z for z, m in marks.items() if m in ("z->X", "z o->X")]
            confounded = any(m == "z<->X" for m in marks.values())
            p0, p1 = p_y_do_x(sub, x, Z)
            d0, d1 = p_y_do_x(sub, x, definite)
            q0, q1 = p_y_do_x(sub, x, possible)
            rr = p1 / p0 if p0 > 0 else float("nan")
            rows.append({
                "factor": x, "pc_adjustment_set": Z, "fci_marks": marks,
                "n_definite": len(definite), "n_possible": len(possible),
                "bidirected_into_X": confounded,
                "crude": iv["crude_effect"], "adjusted_pc": iv["adjusted_effect"],
                "adjusted_definite_parents": round(d1 - d0, 4),
                "adjusted_possible_parents": round(q1 - q0, 4),
                "risk_ratio_pc": round(rr, 3),
                "e_value_pc": round(e_value(rr), 3) if rr == rr else None,
                "e_value_ci_bound": None,
            })
        # E-value for the CI limit closest to the null, on the risk-difference
        # interval mapped to a ratio at the do(X=0) level
        for r, iv in zip(rows, [i for i in bd["interventions"]
                                if i.get("adjustment_set") and i["factor"] in idx]):
            lo, hi = iv["ci95"]
            p0 = iv["p_serious_do_absent"]
            bound = lo if iv["adjusted_effect"] > 0 else hi
            rr_b = (p0 + bound) / p0 if p0 > 0 else float("nan")
            r["e_value_ci_bound"] = (round(e_value(rr_b), 3)
                                     if rr_b == rr_b and (rr_b - 1) * (iv["adjusted_effect"]) > 0 else 1.0)

        out = {"category": cat, "n_records": int(len(sub)), "n_candidates": len(cands),
               "alpha": args.alpha, "pag_edge_census": census,
               "n_factors_with_pc_adjustment": len(rows),
               "summary": {
                   "definite_parent_relations": sum(sum(1 for m in r["fci_marks"].values() if m == "z->X") for r in rows),
                   "possible_parent_relations": sum(sum(1 for m in r["fci_marks"].values() if m == "z o->X") for r in rows),
                   "bidirected_relations": sum(sum(1 for m in r["fci_marks"].values() if m == "z<->X") for r in rows),
                   "undetermined_relations": sum(sum(1 for m in r["fci_marks"].values() if m == "z o-o X") for r in rows),
                   "reversed_relations": sum(sum(1 for m in r["fci_marks"].values() if m in ("X->z", "X o->z")) for r in rows),
                   "absent_relations": sum(sum(1 for m in r["fci_marks"].values() if m == "absent") for r in rows),
                   "factors_bidirected_into_X": sum(1 for r in rows if r["bidirected_into_X"]),
                   "mean_abs_shift_pc_vs_definite": round(float(np.mean(
                       [abs(r["adjusted_pc"] - r["adjusted_definite_parents"]) for r in rows])), 4) if rows else None,
               },
               "factors": rows}
        with open(f"event_extraction/out/causation_kg/fci_check_{cat}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({k: v for k, v in out.items() if k != "factors"}, indent=2))
        for r in rows:
            print(f"  {r['factor']:34s} Z={r['pc_adjustment_set']} marks={r['fci_marks']} "
                  f"crude={r['crude']:.3f} pc={r['adjusted_pc']:.3f} "
                  f"def={r['adjusted_definite_parents']:.3f} poss={r['adjusted_possible_parents']:.3f} "
                  f"E={r['e_value_pc']} E_ci={r['e_value_ci_bound']}")


if __name__ == "__main__":
    main()
