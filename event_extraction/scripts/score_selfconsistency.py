"""Score the self-consistency runs of run_selfconsistency.sh (review 2026-09-14,
comment 5): agreement of the extracted chains across repeated narrative-only
extractions of the 2,000-report evaluation sample, and the spread of the
judge-scored recall.

Per record and per pair of runs:
  * Jaccard index of the sets of factor types in the two chains;
  * agreement of the factor type of the first primary-cause node;
  * absolute difference in chain length.
Aggregated over records, together with the recall and precision of every run
from the judge summaries. The production run (temperature 0.2) is compared
with itself across three repetitions and with one run at temperature 0.7.

Output: event_extraction/out/selfconsistency/selfconsistency_summary.json
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "event_extraction/out/selfconsistency"
BASE = ROOT / "event_extraction/out/calibration_2k_v4_narrative.jsonl"
RUNS = {"base": BASE, "run1": D / "calibration_2k_v4_narrative_run1.jsonl",
        "run2": D / "calibration_2k_v4_narrative_run2.jsonl",
        "run3": D / "calibration_2k_v4_narrative_run3.jsonl",
        "t07": D / "calibration_2k_v4_narrative_t07.jsonl"}


def load(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("chain"):
                out[r["record_id"]] = r["chain"]
    return out


def primary(chain) -> str | None:
    for n in chain:
        if n.get("cause_role") == "primary":
            return n.get("factor_type")
    return None


def pair_stats(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    jac, prim, dlen = [], [], []
    for rid in ids:
        sa = {n["factor_type"] for n in a[rid]}
        sb = {n["factor_type"] for n in b[rid]}
        jac.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
        pa, pb = primary(a[rid]), primary(b[rid])
        prim.append(float(pa == pb) if pa and pb else np.nan)
        dlen.append(abs(len(a[rid]) - len(b[rid])))
    return {"n_records": len(ids), "jaccard_mean": round(float(np.mean(jac)), 4),
            "jaccard_median": round(float(np.median(jac)), 4),
            "jaccard_at_least_0.5": round(float(np.mean(np.array(jac) >= 0.5)), 4),
            "primary_cause_agreement": round(float(np.nanmean(prim)), 4),
            "mean_abs_length_diff": round(float(np.mean(dlen)), 3)}


def main() -> None:
    chains = {k: load(p) for k, p in RUNS.items() if p.exists()}
    out = {"runs": {}, "pairs": {}}
    for k, c in chains.items():
        summ = D / f"semantic_eval_{k}.summary.json"
        if k == "base":
            summ = ROOT / "event_extraction/out/semantic_eval_v4_narrative.summary.json"
        s = json.load(open(summ)) if summ.exists() else {}
        out["runs"][k] = {"n_ok": len(c),
                          "median_chain_length": float(np.median([len(v) for v in c.values()])),
                          "recall": round(s.get("recall", float("nan")), 4),
                          "precision": round(s.get("precision", float("nan")), 4)}
    for a, b in combinations([k for k in chains if k != "t07"], 2):
        out["pairs"][f"{a}-{b}"] = pair_stats(chains[a], chains[b])
    if "t07" in chains:
        for a in ("base", "run1"):
            if a in chains:
                out["pairs"][f"{a}-t07"] = pair_stats(chains[a], chains["t07"])
    prod = [v for k, v in out["pairs"].items() if "t07" not in k]
    out["summary"] = {
        "production_temperature_pairs": len(prod),
        "jaccard_mean_over_pairs": round(float(np.mean([p["jaccard_mean"] for p in prod])), 4),
        "primary_agreement_over_pairs": round(float(np.mean([p["primary_cause_agreement"] for p in prod])), 4),
        "recall_range_production": [round(min(out["runs"][k]["recall"] for k in out["runs"] if k != "t07"), 4),
                                    round(max(out["runs"][k]["recall"] for k in out["runs"] if k != "t07"), 4)],
        "recall_t07": out["runs"].get("t07", {}).get("recall"),
    }
    with open(D / "selfconsistency_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
