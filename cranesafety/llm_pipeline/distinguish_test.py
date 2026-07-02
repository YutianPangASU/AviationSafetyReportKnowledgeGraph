"""Can we distinguish the human labeler from the AI labelers?

We treat the four labelers on each (report, cohort) item as a panel:
    Human (gold), Qwen3.6, Opus4.8, Fable5   -> binary IN+MAYBE vs OUT.

Three complementary tests:

(1) Pairwise agreement / Cohen-κ matrix — descriptive: do the AIs cluster
    together (high AI–AI agreement) while the human sits apart (lower human–AI)?

(2) Consensus-outlier PERMUTATION test — the main test. For each item and each
    labeler, "outlier" = its label disagrees with the majority of the OTHER three
    labelers (3 others → no ties). Statistic
        T = (human outlier rate) − (mean AI outlier rate).
    Null: labeler identity is exchangeable. We permute, per item, which of the 4
    labels sits in the "human" slot vs the 3 "AI" slots, keeping the item's label
    multiset fixed, and recompute T. p = P(|T_perm| ≥ |T_obs|).

(3) Classifier two-sample test (C2ST) — robustness. Predict human(0) vs AI(1)
    from report features (TF-IDF + cohort + label) with 5-fold CV balanced
    accuracy; permutation null by shuffling the human/AI target.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

CORPUS = Path("../labeled_crane_corpus_sample_ai_labeled.jsonl")
OUT = Path("out")
RNG = np.random.default_rng(20260702)
N_PERM = 20000


def b(v):
    return 0 if v == "OUT" else 1  # 1 = IN (IN or MAYBE)


# ---------- assemble the 4-labeler panel per (report, cohort) ----------
def load_doc(paths):
    d = {}
    for p in paths:
        for line in Path(p).open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") is False:
                continue
            lab = str(r.get("label", "OUT")).upper()
            d[(r["record_id"], r["cohort"])] = lab if lab in {"IN", "OUT", "MAYBE"} else "OUT"
    return d


records = [json.loads(l) for l in CORPUS.open()]
gold, text = {}, {}
for r in records:
    per = defaultdict(list)
    for s in r["sentences"]:
        for lb in s["labels"]:
            if lb["author"] in ("Qwen3.6", "Opus4.8", "Fable5"):
                continue  # skip AI rows if this is the augmented file
            per[lb["label"]].append(lb["value"])
    for coh, vals in per.items():
        gold[(r["record_id"], coh)] = "IN" if "IN" in vals else ("MAYBE" if "MAYBE" in vals else "OUT")
        text[(r["record_id"], coh)] = " ".join(s["text"] for s in r["sentences"])

ai = {
    "Qwen3.6": load_doc([OUT / "doc_qwen.jsonl"]),
    "Opus4.8": load_doc(sorted(glob.glob(str(OUT / "doc_opus" / "*.jsonl")))),
    "Fable5": load_doc(sorted(glob.glob(str(OUT / "doc_fable" / "*.jsonl")))),
}
keys = [k for k in gold if all(k in a for a in ai.values())]
LABELERS = ["Human", "Qwen3.6", "Opus4.8", "Fable5"]
# M[item] = [human, qwen, opus, fable] as 0/1
M = np.array([[b(gold[k])] + [b(ai[n][k]) for n in ["Qwen3.6", "Opus4.8", "Fable5"]]
              for k in keys])
n = len(keys)
print(f"panel: {n} (report, cohort) items x {len(LABELERS)} labelers\n")

# ---------- (1) pairwise agreement + kappa ----------
print("(1) Pairwise agreement (upper) / Cohen κ (lower):")
print("           " + "".join(f"{x[:7]:>9}" for x in LABELERS))
for i, a in enumerate(LABELERS):
    row = []
    for j, c in enumerate(LABELERS):
        if i == j:
            row.append(f"{'—':>9}")
        elif j > i:
            row.append(f"{(M[:, i] == M[:, j]).mean():>9.3f}")
        else:
            row.append(f"{cohen_kappa_score(M[:, i], M[:, j]):>9.3f}")
    print(f"  {a:<8} " + "".join(row))
ai_pairs = [(1, 2), (1, 3), (2, 3)]
hu_pairs = [(0, 1), (0, 2), (0, 3)]
aa = np.mean([(M[:, i] == M[:, j]).mean() for i, j in ai_pairs])
ha = np.mean([(M[:, i] == M[:, j]).mean() for i, j in hu_pairs])
print(f"\n  mean AI–AI agreement   = {aa:.3f}")
print(f"  mean Human–AI agreement = {ha:.3f}   (gap = {aa - ha:+.3f})")


# ---------- (2) consensus-outlier permutation test ----------
def outlier_rates(mat):
    """For each labeler, fraction of items where it differs from the majority of
    the other three (3 others -> strict majority, no ties)."""
    rates = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        maj = (mat[:, others].sum(axis=1) >= 2).astype(int)
        rates.append((mat[:, i] != maj).mean())
    return np.array(rates)


obs = outlier_rates(M)
T_obs = obs[0] - obs[1:].mean()
print("\n(2) Consensus-outlier permutation test")
print("    outlier rate (disagrees with majority of other 3):")
for name, r in zip(LABELERS, obs):
    print(f"      {name:<8} {r:.3f}")
print(f"    T_obs = human − mean(AI) = {T_obs:+.4f}")

# permute labeler identity within each item
Tperm = np.empty(N_PERM)
for p in range(N_PERM):
    perm = M.copy()
    # independent per-row permutation of the 4 columns
    idx = np.argsort(RNG.random((n, 4)), axis=1)
    perm = np.take_along_axis(M, idx, axis=1)
    r = outlier_rates(perm)
    Tperm[p] = r[0] - r[1:].mean()
p_two = (np.abs(Tperm) >= abs(T_obs)).mean()
p_one = (Tperm >= T_obs).mean() if T_obs > 0 else (Tperm <= T_obs).mean()
print(f"    permutation null: mean={Tperm.mean():+.4f} sd={Tperm.std():.4f}")
print(f"    two-sided p = {p_two:.4f}   one-sided p = {p_one:.4f}   (N={N_PERM})")


# ---------- (3) "spot the human" — when the panel splits 3-vs-1, is the lone
#                dissenter the human more often than chance (25%)? ----------
from scipy.stats import binomtest

s = M.sum(axis=1)
split31 = np.isin(s, [1, 3])
minority_val = np.where(s[split31] == 1, 1, 0)   # the lone label
sub = M[split31]
minority_idx = np.array([np.where(row == mv)[0][0] for row, mv in zip(sub, minority_val)])
n31 = split31.sum()
human_is_odd = int((minority_idx == 0).sum())
bt = binomtest(human_is_odd, n31, 0.25, alternative="greater")
n22 = int((s == 2).sum())
unan = int(np.isin(s, [0, 4]).sum())
print("\n(3) 'Spot the human' — on 3-vs-1 split items, who is the lone dissenter?")
print(f"    unanimous items: {unan} | 2-2 splits: {n22} | 3-1 splits: {n31}")
print(f"    lone dissenter is the Human: {human_is_odd}/{n31} = {human_is_odd/n31:.3f} "
      f"(chance = 0.25)")
counts = {name: int((minority_idx == i).sum()) for i, name in enumerate(LABELERS)}
print(f"    dissenter identity counts: {counts}")
print(f"    binomial p (human > chance) = {bt.pvalue:.2e}")

# ---------- save ----------
(OUT / "distinguish_summary.json").write_text(json.dumps({
    "n_items": n,
    "mean_AI_AI_agreement": float(aa),
    "mean_Human_AI_agreement": float(ha),
    "outlier_rates": dict(zip(LABELERS, obs.tolist())),
    "T_obs": float(T_obs), "perm_p_two_sided": float(p_two),
    "split_3v1_items": int(n31), "human_is_lone_dissenter": human_is_odd,
    "human_dissenter_rate": human_is_odd / n31, "dissenter_counts": counts,
    "spot_human_binom_p": float(bt.pvalue),
}, indent=2))
print(f"\nwrote {OUT/'distinguish_summary.json'}")
