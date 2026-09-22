"""Physical-violation rate of the extracted chains (paper Sec. 3.4 / Sec. 6).

Runs the admissibility monitor over every transition of every extracted
corpus chain: a transition is checkable when it carries a guard and the
guard's drivers are recorded for that case (weather from the NTSB events
table, phase from the chain node); a checkable transition is a violation
when its availability Lambda_e is zero -- the chain, as reconstructed, needs
a step the physics rules out at the recorded state.

Reports the two corpus statistics the formalism defines:
  * guard coverage  -- fraction of transitions (support-weighted) that are
    checkable under the guards and the recorded drivers, broken down into
    guarded-and-checkable / guarded-but-drivers-missing / unguarded;
  * violation rate  -- violations / checkable, overall and per guard.

Corrupted-guard control: the same monitor re-run with sign-inverted margins
(--arm negate), with driver tuples permuted across the guarded work list
(--arm permute, fixed seed), and with driver tuples drawn from the whole
population of driver-recorded NTSB events (--arm population, fixed seed).
The population arm is the null the manuscript reports (review 2026-09-14,
comment 8): it asks how often a chain would be rejected if it were paired
with the weather of an arbitrary accident rather than its own, and a
binomial tail against that rate gives the p-value of the true-arm count. The
true margins should produce a low violation rate; the corrupted arms should
not. By default all four arms run in one pass.

Usage (qwen-vllm env has jsbsim for the phase-prior stall guard):
  ~/miniconda3/envs/qwen-vllm/bin/python physics/violation_rate.py
Output: physics/out/violation_rate.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.guards import Drivers, guard_for_edge  # noqa: E402

CHAINS = "event_extraction/out/full_corpus_v4.jsonl"
DBS = ["data/NTSB_ASRS/avall.mdb", "data/NTSB_ASRS/Pre2008.mdb"]
OUT = "physics/out/violation_rate.json"
ARMS = ("true", "negate", "permute", "population")
PERMUTE_SEED = 20260810
# Availability below EPS counts as zero: for the exceedance-tail guards this
# means the recorded state puts the required gust beyond a one-in-a-billion
# encounter, the numerical reading of "unavailable as reconstructed".
EPS = 1e-9


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def build_driver_index() -> dict[str, Drivers]:
    """ev_id -> recorded drivers from the NTSB events table.

    Same hygiene as validate_carb_icing: (0,0) temperature/dewpoint pairs are
    missing, dewpoint above temperature is implausible. Wind (0 kt, 0 deg) is
    likewise treated as unrecorded rather than calm, so a calm-wind violation
    can never be an artifact of a blank field.
    """
    idx: dict[str, Drivers] = {}
    for db in DBS:
        if not os.path.exists(db):
            continue
        p = subprocess.Popen(["mdb-export", db, "events"],
                             stdout=subprocess.PIPE, text=True,
                             stderr=subprocess.DEVNULL)
        r = csv.reader(p.stdout)
        hdr = next(r)
        ix = {c: i for i, c in enumerate(hdr)}

        def fget(row, col):
            v = row[ix[col]].strip() if col in ix else ""
            try:
                return float(v) if v else None
            except ValueError:
                return None

        for row in r:
            d = Drivers()
            t, dp = fget(row, "wx_temp"), fget(row, "wx_dew_pt")
            if (t is not None and dp is not None and not (t == 0 and dp == 0)
                    and -60 < t < 130 and -60 < dp < 130 and dp <= t + 2):
                d.temp_c, d.dew_c = f_to_c(t), f_to_c(min(dp, t))
            w, wd = fget(row, "wind_vel_kts"), fget(row, "wind_dir_deg")
            if w is not None and 0 <= w < 200 and not (w == 0 and not wd):
                d.wind_kts = w
            da = fget(row, "wx_dens_alt")
            if da is not None and -2000 < da < 30000:
                d.dens_alt_ft = da
            if (d.temp_c, d.wind_kts, d.dens_alt_ft) != (None, None, None):
                idx[row[ix["ev_id"]]] = d
        p.wait()
    return idx


def chain_transitions(rec: dict):
    """(src_factor, dst_factor, dst_phase) for every transition of the chain.

    Every caused_by link is a transition; a root node's onset is the
    transition entering the automaton from the quiescent mode ("__START__"),
    which is where guarded factors like carburetor icing usually sit.
    """
    nodes = {n["idx"]: n for n in rec.get("chain", [])}
    for n in rec.get("chain", []):
        links = n.get("caused_by", []) or []
        if not links:
            yield ("__START__", n.get("factor_type", ""),
                   n.get("phase_of_flight"))
        for e in links:
            src = nodes.get(e.get("src"))
            if src is not None:
                yield (src.get("factor_type", ""), n.get("factor_type", ""),
                       n.get("phase_of_flight"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N records only (smoke test)")
    args = ap.parse_args()

    drivers_by_ev = build_driver_index()
    print(f"driver index: {len(drivers_by_ev)} NTSB events", flush=True)

    # Collect the checkable work list first so the permute arm can shuffle
    # driver assignments across exactly the same transitions.
    work = []  # (guard, drivers, phase, src, dst)
    n_transitions = 0
    n_unguarded = 0
    n_guarded_unrecorded = 0
    with open(CHAINS) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            ev_id = rec["record_id"].rsplit("_", 1)[0]
            d = drivers_by_ev.get(ev_id)
            for src, dst, phase in chain_transitions(rec):
                n_transitions += 1
                g = guard_for_edge(src, dst)
                if g is None:
                    n_unguarded += 1
                    continue
                dd = Drivers(**vars(d)) if d is not None else Drivers()
                dd.phase = phase
                if not g.checkable(dd):
                    n_guarded_unrecorded += 1
                    continue
                work.append((g, dd, src, dst))

    n_checkable = len(work)
    out = {
        "n_records_scanned": args.limit or "all",
        "availability_zero_eps": EPS,
        "n_transitions": n_transitions,
        "guard_coverage": {
            "checkable": n_checkable,
            "checkable_frac": round(n_checkable / max(n_transitions, 1), 4),
            "guarded_drivers_unrecorded": n_guarded_unrecorded,
            "unguarded": n_unguarded,
        },
        "arms": {},
    }

    # Permutation: shuffle which case's drivers each checkable transition
    # sees, holding the transition structure fixed.
    perm = list(range(n_checkable))
    random.Random(PERMUTE_SEED).shuffle(perm)
    # Population null: each checkable transition is paired with the drivers
    # of an arbitrary driver-recorded event, drawn with replacement.
    pop_rng = random.Random(PERMUTE_SEED + 1)
    pool = list(drivers_by_ev.values())

    for arm in args.arms:
        per_guard = defaultdict(lambda: {"checkable": 0, "violations": 0})
        for j, (g, dd, src, dst) in enumerate(work):
            if arm == "permute":
                other = work[perm[j]][1]
                dd = Drivers(temp_c=other.temp_c, dew_c=other.dew_c,
                             wind_kts=other.wind_kts,
                             dens_alt_ft=other.dens_alt_ft, phase=dd.phase)
                if not g.checkable(dd):
                    continue  # permuted case lacks this guard's drivers
            elif arm == "population":
                for _ in range(50):
                    other = pool[pop_rng.randrange(len(pool))]
                    cand = Drivers(temp_c=other.temp_c, dew_c=other.dew_c,
                                   wind_kts=other.wind_kts,
                                   dens_alt_ft=other.dens_alt_ft, phase=dd.phase)
                    if g.checkable(cand):
                        dd = cand
                        break
                else:
                    continue
            corrupt = "negate" if arm == "negate" else None
            lam = g.availability(dd, corrupt)
            per_guard[g.name]["checkable"] += 1
            if lam < EPS:
                per_guard[g.name]["violations"] += 1
        tot_c = sum(v["checkable"] for v in per_guard.values())
        tot_v = sum(v["violations"] for v in per_guard.values())
        out["arms"][arm] = {
            "checkable": tot_c,
            "violations": tot_v,
            "violation_rate": round(tot_v / max(tot_c, 1), 4),
            "per_guard": {
                k: {**v, "rate": round(v["violations"] / max(v["checkable"], 1), 4)}
                for k, v in sorted(per_guard.items())
            },
        }
        print(f"arm={arm}: {tot_v}/{tot_c} violations "
              f"({out['arms'][arm]['violation_rate']:.2%})", flush=True)

    # Binomial tail of the true-arm count against the population-null rate,
    # per guard: P(X <= v_true | n_true, rate_population).
    if "true" in out["arms"] and "population" in out["arms"]:
        try:
            from scipy.stats import binom
            pv = {}
            for gname, t in out["arms"]["true"]["per_guard"].items():
                pop = out["arms"]["population"]["per_guard"].get(gname)
                if not pop or pop["checkable"] == 0:
                    continue
                pv[gname] = float(binom.cdf(t["violations"], t["checkable"],
                                            pop["rate"]))
            out["p_value_true_vs_population"] = pv
        except ImportError:
            pass

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
