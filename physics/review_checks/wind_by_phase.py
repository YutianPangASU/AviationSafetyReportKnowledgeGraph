"""Arno #6/#8: wind-guard violation rates by node phase (true arm), to test phase gating."""
import sys, json
sys.path.insert(0,".")
from collections import defaultdict
from physics.violation_rate import build_driver_index, chain_transitions, CHAINS, EPS
from physics.guards import Drivers, guard_for_edge
drv=build_driver_index()
per=defaultdict(lambda: defaultdict(lambda:[0,0]))
with open(CHAINS) as f:
    for line in f:
        rec=json.loads(line)
        if not rec.get("ok"): continue
        d=drv.get(rec["record_id"].rsplit("_",1)[0])
        for src,dst,phase in chain_transitions(rec):
            g=guard_for_edge(src,dst)
            if g is None or g.name not in ("gust_stall","shear_stall","gust_overload"): continue
            dd=Drivers(**vars(d)) if d else Drivers(); dd.phase=phase
            if not g.checkable(dd): continue
            lam=g.availability(dd,None); c=per[g.name][phase or "None"]; c[0]+=1; c[1]+= (lam<EPS)
for g,ph in per.items():
    print("==",g)
    for p,(n,v) in sorted(ph.items(), key=lambda kv:-kv[1][0]): print(f"   {p:18s} n={n:4d} viol={v:4d} rate={v/max(n,1):.2f}")
    low={"approach","landing","takeoff","initial_climb","go_around","climb","maneuvering","emergency_descent"}
    n=sum(v[0] for p,v in ph.items() if p in low); vv=sum(v[1] for p,v in ph.items() if p in low)
    print(f"   LOW-ALTITUDE PHASES: n={n} viol={vv} rate={vv/max(n,1):.2f}")
