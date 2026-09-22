"""Arno #8: exact permutation p-value for the icing guard contrast (28/1222 true vs permuted)."""
import sys, json, random, numpy as np
sys.path.insert(0,".")
from physics.violation_rate import build_driver_index, chain_transitions, CHAINS, EPS
from physics.guards import Drivers, guard_for_edge
from math import comb, log, exp
drv=build_driver_index()
work=[]
with open(CHAINS) as f:
    for line in f:
        rec=json.loads(line)
        if not rec.get("ok"): continue
        d=drv.get(rec["record_id"].rsplit("_",1)[0])
        for src,dst,phase in chain_transitions(rec):
            g=guard_for_edge(src,dst)
            if g is None or g.name!="carb_icing": continue
            dd=Drivers(**vars(d)) if d else Drivers(); dd.phase=phase
            if g.checkable(dd): work.append((g,dd))
n=len(work); avail=np.array([g.availability(dd,None) for g,dd in work])
viol=int((avail<EPS).sum()); print("icing checkable",n,"true violations",viol, round(viol/n,4))
# Under permutation of driver tuples among icing-checkable items, the violation count is the
# number of items assigned a never-ices tuple = K (fixed) -> identical! So permute across the
# full driver pool instead: draw driver tuples from ALL weather-recorded events (the population).
pool=[d for d in drv.values() if d.temp_c is not None]
pa=np.array([guard_for_edge("__START__","CARBURETOR_OR_INDUCTION_ICING").availability(d,None) for d in pool])
p_never=float((pa<EPS).mean()); print("population never-ices fraction",round(p_never,4),"of",len(pool))
# exact binomial tail
from scipy.stats import binom
print("expected violations under random pairing",round(n*p_never,1)," P(X<=%d)="%viol, binom.cdf(viol,n,p_never))
# Monte Carlo permutation restricted to the 1685-item mixed work list is what the paper did (5.6%);
# replicate that arm's null with 20000 resamples from the population pool
rng=np.random.default_rng(0); sims=rng.binomial(n,p_never,20000); print("MC p:",(sims<=viol).mean())
