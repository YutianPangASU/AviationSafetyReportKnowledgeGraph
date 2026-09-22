"""Arno #2: parent coverage of LOC / EF chains and a leaky noisy-OR fit."""
import json, csv, numpy as np, pandas as pd
from collections import Counter
from scipy.optimize import minimize
KG="event_extraction/out/causation_kg"
df=pd.read_parquet(KG+"/factor_vectors.parquet")
N=len(df); print("records",N)
LOC=["STALL","SPATIAL_DISORIENTATION","CONTROL_SURFACE_ANOMALY","AIRFRAME_STRUCTURAL_FAILURE",
 "PILOT_INCAPACITATION_OR_IMPAIRMENT","TURBULENCE_ENCOUNTER","CONTROL_INPUT_IMPROPER","ENGINE_FAILURE",
 "WIND_SHEAR_OR_GUST","DECISION_INAPPROPRIATE"]
EF=["FUEL_EXHAUSTION_OR_STARVATION","LATENT_MECHANICAL_DEFECT","PROCEDURE_NOT_FOLLOWED","OTHER_SYSTEM_FAILURE",
 "CARBURETOR_OR_INDUCTION_ICING","FUEL_SYSTEM_ANOMALY","FUEL_CONTAMINATION","MAINTENANCE_INADEQUATE"]
edges=list(csv.DictReader(open(KG+"/causation_edges.csv")))
def inedges(t):
    return sorted([(e["src"],int(e["support"]),float(e["p_dst_given_src"]),float(e["lift"])) for e in edges if e["dst"]==t], key=lambda r:-r[1])
for T,F in [("LOSS_OF_CONTROL_INFLIGHT",LOC),("ENGINE_FAILURE",EF)]:
    print("\n=====",T)
    ie=inedges(T)
    print("top in-edges by support (src, support, q, lift):")
    for r in ie[:14]: print("  ",r, "<-- in set" if r[0] in F else "")
    print("set members' rank by support:", [ [x[0] for x in ie].index(f)+1 for f in F])
    # base rates and presence
    y=df[T].values.astype(int)
    X=df[F].values.astype(int)
    print("prevalence",y.mean().round(4)," base rates:",{f:round(df[f].mean(),3) for f in F})
    q={r[0]:r[2] for r in ie}
    # noisy-OR at base rates (paper)
    P0=1-np.prod([1-df[f].mean()*q[f] for f in F]); print("paper noisy-OR at base rates:",round(P0,4))
    # coverage: fraction of T records with any set member present
    anyp=(X.sum(1)>0)
    print("P(any set member present | T present):",round(anyp[y==1].mean(),4), " | T absent:",round(anyp[y==0].mean(),4))
    # per-record leaky noisy-OR MLE: P(y=1|x)=1-(1-l)*prod(1-p_i)^x_i
    def nll(theta):
        l=1/(1+np.exp(-theta[0])); p=1/(1+np.exp(-theta[1:]))
        s=np.log1p(-l)+X@np.log1p(-p); py=1-np.exp(s)
        py=np.clip(py,1e-9,1-1e-9)
        return -(y*np.log(py)+(1-y)*np.log(1-py)).sum()
    th0=np.concatenate([[-3],np.full(len(F),-1.0)])
    res=minimize(nll,th0,method="L-BFGS-B")
    l=1/(1+np.exp(-res.x[0])); p=1/(1+np.exp(-res.x[1:]))
    print("leaky noisy-OR MLE: leak=",round(l,4)," per-factor p_i:",{f:round(v,3) for f,v in zip(F,p)})
    s=np.log1p(-l)+X@np.log1p(-p); py=1-np.exp(s)
    print("  mean predicted:",round(py.mean(),4)," realized:",round(y.mean(),4), " nll/N:",round(res.fun/N,4))
    # non-leaky MLE for comparison
    def nll0(theta):
        p=1/(1+np.exp(-theta)); s=X@np.log1p(-p); py=np.clip(1-np.exp(s),1e-9,1-1e-9)
        return -(y*np.log(py)+(1-y)*np.log(1-py)).sum()
    r0=minimize(nll0,np.full(len(F),-1.0),method="L-BFGS-B"); p0=1/(1+np.exp(-r0.x))
    py0=1-np.exp(X@np.log1p(-p0)); print("non-leaky MLE mean predicted:",round(py0.mean(),4)," nll/N:",round(r0.fun/N,4))
    # implied base rates from table: support/(q*N)
    print("implied base rate support/(q N) vs actual:",{f:(round(dict((r[0],r[1]) for r in ie)[f]/(q[f]*N),3), round(df[f].mean(),3)) for f in F})
# chain-level: for LOC records, do the explicit caused_by parents of the LOC node intersect the set?
print("\n===== chain-level parent coverage (explicit caused_by links into the node)")
cnt={"LOSS_OF_CONTROL_INFLIGHT":Counter(),"ENGINE_FAILURE":Counter()}
sets={"LOSS_OF_CONTROL_INFLIGHT":set(LOC),"ENGINE_FAILURE":set(EF)}
with open(KG+"/per_accident_chains.jsonl") as f:
    for line in f:
        rec=json.loads(line); nodes={n["idx"]:n for n in rec["chain"]}
        for n in rec["chain"]:
            t=n["factor_type"]
            if t in cnt:
                par={nodes[e["src"]]["factor_type"] for e in n.get("caused_by",[]) if e["src"] in nodes}
                cnt[t]["n"]+=1
                if not par: cnt[t]["root"]+=1
                elif par & sets[t]: cnt[t]["parent_in_set"]+=1
                else: cnt[t]["parents_outside_set"]+=1
for t,c in cnt.items():
    n=c["n"]; print(t, dict(c), {k:round(v/n,3) for k,v in c.items() if k!="n"})
