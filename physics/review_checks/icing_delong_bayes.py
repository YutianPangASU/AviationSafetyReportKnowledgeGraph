"""Arno #7: DeLong CI on the icing AUCs, paired test physics vs logistic, and an out-of-fold
density-ratio (Bayes-optimal) AUC estimate on (T, Td)."""
import sys, numpy as np
sys.path.insert(0,".")
from physics.ceiling_carb_icing import load_joined
from physics.carb_icing_model import p_carb_icing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
df=load_joined(); y=df["CARBURETOR_OR_INDUCTION_ICING"].to_numpy(int)
T=df["temp_c"].to_numpy(float); Td=df["dew_c"].to_numpy(float); X=np.column_stack([T,Td])
phys=np.array([p_carb_icing(t,d,"descent").p_ice for t,d in zip(T,Td)])
def delong(y,s1,s2=None):
    """DeLong variance of AUC (and covariance for paired scores)."""
    pos=y==1; neg=~pos
    def comps(s):
        sp,sn=s[pos],s[neg]
        # V10: for each positive, fraction of negatives below (ties 0.5)
        order=np.argsort(sn); sn_s=sn[order]
        v10=(np.searchsorted(sn_s,sp,'left')+np.searchsorted(sn_s,sp,'right'))/(2*len(sn))
        order=np.argsort(sp); sp_s=sp[order]
        v01=1-(np.searchsorted(sp_s,sn,'left')+np.searchsorted(sp_s,sn,'right'))/(2*len(sp))
        return v10,v01
    a1,b1=comps(s1); auc1=a1.mean(); m,n=pos.sum(),neg.sum()
    var1=a1.var(ddof=1)/m+b1.var(ddof=1)/n
    if s2 is None: return auc1,np.sqrt(var1)
    a2,b2=comps(s2); auc2=a2.mean(); var2=a2.var(ddof=1)/m+b2.var(ddof=1)/n
    cov=np.cov(a1,a2)[0,1]/m+np.cov(b1,b2)[0,1]/n
    z=(auc1-auc2)/np.sqrt(var1+var2-2*cov); return auc1,np.sqrt(var1),auc2,np.sqrt(var2),z,2*norm.sf(abs(z))
cv=StratifiedKFold(5,shuffle=True,random_state=0)
lr=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),X,y,cv=cv,method="predict_proba")[:,1]
a,se=delong(y,phys); print(f"physics descent AUC {a:.4f} DeLong 95% CI ({a-1.96*se:.4f}, {a+1.96*se:.4f})")
a,se=delong(y,lr); print(f"logistic OOF AUC {a:.4f} DeLong 95% CI ({a-1.96*se:.4f}, {a+1.96*se:.4f})")
print("paired DeLong physics vs logistic: auc1, se1, auc2, se2, z, p =", [round(v,4) for v in delong(y,phys,lr)])
# density-ratio Bayes-optimal AUC, out of fold, KDE per class with bandwidth chosen by CV on a grid
for bw in (0.5,1.0,2.0,3.0):
    s=np.zeros(len(y))
    for tr,te in cv.split(X,y):
        kp=KernelDensity(bandwidth=bw).fit(X[tr][y[tr]==1]); kn=KernelDensity(bandwidth=bw).fit(X[tr][y[tr]==0])
        s[te]=kp.score_samples(X[te])-kn.score_samples(X[te])
    a,se=delong(y,s); print(f"KDE density-ratio bw={bw}: OOF AUC {a:.4f} ({a-1.96*se:.4f}, {a+1.96*se:.4f})")
# in-sample (optimistic) density-ratio AUC as the upper envelope
for bw in (1.0,2.0):
    kp=KernelDensity(bandwidth=bw).fit(X[y==1]); kn=KernelDensity(bandwidth=bw).fit(X[y==0])
    s=kp.score_samples(X)-kn.score_samples(X); print(f"KDE in-sample bw={bw}: AUC {roc_auc_score(y,s):.4f}")
