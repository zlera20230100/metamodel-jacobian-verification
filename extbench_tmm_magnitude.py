# -*- coding: utf-8 -*-
# Check whether sign-wrong components have small gradient magnitude in this TMM benchmark.
# ones, so that a magnitude threshold / smallest-|grad|-first ties or beats the gate? Re-runs the exact
# extbench_tmm pipeline (same seeds) and reports |ad| of sign-wrong vs sign-right, the magnitude AUC,
# and the smallest-|grad|-first caught curve vs the gate.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
LAM0=1550.0; nH,nL,nsub,n0=2.35,1.45,1.52,1.0
IDX=np.array([nH,nL,nH,nL,nH,nL,nH,nL,nH,nL,nH]); QW=LAM0/(4.0*IDX); NOMINAL=QW.copy(); NOMINAL[5]*=2.0; K=IDX.size
def tmm_R(thick,lam):
    M=np.eye(2,dtype=complex)
    for nj,dj in zip(IDX,thick):
        d=2*np.pi*nj*dj/lam; c,s=np.cos(d),np.sin(d); M=M@np.array([[c,1j*s/nj],[1j*nj*s,c]],dtype=complex)
    B=M[0,0]+M[0,1]*nsub; C=M[1,0]+M[1,1]*nsub; return float(np.abs((n0*B-C)/(n0*B+C))**2)
_scan=np.linspace(LAM0-60,LAM0+60,481); _Rs=np.array([tmm_R(NOMINAL,l) for l in _scan]); _up=_scan>LAM0
LAM_EVAL=float(_scan[_up][np.argmin(np.abs(_Rs[_up]-0.4))])
def fd(thick,h=0.5):
    g=np.zeros(K)
    for k in range(K):
        tp=thick.copy(); tp[k]+=h; tm=thick.copy(); tm[k]-=h; g[k]=(tmm_R(tp,LAM_EVAL)-tmm_R(tm,LAM_EVAL))/(2*h)
    return g
RNG=np.random.default_rng(7); NS=600; span=0.30
X=NOMINAL[None,:]*(1.0+span*(2*RNG.random((NS,K))-1)); Y=np.array([tmm_R(x,LAM_EVAL) for x in X])
xm,xs=X.mean(0),X.std(0); ym,ysd=Y.mean(),Y.std()
Xt=torch.tensor((X-xm)/xs,dtype=torch.float32); Yt=torch.tensor((Y-ym)/ysd,dtype=torch.float32).view(-1,1)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(K,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
    def forward(s,x): return s.net(x)
def train_one(seed):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed); net=MLP(); opt=torch.optim.Adam(net.parameters(),lr=2e-3); lf=nn.MSELoss()
    n=Xt.size(0); idx=torch.randperm(n,generator=g); tr=idx[:int(0.9*n)]
    for ep in range(400):
        for b in tr[torch.randperm(tr.numel(),generator=g)].split(256):
            opt.zero_grad(); lf(net(Xt[b]),Yt[b]).backward(); opt.step()
    return net
nets=[train_one(s) for s in range(10)]
def ad_grad(net,xp):
    xn=torch.tensor(((xp-xm)/xs),dtype=torch.float32).view(1,-1).requires_grad_(True); net(xn).backward(); return (xn.grad.detach().numpy().ravel())*(ysd/xs)
NQ=80; QP=NOMINAL[None,:]*(1.0+0.12*(2*RNG.random((NQ,K))-1))
all_sa=[]; all_cor=[]; all_absad=[]
caught_gate=np.zeros((NQ,K+1)); caught_small=np.zeros((NQ,K+1)); nwrong=np.zeros(NQ)
for q,xq in enumerate(QP):
    Gm=np.array([ad_grad(net,xq) for net in nets]); ad=Gm.mean(0); sa=np.maximum((Gm>0).mean(0),(Gm<0).mean(0)); f=fd(xq)
    wrong=(np.sign(ad)!=np.sign(f)); nwrong[q]=wrong.sum()
    all_sa.append(sa); all_cor.append((~wrong).astype(int)); all_absad.append(np.abs(ad))
    og=np.argsort(sa); osm=np.argsort(np.abs(ad))   # gate: least-agreement first ; baseline: SMALLEST-|ad| first
    for n in range(K+1):
        if wrong.sum()>0:
            caught_gate[q,n]=wrong[og[:n]].sum()/wrong.sum(); caught_small[q,n]=wrong[osm[:n]].sum()/wrong.sum()
sa=np.concatenate(all_sa); cor=np.concatenate(all_cor); absad=np.concatenate(all_absad)
print("sign-correct frac:",cor.mean(),"  n_wrong:",int((1-cor).sum()))
print("median |ad|  sign-RIGHT=%.2e  sign-WRONG=%.2e  (ratio %.1fx)"%(np.median(absad[cor==1]),np.median(absad[cor==0]),np.median(absad[cor==1])/max(np.median(absad[cor==0]),1e-30)))
print("AUC gate (sign-agreement) =%.3f"%roc_auc_score(cor,sa))
print("AUC magnitude (|ad|, large=trust) =%.3f"%roc_auc_score(cor,absad))
hasw=nwrong>0; qg=caught_gate[hasw].mean(0); qs=caught_small[hasw].mean(0)
def b80(c): i=np.where(c>=0.8)[0]; return int(i[0]) if len(i) else K
print("FD to catch >=80%% sign-wrong: gate=%d  smallest-|ad|-first=%d (of %d)"%(b80(qg),b80(qs),K))
print("caught@2: gate=%.2f small=%.2f ; caught@3: gate=%.2f small=%.2f"%(qg[2],qs[2],qg[3],qs[3]))
np.savez('extbench_tmm_magnitude.npz',
         gate_auc=roc_auc_score(cor, sa), mag_auc=roc_auc_score(cor, absad),
         median_absad_right=float(np.median(absad[cor == 1])),
         median_absad_wrong=float(np.median(absad[cor == 0])),
         b80_gate=b80(qg), b80_smallest=b80(qs), caught_gate=qg, caught_smallest=qs, K=K, NQ=NQ)
print("saved extbench_tmm_magnitude.npz")
