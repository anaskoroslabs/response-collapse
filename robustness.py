"""E9: robustness of the decay laws.
Axes: resolution schedule k(T)=T^alpha; perturbation scale sigma=s*h;
prior mass a; truth family; bootstrap CIs on slopes.
Predictions (generalized theorem): pointwise slope = alpha-1, smooth scalar = -1,
full predictive = 3*alpha/2 - 1; alpha=1/2 recovers the paper rates."""
import numpy as np
from skeleton import Skeleton
import skeleton as SK
from leverage import LX_pointwise_IS, LX_scalar, LX_full

def slopes_at(Ts, reps, n, seed, alpha=None, a=1.0):
    A,B,C=[],[],[]
    for T in Ts:
        a_,b_,c_=[],[],[]
        for r in range(reps):
            rng=np.random.default_rng(seed*997+r)
            k=int(np.ceil(T**alpha)) if alpha else None
            sk=Skeleton(T,a=a,rng=rng,k=k)
            sig=sk.delta/4; i0=sk.k//2
            a_.append(LX_pointwise_IS(sk,i0,sig,n,rng))
            b_.append(LX_scalar(sk.bayes_mean,sk,sig,n,rng))
            c_.append(LX_full(sk,sig,n,rng))
        A.append(a_);B.append(b_);C.append(c_)
    def fit(M):
        m=np.log10(np.mean(M,axis=1)); return round(float(np.polyfit(np.log10(Ts),m,1)[0]),3)
    def boot(M,nb=1000,seed=5):
        rng=np.random.default_rng(seed); reps_=len(M[0]); sl=[]
        Marr=np.array(M)
        for _ in range(nb):
            idx=rng.integers(0,reps_,reps_)
            m=np.log10(Marr[:,idx].mean(axis=1))
            sl.append(np.polyfit(np.log10(Ts),m,1)[0])
        lo,hi=np.percentile(sl,[2.5,97.5]); return [round(float(lo),3),round(float(hi),3)]
    return {"pointwise":fit(A),"mean":fit(B),"full":fit(C),
            "ci_pointwise":boot(A),"ci_mean":boot(B),"ci_full":boot(C)}

def E9(seed=61):
    out={}
    Ts=(10**3,10**4,10**5)
    # (a) alpha sweep
    alphas=[0.35,0.45,0.50,0.60]; sweep={}
    for al in alphas:
        r=slopes_at(Ts,reps=4,n=3500,seed=seed,alpha=al)
        r["predicted"]={"pointwise":round(al-1,3),"mean":-1.0,"full":round(1.5*al-1,3)}
        sweep[str(al)]=r
    out["alpha_sweep"]=sweep
    # (b) sigma plateau at alpha=1/2 (admissible range s<=1; s=2,4 shown as beyond-range)
    sig_out={}
    for s_ in (0.5,1.0,2.0,4.0):
        A,C=[],[]
        for T in Ts:
            a_,c_=[],[]
            for r in range(4):
                rng=np.random.default_rng(seed*31+r); sk=Skeleton(T,rng=rng)
                sig=s_*sk.delta/4
                a_.append(LX_pointwise_IS(sk,sk.k//2,sig,3500,rng))
                c_.append(LX_full(sk,sig,3500,rng))
            A.append(np.mean(a_)); C.append(np.mean(c_))
        lT=np.log10(Ts)
        sig_out[str(s_)]={"pointwise":round(float(np.polyfit(lT,np.log10(A),1)[0]),3),
                          "full":round(float(np.polyfit(lT,np.log10(C),1)[0]),3)}
    out["sigma_scan_s_times_h_over_4"]=sig_out
    # (c) prior mass
    pri={}
    for a in (0.1,1.0,10.0):
        r=slopes_at(Ts,reps=4,n=3500,seed=seed+7,alpha=None,a=a)
        pri[str(a)]={k:r[k] for k in ("pointwise","mean","full")}
    out["prior_mass"]=pri
    # (d) truth families: ordering preserved
    truths=[lambda y:1+0.5*np.sin(2*np.pi*y), lambda y:1+0.3*np.sin(4*np.pi*y),
            lambda y:1+0.25*np.sin(2*np.pi*y)+0.15*np.cos(6*np.pi*y),
            lambda y:1.2-0.4*np.cos(2*np.pi*y), lambda y:1+0.45*np.sin(2*np.pi*y+0.7)]
    fam=[]
    orig=SK.truth_density
    for i,f in enumerate(truths):
        SK.truth_density=f
        r=slopes_at((10**3,10**4,10**5),reps=3,n=3000,seed=seed+i,alpha=None)
        fam.append({k:r[k] for k in ("pointwise","mean","full")})
    SK.truth_density=orig
    out["truth_families"]=fam
    out["ordering_preserved_all_families"]=all(f["full"]>f["pointwise"]>f["mean"] for f in fam)
    return out
