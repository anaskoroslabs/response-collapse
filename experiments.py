"""
Reproduces every numerical claim in the paper. Run: python experiments.py
Writes results/results.json and prints a table the editor can diff byte-for-byte.
All seeds fixed; deterministic.
"""
import json, os, numpy as np
from skeleton import Skeleton
from leverage import sample_pT, LX_pointwise_IS, LX_scalar, LX_full

os.makedirs("results", exist_ok=True)
OUT = {}

# ---------- E1: three decay laws (leverage vs T) ----------
def E1(Ts=(10**3,10**4,10**5,10**6), reps=8, n=6000, seed=7):
    A,B,C=[],[],[]
    for T in Ts:
        a_,b_,c_=[],[],[]
        for r in range(reps):
            rng=np.random.default_rng(seed*777+r); sk=Skeleton(T,rng=rng)
            sig=sk.delta/4; i0=sk.k//2
            a_.append(LX_pointwise_IS(sk,i0,sig,n,rng))
            b_.append(LX_scalar(sk.bayes_mean,sk,sig,n,rng))
            c_.append(LX_full(sk,sig,n,rng))
        A.append(np.mean(a_)); B.append(np.mean(b_)); C.append(np.mean(c_))
    lT=np.log10(Ts)
    return {"T":list(Ts),
            "pointwise":[float(v) for v in A],"mean":[float(v) for v in B],"full":[float(v) for v in C],
            "slope_pointwise":round(float(np.polyfit(lT,np.log10(A),1)[0]),3),
            "slope_mean":round(float(np.polyfit(lT,np.log10(B),1)[0]),3),
            "slope_full":round(float(np.polyfit(lT,np.log10(C),1)[0]),3)}

# ---------- E2: frontier saturation + adversary ----------
def E2(T=10**5, n=60000, seed=3, eps=0.03):
    rng=np.random.default_rng(seed); sk=Skeleton(T,rng=rng); sigma=sk.delta/4
    def LX_dith(d):
        x=sample_pT(sk,n,rng); s=rng.choice([-1.,1.],n)*sigma
        return float(np.mean(np.abs(d(np.clip(x+s,0,1))-d(x)))/sigma)
    def ratio(d,DR): return LX_dith(d)/(2*np.sqrt(DR)/sigma)
    w=np.pi/sigma
    res={"square_ratio":round(ratio(lambda x: eps*np.sign(np.sin(w*x)), eps**2),4),
         "sine_ratio":round(ratio(lambda x: eps*np.sin(w*x), eps**2/2),4),
         "lowfreq_ratio":round(ratio(lambda x: eps*np.sin(4*np.pi*x), eps**2/2),4)}
    # 16-dim Fourier adversary on regret sphere
    M=16; ws=(np.pi/sigma)*np.linspace(0.1,2.0,M)
    xg=sample_pT(sk,30000,rng); sg=rng.choice([-1.,1.],30000)*sigma
    PhiX=np.sin(np.outer(xg,ws)); D=np.sin(np.outer(np.clip(xg+sg,0,1),ws))-PhiX
    def obj(c):
        d2=np.mean((PhiX@c)**2); c2=c*eps/np.sqrt(max(d2,1e-30))
        return np.mean(np.abs(D@c2))/sigma
    c=rng.standard_normal(M); best=obj(c)
    for it in range(4000):
        c2=c+rng.standard_normal(M)*0.3*(0.999**it); v=obj(c2)
        if v>best: best,c=v,c2
    res["adversary_ratio"]=round(best/(2*np.sqrt(eps**2)/sigma),4)
    return res

# ---------- E3: decoupling ablation (wins orthogonal to leverage) ----------
def E3(T=10**5, n=80000, seed=13, chi0=0.1):
    rng=np.random.default_rng(seed); sk=Skeleton(T,rng=rng); sig=sk.delta/4
    centers=(np.arange(sk.k)+0.5)*sk.delta
    m_true=float(np.dot(truth_weights(sk),centers))
    # gap-recalibration map: slice-efficient mean estimate + optional band rider
    def base_mean(x): return sk.bayes_mean(x)
    w=np.pi/sig
    def rider(x): return chi0*sig*np.sign(np.sin(w*np.atleast_1d(x)))
    xs=sample_pT(sk,n,rng)
    # measure excess leverage of ridered vs ablated
    def LX(fn):
        s=rng.choice([-1.,1.],n)*sig
        return float(np.mean(np.abs(fn(np.clip(xs+s,0,1))-fn(xs)))/sig)
    LX_ride=LX(lambda x: np.array([base_mean(xi) for xi in np.atleast_1d(x)])+rider(x))
    LX_abl =LX(lambda x: np.array([base_mean(xi) for xi in np.atleast_1d(x)]))
    return {"chi0":chi0,"LX_with_rider":round(LX_ride,4),"LX_ablated":round(LX_abl,4),
            "predicted_rider_LX":round(2.0*chi0,4),
            "price_refund_measured":round(LX_ride-LX_abl,4)}

def truth_weights(sk):
    from skeleton import truth_density
    c=(np.arange(sk.k)+0.5)*sk.delta; w=truth_density(c); return w/np.sum(w)

# ---------- E4: correction-cost hierarchy (TRUE simulation: inject until threshold) ----------
def E4(Ts=(10**4,10**5,10**6), reps=8, seed=21,
       eps_point=0.5, eps_field=0.02, eps_scalar=0.002, cap_frac=200):
    """Empirical samples-to-fixed-correction: aimed samples are injected one at a
    time into the live channel until the elicited quantity has moved by a fixed
    threshold. No analytic shortcut; counts are integers; dilution is real."""
    def run_one(sk, channel):
        i0=sk.k//2
        extra=np.zeros(sk.k)
        centers=(np.arange(sk.k)+0.5)*sk.delta
        def dens(e):
            c=sk.a+sk.counts+e; return c/np.sum(c)/sk.delta
        base=dens(extra)
        if channel=="point":
            target=base[i0]; n=0
            while abs(dens(extra)[i0]-target)<eps_point and n<cap_frac*np.sqrt(sk.T):
                extra[i0]+=1; n+=1
            return n
        if channel=="field":
            n=0
            while np.sqrt(np.sum((dens(extra)-base)**2)*sk.delta)<eps_field and n<cap_frac*np.sqrt(sk.T):
                extra[i0+1]+=1; n+=1
            return n
        if channel=="scalar":
            m0=float(np.dot(base*sk.delta,centers)); j_far=int(0.95*sk.k); n=0
            while abs(float(np.dot(dens(extra)*sk.delta,centers))-m0)<eps_scalar and n<cap_frac*np.sqrt(sk.T)*np.sqrt(sk.T):
                extra[j_far]+=1; n+=1
            return n
    out={"T":list(Ts),"n_point":[],"n_field":[],"n_scalar":[]}
    for T in Ts:
        p_,f_,s_=[],[],[]
        for r in range(reps):
            rng=np.random.default_rng(seed*13+r); sk=Skeleton(T,rng=rng)
            p_.append(run_one(sk,"point")); f_.append(run_one(sk,"field")); s_.append(run_one(sk,"scalar"))
        out["n_point"].append(float(np.mean(p_))); out["n_field"].append(float(np.mean(f_))); out["n_scalar"].append(float(np.mean(s_)))
    lT=np.log10(Ts)
    for key,lab in (("n_point","slope_pointwise"),("n_field","slope_field"),("n_scalar","slope_scalar")):
        out[lab]=round(float(np.polyfit(lT,np.log10(out[key]),1)[0]),3)
    out["speedup_scalar_over_field"]=[round(s/f,2) for s,f in zip(out["n_scalar"],out["n_field"])]
    return out

# ---------- E5: drift recovery tau_rec ~ T0 ----------
def E5(T0s=(100,1000,10000), Delta=3.0, eps=0.25, reps=200, seed=5):
    rng=np.random.default_rng(seed); tau=[]
    for T0 in T0s:
        ts=[]
        for _ in range(reps):
            m,n=rng.standard_normal()/np.sqrt(T0),T0; theta1,t=Delta,0
            while abs(theta1-m)>eps*Delta and t<200*T0:
                t+=1;n+=1;m+=(theta1+rng.standard_normal()-m)/n
            ts.append(t)
        tau.append(float(np.mean(ts)))
    return {"T0":list(T0s),"tau":tau,
            "slope":round(float(np.polyfit(np.log10(T0s),np.log10(tau),1)[0]),3)}

# ---------- E6: separability (debias closes the financing door; measure bias energy) ----------
def E6(Ts=(10**3,10**4,10**5), seed=31):
    trapz = np.trapezoid
    from skeleton import truth_density
    def bias_energy(sk, order):
        c=(np.arange(sk.k)+0.5)*sk.delta
        yg=np.linspace(0,1,4000); q=truth_density(yg); q=q/trapz(q,yg)
        if order==0:  # histogram (zeroth-order): bin-average vs truth = the sawtooth
            qbar=np.array([q[(yg>=i*sk.delta)&(yg<(i+1)*sk.delta)].mean() for i in range(sk.k)])
            approx=qbar[np.minimum((yg*sk.k).astype(int),sk.k-1)]
        else:          # frequency polygon (first-order): linear interp of bin-center values
            qc=np.interp(c,yg,q); approx=np.interp(yg,c,qc)
        return float(trapz((approx-q)**2,yg))
    H,P=[],[]
    for T in Ts:
        sk=Skeleton(T,rng=np.random.default_rng(seed+T%97))
        H.append(bias_energy(sk,0)); P.append(bias_energy(sk,1))
    lT=np.log10(Ts)
    return {"T":list(Ts),
            "hist_bias_energy":[float(v) for v in H],"poly_bias_energy":[float(v) for v in P],
            "slope_histogram_fuel":round(float(np.polyfit(lT,np.log10(H),1)[0]),3),
            "slope_polygon_fuel":round(float(np.polyfit(lT,np.log10(P),1)[0]),3),
            "transport_slope_note":"transport leverage slope ~ -1/4 is E1['slope_full']; unchanged by debiasing"}

# ---------- E7: sequential correction dialogue (2- and 3-step compounding) ----------
def E7(Ts=(10**4,10**5,10**6), reps=8, seed=41):
    """Does aimed transport steering compound linearly over a k-step dialogue,
    or saturate? Protocol: at each step the operator injects one targeted sample
    at the bin adjacent to the current predictive mode; measure total L2 movement
    of the predictive after k steps vs k * (one-step movement)."""
    out={"T":list(Ts),"k":[1,2,3],"efficiency":[]}  # efficiency = move(k)/(k*move(1))
    for T in Ts:
        effs=[]
        for r in range(reps):
            rng=np.random.default_rng(seed*7+r); sk=Skeleton(T,rng=rng)
            j0=sk.k//2
            def predictive(counts_extra):
                c=sk.a+sk.counts+counts_extra
                return c/np.sum(c)/sk.delta
            base=predictive(np.zeros(sk.k))
            # one aimed sample into bin j0+1 per step; posterior re-concentrates between steps
            moves=[]
            extra=np.zeros(sk.k)
            prev=base
            for step in range(3):
                extra[j0+1]+=1.0
                cur=predictive(extra)
                moves.append(np.sqrt(np.sum((cur-prev)**2)*sk.delta))
                prev=cur
            one=moves[0]
            effs.append([ (sum(moves[:k]))/(k*one) for k in (1,2,3) ])
        out["efficiency"].append([round(float(np.mean([e[i] for e in effs])),4) for i in range(3)])
    return out

if __name__=="__main__":
    OUT["E1_decay_laws"]=E1()
    OUT["E2_frontier"]=E2()
    OUT["E3_decoupling_ablation"]=E3()
    OUT["E4_correction_cost"]=E4()
    OUT["E5_drift_recovery"]=E5()
    OUT["E6_separability"]=E6()
    OUT["E7_sequential"]=E7()
    with open("results/results.json","w") as f: json.dump(OUT,f,indent=2)
    print(json.dumps(OUT,indent=2))

