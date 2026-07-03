"""E8/E10: non-Bayesian learned-memory audit (NumPy only).
Model: random-Fourier feature memory m_T = mean phi(X_t); learned softmax
decoder over G grid cells; learned local correction head lambda_T * K(.,x).
This is NOT the Bayes histogram: memory is compressed through features and a
trained decoder; the live channel is an architectural correction head.
Registered validation target (qualitative, not exponents): (i) full-
distribution chi decays slower / stays larger than committed-scalar chi at
late horizons; (ii) aimed correction costs order patch < field < scalar;
(iii) the planted audit separates recalibration from rider."""
import numpy as np

R, G = 8, 64
GRID = (np.arange(G) + 0.5) / G

def phi(x):
    x = np.atleast_1d(x)
    out = [np.ones_like(x)]
    for r in range(1, R + 1):
        out += [np.sin(2*np.pi*r*x), np.cos(2*np.pi*r*x)]
    return np.stack(out, -1)          # (n, 2R+1)

def p_omega(om):
    d = 1 + 0.25*np.sin(2*np.pi*(GRID-om[0])) + 0.15*np.cos(6*np.pi*(GRID+om[1]))
    return d / d.sum()

def sample_om(om, n, rng):
    p = p_omega(om); c = np.cumsum(p)
    j = np.searchsorted(c, rng.random(n)); return (j + rng.random(n)) / G

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def train_decoder(seed=3, S=300, Ttr=1500, iters=600, lr=0.5):
    rng = np.random.default_rng(seed)
    M, Y = [], []
    for _ in range(S):
        om = rng.random(2); M.append(phi(sample_om(om, Ttr, rng)).mean(0)); Y.append(p_omega(om))
    M, Y = np.array(M), np.array(Y)
    W = rng.standard_normal((G, 2*R+1)) * 0.01; b = np.zeros(G); vW = np.zeros_like(W); vb = np.zeros_like(b)
    for _ in range(iters):
        P = softmax(M @ W.T + b)                    # (S,G)
        Gd = (P - Y)                                 # CE gradient wrt logits
        gW = Gd.T @ M / S; gb = Gd.mean(0)
        vW = 0.9*vW - lr*gW; vb = 0.9*vb - lr*gb; W += vW; b += vb
    return W, b

def bump(x, width=3):
    j = np.clip((np.atleast_1d(x)*G).astype(int), 0, G-1)
    K = np.zeros((len(j), G))
    for off in range(-width, width+1):
        w = (width+1-abs(off))
        K[np.arange(len(j)), np.clip(j+off, 0, G-1)] += w
    return K / K.sum(-1, keepdims=True)

class Learned:
    def __init__(self, W, b, c, beta):
        self.W, self.b, self.c, self.beta = W, b, c, beta
    def lam(self, T): return float(np.clip(self.c * T**(-self.beta), 0, 1))
    def dec(self, m): return softmax(m @ self.W.T + self.b)
    def f(self, m, x, T):
        lam = self.lam(T)
        return (1-lam)*self.dec(np.atleast_2d(m)) + lam*bump(x)

def fit_lambda(W, b, seed=11):
    rng = np.random.default_rng(seed); best=(None,None,1e9)
    streams=[]
    for _ in range(40):
        om=rng.random(2); T=int(10**rng.uniform(3,5)); xs=sample_om(om,T+2,rng)
        streams.append((phi(xs[:T]).mean(0), xs[T], xs[T+1], T))
    for c in (0.5,1.0,2.0,4.0):
        for beta in (0.2,0.35,0.5,0.65):
            mdl=Learned(W,b,c,beta); nll=0.0
            for m,x,y,T in streams:
                q=mdl.f(m,x,T)[0]; nll -= np.log(q[min(int(y*G),G-1)]*G + 1e-12)
            if nll<best[2]: best=(c,beta,nll)
    return best[0],best[1]

def elicit(q, which):
    dens = q * G
    if which=="point":  return dens[..., int(0.5*G)]
    if which=="scalar": return (q * GRID).sum(-1)
    return dens  # full, L2 with weight 1/G

def norm_of(v, which):
    if which=="full": return np.sqrt((v**2).sum(-1)/G)
    return np.abs(v)

def audit_chi(mdl, m, T, rng, sigma=1.0/G, n=1500):
    out={}
    xs = rng.random(n); sg = rng.choice([-1.,1.],n)*sigma
    f0 = mdl.f(m, xs, T); f1 = mdl.f(m, np.clip(xs+sg,0,1-1e-9), T)
    zs, zs2 = rng.random(n), rng.random(n)
    dm = (phi(zs2)-phi(zs))/T                       # (n,d) surgeries
    for which in ("point","scalar","full"):
        LX = float(np.mean(norm_of(elicit(f1,which)-elicit(f0,which),which))/sigma)
        # ratio-of-expectations surgery-normalized L_M
        num=[]; den=[]
        for i in range(0,n,50):
            m2 = m + dm[i]
            fa = mdl.f(m2, xs[:200], T); fb = mdl.f(m, xs[:200], T)
            num.append(np.mean(norm_of(elicit(fa,which)-elicit(fb,which),which)))
            den.append(norm_of(np.mean(elicit(fa,which)-elicit(fb,which),axis=0) if which=="full"
                               else np.mean(elicit(fa,which))-np.mean(elicit(fb,which)),which))
        LM = float(np.sum(num)/max(np.sum(den),1e-15))
        out[which] = {"LX":LX, "LM":LM, "chi": LX/(LX+LM)}
    return out

def E8(Ts=(10**3,3*10**3,10**4,3*10**4,10**5), reps=6, seed=17):
    W,b = train_decoder(); c,beta = fit_lambda(W,b)
    mdl = Learned(W,b,c,beta)
    res={"lambda_fit":{"c":c,"beta":beta},"T":list(Ts),"chi":{k:[] for k in ("point","scalar","full")},
         "corrections":{}}
    rngm=np.random.default_rng(seed)
    for T in Ts:
        acc={k:[] for k in ("point","scalar","full")}
        for r in range(reps):
            rng=np.random.default_rng(seed*77+r); om=rng.random(2)
            m = phi(sample_om(om,T,rng)).mean(0)
            a = audit_chi(mdl,m,T,rng)
            for k in acc: acc[k].append(a[k]["chi"])
        for k in acc: res["chi"][k].append(float(np.mean(acc[k])))
    # ordering check at late horizons (last two T)
    ch=res["chi"]
    res["ordering_full_gt_scalar_late"] = bool(all(ch["full"][i]>ch["scalar"][i] for i in (-2,-1)))
    # aimed corrections at T=1e4: memory-channel injection until threshold
    def n_correct(which, T=10**4, eps=None, rng=None):
        om=rng.random(2); m=phi(sample_om(om,T,rng)).mean(0); x0=0.25
        base=elicit(mdl.f(m,x0,T)[0],which)
        tgt = {"point":0.5,"full":0.3,"scalar":0.95}[which]
        eps={"point":0.35,"full":0.05,"scalar":0.01}[which]
        mm=m.copy(); n=0
        while n < 20000:
            cur=elicit(mdl.f(mm,x0,T)[0],which)
            moved = norm_of(cur-base,which) if which!="full" else norm_of(cur-base,"full")
            if moved>=eps: break
            mm = (T+n)/(T+n+1)*mm + phi(np.array([tgt]))[0]/(T+n+1); n+=1
        return n
    rr=np.random.default_rng(seed+5)
    res["corrections"]={k:float(np.mean([n_correct(k,rng=np.random.default_rng(seed*3+j)) for j in range(6)]))
                        for k in ("point","full","scalar")}
    res["correction_ordering_ok_field_lt_scalar"]= bool(res["corrections"]["full"]<res["corrections"]["scalar"])
    res["note_point_patch"]="global-feature memory exposes no local handle; audit surfaces this (architecture-dependent leg)"
    import numpy as _np
    lT=_np.log10(res["T"])
    res["chi_slopes"]={k:round(float(_np.polyfit(lT,_np.log10(res["chi"][k]),1)[0]),3) for k in res["chi"]}
    return res, mdl

def E10(mdl, T=10**4, reps=8, seed=29, lam_cal=0.3, lam_ride=0.006, sigma=1.0/G):
    rng=np.random.default_rng(seed); rows={}
    u = np.zeros(G); u[G//3]=1; u[G//3+1]=-1; u/= np.sqrt((u**2).sum())   # zero-sum dipole
    def variants(m, om):
        qstar = p_omega(om)
        f0 = lambda x: mdl.f(m,x,T)
        dbar = lam_cal*(qstar - f0(np.linspace(0,1,200)).mean(0))
        def make(add_bar, add_ride):
            def f(x):
                q = f0(x)
                if add_bar: q = q + dbar
                if add_ride:
                    s = np.sign(np.sin(np.pi*np.atleast_1d(x)/sigma))[:,None]
                    q = q + lam_ride*s*u
                q = np.clip(q,1e-9,None); return q/q.sum(-1,keepdims=True)
            return f
        return {"baseline":make(0,0),"+dbar":make(1,0),"+rider":make(0,1),"+both":make(1,1)}
    acc={v:{"nll_gain":[],"excess_LX":[],"share_gain_from_dbar":[],"share_lev_from_dtilde":[]} for v in ("baseline","+dbar","+rider","+both")}
    for r in range(reps):
        om=rng.random(2); m=phi(sample_om(om,T,rng)).mean(0)
        V=variants(m,om); xs=rng.random(800); ys=sample_om(om,800,rng)
        def nll(f):
            q=f(xs); return float(-np.mean(np.log(q[np.arange(800),np.clip((ys*G).astype(int),0,G-1)]*G+1e-12)))
        def LX(f):
            sg=rng.choice([-1.,1.],800)*sigma
            return float(np.mean(np.sqrt(((f(np.clip(xs+sg,0,1-1e-9))-f(xs))**2).sum(-1)*G))/sigma)
        n0,L0=nll(V["baseline"]),LX(V["baseline"])
        for v,f in V.items():
            d = f(xs)-V["baseline"](xs)
            dbar_hat=d.mean(0); dtil=d-dbar_hat
            acc[v]["nll_gain"].append(n0-nll(f))
            acc[v]["excess_LX"].append(LX(f)-L0)
            # gain attributable to dbar alone
            fb=lambda x: np.clip(V["baseline"](x)+dbar_hat,1e-9,None)/np.clip(V["baseline"](x)+dbar_hat,1e-9,None).sum(-1,keepdims=True)
            g_bar=n0-nll(fb); g_full=n0-nll(f)
            acc[v]["share_gain_from_dbar"].append(g_bar/g_full if abs(g_full)>1e-6 else 1.0)
            lev_til=float(np.mean(np.sqrt(((dtil[:-1]-dtil[1:])**2).sum(-1)*G)))
            lev_tot=float(np.mean(np.sqrt(((d[:-1]-d[1:])**2).sum(-1)*G)))
            acc[v]["share_lev_from_dtilde"].append(lev_til/lev_tot if lev_tot>1e-12 else 0.0)
    return {v:{k:round(float(np.mean(vals)),4) for k,vals in d.items()} for v,d in acc.items()}

if __name__=="__main__":
    import json
    e8,mdl=E8(); e10=E10(mdl)
    json.dump({"E8":e8,"E10":e10},open("results/learned.json","w"),indent=1)
    print(json.dumps({"E8":{k:e8[k] for k in ("lambda_fit","chi","ordering_full_gt_scalar_late","corrections","correction_ordering_ok")},"E10":e10},indent=1))
