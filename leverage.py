"""Finite-perturbation interventional leverage L_X (Sec. 2)."""
import numpy as np

def sample_pT(sk, n, rng):
    bins = rng.choice(sk.k, size=n, p=sk.pbar)
    return (bins + rng.random(n)) * sk.delta

def LX_pointwise_IS(sk, i0, sigma, n, rng):
    bL, bR = i0 * sk.delta, (i0 + 1) * sk.delta
    half = n // 2
    xs = np.concatenate([bL + (rng.random(half) * 2 - 1) * sigma,
                         bR + (rng.random(n - half) * 2 - 1) * sigma])
    xs = np.clip(xs, 1e-12, 1 - 1e-12)
    s = rng.choice([-1.0, 1.0], size=n) * sigma
    xp = np.clip(xs + s, 1e-12, 1 - 1e-12)
    f = np.array([sk.bayes_pointwise(x, i0) for x in xs])
    fp = np.array([sk.bayes_pointwise(x, i0) for x in xp])
    q = np.abs(fp - f) / sigma
    w = sk.pT(xs); window = 4 * sigma
    return float(np.mean(q * w) * window)

def LX_scalar(f, sk, sigma, n, rng):
    x = sample_pT(sk, n, rng); s = rng.choice([-1.0, 1.0], size=n) * sigma
    xp = np.clip(x + s, 0, 1 - 1e-12)
    fx = np.array([f(xi) for xi in x]); fxp = np.array([f(xi) for xi in xp])
    return float(np.mean(np.abs(fxp - fx)) / sigma)

def LX_full(sk, sigma, n, rng):
    x = sample_pT(sk, n, rng); s = rng.choice([-1.0, 1.0], size=n) * sigma
    xp = np.clip(x + s, 0, 1 - 1e-12)
    F, Fp = sk.bayes_full(x), sk.bayes_full(xp)
    l2 = np.sqrt(np.sum((Fp - F) ** 2, axis=1) * sk.delta)
    return float(np.mean(l2) / sigma)
