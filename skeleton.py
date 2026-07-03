"""
Dirichlet-histogram skeleton of a slowly-learnable (BNT-type) source.
k(T)=ceil(sqrt(T)) bins on [0,1]; Dirichlet(a) prior; truth bounded in [c,C].
"""
import numpy as np

def truth_density(y):
    return 1.0 + 0.5 * np.sin(2 * np.pi * y)

def sample_truth(T, rng):
    g = np.linspace(0, 1, 20001)
    pdf = truth_density(g); cdf = np.cumsum(pdf); cdf /= cdf[-1]
    return np.interp(rng.random(T), cdf, g)

class Skeleton:
    def __init__(self, T, a=1.0, rng=None):
        self.T = T
        self.k = int(np.ceil(np.sqrt(T)))
        self.delta = 1.0 / self.k
        self.a = a
        self.rng = rng or np.random.default_rng(0)
        x = sample_truth(T, self.rng)
        self.counts = np.bincount(np.minimum((x * self.k).astype(int), self.k - 1),
                                  minlength=self.k).astype(float)
        self.N0 = self.k * a + T
        self.pbar = (a + self.counts) / self.N0

    def bin_of(self, x):
        return np.minimum((np.asarray(x) * self.k).astype(int), self.k - 1)

    def pT(self, x):
        return self.pbar[self.bin_of(x)] / self.delta

    def bayes_pointwise(self, x, i0):
        j = self.bin_of(x)
        num = self.a + self.counts[i0] + (j == i0)
        return num / (self.N0 + 1) / self.delta

    def bayes_mean(self, x):
        j = self.bin_of(x)
        centers = (np.arange(self.k) + 0.5) * self.delta
        base = float(np.dot(self.a + self.counts, centers))
        return (base + centers[j]) / (self.N0 + 1)

    def bayes_full(self, x):
        j = np.atleast_1d(self.bin_of(x))
        out = np.tile(self.a + self.counts, (len(j), 1))
        out[np.arange(len(j)), j] += 1.0
        return out / (self.N0 + 1) / self.delta
