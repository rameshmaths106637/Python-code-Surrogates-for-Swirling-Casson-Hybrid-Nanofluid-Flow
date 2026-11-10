#!/usr/bin/env python3
"""
Revised benchmark for main.tex: robust ANFIS (PSO / GA / gradient-based) vs MLP ensemble,
HistGB and a Gaussian process, with repeated random splits and paired statistics.

Usage:
    python anfis_benchmark.py --csv dataset.csv --inputs Re,A1,... --outputs Cf1,Cf2,Nu1,Nu2 \
        --splits 20 --out results.csv
    python anfis_benchmark.py --selftest            # runs on synthetic data (quick)

Fixes relative to the first version of the study
  * Gaussian widths are bounded (sigma in [0.15, 4] on standardised inputs) -> no collapsing widths
  * firing strengths use log-sum-exp normalisation (no underflow in 14 dimensions)
  * ridge lambda 1e-3 (was 1e-6) for the linear consequents
  * rule grid extended (default 3,5,8,12,16,24), chosen on a validation split only
  * a gradient-based ANFIS (LSE for consequents + L-BFGS on antecedents) is added as the classic baseline
  * identical evaluation budget for PSO and GA; PSO has velocity clamping
  * Gaussian-process baseline (Matern-5/2, ARD)
  * repeated random splits + paired Wilcoxon tests and win counts
"""
import argparse, time, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")
SMIN, SMAX = 0.15, 4.0

# ----------------------------------------------------------------- ANFIS
class ANFIS:
    def __init__(self, n_rules, ridge=1e-3, seed=0):
        self.nr, self.ridge, self.rng = n_rules, ridge, np.random.default_rng(seed)
    def _unpack(self, th, d):
        c = th[:self.nr*d].reshape(self.nr, d)
        s = np.clip(np.exp(th[self.nr*d:]).reshape(self.nr, d), SMIN, SMAX)
        return c, s
    def _design(self, X, th):
        n, d = X.shape; c, s = self._unpack(th, d)
        logw = -0.5*(((X[:, None, :]-c[None])/s[None])**2).sum(2) - np.log(s).sum(1)[None]*0   # (n, nr)
        logw -= logw.max(1, keepdims=True)                                                   # log-sum-exp
        w = np.exp(logw); w /= w.sum(1, keepdims=True)
        Xa = np.hstack([X, np.ones((n, 1))])
        return (w[:, :, None]*Xa[:, None, :]).reshape(n, -1)
    def _fit_lse(self, X, y, th):
        A = self._design(X, th)
        beta = np.linalg.solve(A.T@A + self.ridge*np.eye(A.shape[1]), A.T@y)
        return beta, np.mean((A@beta-y)**2)
    def _fitness(self, th, X, y):
        try: return self._fit_lse(X, y, th)[1]
        except np.linalg.LinAlgError: return 1e6
    def _init(self, X):
        km = KMeans(self.nr, n_init=3, random_state=int(self.rng.integers(1e6))).fit(X)
        return np.concatenate([km.cluster_centers_.ravel(), np.zeros(self.nr*X.shape[1])])   # sigma=1
    def fit(self, X, y, method="pso", n_pop=30, n_iter=100):
        th0 = self._init(X); D = th0.size; rng = self.rng
        f = lambda t: self._fitness(t, X, y)
        lo = np.concatenate([np.full(self.nr*X.shape[1], X.min()-1), np.full(self.nr*X.shape[1], np.log(SMIN))])
        hi = np.concatenate([np.full(self.nr*X.shape[1], X.max()+1), np.full(self.nr*X.shape[1], np.log(SMAX))])
        pop = th0 + 0.3*rng.standard_normal((n_pop, D)); pop[0] = th0; pop = np.clip(pop, lo, hi)
        if method == "pso":
            w_, c1, c2 = 0.7, 1.5, 1.5; vmax = 0.2*(hi-lo)
            v = np.zeros_like(pop); pb = pop.copy(); pf = np.array([f(p) for p in pop])
            g = pb[pf.argmin()].copy(); gf = pf.min()
            for _ in range(n_iter):
                r1, r2 = rng.random((2, n_pop, D))
                v = np.clip(w_*v + c1*r1*(pb-pop) + c2*r2*(g-pop), -vmax, vmax)
                pop = np.clip(pop+v, lo, hi); fp = np.array([f(p) for p in pop])
                m = fp < pf; pb[m], pf[m] = pop[m], fp[m]
                if pf.min() < gf: gf, g = pf.min(), pb[pf.argmin()].copy()
            th = g
        elif method == "ga":
            fit = np.array([f(p) for p in pop])
            for _ in range(n_iter):
                el = pop[fit.argmin()].copy()
                idx = rng.integers(0, n_pop, (n_pop, 2)); par = np.where((fit[idx[:, 0]] < fit[idx[:, 1]])[:, None], pop[idx[:, 0]], pop[idx[:, 1]])
                mates = par[rng.permutation(n_pop)]; mask = rng.random((n_pop, D)) < 0.5
                kids = np.where(mask, par, mates)
                mut = rng.random((n_pop, D)) < 0.1
                kids = np.clip(kids + mut*0.1*(hi-lo)*0.1*rng.standard_normal((n_pop, D)), lo, hi)
                kids[0] = el; pop = kids; fit = np.array([f(p) for p in pop])
            th = pop[fit.argmin()]
        elif method == "grad":       # LSE + L-BFGS on antecedents (variable projection), same budget order
            res = minimize(f, th0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=n_iter, maxfun=n_iter*30))
            th = res.x
        else: raise ValueError(method)
        self.th = th; self.beta, _ = self._fit_lse(X, y, th); return self
    def predict(self, X): return self._design(X, self.th)@self.beta

# ----------------------------------------------------------------- helpers
def std(a, ref): m, s = ref.mean(0), ref.std(0)+1e-12; return (a-m)/s, m, s
def anfis_select(Xtr, ytr, Xva, yva, method, rules, seed, n_iter):
    best = (np.inf, None)
    for nr in rules:
        m = ANFIS(nr, seed=seed).fit(Xtr, ytr, method, n_iter=n_iter)
        e = mean_squared_error(yva, m.predict(Xva))
        if e < best[0]: best = (e, m, nr)
    return best[1], best[2]

def run_split(X, Y, names, seed, rules, n_iter, models):
    rng = np.random.default_rng(seed); n = len(X); p = rng.permutation(n)
    ntr, nva = int(.7*n), int(.15*n); tr, va, te = p[:ntr], p[ntr:ntr+nva], p[ntr+nva:]
    Xs, mx, sx = std(X, X[tr]); rows = []
    for j, out in enumerate(names):
        y = Y[:, j]; ys, my, sy = std(y[:, None], y[tr, None]); ys = ys.ravel()
        res = {}
        for name in models:
            t0 = time.time()
            if name.startswith("ANFIS"):
                meth = {"ANFIS-PSO": "pso", "ANFIS-GA": "ga", "ANFIS-GRAD": "grad"}[name]
                mdl, nr = anfis_select(Xs[tr], ys[tr], Xs[va], ys[va], meth, rules, seed, n_iter)
                pred = mdl.predict(Xs[te]); extra = nr
            elif name == "MLP-ens":
                preds = []
                for b in range(5):
                    bi = rng.choice(tr, len(tr)); m = MLPRegressor(hidden_layer_sizes=(48, 48), activation="tanh", alpha=1e-4, max_iter=500, random_state=b).fit(Xs[bi], ys[bi]); preds.append(m.predict(Xs[te]))
                pred = np.mean(preds, 0); extra = ""
            elif name == "HistGB":
                pred = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xs[tr], ys[tr]).predict(Xs[te]); extra = ""
            elif name == "GP":
                sub = tr[:min(len(tr), 800)]
                k = C(1.0)*Matern(length_scale=np.ones(X.shape[1]), nu=2.5)+WhiteKernel(1e-3)
                g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=0).fit(Xs[sub], ys[sub])
                g2 = GaussianProcessRegressor(g.kernel_, optimizer=None).fit(Xs[tr], ys[tr]); pred = g2.predict(Xs[te]); extra = ""
            yt = ys[te]
            rows.append(dict(split=seed, output=out, model=name, R2=r2_score(yt, pred), RMSE=np.sqrt(mean_squared_error(yt*sy+my, pred*sy+my)),
                             rules=extra, sec=round(time.time()-t0, 1)))
    return rows

def summarise(df):
    out = []
    for o, g in df.groupby("output"):
        piv = g.pivot(index="split", columns="model", values="R2")
        for m in piv.columns:
            out.append(dict(output=o, model=m, mean_R2=piv[m].mean(), sd=piv[m].std(), median=piv[m].median(), min=piv[m].min()))
        for a, b in [("ANFIS-GA", "ANFIS-PSO"), ("ANFIS-GRAD", "ANFIS-PSO"), ("ANFIS-GRAD", "ANFIS-GA")]:
            if a in piv and b in piv and len(piv) >= 6:
                d = piv[a]-piv[b]
                try: p = wilcoxon(d).pvalue
                except ValueError: p = np.nan
                out.append(dict(output=o, model=f"{a} vs {b}", mean_diff_R2=d.mean(), sd_diff=d.std(), wins_first=int((d > 0).sum()), n_splits=len(d), p_wilcoxon=p))
    return pd.DataFrame(out)

def selftest():
    rng = np.random.default_rng(1); X = rng.random((400, 14))
    y1 = np.sin(3*X[:, 0])+X[:, 1]*X[:, 2]+0.5*X[:, 3]**2; y2 = np.exp(X[:, 4])+X[:, 5]
    return X, np.c_[y1, y2], ["y1", "y2"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--inputs"); ap.add_argument("--outputs")
    ap.add_argument("--splits", type=int, default=20); ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rules", default="3,5,8,12,16,24"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", default="ANFIS-PSO,ANFIS-GA,ANFIS-GRAD,MLP-ens,HistGB,GP"); ap.add_argument("--out", default="benchmark_results.csv")
    a = ap.parse_args()
    if a.selftest: X, Y, names = selftest(); a.splits, a.iters, rules = 6, 15, [3, 5]
    else:
        d = pd.read_csv(a.csv); X = d[a.inputs.split(",")].values; names = a.outputs.split(","); Y = d[names].values; rules = [int(r) for r in a.rules.split(",")]
    allrows = []
    for s in range(a.splits):
        allrows += run_split(X, Y, names, s, rules, a.iters, a.models.split(",")); print("split", s, "done", flush=True)
    df = pd.DataFrame(allrows); df.to_csv(a.out, index=False)
    print(summarise(df).round(4).to_string(index=False))
