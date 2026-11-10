#!/usr/bin/env python3
"""Independent check of the Chebyshev solver: re-solve random rows of the data set with SciPy solve_bvp.
Usage: python compare_dataset.py --csv dataset.csv --n 40
CSV columns (rename in COLS if yours differ): Re,A1,A2,Om,M,chi,beta,Lam,Rd,gt,Ec,em,phi1,phi2,Cf1,Cf2,Nu1,Nu2
Fixed in the study: Pr=6.2, ek=0.2, z1=z2=0.5, tw=0.1."""
import argparse, numpy as np, pandas as pd
from independent_solver import solve
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
d = pd.read_csv(a.csv); idx = np.random.default_rng(a.seed).choice(len(d), a.n, replace=False)
IN = ["Re","A1","A2","Om","M","chi","beta","Lam","Rd","gt","Ec","em","phi1","phi2"]; OUT = ["Cf1","Cf2","Nu1","Nu2"]
errs, fail = [], 0
for i in idx:
    row = d.iloc[i]
    try: r = solve(**{k: float(row[k]) for k in IN})
    except Exception as e: fail += 1; continue
    errs.append([abs(r[k]-row[k])/abs(row[k]) for k in OUT])
E = np.array(errs); print(f"solved {len(E)} of {a.n} (failed to converge: {fail})")
if len(E): print("max relative deviation per output:", dict(zip(OUT, E.max(0).round(8))), " median:", dict(zip(OUT, np.median(E, 0).round(8))))
