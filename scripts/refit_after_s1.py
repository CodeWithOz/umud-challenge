"""After submission 1 (PA->13) gets an LB score, fold it into the fit to pin the
true PA center (the first point with PA != ~4 deg breaks the identifiability),
then recommend submission-2 parameters.

Usage: .venv/bin/python scripts/refit_after_s1.py <S1_LB_SCORE> [s1_debug_csv]
Default s1 debug: data/kaggle-outputs/block9-s1/submission_debug.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import fit_truth as ft

ROOT = Path(__file__).resolve().parent.parent
TOL = ft.TOL


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    s1_lb = float(sys.argv[1])
    s1_csv = sys.argv[2] if len(sys.argv) > 2 else "data/kaggle-outputs/block9-s1/submission_debug.csv"
    s1_csv = ROOT / s1_csv

    pts = ft.build()
    # Add the new point: submission 1's actual calibrated predictions vs its LB.
    d = pd.read_csv(s1_csv)
    pa = d["pa_deg"].to_numpy(float)   # already calibrated (==13 const) in submission.csv path
    # debug keeps calibrated pa_deg/fl_mm/mt_mm; use them directly
    fl = d["fl_mm"].to_numpy(float)
    mt = d["mt_mm"].to_numpy(float)
    m = np.isfinite(pa) & np.isfinite(fl) & np.isfinite(mt)
    pts.append(dict(label="s1_PA13", lb=s1_lb, pa=pa[m], fl=fl[m], mt=mt[m]))
    y = np.array([p["lb"] for p in pts])

    def loss(params):
        return np.sum((ft.lb_model(params, pts) - y) ** 2)

    res = minimize(loss, [0.0, 13, 77, 20], method="Nelder-Mead",
                   options=dict(maxiter=30000, xatol=1e-7, fatol=1e-12))
    c0, mu_pa, mu_fl, mu_mt = res.x
    pred = ft.lb_model(res.x, pts)
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    print(f"REFIT with s1 (PA=13, LB={s1_lb}):")
    print(f"  mu_pa={mu_pa:.2f}  mu_fl={mu_fl:.2f}  mu_mt={mu_mt:.2f}  c0={c0:.3f}  RMSE={rmse:.4f}")
    print(f"  s1 predicted {pred[-1]:.3f} vs actual {s1_lb}")
    print()
    print("Recommended submission 2: set PA -> mu_pa (rounded), keep FL/MT calibration,")
    print(f"  i.e. PA_TARGET = {mu_pa:.1f}; consider MT_SHRINK 0.3-0.4 if MT still loose.")


if __name__ == "__main__":
    main()
