"""Identifiability check: how strongly do the LB data prefer a high mu_pa
(PA miscalibration) vs. an irreducible floor (c0) with PA left near ~4 deg?

For each fixed mu_pa, refit (c0, mu_fl, mu_mt) and report best RMSE.
Run: .venv/bin/python scripts/sensitivity_pa.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import fit_truth as ft

TOL = ft.TOL


def main():
    pts = ft.build()
    y = np.array([p["lb"] for p in pts])

    def lb_model(c0, mu_pa, mu_fl, mu_mt):
        out = []
        for p in pts:
            g_pa = np.mean(np.abs(p["pa"] - mu_pa)) / TOL["pa"]
            g_fl = np.mean(np.abs(p["fl"] - mu_fl)) / TOL["fl"]
            g_mt = np.mean(np.abs(p["mt"] - mu_mt)) / TOL["mt"]
            out.append(c0 + (g_pa + g_fl + g_mt) / 3.0)
        return np.array(out)

    print(f"  {'mu_pa':>6} {'c0':>7} {'mu_fl':>7} {'mu_mt':>7} {'RMSE':>7}")
    for mu_pa in [3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 25]:
        def loss(p, mu_pa=mu_pa):
            return np.sum((lb_model(p[0], mu_pa, p[1], p[2]) - y) ** 2)
        r = minimize(loss, [0.5, 77, 20], method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-7, fatol=1e-12))
        c0, mu_fl, mu_mt = r.x
        rmse = np.sqrt(np.mean((lb_model(c0, mu_pa, mu_fl, mu_mt) - y) ** 2))
        print(f"  {mu_pa:6.1f} {c0:7.3f} {mu_fl:7.2f} {mu_mt:7.2f} {rmse:7.4f}")

    print("\nNote: c0 is the implied irreducible floor. A physically sensible fit")
    print("needs c0 >= 0 and ideally small. If high mu_pa forces c0<0, that's")
    print("the model 'paying back' an over-attributed PA term.")


if __name__ == "__main__":
    main()
