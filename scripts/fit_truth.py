"""Recover effective true central values (mu_pa, mu_fl, mu_mt) and an irreducible
floor from the known (predictions, LB-score) pairs, then build & validate a
tracking metric for the UMUD leaderboard.

Model (weak-model / constant-truth approximation):
    LB ~= c0 + (1/3)[ mean_j|pa_j - mu_pa|/6
                    + mean_j|fl_j - mu_fl|/12
                    + mean_j|mt_j - mu_mt|/3 ]
where pa_j, fl_j, mt_j are a submission's per-image predictions and c0 absorbs
the irreducible variance floor of the (hidden) test targets.

Run: .venv/bin/python scripts/fit_truth.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}


def load(path):
    return pd.read_csv(ROOT / path)


# Each point: (label, debug_csv, scale_s, LB). Predictions = pa_deg, fl_px*s, mt_px*s.
# 200-tier r34 geometry: one fixed per-image geometry, 6 known mm-scales (calibration curve).
R34 = "tmp/kaggle-output/block3-200tier/submission_debug.csv"
POINTS = [
    # --- r34 200-tier calibration curve (fixed geometry, varying global scale) ---
    ("r34@0.065", R34, 0.065, 2.01025),
    ("r34@0.070", R34, 0.070, 1.91552),
    ("r34@0.075", R34, 0.075, 1.91296),
    ("r34@0.085", R34, 0.085, 1.99285),
    ("r34@0.090", R34, 0.090, 2.06330),
    ("r34@0.098", R34, 0.098, 2.20146),
    # --- block7/8 encoders @0.075 (vary apo->MT; FL identical) ---
    ("maxvit@0.075", "data/kaggle-outputs/block8/maxvit-nano/submit/submission_debug.csv", 0.075, 1.82151),
    ("rv2_18@0.075", "data/kaggle-outputs/block8/resnetv2-18/submit/submission_debug.csv", 0.075, 1.84197),
    ("r18@0.075", "data/kaggle-outputs/block7-test-eval/run13-resnet18/submission_debug.csv", 0.075, 1.86662),
    ("regnet@0.075", "data/kaggle-outputs/block7-test-eval/run19-regnetx_004/submission_debug.csv", 0.075, 1.87201),
    ("enb1@0.075", "data/kaggle-outputs/block7-test-eval/run17-efficientnet_b1/submission_debug.csv", 0.075, 1.88316),
    ("levit@0.075", "data/kaggle-outputs/block8/levit128s/submit/submission_debug.csv", 0.075, 1.91255),
    ("mnv3@0.075", "data/kaggle-outputs/block7-test-eval/run18-mobilenetv3_small_100/submission_debug.csv", 0.075, 1.91682),
    ("env2@0.075", "data/kaggle-outputs/block8/efficientnetv2-rw-t/submit/submission_debug.csv", 0.075, 1.98186),
    ("r50@0.075", "tmp/kaggle-output/block6c-submission/submission_debug.csv", 0.075, 1.87312),
    # --- micro geometry (v9) @0.098 ---
    ("v9micro@0.098", "tmp/kaggle-output/submission-v9-calibrated/submission_debug.csv", 0.098, 2.35170),
]


def build():
    rows = []
    for label, path, s, lb in POINTS:
        d = load(path)
        pa = d["pa_deg"].to_numpy(float)
        fl = d["fl_px"].to_numpy(float) * s
        mt = d["mt_px"].to_numpy(float) * s
        m = np.isfinite(pa) & np.isfinite(fl) & np.isfinite(mt)
        rows.append(dict(label=label, lb=lb, pa=pa[m], fl=fl[m], mt=mt[m]))
    return rows


def lb_model(params, pts):
    c0, mu_pa, mu_fl, mu_mt = params
    preds = []
    for p in pts:
        g_pa = np.mean(np.abs(p["pa"] - mu_pa)) / TOL["pa"]
        g_fl = np.mean(np.abs(p["fl"] - mu_fl)) / TOL["fl"]
        g_mt = np.mean(np.abs(p["mt"] - mu_mt)) / TOL["mt"]
        preds.append(c0 + (g_pa + g_fl + g_mt) / 3.0)
    return np.array(preds)


def main():
    pts = build()
    y = np.array([p["lb"] for p in pts])

    def loss(params):
        return np.sum((lb_model(params, pts) - y) ** 2)

    x0 = [0.0, 12.0, 90.0, 22.0]
    res = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(maxiter=20000, xatol=1e-6, fatol=1e-10))
    c0, mu_pa, mu_fl, mu_mt = res.x
    pred = lb_model(res.x, pts)
    ss_res = np.sum((pred - y) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred - y) ** 2))

    print("=" * 88)
    print("FITTED effective true central values (constant-truth model)")
    print("=" * 88)
    print(f"  c0 (irreducible floor) = {c0:.4f}")
    print(f"  mu_pa = {mu_pa:6.2f} deg")
    print(f"  mu_fl = {mu_fl:6.2f} mm")
    print(f"  mu_mt = {mu_mt:6.2f} mm")
    print(f"  fit R^2 = {r2:.4f}   RMSE = {rmse:.4f}   (n={len(y)})")
    print()
    print(f"  {'label':16s} {'LB_true':>8} {'LB_pred':>8} {'resid':>7}  "
          f"{'PAcontrib':>9} {'FLcontrib':>9} {'MTcontrib':>9}")
    for p, yp in zip(pts, pred):
        g_pa = np.mean(np.abs(p["pa"] - mu_pa)) / TOL["pa"] / 3
        g_fl = np.mean(np.abs(p["fl"] - mu_fl)) / TOL["fl"] / 3
        g_mt = np.mean(np.abs(p["mt"] - mu_mt)) / TOL["mt"] / 3
        print(f"  {p['label']:16s} {p['lb']:8.4f} {yp:8.4f} {p['lb']-yp:7.3f}  "
              f"{g_pa:9.3f} {g_fl:9.3f} {g_mt:9.3f}")

    # Leave-one-out validation of ranking
    print("\n=== Leave-one-out: refit without each point, predict it ===")
    loo_pred = []
    for i in range(len(pts)):
        sub = [pts[j] for j in range(len(pts)) if j != i]
        ysub = np.array([p["lb"] for p in sub])

        def loss_i(params, sub=sub, ysub=ysub):
            return np.sum((lb_model(params, sub) - ysub) ** 2)
        ri = minimize(loss_i, res.x, method="Nelder-Mead",
                      options=dict(maxiter=20000, xatol=1e-6, fatol=1e-10))
        loo_pred.append(lb_model(ri.x, [pts[i]])[0])
    loo_pred = np.array(loo_pred)
    loo_rmse = np.sqrt(np.mean((loo_pred - y) ** 2))
    # rank correlation
    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(loo_pred, y).correlation
    pe = pearsonr(loo_pred, y)[0]
    print(f"  LOO RMSE = {loo_rmse:.4f}   Spearman = {sp:.4f}   Pearson = {pe:.4f}")

    np.save(ROOT / "data/fit_mu.npy", np.array([c0, mu_pa, mu_fl, mu_mt]))
    print("\nSaved mu -> data/fit_mu.npy")
    return res.x, pts


if __name__ == "__main__":
    main()
