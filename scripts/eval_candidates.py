"""Score candidate post-processing recalibrations of the maxvit (LB 1.82151)
submission using the validated tracking metric, and decompose by target.

Tracking metric: LB ~= c0 + (1/3)[mean|pa-mu_pa|/6 + mean|fl-mu_fl|/12 + mean|mt-mu_mt|/3]
mu/c0 from scripts/fit_truth.py (data/fit_mu.npy).

Run: .venv/bin/python scripts/eval_candidates.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
c0, MU_PA, MU_FL, MU_MT = np.load(ROOT / "data/fit_mu.npy")

MAXVIT = ROOT / "data/kaggle-outputs/block8/maxvit-nano/submit/submission_debug.csv"


def track(pa, fl, mt):
    g_pa = np.mean(np.abs(pa - MU_PA)) / TOL["pa"]
    g_fl = np.mean(np.abs(fl - MU_FL)) / TOL["fl"]
    g_mt = np.mean(np.abs(mt - MU_MT)) / TOL["mt"]
    return c0 + (g_pa + g_fl + g_mt) / 3.0, (g_pa / 3, g_fl / 3, g_mt / 3)


def main():
    d = pd.read_csv(MAXVIT)
    pa0 = d["pa_deg"].to_numpy(float)
    flpx = d["fl_px"].to_numpy(float)
    mtpx = d["mt_px"].to_numpy(float)
    pa_med = np.nanmedian(pa0)
    print(f"mu_pa={MU_PA:.2f}  mu_fl={MU_FL:.2f}  mu_mt={MU_MT:.2f}  c0={c0:.3f}")
    print(f"maxvit pred medians: PA={pa_med:.2f}  FLpx={np.nanmedian(flpx):.1f}  MTpx={np.nanmedian(mtpx):.1f}\n")

    s_fl_cal = MU_FL / np.nanmedian(flpx)   # center FL median on mu_fl
    s_mt_cal = MU_MT / np.nanmedian(mtpx)   # center MT median on mu_mt
    print(f"calibration scales: s_fl={s_fl_cal:.4f}  s_mt={s_mt_cal:.4f}\n")

    candidates = {
        "0 baseline maxvit (0.075/0.075, PA raw)":
            (pa0, flpx * 0.075, mtpx * 0.075),
        "1 FL->0.091 only":
            (pa0, flpx * s_fl_cal, mtpx * 0.075),
        "2 MT->cal only":
            (pa0, flpx * 0.075, mtpx * s_mt_cal),
        "3 FL+MT calibrated, PA raw":
            (pa0, flpx * s_fl_cal, mtpx * s_mt_cal),
        "4 PA->15 const only":
            (np.full_like(pa0, 15.0), flpx * 0.075, mtpx * 0.075),
        "5 FULL: PA=15, FL+MT cal":
            (np.full_like(pa0, 15.0), flpx * s_fl_cal, mtpx * s_mt_cal),
        "6 FULL: PA=mu(17.15), FL+MT cal":
            (np.full_like(pa0, MU_PA), flpx * s_fl_cal, mtpx * s_mt_cal),
        "7 FULL + MT shrink a=0.6":
            (np.full_like(pa0, 15.0), flpx * s_fl_cal,
             MU_MT + 0.6 * (mtpx * s_mt_cal - np.nanmedian(mtpx * s_mt_cal))),
        "8 FULL + FL shrink a=0.7, MT shrink 0.6":
            (np.full_like(pa0, 15.0),
             MU_FL + 0.7 * (flpx * s_fl_cal - np.nanmedian(flpx * s_fl_cal)),
             MU_MT + 0.6 * (mtpx * s_mt_cal - np.nanmedian(mtpx * s_mt_cal))),
    }
    print(f"  {'candidate':42s} {'track_LB':>8}  {'PA':>6} {'FL':>6} {'MT':>6}")
    for name, (pa, fl, mt) in candidates.items():
        s, (gpa, gfl, gmt) = track(pa, fl, mt)
        print(f"  {name:42s} {s:8.3f}  {gpa:6.3f} {gfl:6.3f} {gmt:6.3f}")

    print("\nNOTE: tracking metric is reliable for FL/MT terms (identified by the")
    print("calibration curve) and for ranking realistic-spread submissions. The PA")
    print("term is NOT offline-identifiable (flat sensitivity); its true effect is")
    print("what submission 1 tests. Absolute values understate the irreducible floor.")


if __name__ == "__main__":
    main()
