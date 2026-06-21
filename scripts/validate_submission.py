"""Validate calibrated submissions locally against the maxvit (LB 1.82151) base:
tracking-metric prediction, target distributions, NaN count, format checks.

Run: .venv/bin/python scripts/validate_submission.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import calibrate as cal

ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
c0, MU_PA, MU_FL, MU_MT = np.load(ROOT / "data/fit_mu.npy")
BASE = ROOT / "data/kaggle-outputs/block8/maxvit-nano/submit/submission_debug.csv"


def track(pa, fl, mt):
    g_pa = np.mean(np.abs(pa - MU_PA)) / TOL["pa"]
    g_fl = np.mean(np.abs(fl - MU_FL)) / TOL["fl"]
    g_mt = np.mean(np.abs(mt - MU_MT)) / TOL["mt"]
    return c0 + (g_pa + g_fl + g_mt) / 3.0


def main():
    df = pd.read_csv(BASE)
    print(f"Base: maxvit (real LB 1.82151), tracking metric mu=({MU_PA:.1f},{MU_FL:.1f},{MU_MT:.1f})")
    print(f"S_FL={cal.S_FL:.4f}  S_MT={cal.S_MT:.4f}\n")
    print(f"  {'variant':38s} {'trackLB':>8} {'PA_med':>6} {'FL_med':>6} {'MT_med':>6} {'MT_std':>6} {'NaN':>4}")

    variants = [
        ("baseline maxvit (0.075, PA raw)", None, 1.0, "raw075"),
        ("S1: FL+MT recenter, PA raw", None, 1.0, "cal"),
        ("S1: FL+MT recenter +MTshrink.5, PA raw", None, 0.5, "cal"),
        ("S1b: + PA->10", 10.0, 0.5, "cal"),
        ("S1c: + PA->13", 13.0, 0.5, "cal"),
        ("S2: PA->15, MTshrink.5", 15.0, 0.5, "cal"),
        ("S2b: PA->17, MTshrink.4", 17.15, 0.4, "cal"),
        ("S2c: PA->17, MTshrink.3", 17.15, 0.3, "cal"),
    ]
    for name, pa_t, shrink, mode in variants:
        if mode == "raw075":
            pa = df["pa_deg"].to_numpy(float)
            fl = df["fl_px"].to_numpy(float) * 0.075
            mt = df["mt_px"].to_numpy(float) * 0.075
        else:
            pa, fl, mt = cal.calibrate_frame(df, pa_target=pa_t, mt_shrink=shrink)
        nan = int(np.sum(~np.isfinite(pa)) + np.sum(~np.isfinite(fl)) + np.sum(~np.isfinite(mt)))
        print(f"  {name:38s} {track(pa,fl,mt):8.3f} {np.median(pa):6.1f} "
              f"{np.median(fl):6.1f} {np.median(mt):6.1f} {np.std(mt):6.2f} {nan:4d}")

    # Build the actual S1 submission.csv and check format
    pa, fl, mt = cal.calibrate_frame(df, pa_target=13.0, mt_shrink=0.5)
    out = pd.DataFrame({"image_id": df["image_id"], "pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
    assert len(out) == len(df)
    assert out[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all(), "NaN in submission!"
    outp = ROOT / "data/submission_s1_local_preview.csv"
    out.to_csv(outp, index=False)
    print(f"\nWrote preview {outp.name} ({len(out)} rows, 0 NaN). Sample:")
    print(out.head(3).to_string(index=False))
    print("\nReminder: tracking metric understates the irreducible floor for the")
    print("PA term (offline-unidentifiable). FL/MT terms are LB-identified & reliable.")


if __name__ == "__main__":
    main()
