"""Design submission 2: evaluate candidates against the refit truth model that
now includes the real s1 (PA=13 -> LB 1.07757) data point. Report predicted LB
and per-target contributions under BOTH the conservative (pre-s1) and refit
centers, so the chosen shot is robust to center uncertainty.

Run: .venv/bin/python scripts/eval_s2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import calibrate as cal

ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
BASE = ROOT / "data/kaggle-outputs/block9-s1/submission_debug.csv"  # has raw fl_px/mt_px

# Two center estimates: conservative (pre-s1 fit) and refit (post-s1).
CENTERS = {
    "conservative": dict(c0=0.024, mu_pa=17.15, mu_fl=76.89, mu_mt=19.76),
    "refit": dict(c0=-0.036, mu_pa=20.85, mu_fl=72.21, mu_mt=22.17),
}


def track(pa, fl, mt, ctr):
    g_pa = np.mean(np.abs(pa - ctr["mu_pa"])) / TOL["pa"]
    g_fl = np.mean(np.abs(fl - ctr["mu_fl"])) / TOL["fl"]
    g_mt = np.mean(np.abs(mt - ctr["mu_mt"])) / TOL["mt"]
    return ctr["c0"] + (g_pa + g_fl + g_mt) / 3.0, (g_pa / 3, g_fl / 3, g_mt / 3)


def main():
    d = pd.read_csv(BASE)
    flpx = d["fl_px"].to_numpy(float)
    mtpx = d["mt_px"].to_numpy(float)

    # s1 actually-submitted config, as anchor (real LB 1.07757).
    pa_s1 = np.full(len(d), 13.0)
    fl_s1 = np.where(np.isfinite(flpx), flpx * cal.S_FL, cal.MU_FL)
    mt_s1 = np.where(np.isfinite(mtpx), cal.MU_MT + 0.5 * (mtpx * cal.S_MT - cal.MU_MT), cal.MU_MT)

    # Candidate builder: pa_target const, fl scale, mt center+shrink.
    def build(pa_t, s_fl, mt_center, shrink):
        pa = np.full(len(d), float(pa_t))
        fl = np.where(np.isfinite(flpx), flpx * s_fl, mt_center)  # fallback minor
        fl = np.where(np.isfinite(flpx), flpx * s_fl, 75.0)
        s_mt = mt_center / 296.4
        mt = np.where(np.isfinite(mtpx), mt_center + shrink * (mtpx * s_mt - mt_center), mt_center)
        return pa, fl, mt

    cands = {
        "s1 ACTUAL (LB 1.07757)": (pa_s1, fl_s1, mt_s1),
        "A PA17 FL0.088 MT21 shr.5": build(17, 0.088, 21.0, 0.5),
        "B PA18 FL0.087 MT21.5 shr.45": build(18, 0.087, 21.5, 0.45),
        "C PA19 FL0.086 MT22 shr.4": build(19, 0.086, 22.0, 0.4),
        "D PA20 FL0.086 MT22 shr.4": build(20, 0.086, 22.0, 0.4),
        "E PA17 FL0.091 MT19.8 shr.5 (s1+PA)": build(17, 0.0908, 19.76, 0.5),
        "F PA18 FL0.088 MT21 shr.5": build(18, 0.088, 21.0, 0.5),
    }

    for label in ("conservative", "refit"):
        ctr = CENTERS[label]
        print(f"\n=== centers: {label}  mu=({ctr['mu_pa']},{ctr['mu_fl']},{ctr['mu_mt']}) ===")
        print(f"  {'candidate':38s} {'predLB':>7}  {'PA':>5} {'FL':>5} {'MT':>5}  {'PAmed':>5} {'MTmed':>5} {'MTstd':>5}")
        for name, (pa, fl, mt) in cands.items():
            s, (gpa, gfl, gmt) = track(pa, fl, mt, ctr)
            print(f"  {name:38s} {s:7.3f}  {gpa:5.3f} {gfl:5.3f} {gmt:5.3f}  "
                  f"{np.median(pa):5.1f} {np.median(mt):5.1f} {np.std(mt):5.2f}")

    print("\nAnchor: s1 ACTUAL scored 1.07757. Read candidates by how much they beat")
    print("the s1 row under BOTH center sets; prefer ones that improve under both.")


if __name__ == "__main__":
    main()
