"""Compute the geometry pipeline on GROUND-TRUTH training masks to recover the
true (PA, FL_px, MT_px) target distributions, and evaluate the optimal-constant
predictor under the UMUD metric at various mm scales.

Run: .venv/bin/python scripts/gt_geometry_dist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import segment_geometry as sg  # noqa: E402

APO_IMG = ROOT / "data/umud-challenge/apo_imgs_v1/apo_images_new_model_v1"
APO_MASK = ROOT / "data/umud-challenge/apo_masks_v1/apo_masks_new_model_v1"
FASC_MASK = ROOT / "data/umud-challenge/fasc_masks_v1/fasc_masks_new_model_v1"

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def main():
    apo_df = pd.read_csv(ROOT / "tmp/geometry-local-output/train_apo_all.csv")
    fasc_df = pd.read_csv(ROOT / "tmp/geometry-local-output/train_fasc_clean.csv")
    apo_files = set(apo_df["filename"])
    fasc_files = set(fasc_df["filename"])
    dual = sorted(apo_files & fasc_files)
    print(f"apo={len(apo_files)} fasc={len(fasc_files)} dual-track={len(dual)}")

    rows = []
    for i, fn in enumerate(dual):
        ap_img = APO_IMG / fn
        ap_msk = APO_MASK / fn
        fa_msk = FASC_MASK / fn
        if not (ap_img.exists() and ap_msk.exists() and fa_msk.exists()):
            continue
        try:
            img = sg.load_gray(ap_img)
            h, w = img.shape
            apo_raw = sg.load_mask(ap_msk)
            fasc_raw = sg.load_mask(fa_msk)
            mm = sg.gt_geometry_from_masks(fasc_raw, apo_raw, (h, w), mm_per_pixel=1.0)
            rows.append({
                "filename": fn, "img_h": h, "img_w": w,
                "pa_deg": mm["pa_deg"], "fl_px": mm["fl_mm"], "mt_px": mm["mt_mm"],
            })
        except Exception as e:
            rows.append({"filename": fn, "err": str(e)})
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(dual)}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/gt_geometry_dual.csv", index=False)
    ok = df.dropna(subset=["pa_deg", "fl_px", "mt_px"]) if "pa_deg" in df else df
    print(f"\nComputed geometry on {len(ok)}/{len(df)} dual-track images "
          f"(finite PA/FL/MT). Saved -> data/gt_geometry_dual.csv")

    print("\n=== GT distributions (px for FL/MT, deg for PA) ===")
    for c, unit in [("pa_deg", "deg"), ("fl_px", "px"), ("mt_px", "px")]:
        a = ok[c].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"  {c:6s} [{unit}] n={len(a)} median={np.median(a):8.2f} mean={np.mean(a):8.2f} "
              f"p05={np.percentile(a,5):7.2f} p25={np.percentile(a,25):7.2f} "
              f"p75={np.percentile(a,75):7.2f} p95={np.percentile(a,95):8.2f}")

    # Resolution cohorts
    print("\n=== by resolution cohort ===")
    ok2 = ok.copy()
    ok2["res"] = ok2["img_h"].astype(str) + "x" + ok2["img_w"].astype(str)
    for res, g in ok2.groupby("res"):
        if len(g) < 10:
            continue
        print(f"  {res:10s} n={len(g):4d}  PA_med={g['pa_deg'].median():6.2f}  "
              f"FLpx_med={g['fl_px'].median():7.1f}  MTpx_med={g['mt_px'].median():7.1f}")

    # Optimal-constant predictor score under UMUD metric, sweeping mm scale.
    # PA constant = median PA (deg). FL/MT constant = median * scale (mm).
    print("\n=== Optimal-CONSTANT predictor: UMUD score vs mm-scale ===")
    print("  (predict the GT median for every image; MAE = mean|x - median|)")
    pa = ok["pa_deg"].to_numpy(float); pa = pa[np.isfinite(pa)]
    flpx = ok["fl_px"].to_numpy(float); flpx = flpx[np.isfinite(flpx)]
    mtpx = ok["mt_px"].to_numpy(float); mtpx = mtpx[np.isfinite(mtpx)]
    pa_med = np.median(pa)
    pa_mae = np.mean(np.abs(pa - pa_med))
    print(f"  PA: median={pa_med:.2f} deg, constant-MAE={pa_mae:.3f} deg -> norm {pa_mae/6:.3f}")
    print(f"  {'scale':>7} {'FLmm_med':>9} {'FL_nMAE':>8} {'MTmm_med':>9} {'MT_nMAE':>8} {'PA_nMAE':>8} {'SCORE':>8}")
    for scale in [0.060, 0.065, 0.070, 0.075, 0.080, 0.090, 0.100, 0.110, 0.120, 0.135]:
        fl = flpx * scale; mt = mtpx * scale
        fl_med, mt_med = np.median(fl), np.median(mt)
        fl_nmae = np.mean(np.abs(fl - fl_med)) / TOL["fl_mm"]
        mt_nmae = np.mean(np.abs(mt - mt_med)) / TOL["mt_mm"]
        pa_nmae = pa_mae / TOL["pa_deg"]
        score = (pa_nmae + fl_nmae + mt_nmae) / 3
        print(f"  {scale:7.3f} {fl_med:9.2f} {fl_nmae:8.3f} {mt_med:9.2f} {mt_nmae:8.3f} {pa_nmae:8.3f} {score:8.4f}")


if __name__ == "__main__":
    main()
