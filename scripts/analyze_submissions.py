"""Decompose scored submissions: isolate per-target error structure and
compare predicted target distributions to GT-mask-derived distributions.

Run: .venv/bin/python scripts/analyze_submissions.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (label, LB_score, debug_csv_path)
SUBMISSIONS = [
    ("maxvit_nano", 1.82151, "data/kaggle-outputs/block8/maxvit-nano/submit/submission_debug.csv"),
    ("resnetv2_18", 1.84197, "data/kaggle-outputs/block8/resnetv2-18/submit/submission_debug.csv"),
    ("resnet18", 1.86662, "data/kaggle-outputs/block7-test-eval/run13-resnet18/submission_debug.csv"),
    ("regnetx_004", 1.87201, "data/kaggle-outputs/block7-test-eval/run19-regnetx_004/submission_debug.csv"),
    ("efficientnet_b1", 1.88316, "data/kaggle-outputs/block7-test-eval/run17-efficientnet_b1/submission_debug.csv"),
    ("levit128s", 1.91255, "data/kaggle-outputs/block8/levit128s/submit/submission_debug.csv"),
    ("mobilenetv3", 1.91682, "data/kaggle-outputs/block7-test-eval/run18-mobilenetv3_small_100/submission_debug.csv"),
    ("efficientnetv2_rw_t", 1.98186, "data/kaggle-outputs/block8/efficientnetv2-rw-t/submit/submission_debug.csv"),
    ("v9_calibrated", 2.35170, "tmp/kaggle-output/submission-v9-calibrated/submission_debug.csv"),
]

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def load_sub(path):
    df = pd.read_csv(ROOT / path)
    return df


def describe(col):
    a = np.asarray(col, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(n=0)
    return dict(
        n=len(a),
        mean=float(np.mean(a)),
        median=float(np.median(a)),
        mad=float(np.median(np.abs(a - np.median(a)))),
        std=float(np.std(a)),
        p05=float(np.percentile(a, 5)),
        p95=float(np.percentile(a, 95)),
        min=float(np.min(a)),
        max=float(np.max(a)),
    )


def main():
    print("=" * 100)
    print("PART 1: Per-submission predicted distributions (309 test rows each)")
    print("=" * 100)
    subs = {}
    for label, lb, path in SUBMISSIONS:
        df = load_sub(path)
        subs[label] = (lb, df)
        nan_counts = {c: int(df[c].isna().sum()) for c in ("pa_deg", "fl_mm", "mt_mm")}
        print(f"\n### {label}  LB={lb}  rows={len(df)}  NaN={nan_counts}")
        for c in ("pa_deg", "fl_mm", "mt_mm"):
            d = describe(df[c])
            print(f"  {c:7s} median={d['median']:8.3f} mean={d['mean']:8.3f} "
                  f"mad={d['mad']:7.3f} std={d['std']:7.3f} "
                  f"p05={d['p05']:7.2f} p95={d['p95']:8.2f}")

    print("\n" + "=" * 100)
    print("PART 2: Are PA / FL identical across encoder swaps? (shared fasc model)")
    print("=" * 100)
    ref_label = "maxvit_nano"
    _, ref = subs[ref_label]
    ref = ref.set_index("image_id")
    for label, (lb, df) in subs.items():
        if label == ref_label:
            continue
        d = df.set_index("image_id")
        common = ref.index.intersection(d.index)
        for c in ("pa_deg", "fl_mm", "mt_mm"):
            diff = np.abs(ref.loc[common, c].values - d.loc[common, c].values)
            diff = diff[np.isfinite(diff)]
            maxd = float(np.max(diff)) if len(diff) else float("nan")
            meand = float(np.mean(diff)) if len(diff) else float("nan")
            tag = "IDENTICAL" if maxd < 1e-6 else f"differs max={maxd:.3f} mean={meand:.4f}"
            print(f"  {ref_label} vs {label:20s} {c:7s}: {tag}")
        print()


if __name__ == "__main__":
    main()
