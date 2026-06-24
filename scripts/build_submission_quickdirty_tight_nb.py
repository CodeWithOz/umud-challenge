"""Generate notebooks/submission-quickdirty-tight/submission-quickdirty-tight.ipynb.

Block 16 candidate: tighter calibrated quick-dirty image geometry.

Blocks 14/15 showed that raw quick-dirty has useful signal, but the old proxy was
too optimistic. This candidate tests whether the remaining per-image variation is
still too noisy by shrinking all three targets more aggressively toward the best
post-Block-15 centres.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_SRC = (ROOT / "scripts/quickdirty_geometry.py").read_text()


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


cells = [
    md(
        """# UMUD — Submission QuickDirty Tight

CPU notebook. This Block 16 candidate runs the same quick-dirty image-geometry
estimator as Blocks 13/14, then applies a tighter center/shrink calibration.

Rationale:

- Block 13 raw quick-dirty scored poorly because its FL/MT scale and spread were too large.
- Block 14 calibration improved to 0.96243.
- Block 15 blend improved to 0.93837, but the qdc/cxs8 blend curve is already near its optimum.

This notebook tests the next hypothesis: the remaining quick-dirty per-image
movement is still too noisy. It keeps a small amount of variation but pulls every
target closer to the best current centres."""
    ),
    md("## Embedded quickdirty geometry module"),
    code(MODULE_SRC),
    md("## Tight calibration"),
    code(
        """from pathlib import Path

import numpy as np


COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
OUT = Path("/kaggle/working")


def calibrate_quickdirty_tight(raw_df):
    out = raw_df.copy()
    raw_pa = out["pa_deg"].astype(float)
    raw_fl = out["fl_mm"].astype(float)
    raw_mt = out["mt_mm"].astype(float)

    out["pa_deg_raw"] = raw_pa
    out["fl_mm_raw"] = raw_fl
    out["mt_mm_raw"] = raw_mt

    # Raw medians are fixed from the public-test quickdirty run, not recomputed
    # dynamically from hidden data. The centres come from the best Block 15 output.
    out["pa_deg"] = np.clip(18.0 + 0.25 * (raw_pa - QD_RAW_PA_MEDIAN), 5.0, 45.0)
    out["fl_mm"] = 75.2 + 0.10 * (raw_fl - QD_RAW_FL_MEDIAN)
    out["mt_mm"] = 20.4 + 0.10 * (raw_mt - QD_RAW_MT_MEDIAN)

    out["fl_mm"] = np.clip(out["fl_mm"], 30.0, 200.0)
    out["mt_mm"] = np.clip(out["mt_mm"], 10.0, 50.0)
    assert out[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()
    return out


QD_RAW_PA_MEDIAN = 11.310210727446622

raw_df = predict_quickdirty(COMPETITION_DIR)
pred_df = calibrate_quickdirty_tight(raw_df)

raw_df.to_csv(OUT / "submission_raw_debug.csv", index=False)
pred_df.to_csv(OUT / "submission_debug.csv", index=False)
write_submission(pred_df, OUT / "submission.csv")

submission = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
assert len(submission) == 309, len(submission)
assert submission["image_id"].is_unique
assert submission[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()

print("Wrote submission.csv, submission_debug.csv, and submission_raw_debug.csv")
print("Rows:", len(submission), "NaNs:", int(submission.isna().sum().sum()))
display(submission.head())
display(pred_df[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
display(raw_df[["pa_deg", "fl_mm", "mt_mm", "mm_per_px"]].describe().round(3))
display(pred_df.groupby(["height", "width"]).size().rename("n").reset_index())
"""
    ),
]


def main() -> None:
    out = ROOT / "notebooks/submission-quickdirty-tight"
    out.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    (out / "submission-quickdirty-tight.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-quickdirty-tight",
        "title": "UMUD Submission QuickDirty Tight",
        "code_file": "submission-quickdirty-tight.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "None",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
