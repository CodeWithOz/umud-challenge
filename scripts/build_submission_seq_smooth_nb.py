"""Generate notebooks/submission-seq-smooth/submission-seq-smooth.ipynb.

Block 18 candidate: hidden-safe sequence smoothing on the Block 15 qdc+cxs8
blend. The test images are numbered consecutively, and public baselines note
5-frame sequence structure. This notebook computes both components from mounted
images, blends them, then smooths predictions over a rolling 5-image window.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import build_submission_nb as prod


ROOT = Path(__file__).resolve().parents[1]
MODULE_SRC = (ROOT / "scripts/quickdirty_geometry.py").read_text()
BLEND_QD_WEIGHT = 0.70
ROLL_WINDOW = 5


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


cells = copy.deepcopy(prod.cells)
for cell in cells:
    if cell.get("cell_type") in {"code", "markdown"}:
        cell["source"] = [
            line.replace("apo_gray55_line_200_cxs.pkl", "apo_gray55_line_200_cxs8.pkl")
            .replace("convnext_small + Block 9 s2 (LB 1.04862)", "convnext_small 8ep + Block 9 s2")
            .replace("production cxs-s2", "cxs8-s2")
            .replace("cxs-s2", "cxs8-s2")
            for line in cell["source"]
        ]
cells[0] = md(
    """# UMUD — Submission Sequence Smooth QDC + CXS8

GPU notebook. Runs the hidden-safe cxs8 segment-then-measure pipeline, computes
calibrated quick-dirty image geometry on the same mounted images, blends them as
Block 15 did, then applies rolling 5-image smoothing by numeric image order.

`submission.csv` is the rolling-mean smoothed blend:

`smooth5_mean(0.70 * quickdirty_cal + 0.30 * cxs8_s2)`.

This does not embed public-test predictions; all component predictions are
computed from the images mounted in the notebook."""
)

cells.extend(
    [
        md("## Embedded quickdirty geometry module"),
        code(MODULE_SRC),
        md("## Blend cxs8-s2 with calibrated quickdirty and smooth by sequence order"),
        code(
            f"""import re


# pred_df currently contains the calibrated cxs8-s2 output from the prior cells.
cxs_df = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
cxs_df = cxs_df.rename(columns={{
    "pa_deg": "pa_cxs",
    "fl_mm": "fl_cxs",
    "mt_mm": "mt_cxs",
}})

qd_raw = predict_quickdirty(COMPETITION_DIR)
qd_cal = calibrate_quickdirty(qd_raw)
qd_df = qd_cal[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
qd_df = qd_df.rename(columns={{
    "pa_deg": "pa_qd",
    "fl_mm": "fl_qd",
    "mt_mm": "mt_qd",
}})

blend = cxs_df.merge(qd_df, on="image_id", how="inner", validate="one_to_one")
assert len(blend) == len(cxs_df) == len(qd_df), (len(blend), len(cxs_df), len(qd_df))

QD_WEIGHT = {BLEND_QD_WEIGHT}
CXS_WEIGHT = 1.0 - QD_WEIGHT
blend["pa_deg"] = QD_WEIGHT * blend["pa_qd"] + CXS_WEIGHT * blend["pa_cxs"]
blend["fl_mm"] = QD_WEIGHT * blend["fl_qd"] + CXS_WEIGHT * blend["fl_cxs"]
blend["mt_mm"] = QD_WEIGHT * blend["mt_qd"] + CXS_WEIGHT * blend["mt_cxs"]

def _image_num(image_id):
    m = re.search(r"(\\d+)", str(image_id))
    if not m:
        raise ValueError(f"Cannot parse image number from {{image_id!r}}")
    return int(m.group(1))

ROLL_WINDOW = {ROLL_WINDOW}
value_cols = ["pa_deg", "fl_mm", "mt_mm"]
smooth = blend.sort_values("image_id").copy()
smooth["image_num"] = smooth["image_id"].map(_image_num)
smooth = smooth.sort_values("image_num").reset_index(drop=True)

smooth_mean = smooth.copy()
smooth_median = smooth.copy()
for col in value_cols:
    smooth_mean[col] = smooth[col].rolling(ROLL_WINDOW, center=True, min_periods=1).mean()
    smooth_median[col] = smooth[col].rolling(ROLL_WINDOW, center=True, min_periods=1).median()

submit = smooth_mean[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
assert len(submit) == len(smooth) == 309, len(submit)
assert submit["image_id"].is_unique
assert submit[value_cols].notna().all().all()

submit.to_csv("/kaggle/working/submission.csv", index=False)
blend.to_csv("/kaggle/working/submission_debug_blend_unsmoothed.csv", index=False)
smooth_mean.to_csv("/kaggle/working/submission_debug_smooth5_mean.csv", index=False)
smooth_median.to_csv("/kaggle/working/submission_debug_smooth5_median.csv", index=False)
qd_raw.to_csv("/kaggle/working/submission_quickdirty_raw_debug.csv", index=False)
qd_cal.to_csv("/kaggle/working/submission_quickdirty_cal_debug.csv", index=False)

print(f"Overwrote submission.csv with rolling-{{ROLL_WINDOW}} mean sequence-smoothed blend")
print(f"Blend weights: QD={{QD_WEIGHT:.2f}}, CXS={{CXS_WEIGHT:.2f}}")
print("Rows:", len(submit), "NaNs:", int(submit.isna().sum().sum()))
display(submit[value_cols].describe().round(3))
display(submit.head())
"""
        ),
    ]
)


def main() -> None:
    out = ROOT / "notebooks/submission-seq-smooth"
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
    (out / "submission-seq-smooth.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-seq-smooth",
        "title": "UMUD Submission Sequence Smooth",
        "code_file": "submission-seq-smooth.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [
            "ucheozoemena/umud-train-mounted-phase-3",
            "ucheozoemena/umud-train-apo-gray55-phase-3",
        ],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
