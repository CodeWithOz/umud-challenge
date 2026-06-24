"""Generate a hidden-safe qdc+cxs5 blend submission notebook.

Block 19 candidate: refresh the stronger cxs5 segment-then-measure model in a
separate train kernel, then blend it with calibrated quick-dirty geometry.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import build_submission_nb as prod


ROOT = Path(__file__).resolve().parents[1]
MODULE_SRC = (ROOT / "scripts/quickdirty_geometry.py").read_text()
TRAIN_KERNEL_SLUG = "umud-train-apo-gray55-cxs5-refresh"
BLEND_QD_WEIGHT = 0.72


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


def main() -> None:
    cells = copy.deepcopy(prod.cells)
    for cell in cells:
        if cell.get("cell_type") in {"code", "markdown"}:
            cell["source"] = [
                line.replace("umud-train-apo-gray55-phase-3", TRAIN_KERNEL_SLUG)
                .replace(
                    "Phase 4 Block 10 prod — convnext_small + Block 9 s2 (LB 1.04862)",
                    "Block 19 — refreshed cxs5 + Block 9 s2",
                )
                .replace("production cxs-s2", "refreshed cxs5-s2")
                .replace("cxs-s2", "cxs5-s2")
                for line in cell["source"]
            ]

    cells[0] = md(
        """# UMUD — Submission Blend QuickDirtyCal + CXS5 Refresh

GPU notebook. Runs the refreshed cxs5-s2 segment-then-measure pipeline, computes
calibrated quick-dirty image geometry on the same mounted test images, and writes
`submission.csv` as:

`0.72 * quickdirty_cal + 0.28 * refreshed_cxs5_s2`.

This is a real inference path, not a public-test CSV lookup: both components are
computed from the images present in `/kaggle/input/competitions/...`."""
    )

    cells.extend(
        [
            md("## Embedded quickdirty geometry module"),
            code(MODULE_SRC),
            md("## Blend refreshed cxs5-s2 with calibrated quickdirty"),
            code(
                f"""# pred_df currently contains the calibrated refreshed cxs5-s2 output from the prior cells.
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

submit = blend[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
assert submit[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()

submit.to_csv("/kaggle/working/submission.csv", index=False)
blend.to_csv("/kaggle/working/submission_debug_blend.csv", index=False)
qd_raw.to_csv("/kaggle/working/submission_quickdirty_raw_debug.csv", index=False)
qd_cal.to_csv("/kaggle/working/submission_quickdirty_cal_debug.csv", index=False)

print(f"Overwrote submission.csv with blend: QD={{QD_WEIGHT:.2f}}, CXS={{CXS_WEIGHT:.2f}}")
print("Rows:", len(submit), "NaNs:", int(submit.isna().sum().sum()))
display(submit[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
display(submit.head())
"""
            ),
        ]
    )

    out = ROOT / "notebooks/submission-blend-qdc-cxs5"
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
    (out / "submission-blend-qdc-cxs5.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-blend-qdc-cxs5",
        "title": "UMUD Submission Blend QDC CXS5",
        "code_file": "submission-blend-qdc-cxs5.ipynb",
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
            f"ucheozoemena/{TRAIN_KERNEL_SLUG}",
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
