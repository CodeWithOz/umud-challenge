"""Generate notebooks/submission-quickdirty-cal/submission-quickdirty-cal.ipynb.

Block 14 candidate: calibrated quick-dirty image geometry. This keeps the
AmbrosM-style per-image measurements but applies fixed PA shift + FL/MT
center-shrink calibration found from the Block 13 raw output distribution and
the leaderboard-fit centers.
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
        """# UMUD — Submission QuickDirty Calibrated

CPU notebook. This candidate runs the quick-dirty image-geometry estimator, then
applies a fixed center/shrink calibration:

- `PA = clip(raw_PA + 5, 5, 45)`
- `FL = 76.9 + 0.20 * (raw_FL - 118.9114)`
- `MT = 19.76 + 0.20 * (raw_MT - 24.3678)`

The goal is to preserve the per-image signal from quick-dirty while correcting
its raw high FL/MT centers and excessive spread."""
    ),
    md("## Embedded quickdirty geometry module"),
    code(MODULE_SRC),
    md("## Run inference, calibrate, and write submission"),
    code(
        """from pathlib import Path

COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
OUT = Path("/kaggle/working")

raw_df = predict_quickdirty(COMPETITION_DIR)
pred_df = calibrate_quickdirty(raw_df)

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
    out = ROOT / "notebooks/submission-quickdirty-cal"
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
    (out / "submission-quickdirty-cal.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-quickdirty-cal",
        "title": "UMUD Submission QuickDirty Cal",
        "code_file": "submission-quickdirty-cal.ipynb",
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
