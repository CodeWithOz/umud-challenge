"""Generate notebooks/submission-quickdirty/submission-quickdirty.ipynb.

Block 13 candidate: AmbrosM-style image-geometry baseline using real viewport
scale, brightness aponeurosis depths, texture-correlation PA, and FL = MT/sin(PA).
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
        """# UMUD — Submission QuickDirty Block 13

CPU notebook. This is a direct code-competition submission candidate based on
image metadata and ultrasound texture, not segmentation masks:

1. Recover per-image pixel scale and ultrasound viewport from ruler/tick marks.
2. Estimate superficial/deep aponeurosis depths from horizontal brightness.
3. Estimate fascicle slope from shifted vertical-patch correlation.
4. Predict `MT`, `PA`, and `FL = MT / sin(PA)`.

This is adapted from AmbrosM's public Kaggle notebook `umud-quick-and-dirty`,
with format assertions and debug output added for this repo."""
    ),
    md("## Embedded quickdirty geometry module"),
    code(MODULE_SRC),
    md("## Run inference and write submission"),
    code(
        """from pathlib import Path

COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
OUT = Path("/kaggle/working")

pred_df = predict_quickdirty(COMPETITION_DIR)
pred_df.to_csv(OUT / "submission_debug.csv", index=False)
write_submission(pred_df, OUT / "submission.csv")

submission = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
assert len(submission) == 309, len(submission)
assert submission["image_id"].is_unique
assert submission[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()

print("Wrote submission.csv and submission_debug.csv")
print("Rows:", len(submission), "NaNs:", int(submission.isna().sum().sum()))
display(submission.head())
display(pred_df[["pa_deg", "fl_mm", "mt_mm", "mm_per_px"]].describe().round(3))
display(pred_df.groupby(["height", "width"]).size().rename("n").reset_index())
"""
    ),
]


def main() -> None:
    out = ROOT / "notebooks/submission-quickdirty"
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
    (out / "submission-quickdirty.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-quickdirty",
        "title": "UMUD Submission QuickDirty",
        "code_file": "submission-quickdirty.ipynb",
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
