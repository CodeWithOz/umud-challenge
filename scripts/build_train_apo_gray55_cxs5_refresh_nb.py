"""Generate a separate cxs5 refresh training notebook.

This keeps the mutable production gray55 training kernel untouched. The refreshed
kernel exports the same file name as the original cxs5 run:
`apo_gray55_line_200_cxs.pkl`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import build_train_apo_gray55_nb as base


ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN = 15
OUT_DIR = ROOT / "notebooks/train-apo-gray55-cxs5-refresh"
KERNEL_ID = "ucheozoemena/umud-train-apo-gray55-cxs5-refresh"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = copy.deepcopy(base.cells)
    for cell in cells:
        if cell.get("cell_type") in {"code", "markdown"}:
            cell["source"] = [
                line.replace(
                    "TRAIN_RUN = 21  # Block 7 encoder sweep — see TRAIN_PROFILES",
                    f"TRAIN_RUN = {TRAIN_RUN}  # Block 19 cxs5 refresh — see TRAIN_PROFILES",
                ).replace(
                    "# UMUD — Train Apo U-Net on Gray55 Prep Dataset",
                    "# UMUD — Train Apo U-Net CXS5 Refresh",
                )
                for line in cell["source"]
            ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    (OUT_DIR / "train-apo-gray55-cxs5-refresh.ipynb").write_text(json.dumps(nb, indent=1))

    meta = {
        "id": KERNEL_ID,
        "title": "UMUD Train Apo Gray55 CXS5 Refresh",
        "code_file": "train-apo-gray55-cxs5-refresh.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [base.DATASET_SLUG_BY_RUN[TRAIN_RUN]],
        "kernel_sources": ["ucheozoemena/umud-train-mounted-phase-3"],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
