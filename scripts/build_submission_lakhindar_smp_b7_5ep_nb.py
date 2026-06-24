"""Generate a bounded Block21 SMP U-Net++ submission notebook.

This reuses the Block20 SMP notebook structure, but cuts EfficientNet-B7
training from 30 epochs to 5 epochs. The Block20 timing benchmark projects this
at roughly two hours of training before overhead, so this variant is a feasible
hidden-safe public-LB probe.
"""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts/build_submission_lakhindar_smp_nb.py"
OUT_DIR = ROOT / "notebooks/submission-lakhindar-smp-b7-5ep"


def replace_in_cell_sources(cells: list[dict], replacements: dict[str, str]) -> list[dict]:
    updated = copy.deepcopy(cells)
    for cell in updated:
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        new_source = []
        for line in source:
            for old, new in replacements.items():
                line = line.replace(old, new)
            new_source.append(line)
        cell["source"] = new_source
    return updated


def main() -> None:
    ns = runpy.run_path(str(BASE_BUILDER))
    cells = replace_in_cell_sources(
        ns["cells"],
        {
            "# UMUD - Block20 SMP U-Net++ Geometry": "# UMUD - Block21 SMP B7 5ep Geometry",
            "GPU notebook adapted from the public EfficientNet-B7 U-Net++ baseline. It trains": (
                "Bounded GPU notebook adapted from the public EfficientNet-B7 U-Net++ baseline. It trains"
            ),
            "both fascicle and aponeurosis segmenters from the competition masks, predicts": (
                "both fascicle and aponeurosis segmenters for 5 epochs, predicts"
            ),
            "    EPOCHS = 30": "    EPOCHS = 5",
        },
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    (OUT_DIR / "submission-lakhindar-smp-b7-5ep.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-lakhindar-smp-b7-5ep",
        "title": "UMUD Submission Lakhindar SMP B7 5ep",
        "code_file": "submission-lakhindar-smp-b7-5ep.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
