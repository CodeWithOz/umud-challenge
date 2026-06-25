"""Generate a Block22 timing benchmark notebook for SMP U-Net++ EfficientNet-B3."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts/build_lakhindar_smp_timing_nb.py"
OUT_DIR = ROOT / "notebooks/bench-lakhindar-smp-b3-timing"


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
            "# UMUD - Block20 SMP Timing Benchmark": "# UMUD - Block22 SMP B3 Timing Benchmark",
            "EfficientNet-B7 U-Net++": "EfficientNet-B3 U-Net++",
            '    BACKBONE = "efficientnet-b7"': '    BACKBONE = "efficientnet-b3"',
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
    (OUT_DIR / "bench-lakhindar-smp-b3-timing.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-bench-lakhindar-smp-timing",
        "title": "UMUD Bench Lakhindar SMP B3 Timing",
        "code_file": "bench-lakhindar-smp-b3-timing.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "benchmark"],
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
