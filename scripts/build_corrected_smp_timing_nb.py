"""Generate a corrected-config SMP timing/feasibility benchmark (Block27).

Differences vs the Block20 benchmark:
- **Apo mask polarity fix**: ~40% of apo masks are inverted (aponeurosis = thin BLACK
  bands on a white field). We normalize every mask so the thin minority structure is
  always foreground (`if fg-fraction > 0.5: invert`). Fascicle masks (~0.3% fg) are
  never inverted, so applying universally is safe.
- **Higher resolution** 640x960 (vs 512x768) with BATCH=1/ACCUM=16 so thin fascicles
  keep more detail; benchmark confirms sec/pair + peak GPU memory before the full run.
"""
from __future__ import annotations

import json
from pathlib import Path

import build_lakhindar_smp_timing_nb as base


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks/bench-corrected-smp-timing"
KERNEL_ID = "ucheozoemena/umud-bench-corrected-smp-timing"


def _patch(src: str) -> str:
    src = src.replace("    H = 512\n", "    H = 640\n")
    src = src.replace("    W = 768\n", "    W = 960\n")
    src = src.replace("    BATCH = 2\n", "    BATCH = 1\n")
    src = src.replace("    ACCUM = 8\n", "    ACCUM = 16\n")
    # Apo polarity fix: normalize so the thin structure is always foreground.
    src = src.replace(
        "        mask = (mask > 0).astype(np.float32)\n",
        "        mask = (mask > 127).astype(np.float32)\n"
        "        if mask.mean() > 0.5:  # inverted apo label -> aponeurosis is the thin minority\n"
        "            mask = 1.0 - mask\n",
    )
    return src


def main() -> None:
    cells = []
    for cell in base.cells:
        c = dict(cell)
        c["source"] = [_patch("".join(cell["source"]))]
        # split back into line list with newlines preserved
        text = c["source"][0]
        lines = text.split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        cells.append(c)

    # Title tweak
    cells[0]["source"] = [
        "# UMUD - Block27 Corrected SMP Timing Benchmark\n",
        "\n",
        "Apo polarity fix + 640x960 (B7, batch 1, accum 16). Trains one capped epoch\n",
        "for fascicle and aponeurosis, reports sec/pair/epoch and peak GPU memory so we\n",
        "can pick a feasible epoch count for the full corrected retrain.\n",
    ]

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
    (OUT_DIR / "bench-corrected-smp-timing.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": KERNEL_ID,
        "title": "UMUD Bench Corrected SMP Timing",
        "code_file": "bench-corrected-smp-timing.ipynb",
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
