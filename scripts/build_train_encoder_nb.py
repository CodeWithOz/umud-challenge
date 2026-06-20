"""Generate one train notebook + kernel-metadata per Block 8 encoder."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from block8_encoders import BLOCK8_ENCODERS, DATASET_SLUG, EPOCHS, EncoderSpec

_DOCKER = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9"
)


def _load_gray55_builder():
    path = ROOT / "scripts/build_train_apo_gray55_nb.py"
    spec = importlib.util.spec_from_file_location("build_train_apo_gray55_nb", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def config_cell(enc: EncoderSpec) -> str:
    return f"""# --- Block 8 single-encoder train ---
RANDOM_SEED = 42
ENCODER_SLUG = "{enc.slug}"
ARCH = "{enc.arch}"
FAMILY = "{enc.family}"
EXPORT_NAME = "{enc.export_name}"
DATASET_SLUG = "{DATASET_SLUG}"
EPOCHS = {EPOCHS}

VALID_PCT = 0.20
STRATIFY_VAL_BY_RESOLUTION = True
BATCH_SIZE = 8
IMG_SIZE = 256
MM_PER_PIXEL = 0.075
USE_CLASS_WEIGHTS = True
APO_FG_WEIGHT = 15.0

print(f"Block 8 | {{FAMILY}} | arch={{ARCH}} | slug={{ENCODER_SLUG}} | epochs={{EPOCHS}} | dataset={{DATASET_SLUG}}")
"""


def train_cell() -> str:
    return """t_train = time.perf_counter()
import torch
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

if USE_CLASS_WEIGHTS:
    loss_weights = torch.tensor([1.0, APO_FG_WEIGHT])
    loss_func = CrossEntropyLossFlat(axis=1, weight=loss_weights)
    print(f"Class weights: background=1.0, structure={APO_FG_WEIGHT}")
else:
    loss_func = CrossEntropyLossFlat(axis=1)

learn = timm_unet_learner(
    dls,
    ARCH,
    metrics=[Dice()],
    loss_func=loss_func,
    bottleneck="conv",
)
learn.fine_tune(EPOCHS)
train_sec = time.perf_counter() - t_train
print(f"Train wall-clock: {train_sec:.1f}s")
learn.export(WORKING / EXPORT_NAME)

val_losses, val_metrics = learn.validate(dl=dls.valid)
if isinstance(val_metrics, (list, tuple)):
    val_dice = float(val_metrics[0]) if val_metrics else float("nan")
else:
    val_dice = float(val_metrics)
print(f"Val Dice (reference): {val_dice:.4f}")

timing = pd.DataFrame(
    [
        {
            "encoder_slug": ENCODER_SLUG,
            "family": FAMILY,
            "arch": ARCH,
            "n_pairs": len(fnames),
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "val_dice": round(val_dice, 4) if val_dice == val_dice else None,
            "total_sec": round(train_sec, 1),
            "sec_per_pair_epoch": round(train_sec / max(1, len(fnames) * EPOCHS), 3),
            "dataset": DATASET_SLUG,
        }
    ]
)
timing.to_csv(WORKING / "timing_report.csv", index=False)
display(timing)
"""


def build_cells(g55, enc: EncoderSpec) -> list[dict]:
    shared = g55.cells
    title = f"""# UMUD — Block 8 Train — {enc.family}

**GPU notebook** — single encoder **`{enc.arch}`** @ 200×5ep on gray55+line prep.

Outputs: `{enc.export_name}`, `timing_report.csv`, `val_umud_report.csv` (val UMUD + mt_ok)."""
    return [
        g55.md(title),
        g55.md("## Configuration"),
        g55.code(config_cell(enc)),
        shared[3],
        shared[4],
        shared[5],
        shared[6],
        g55.md("## Timm U-Net"),
        shared[8],
        g55.code(train_cell()),
        shared[10],
        shared[11],
        shared[12],
    ]


def write_notebook(g55, enc: EncoderSpec) -> Path:
    out_dir = ROOT / enc.notebook_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / enc.ipynb_name
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": build_cells(g55, enc),
    }
    nb_path.write_text(json.dumps(nb, indent=1))
    meta = {
        "id": enc.kernel_id,
        "title": f"UMUD Train Encoder {enc.family} Phase 3",
        "code_file": enc.ipynb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [DATASET_SLUG],
        "kernel_sources": ["ucheozoemena/umud-train-mounted-phase-3"],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": _DOCKER,
        "machine_shape": "NvidiaTeslaT4",
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {nb_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="Build one encoder notebook (default: all Block 8)")
    args = parser.parse_args()
    g55 = _load_gray55_builder()
    specs = [e for e in BLOCK8_ENCODERS if not args.slug or e.slug == args.slug]
    if args.slug and not specs:
        raise SystemExit(f"Unknown slug: {args.slug}")
    for enc in specs:
        write_notebook(g55, enc)


if __name__ == "__main__":
    main()
