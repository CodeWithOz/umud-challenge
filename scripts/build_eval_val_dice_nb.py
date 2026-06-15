"""Generate notebooks/eval-val-dice/eval-val-dice-phase-3.ipynb — validation Dice for fasc + apo baselines."""
import json
from pathlib import Path


def md(source: str) -> dict:
    lines = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


cells: list[dict] = []

cells.append(
    md(
        """# UMUD — Validation Dice (Phase 3 Baselines)

**GPU notebook** — loads exported **fasc** and **apo** learners from train kernel outputs, rebuilds the same 80/20 val split (seed 42), and reports **Dice** on held-out validation masks.

Inputs:
- Prep datasets (`umud-aligned-fasc-full`, `umud-aligned-apo-full`)
- Train kernel outputs (`fasc_baseline.pkl`, `apo_baseline.pkl`)

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        """from pathlib import Path

RANDOM_SEED = 42
VALID_PCT = 0.20
BATCH_SIZE = 8
IMG_SIZE = 256

FASC_DATASET = "ucheozoemena/umud-aligned-fasc-full"
APO_DATASET = "ucheozoemena/umud-aligned-apo-full"

FASC_MODEL_PATH = Path(
    "/kaggle/input/notebooks/ucheozoemena/umud-train-mounted-phase-3/fasc_baseline.pkl"
)
APO_MODEL_PATH = Path(
    "/kaggle/input/notebooks/ucheozoemena/umud-train-apo-mounted-phase-3/apo_baseline.pkl"
)
"""
    )
)

cells.append(
    code(
        """from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from fastai.data.block import DataBlock
from fastai.vision.all import (
    AddMaskCodes,
    Dice,
    IntToFloatTensor,
    PILImage,
    PILMask,
    RandomSplitter,
    Resize,
    TransformBlock,
    aug_transforms,
    get_image_files,
    load_learner,
)

SEG_CODES = ["background", "structure"]


def resolve_subdir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    candidates = [p for p in root.rglob(name) if p.is_dir() and p.name == name]
    if not candidates:
        raise FileNotFoundError(f"Could not find {name}/ under {root}")
    return candidates[0]


def dataset_root(slug: str) -> Path:
    mount = Path(f"/kaggle/input/datasets/{slug}")
    if mount.exists():
        return mount
    return Path(kagglehub.dataset_download(slug))


def open_image_pil(fn):
    gray = np.array(PILImage.create(fn))
    if gray.ndim == 3:
        gray = gray[..., 0]
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    return PILImage.create(rgb)


def open_mask_pil(fn):
    arr = np.array(PILImage.create(fn))
    if arr.ndim == 3:
        arr = arr[..., 0]
    binary = (arr > 0).astype(np.uint8)
    return PILMask.create(binary)


def make_dls(img_dir: Path, msk_dir: Path, fnames, valid_pct=0.20, bs=8, seed=42):
    block = DataBlock(
        blocks=(
            TransformBlock(type_tfms=open_image_pil, batch_tfms=IntToFloatTensor),
            TransformBlock(
                type_tfms=open_mask_pil,
                item_tfms=AddMaskCodes(codes=SEG_CODES),
                batch_tfms=IntToFloatTensor,
            ),
        ),
        get_items=lambda _: fnames,
        get_x=lambda f: img_dir / f.name,
        get_y=lambda f: msk_dir / f.name,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=False),
    )
    return block.dataloaders(fnames, bs=bs, num_workers=2)


def pair_fnames(img_dir: Path, msk_dir: Path):
    img_fnames = get_image_files(img_dir)
    msk_names = {p.name for p in get_image_files(msk_dir)}
    return [f for f in img_fnames if f.name in msk_names]


def eval_track(name: str, dataset_slug: str, model_path: Path) -> dict:
    root = dataset_root(dataset_slug)
    img_dir = resolve_subdir(root, "images")
    msk_dir = resolve_subdir(root, "masks")
    fnames = pair_fnames(img_dir, msk_dir)
    dls = make_dls(img_dir, msk_dir, fnames, valid_pct=VALID_PCT, bs=BATCH_SIZE, seed=RANDOM_SEED)
    learn = load_learner(model_path)
    learn.dls = dls
    results = learn.validate(dl=dls.valid)
    metric_names = [getattr(m, "name", None) or type(m).__name__.lower() for m in learn.metrics]
    out = {
        "track": name,
        "n_pairs": len(fnames),
        "n_val": len(dls.valid_ds),
        "loss": float(results[0]),
        "dataset": dataset_slug,
        "model_path": str(model_path),
    }
    for i, mname in enumerate(metric_names):
        out[mname] = float(results[i + 1])
    return out
"""
    )
)

cells.append(
    code(
        """assert FASC_MODEL_PATH.exists(), f"Missing fasc model: {FASC_MODEL_PATH}"
assert APO_MODEL_PATH.exists(), f"Missing apo model: {APO_MODEL_PATH}"

rows = [
    eval_track("fasc", FASC_DATASET, FASC_MODEL_PATH),
    eval_track("apo", APO_DATASET, APO_MODEL_PATH),
]
report = pd.DataFrame(rows)
display(report)
report.to_csv("/kaggle/working/val_dice_report.csv", index=False)
print("Wrote /kaggle/working/val_dice_report.csv")
"""
    )
)


def write_nb(path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path} ({len(cells)} cells)")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/eval-val-dice"
    write_nb(out / "eval-val-dice-phase-3.ipynb")


if __name__ == "__main__":
    main()
