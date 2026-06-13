"""Generate notebooks/train-mounted/train-mounted-phase-3.ipynb — train from prep dataset."""
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
        """# UMUD — Train Fasc U-Net from Prep Dataset

**GPU notebook** — mounts a **prep notebook** dataset (`dataset_sources`). No competition TIFF scan or inline align.

BirdCLEF pattern: `multilabel-234-v2` mounts `species-v2-*` datasets and uses `get_image_files`.

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42
TRAIN_RUN = 3  # 1=timing-50, 2=timing-200, 3=timing-1374 (50% fasc)

VALID_PCT = 0.20
BATCH_SIZE = 8
ARCH = "resnet34"
IMG_SIZE = 256  # must match prep dataset
FASC_FULL_CLEAN = 2749
FULL_EPOCHS = 10

TRAIN_PROFILES = {
    1: {
        "dataset_slug": "ucheozoemena/umud-aligned-fasc-timing-50",
        "mount_name": "umud-aligned-fasc-timing-50",
        "epochs": 1,
        "label": "T1 fasc 50×1ep",
    },
    2: {
        "dataset_slug": "ucheozoemena/umud-aligned-fasc-timing-200",
        "mount_name": "umud-aligned-fasc-timing-200",
        "epochs": 1,
        "label": "T2 fasc 200×1ep",
    },
    3: {
        "dataset_slug": "ucheozoemena/umud-aligned-fasc-timing-1374",
        "mount_name": "umud-aligned-fasc-timing-1374",
        "epochs": FULL_EPOCHS // 2,
        "label": "T3 fasc 1374×5ep (50% data, 50% epochs)",
    },
}

profile = TRAIN_PROFILES[TRAIN_RUN]
DATASET_SLUG = profile["dataset_slug"]
MOUNT_NAME = profile["mount_name"]
EPOCHS = profile["epochs"]
print(f"TRAIN_RUN={TRAIN_RUN} | {profile['label']} | dataset={DATASET_SLUG} | epochs={EPOCHS}")
"""
    )
)

cells.append(
    code(
        """from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
import kagglehub
from fastai.vision.all import (
    AddMaskCodes,
    CrossEntropyLossFlat,
    Dice,
    IntToFloatTensor,
    PILImage,
    PILMask,
    RandomSplitter,
    Resize,
    TransformBlock,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    resnet34,
    resnet50,
    unet_learner,
)
from fastai.data.block import DataBlock

DATASET_ROOT = Path(f"/kaggle/input/datasets/{DATASET_SLUG}")
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path(kagglehub.dataset_download(DATASET_SLUG))
WORKING = Path("/kaggle/working")

print(f"Dataset root: {DATASET_ROOT} (exists={DATASET_ROOT.exists()})")
"""
    )
)

cells.append(
    code(
        """def resolve_subdir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    candidates = [p for p in root.rglob(name) if p.is_dir() and p.name == name]
    if not candidates:
        raise FileNotFoundError(f"Could not find {name}/ under {root}")
    return candidates[0]

IMG_DIR = resolve_subdir(DATASET_ROOT, "images")
MSK_DIR = resolve_subdir(DATASET_ROOT, "masks")
print(f"images: {IMG_DIR}")
print(f"masks: {MSK_DIR}")
"""
    )
)

cells.append(
    code(
        """SEG_CODES = ["background", "structure"]


def encoder():
    return resnet50 if ARCH == "resnet50" else resnet34


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


def make_dls(fnames, valid_pct=0.20, bs=8, seed=42):
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
        get_x=lambda f: IMG_DIR / f.name,
        get_y=lambda f: MSK_DIR / f.name,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=True),
    )
    return block.dataloaders(fnames, bs=bs, num_workers=2)

img_fnames = get_image_files(IMG_DIR)
msk_lookup = {p.name for p in get_image_files(MSK_DIR)}
fnames = [f for f in img_fnames if f.name in msk_lookup]
print(f"Pairs: {len(fnames)}")
assert len(fnames) > 0, "No image/mask pairs in mounted dataset"
"""
    )
)

cells.append(
    code(
        """t0 = time.perf_counter()
dls = make_dls(fnames, valid_pct=VALID_PCT, bs=BATCH_SIZE, seed=RANDOM_SEED)
_ = dls.one_batch()
print(f"Dataloader ready: {time.perf_counter() - t0:.1f}s")
dls.show_batch(max_n=4)
"""
    )
)

cells.append(
    code(
        """t_train = time.perf_counter()
learn = unet_learner(
    dls,
    encoder(),
    metrics=[Dice()],
    loss_func=CrossEntropyLossFlat(axis=1),
    self_attention=True,
)
learn.fine_tune(EPOCHS)
t1 = time.perf_counter()
train_sec = t1 - t_train
print(f"Train wall-clock: {train_sec:.1f}s ({train_sec / max(1, len(fnames)) / max(1, EPOCHS):.2f}s/pair/epoch)")

learn.export(WORKING / "fasc_baseline.pkl")

timing = pd.DataFrame(
    [
        {
            "train_run": TRAIN_RUN,
            "n_pairs": len(fnames),
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "total_sec": round(train_sec, 1),
            "sec_per_pair_epoch": round(train_sec / max(1, len(fnames) * EPOCHS), 3),
            "dataset": DATASET_SLUG,
        }
    ]
)
timing.to_csv(WORKING / "timing_report.csv", index=False)
display(timing)
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
    out = Path(__file__).resolve().parents[1] / "notebooks/train-mounted"
    write_nb(out / "train-mounted-phase-3.ipynb")


if __name__ == "__main__":
    main()
