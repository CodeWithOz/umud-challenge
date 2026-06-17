"""Generate notebooks/train-apo-gray55/train-apo-gray55-phase-3.ipynb — train on gray55 prep dataset."""
import json
from pathlib import Path

BUILD_TRAIN_RUN = 7

DATASET_SLUG_BY_RUN = {
    1: "ucheozoemena/umud-aligned-apo-gray55-timing-50",
    2: "ucheozoemena/umud-aligned-apo-gray55-timing-200",
    3: "ucheozoemena/umud-aligned-apo-gray55-timing-524",
    4: "ucheozoemena/umud-aligned-apo-gray55-full",
    5: "ucheozoemena/umud-aligned-apo-gray55-line-timing-50",
    6: "ucheozoemena/umud-aligned-apo-gray55-line-full",
    7: "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
}


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


cells: list[dict] = [
    md(
        """# UMUD — Train Apo U-Net on Gray55 Prep Dataset

**GPU notebook** — mounts a **gray55 apo prep** dataset. Images were preprocessed with RGB(55,55,55) outside the ultrasound bbox before 256px resize.

> Edit *Configuration*, then re-run from there downward."""
    ),
    md("## Configuration"),
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42
TRAIN_RUN = 7  # 5=micro 50×5ep; 6=full 1044×10ep; 7=200×5ep stratified val (Block 2)

VALID_PCT = 0.20
STRATIFY_VAL_BY_RESOLUTION = True  # uses manifest resolution_cohort when True
BATCH_SIZE = 8
ARCH = "resnet34"
IMG_SIZE = 256
APO_FULL = 1044
FULL_EPOCHS = 10

USE_CLASS_WEIGHTS = True
APO_FG_WEIGHT = 15.0

TRAIN_PROFILES = {
    1: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-timing-50",
        "epochs": 1,
        "label": "GAT1 gray55 apo 50×1ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    2: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-timing-200",
        "epochs": 1,
        "label": "GAT2 gray55 apo 200×1ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    3: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-timing-524",
        "epochs": FULL_EPOCHS // 2,
        "label": "GAT3 gray55 apo 524×5ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    4: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-full",
        "epochs": FULL_EPOCHS,
        "label": "GAT4 gray55 apo full 1044×10ep",
        "export_name": "apo_gray55_baseline.pkl",
    },
    5: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-50",
        "epochs": 5,
        "label": "GAT5 gray55+line apo 50×5ep micro",
        "export_name": "apo_gray55_line_baseline.pkl",
    },
    6: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-full",
        "epochs": FULL_EPOCHS,
        "label": "GAT6 gray55+line apo full 1044×10ep",
        "export_name": "apo_gray55_line_baseline.pkl",
    },
    7: {
        "dataset_slug": "ucheozoemena/umud-aligned-apo-gray55-line-timing-200",
        "epochs": 5,
        "label": "GAT7 gray55+line apo 200×5ep stratified val",
        "export_name": "apo_gray55_line_200.pkl",
    },
}

profile = TRAIN_PROFILES[TRAIN_RUN]
DATASET_SLUG = profile["dataset_slug"]
EPOCHS = profile["epochs"]
EXPORT_NAME = profile["export_name"]
print(f"TRAIN_RUN={TRAIN_RUN} | {profile['label']} | dataset={DATASET_SLUG} | epochs={EPOCHS}")
"""
    ),
    code(
        """from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
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
    resnet34,
    resnet50,
    unet_learner,
)
from fastai.data.block import DataBlock
from fastai.data.transforms import FuncSplitter

DATASET_ROOT = Path(f"/kaggle/input/datasets/{DATASET_SLUG}")
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path(kagglehub.dataset_download(DATASET_SLUG))
WORKING = Path("/kaggle/working")

print(f"Dataset root: {DATASET_ROOT} (exists={DATASET_ROOT.exists()})")
"""
    ),
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
    ),
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


def stratified_train_valid_stems(
    stems: list[str],
    labels: list[str],
    valid_pct: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    from collections import Counter
    import random

    counts = Counter(labels)
    # sklearn needs ≥2 per class; bucket rare native resolutions together.
    collapsed = [label if counts[label] >= 2 else "other" for label in labels]
    counts = Counter(collapsed)
    if min(counts.values()) < 2:
        collapsed = [
            label if counts[label] >= 2 else "other_bucket" for label in collapsed
        ]
        counts = Counter(collapsed)
    if min(counts.values()) >= 2:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            stems,
            test_size=valid_pct,
            random_state=seed,
            stratify=collapsed,
        )

    rng = random.Random(seed)
    by_cohort: dict[str, list[str]] = {}
    for stem, label in zip(stems, labels):
        by_cohort.setdefault(label, []).append(stem)
    train: list[str] = []
    valid: list[str] = []
    for cohort_stems in by_cohort.values():
        rng.shuffle(cohort_stems)
        if len(cohort_stems) == 1:
            train.extend(cohort_stems)
            continue
        n_val = max(1, round(len(cohort_stems) * valid_pct))
        valid.extend(cohort_stems[:n_val])
        train.extend(cohort_stems[n_val:])
    print("Stratified sklearn failed — used per-cohort manual split")
    return train, valid


def make_dls(fnames, valid_pct=0.20, bs=8, seed=42, stratify_cohort: dict[str, str] | None = None):
    if stratify_cohort:
        stems = [Path(f).stem for f in fnames]
        labels = [stratify_cohort.get(s, "unknown") for s in stems]
        train_stems, valid_stems = stratified_train_valid_stems(
            stems, labels, valid_pct=valid_pct, seed=seed
        )
        valid_set = set(valid_stems)

        def _split(items):
            train = [i for i, f in enumerate(items) if Path(f).stem not in valid_set]
            valid = [i for i, f in enumerate(items) if Path(f).stem in valid_set]
            return train, valid

        splitter = FuncSplitter(_split)
        print(
            f"Stratified val: {len(valid_stems)} images across "
            f"{len(set(labels))} resolution cohorts"
        )
    else:
        splitter = RandomSplitter(valid_pct=valid_pct, seed=seed)

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
        splitter=splitter,
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=True),
    )
    return block.dataloaders(fnames, bs=bs, num_workers=2)


def load_cohort_by_stem(root: Path) -> dict[str, str]:
    manifest_dir = resolve_subdir(root, "manifests")
    manifest_path = manifest_dir / "train_apo_gray55_line.csv"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path} — falling back to random val split")
        return {}
    manifest = pd.read_csv(manifest_path)
    if "resolution_cohort" in manifest.columns:
        col = "resolution_cohort"
    elif {"img_h", "img_w"}.issubset(manifest.columns):
        manifest["resolution_cohort"] = manifest.apply(
            lambda r: f"{int(r.img_h)}x{int(r.img_w)}", axis=1
        )
        col = "resolution_cohort"
    else:
        print("Manifest missing resolution columns — random val split")
        return {}
    return dict(zip(manifest["stem"].astype(str), manifest[col].astype(str)))


img_fnames = get_image_files(IMG_DIR)
msk_lookup = {p.name for p in get_image_files(MSK_DIR)}
fnames = [f for f in img_fnames if f.name in msk_lookup]
print(f"Pairs: {len(fnames)}")
assert len(fnames) > 0, "No image/mask pairs in mounted dataset"
cohort_by_stem = load_cohort_by_stem(DATASET_ROOT) if STRATIFY_VAL_BY_RESOLUTION else {}
"""
    ),
    code(
        """t0 = time.perf_counter()
dls = make_dls(
    fnames,
    valid_pct=VALID_PCT,
    bs=BATCH_SIZE,
    seed=RANDOM_SEED,
    stratify_cohort=cohort_by_stem or None,
)
_ = dls.one_batch()
print(f"Dataloader ready: {time.perf_counter() - t0:.1f}s")
dls.show_batch(max_n=4)
"""
    ),
    code(
        """t_train = time.perf_counter()
import torch

if USE_CLASS_WEIGHTS:
    loss_weights = torch.tensor([1.0, APO_FG_WEIGHT])
    loss_func = CrossEntropyLossFlat(axis=1, weight=loss_weights)
    print(f"Class weights: background=1.0, structure={APO_FG_WEIGHT}")
else:
    loss_func = CrossEntropyLossFlat(axis=1)

learn = unet_learner(
    dls,
    encoder(),
    metrics=[Dice()],
    loss_func=loss_func,
    self_attention=True,
)
learn.fine_tune(EPOCHS)
t1 = time.perf_counter()
train_sec = t1 - t_train
print(f"Train wall-clock: {train_sec:.1f}s")

learn.export(WORKING / EXPORT_NAME)

# validation Dice snapshot
val_losses, val_metrics = learn.validate(dl=dls.valid)
if isinstance(val_metrics, (list, tuple)):
    val_dice = float(val_metrics[0]) if val_metrics else float("nan")
else:
    val_dice = float(val_metrics)
print(f"Val Dice: {val_dice:.4f}")

timing = pd.DataFrame(
    [
        {
            "train_run": TRAIN_RUN,
            "n_pairs": len(fnames),
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "val_dice": round(val_dice, 4),
            "total_sec": round(train_sec, 1),
            "sec_per_pair_epoch": round(train_sec / max(1, len(fnames) * EPOCHS), 3),
            "dataset": DATASET_SLUG,
        }
    ]
)
timing.to_csv(WORKING / "timing_report.csv", index=False)
display(timing)
"""
    ),
]


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
    out = Path(__file__).resolve().parents[1] / "notebooks/train-apo-gray55"
    write_nb(out / "train-apo-gray55-phase-3.ipynb")
    profile = DATASET_SLUG_BY_RUN[BUILD_TRAIN_RUN]
    meta = {
        "id": "ucheozoemena/umud-train-apo-gray55-phase-3",
        "title": "UMUD Train Apo Gray55 Phase 3",
        "code_file": "train-apo-gray55-phase-3.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [profile],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
