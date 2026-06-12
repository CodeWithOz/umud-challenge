"""Generate notebooks/baseline/baseline-phase-3.ipynb — fastai U-Net segmentation baseline."""
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
        """# UMUD Challenge — Segmentation Baseline (Phase 3)

**Kaggle GPU notebook** — first learned baseline using **fastai U-Net** segmentation.

### Pipeline

1. Load competition data + build clean manifests (same rules as Phase 2 geometry).
2. **Stretch-align** masks when `img.shape != mask.shape`.
3. Train separate **fascicle** and **aponeurosis** U-Net models (segment-then-measure later).
4. Export `fasc_baseline.pkl` and `apo_baseline.pkl` to `/kaggle/working/`.

> Geometry (PA / FL / MT) runs at inference in a later notebook — this kernel only trains masks.

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42
TRAIN_TRACK = "both"  # "fasc", "apo", or "both"
VALID_PCT = 0.20
IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 10
ARCH = "resnet34"  # fastai encoder
FASC_NEAR_EMPTY_THRESHOLD = 0.0005
DEFAULT_ALIGN_MODE = "stretch"
"""
    )
)

cells.append(
    md(
        """## Paths and imports

Competition data via **`kagglehub.competition_download`** (no internet required on Kaggle)."""
    )
)

cells.append(
    code(
        """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

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
    foreground_acc,
    resnet34,
    resnet50,
    show_image,
    unet_learner,
)
from fastai.data.block import DataBlock

COMPETITION_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
DATA_ROOT = Path(kagglehub.competition_download(COMPETITION_SLUG))
print("Competition dir:", DATA_ROOT)

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
print("Output dir:", OUT)

DIRS = {
    "apo_img": DATA_ROOT / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": DATA_ROOT / "apo_masks_v1/apo_masks_new_model_v1",
    "fasc_img": DATA_ROOT / "fasc_imgs_v1/fasc_images_new_model_v1",
    "fasc_mask": DATA_ROOT / "fasc_masks_v1/fasc_masks_new_model_v1",
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


lookups = {k: build_lookup(v) for k, v in DIRS.items()}
display(pd.DataFrame([{"key": k, "n_files": len(v)} for k, v in lookups.items()]))
"""
    )
)

cells.append(
    md(
        """## Alignment utilities (stretch, from Phase 0/1)

When image and mask shapes differ, stretch the mask to the image size before training."""
    )
)

cells.append(
    code(
        """def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def mask_coverage(mask: np.ndarray) -> float:
    return float(mask.mean())


def place_mask_center(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    mh, mw = mask.shape
    y0 = max(0, (target_h - mh) // 2)
    x0 = max(0, (target_w - mw) // 2)
    y1 = min(target_h, y0 + mh)
    x1 = min(target_w, x0 + mw)
    mask_row_start = max(0, (mh - target_h) // 2)
    mask_col_start = max(0, (mw - target_w) // 2)
    canvas[y0:y1, x0:x1] = mask[
        mask_row_start : mask_row_start + (y1 - y0),
        mask_col_start : mask_col_start + (x1 - x0),
    ]
    return canvas


def align_mask(mask: np.ndarray, target_h: int, target_w: int, mode: str = DEFAULT_ALIGN_MODE) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask
    if mode == "stretch":
        return (
            np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (target_w, target_h), Image.NEAREST
                )
            )
            > 0
        ).astype(np.uint8)
    if mode == "center":
        return place_mask_center(mask, target_h, target_w)
    raise ValueError(f"Unknown mode: {mode}")
"""
    )
)

cells.append(md("""## Clean training manifests"""))

cells.append(
    code(
        """def mask_coverage_from_path(path: Path) -> float:
    return mask_coverage(load_mask(path))


print("Scanning fascicle masks for empty / near-empty pairs...")
fasc_common = sorted(set(lookups["fasc_img"]) & set(lookups["fasc_mask"]))
exclude_rows = []
for name in fasc_common:
    cov = mask_coverage_from_path(lookups["fasc_mask"][name])
    if cov <= 0.0:
        exclude_rows.append({"filename": name, "mask_coverage": cov, "reason": "empty"})
    elif cov < FASC_NEAR_EMPTY_THRESHOLD:
        exclude_rows.append({"filename": name, "mask_coverage": cov, "reason": "near_empty"})

exclude_names = {r["filename"] for r in exclude_rows}
train_fasc = [n for n in fasc_common if n not in exclude_names]
train_apo = sorted(set(lookups["apo_img"]) & set(lookups["apo_mask"]))

print(f"Fasc pairs total: {len(fasc_common)} | exclude: {len(exclude_names)} | clean: {len(train_fasc)}")
print(f"Apo pairs: {len(train_apo)}")
"""
    )
)

cells.append(
    md(
        """## fastai dataloaders

Binary masks: background = 0, structure = 1. Images are grayscale ultrasound frames expanded to 3 channels for the ResNet encoder."""
    )
)

cells.append(
    code(
        """SEG_CODES = ["background", "structure"]


def encoder():
    return resnet50 if ARCH == "resnet50" else resnet34


def make_dblock(fnames: list[str], img_key: str, mask_key: str) -> DataBlock:
    def get_items(_):
        return fnames

    def open_img(fname):
        gray = load_gray(lookups[img_key][fname])
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
        return PILImage.create(rgb)

    def open_mask(fname):
        img = load_gray(lookups[img_key][fname])
        mask = load_mask(lookups[mask_key][fname])
        aligned = align_mask(mask, img.shape[0], img.shape[1])
        return PILMask.create(aligned.astype(np.uint8))

    return DataBlock(
        blocks=(
            TransformBlock(type_tfms=open_img, batch_tfms=IntToFloatTensor),
            TransformBlock(
                type_tfms=open_mask,
                item_tfms=AddMaskCodes(codes=SEG_CODES),
                batch_tfms=IntToFloatTensor,
            ),
        ),
        get_items=get_items,
        splitter=RandomSplitter(valid_pct=VALID_PCT, seed=RANDOM_SEED),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=True),
    )


def make_dls(fnames: list[str], img_key: str, mask_key: str):
    dblock = make_dblock(fnames, img_key, mask_key)
    return dblock.dataloaders(fnames, bs=BATCH_SIZE, num_workers=2)
"""
    )
)

cells.append(md("""## Train fascicle model"""))

cells.append(
    code(
        """fasc_learn = None
if TRAIN_TRACK in ("fasc", "both"):
    print(f"Training fascicle U-Net ({len(train_fasc)} pairs)...")
    fasc_dls = make_dls(train_fasc, "fasc_img", "fasc_mask")
    fasc_learn = unet_learner(
        fasc_dls,
        encoder(),
        loss_func=CrossEntropyLossFlat(axis=1),
        metrics=[Dice(), foreground_acc],
        self_attention=True,
    )
    fasc_learn.fine_tune(EPOCHS)
    fasc_export = OUT / "fasc_baseline"
    fasc_learn.export(fasc_export)
    print("Exported:", fasc_export.with_suffix(".pkl"))
else:
    print("Skipping fascicle training (TRAIN_TRACK != fasc/both)")
"""
    )
)

cells.append(md("""## Train aponeurosis model"""))

cells.append(
    code(
        """apo_learn = None
if TRAIN_TRACK in ("apo", "both"):
    print(f"Training aponeurosis U-Net ({len(train_apo)} pairs)...")
    apo_dls = make_dls(train_apo, "apo_img", "apo_mask")
    apo_learn = unet_learner(
        apo_dls,
        encoder(),
        loss_func=CrossEntropyLossFlat(axis=1),
        metrics=[Dice(), foreground_acc],
        self_attention=True,
    )
    apo_learn.fine_tune(EPOCHS)
    apo_export = OUT / "apo_baseline"
    apo_learn.export(apo_export)
    print("Exported:", apo_export.with_suffix(".pkl"))
else:
    print("Skipping apo training (TRAIN_TRACK != apo/both)")
"""
    )
)

cells.append(md("""## Validation preview (first val batch)"""))

cells.append(
    code(
        """def preview_learner(learn, title: str, n: int = 3):
    if learn is None:
        return
    dl = learn.dls.valid
    batch = dl.one_batch()
    preds, _ = learn.get_preds(dl=dl)
    preds = preds.argmax(dim=1)
    imgs, masks = batch
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for i in range(min(n, len(imgs))):
        show_image(imgs[i], ax=axes[i, 0], title=f"{title} image")
        show_image(masks[i], ax=axes[i, 1], title="GT mask")
        show_image(preds[i], ax=axes[i, 2], title="Pred mask")
    plt.tight_layout()
    fig.savefig(OUT / f"preview_{title}.png", dpi=120, bbox_inches="tight")
    plt.show()


preview_learner(fasc_learn, "fasc")
preview_learner(apo_learn, "apo")

print("Done. Artifacts in", OUT)
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(" ", p.name)
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
    out_dir = Path(__file__).resolve().parents[1] / "notebooks/baseline"
    write_nb(out_dir / "baseline-phase-3.ipynb")


if __name__ == "__main__":
    main()
