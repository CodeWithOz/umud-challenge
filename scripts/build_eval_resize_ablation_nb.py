"""Generate notebooks/eval-resize-ablation/eval-resize-ablation-phase-3.ipynb — fasc val Dice for 512px ablation."""
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
        """# UMUD — Resize Ablation Eval (fasc, 50 pairs)

**GPU notebook** — apples-to-apples comparison of **256px vs 512px** prep on the same 50 fasc pairs (seed 42).

Holds everything fixed except resolution:
- Same pairs, val split (80/20, seed 42), batch size 8, resnet34
- Weighted CE (`w_fg=150`), 5 epochs (512 run; 256 baseline from train kernel v12)

Writes `resize_ablation_report.csv` with val Dice + foreground pixel stats.

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
SEG_CODES = ["background", "structure"]

# 256px weighted verify baseline (train-mounted v12) — for reference in report
BASELINE_256 = {
    "img_size": 256,
    "dataset": "ucheozoemena/umud-aligned-fasc-timing-50",
    "val_dice": 0.008,
    "mean_pred_fg_frac": 0.00012,
    "mean_gt_fg_frac": 0.0030,
    "fasc_pca_ok_rate": 0.50,
    "source": "debug-phase-3 v2 + train-mounted v12 (50×5ep weighted @256)",
}

# Current ablation run — model from train-mounted after TRAIN_RUN=5
ABLATION_512 = {
    "img_size": 512,
    "dataset": "ucheozoemena/umud-aligned-fasc-timing-50-512px",
    "model_path": Path(
        "/kaggle/input/notebooks/ucheozoemena/umud-train-mounted-phase-3/fasc_baseline.pkl"
    ),
}
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
    return PILMask.create((arr > 0).astype(np.uint8))


def make_dls(img_dir, msk_dir, fnames, img_size, valid_pct=0.20, bs=8, seed=42):
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
        item_tfms=Resize(img_size),
        batch_tfms=aug_transforms(size=img_size, min_scale=0.75, flip_vert=False, do_flip=False),
    )
    return block.dataloaders(fnames, bs=bs, num_workers=2)


def pair_fnames(img_dir, msk_dir):
    imgs = get_image_files(img_dir)
    msk_names = {p.name for p in get_image_files(msk_dir)}
    return [f for f in imgs if f.name in msk_names]


def analyze_preds(learn, dl, n_batches=30):
    import torch

    pred_fg, gt_fg = [], []
    learn.model.eval()
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        xb, yb = batch
        with torch.no_grad():
            logits = learn.model(xb)
        pred_cls = logits.argmax(dim=1).cpu().numpy()
        gt = yb.cpu().numpy()
        pred_fg.append(float((pred_cls == 1).mean()))
        gt_fg.append(float((gt == 1).mean()))
    return {
        "mean_pred_fg_frac": float(np.mean(pred_fg)),
        "mean_gt_fg_frac": float(np.mean(gt_fg)),
    }


def eval_run(label, dataset_slug, model_path, img_size):
    root = dataset_root(dataset_slug)
    img_dir = resolve_subdir(root, "images")
    msk_dir = resolve_subdir(root, "masks")
    fnames = pair_fnames(img_dir, msk_dir)
    dls = make_dls(img_dir, msk_dir, fnames, img_size, valid_pct=VALID_PCT, bs=BATCH_SIZE, seed=RANDOM_SEED)
    learn = load_learner(model_path)
    learn.dls = dls
    results = learn.validate(dl=dls.valid)
    metric_names = [getattr(m, "name", None) or type(m).__name__.lower() for m in learn.metrics]
    manual = analyze_preds(learn, dls.valid)
    row = {
        "run": label,
        "img_size": img_size,
        "n_pairs": len(fnames),
        "n_val": len(dls.valid_ds),
        "loss": float(results[0]),
        "dataset": dataset_slug,
        **{metric_names[i]: float(results[i + 1]) for i in range(len(metric_names))},
        **manual,
    }
    return row


def gt_mask_stats(dataset_slug, img_size):
    root = dataset_root(dataset_slug)
    msk_dir = resolve_subdir(root, "masks")
    fnames = get_image_files(msk_dir)
    fracs = []
    for f in fnames:
        arr = np.array(PILImage.create(f))
        if arr.ndim == 3:
            arr = arr[..., 0]
        fracs.append(float((arr > 0).mean()))
    return {
        "gt_fg_frac_mean": float(np.mean(fracs)),
        "gt_fg_pixels_mean": float(np.mean(fracs) * img_size * img_size),
    }
"""
    )
)

cells.append(
    code(
        """cfg = ABLATION_512
assert cfg["model_path"].exists(), f"Missing model: {cfg['model_path']}"

row_512 = eval_run("512_ablation", cfg["dataset"], cfg["model_path"], cfg["img_size"])
gt_512 = gt_mask_stats(cfg["dataset"], cfg["img_size"])
row_512.update(gt_512)

row_256_ref = {"run": "256_baseline_ref", **BASELINE_256}
gt_256 = gt_mask_stats(BASELINE_256["dataset"], BASELINE_256["img_size"])
row_256_ref.update(gt_256)
row_256_ref["gt_fg_frac_mean"] = row_256_ref.pop("mean_gt_fg_frac", gt_256["gt_fg_frac_mean"])
row_256_ref["gt_fg_pixels_mean"] = gt_256["gt_fg_pixels_mean"]

report = pd.DataFrame([row_256_ref, row_512])
display(report)

out = Path("/kaggle/working/resize_ablation_report.csv")
report.to_csv(out, index=False)
print(f"Wrote {out}")

if row_512.get("dice", 0) > BASELINE_256["val_dice"]:
    print("512px ablation beats 256px verify baseline on val Dice")
else:
    print("512px ablation did not beat 256px verify baseline on val Dice — review before full 512 prep")
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
    out = Path(__file__).resolve().parents[1] / "notebooks/eval-resize-ablation"
    write_nb(out / "eval-resize-ablation-phase-3.ipynb")
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-eval-resize-ablation-phase-3",
                "title": "UMUD Eval Resize Ablation Phase 3",
                "code_file": "eval-resize-ablation-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "keywords": ["gpu"],
                "dataset_sources": [
                    "ucheozoemena/umud-aligned-fasc-timing-50",
                    "ucheozoemena/umud-aligned-fasc-timing-50-512px",
                ],
                "kernel_sources": ["ucheozoemena/umud-train-mounted-phase-3"],
                "competition_sources": [],
                "model_sources": [],
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
    )
    print(f"Wrote {out / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
