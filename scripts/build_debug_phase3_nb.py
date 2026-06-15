"""Generate notebooks/debug-phase3/debug-phase3.ipynb — Kaggle runtime debug for Dice + submission NaNs."""
import json
from pathlib import Path


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in s.split("\n")]}


def code(s: str) -> dict:
    lines = s.split("\n")
    src = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


cells = [
    md("""# UMUD Phase 3 — Debug (val Dice + submission NaNs)

Runs on Kaggle GPU with same fastai env as training. Writes `debug_report.json` + `debug_pred_stats.csv`."""),
    code(
        """import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import kagglehub
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

LOG_PATH = Path("/kaggle/working/debug_report.json")
RANDOM_SEED = 42
VALID_PCT = 0.20
BATCH_SIZE = 8
IMG_SIZE = 256
SEG_CODES = ["background", "structure"]

FASC_DATASET = "ucheozoemena/umud-aligned-fasc-full"
APO_DATASET = "ucheozoemena/umud-aligned-apo-full"
FASC_MODEL = Path("/kaggle/input/notebooks/ucheozoemena/umud-train-mounted-phase-3/fasc_baseline.pkl")
APO_MODEL = Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-mounted-phase-3/apo_baseline.pkl")
COMP_DIR = Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
TEST_DIR = COMP_DIR / "test_images_v2/test_set_v2"

debug_logs = []


def log(hypothesis_id, location, message, data):
    debug_logs.append({
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    })


def dataset_root(slug):
    p = Path(f"/kaggle/input/datasets/{slug}")
    return p if p.exists() else Path(kagglehub.dataset_download(slug))


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


def make_dls(img_dir, msk_dir, fnames):
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
        splitter=RandomSplitter(valid_pct=VALID_PCT, seed=RANDOM_SEED),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=False),
    )
    return block.dataloaders(fnames, bs=BATCH_SIZE, num_workers=2)


def pair_fnames(img_dir, msk_dir):
    imgs = get_image_files(img_dir)
    msk_names = {p.name for p in get_image_files(msk_dir)}
    return [f for f in imgs if f.name in msk_names]


def metric_label(m):
    return getattr(m, "name", None) or type(m).__name__


def analyze_preds(learn, dl, n_batches=30):
    import torch

    pred_fg, gt_fg = [], []
    pred_pix = {0: 0, 1: 0}
    manual_dice = []
    learn.model.eval()
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        xb, yb = batch
        with torch.no_grad():
            logits = learn.model(xb)
        pred_cls = logits.argmax(dim=1).cpu().numpy()
        gt = yb.cpu().numpy()
        for c in (0, 1):
            pred_pix[c] += int((pred_cls == c).sum())
        pred_fg.append(float((pred_cls == 1).mean()))
        gt_fg.append(float((gt == 1).mean()))
        for p, g in zip(pred_cls, gt):
            p1, g1 = p == 1, g == 1
            denom = p1.sum() + g1.sum()
            manual_dice.append(float((p1 & g1).sum() / denom) if denom else 0.0)
    return {
        "mean_pred_fg_frac": float(np.mean(pred_fg)),
        "mean_gt_fg_frac": float(np.mean(gt_fg)),
        "pred_class_pixels": pred_pix,
        "manual_dice_mean": float(np.mean(manual_dice)),
    }


def eval_track(name, slug, model_path):
    root = dataset_root(slug)
    img_dir, msk_dir = root / "images", root / "masks"
    fnames = pair_fnames(img_dir, msk_dir)
    dls = make_dls(img_dir, msk_dir, fnames)
    learn = load_learner(model_path)
    learn.dls = dls
    results = learn.validate(dl=dls.valid)
    labels = [metric_label(m) for m in learn.metrics]
    manual = analyze_preds(learn, dls.valid)
    row = {
        "track": name,
        "n_pairs": len(fnames),
        "n_val": len(dls.valid_ds),
        "loss": float(results[0]),
        **{labels[i]: float(results[i + 1]) for i in range(len(labels))},
        **manual,
    }
    log("A", "eval_track", "validate + manual", row)
    return learn, row


def tensor_to_mask(pred):
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    arr = np.asarray(pred)
    if arr.ndim == 3:
        arr = arr.argmax(axis=0)
    return (arr > 0).astype(np.uint8)


def fascicle_pca(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    return True


def test_inference_stats(fasc_learn, apo_learn, n=50):
    from PIL import Image

    files = sorted([p for p in TEST_DIR.rglob("*.tif")])[:n]
    rows = []
    for p in files:
        with Image.open(p) as im:
            arr = np.array(im)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        h, w = arr.shape
        pil = open_image_pil(p)
        _, ft, _ = fasc_learn.predict(pil)
        _, at, _ = apo_learn.predict(pil)
        fm = tensor_to_mask(ft)
        am = tensor_to_mask(at)
        fm_up = np.array(Image.fromarray((fm * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
        rows.append({
            "image_id": p.name,
            "fasc_cov_256": float(fm.mean()),
            "apo_cov_256": float(am.mean()),
            "fasc_cov_native": float(fm_up.mean()),
            "fasc_pca_ok": fascicle_pca(fm_up) is not None,
            "apo_cov_native": float(am.mean()),
        })
    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "fasc_cov_256_mean": float(df.fasc_cov_256.mean()),
        "apo_cov_256_mean": float(df.apo_cov_256.mean()),
        "fasc_pca_ok_rate": float(df.fasc_pca_ok.mean()),
        "fasc_all_empty": bool((df.fasc_cov_256 == 0).all()),
    }
    log("E", "test_inference", "test sample stats", summary)
    return df, summary


fasc_learn, fasc_row = eval_track("fasc", FASC_DATASET, FASC_MODEL)
apo_learn, apo_row = eval_track("apo", APO_DATASET, APO_MODEL)

test_df, test_summary = test_inference_stats(fasc_learn, apo_learn, n=80)

report = {"fasc": fasc_row, "apo": apo_row, "test": test_summary, "logs": debug_logs}
LOG_PATH.write_text(json.dumps(report, indent=2))
pd.DataFrame([fasc_row, apo_row]).to_csv("/kaggle/working/debug_val_summary.csv", index=False)
test_df.to_csv("/kaggle/working/debug_pred_stats.csv", index=False)
print(json.dumps({"fasc": fasc_row, "apo": apo_row, "test": test_summary}, indent=2))
"""
    ),
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

out = Path(__file__).resolve().parents[1] / "notebooks/debug-phase3"
out.mkdir(parents=True, exist_ok=True)
(out / "debug-phase3.ipynb").write_text(json.dumps(nb, indent=1))
(out / "kernel-metadata.json").write_text(
    json.dumps(
        {
            "id": "ucheozoemena/umud-debug-phase-3",
            "title": "UMUD Debug Phase 3",
            "code_file": "debug-phase3.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": ["gpu"],
            "dataset_sources": [
                "ucheozoemena/umud-aligned-fasc-full",
                "ucheozoemena/umud-aligned-apo-full",
            ],
            "kernel_sources": [
                "ucheozoemena/umud-train-mounted-phase-3",
                "ucheozoemena/umud-train-apo-mounted-phase-3",
            ],
            "competition_sources": [
                "umud-challenge-muscle-architecture-in-ultrasound-data"
            ],
            "model_sources": [],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
            "machine_shape": "NvidiaTeslaT4",
        },
        indent=2,
    )
)
print(f"Wrote {out}")
