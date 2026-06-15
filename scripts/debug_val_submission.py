"""Local debug: validation Dice + submission geometry. Writes NDJSON to .cursor/debug-4a4da7.log"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastai.data.block import DataBlock
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
    load_learner,
    resnet34,
    unet_learner,
)

REPO = Path(__file__).resolve().parents[1]
LOG_PATH = REPO / ".cursor/debug-4a4da7.log"
SESSION = "4a4da7"
RUN_ID = "local-debug-1"

FASC_DATASET = Path(
    "/Users/ucheozoemena/.cache/kagglehub/datasets/ucheozoemena/umud-aligned-fasc-full/versions/1"
)
APO_DATASET = Path(
    "/Users/ucheozoemena/.cache/kagglehub/datasets/ucheozoemena/umud-aligned-apo-full/versions/1"
)
FASC_MODEL = REPO / "tmp/kaggle-output/models/fasc/fasc_baseline.pkl"
APO_MODEL = REPO / "tmp/kaggle-output/models/apo/apo_baseline.pkl"

RANDOM_SEED = 42
VALID_PCT = 0.20
BATCH_SIZE = 8
IMG_SIZE = 256
SEG_CODES = ["background", "structure"]
APO_REGION_THRESHOLD = 0.50


def log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": SESSION,
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


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


def make_dls(img_dir: Path, msk_dir: Path, fnames, do_aug: bool = True):
    batch_tfms = (
        aug_transforms(size=IMG_SIZE, min_scale=0.75, flip_vert=False, do_flip=do_aug)
        if do_aug
        else None
    )
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
        batch_tfms=batch_tfms,
    )
    return block.dataloaders(fnames, bs=BATCH_SIZE, num_workers=0)


def pair_fnames(img_dir: Path, msk_dir: Path):
    img_fnames = get_image_files(img_dir)
    msk_names = {p.name for p in get_image_files(msk_dir)}
    return [f for f in img_fnames if f.name in msk_names]


def mask_coverage(arr: np.ndarray) -> float:
    return float((arr > 0).mean())


def tensor_to_mask(pred) -> np.ndarray:
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    arr = np.asarray(pred)
    if arr.ndim == 3:
        arr = arr.argmax(axis=0)
    return (arr > 0).astype(np.uint8)


def fascicle_pca(mask: np.ndarray) -> dict | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    return {"length_px": float(projections.max() - projections.min())}


def analyze_batch_preds(learn, dl, n_batches: int = 5) -> dict:
    """Manual Dice + pred class histogram on validation batches."""
    pred_fg = []
    gt_fg = []
    pred_class_counts = {0: 0, 1: 0}
    learn.model.eval()
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        xb, yb = batch
        with learn.no_bar():
            preds = learn.model(xb)
        pred_cls = preds.argmax(dim=1).detach().cpu().numpy()
        # yb: mask codes from AddMaskCodes
        gt = yb.detach().cpu().numpy()
        if gt.ndim == 4:
            gt_cls = gt[:, 0]  # channel 0 is class index
        else:
            gt_cls = gt
        for c in [0, 1]:
            pred_class_counts[c] += int((pred_cls == c).sum())
        pred_fg.append(float((pred_cls == 1).mean()))
        gt_fg.append(float((gt_cls == 1).mean()))
    inter = []
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        xb, yb = batch
        with learn.no_bar():
            preds = learn.model(xb)
        pred_cls = preds.argmax(dim=1).detach().cpu().numpy()
        gt = yb.detach().cpu().numpy()
        gt_cls = gt[:, 0] if gt.ndim == 4 else gt
        for p, g in zip(pred_cls, gt_cls):
            p1 = p == 1
            g1 = g == 1
            denom = p1.sum() + g1.sum()
            inter.append(float((p1 & g1).sum() / denom) if denom else 0.0)
    return {
        "mean_pred_fg_frac": float(np.mean(pred_fg)),
        "mean_gt_fg_frac": float(np.mean(gt_fg)),
        "pred_class_pixel_counts": pred_class_counts,
        "manual_dice_per_image_mean": float(np.mean(inter)),
        "n_images": len(inter),
    }


def eval_track(name: str, dataset_root: Path, model_path: Path) -> dict:
    img_dir = dataset_root / "images"
    msk_dir = dataset_root / "masks"
    fnames = pair_fnames(img_dir, msk_dir)
    dls_aug = make_dls(img_dir, msk_dir, fnames, do_aug=True)
    dls_noaug = make_dls(img_dir, msk_dir, fnames, do_aug=False)

    learn = load_learner(model_path)
    orig_n_train = len(learn.dls.train_ds)
    orig_n_valid = len(learn.dls.valid_ds)

    # H-B: validate with swapped dls (matches eval notebook)
    learn.dls = dls_aug
    results_aug = learn.validate(dl=dls_aug.valid)
    metric_names = [getattr(m, "__name__", type(m).__name__) for m in learn.metrics]

    # H-D: validate without aug batch tfms
    results_noaug = learn.validate(dl=dls_noaug.valid)

    manual = analyze_batch_preds(learn, dls_aug.valid, n_batches=20)

    # H-A: sample single-image predict
    sample = dls_aug.valid_ds[0]
    pil_img = sample[0]
    gt_mask = np.array(sample[1])
    _, pred_t, _ = learn.predict(pil_img)
    pred_mask = tensor_to_mask(pred_t)
    if gt_mask.ndim == 3:
        gt_bin = (gt_mask[..., 0] == 1).astype(np.uint8)
    else:
        gt_bin = (gt_mask == 1).astype(np.uint8)

    log("A", "debug_val_submission:eval_track", "single sample predict", {
        "track": name,
        "gt_fg_frac": mask_coverage(gt_bin),
        "pred_fg_frac": mask_coverage(pred_mask),
        "pred_unique": sorted(map(int, np.unique(pred_mask)).tolist()),
        "gt_unique": sorted(map(int, np.unique(gt_bin)).tolist()),
    })

    log("B", "debug_val_submission:eval_track", "validate swapped dls", {
        "track": name,
        "loss_aug": float(results_aug[0]),
        "metrics_aug": {metric_names[i]: float(results_aug[i + 1]) for i in range(len(metric_names))},
        "loss_noaug": float(results_noaug[0]),
        "metrics_noaug": {metric_names[i]: float(results_noaug[i + 1]) for i in range(len(metric_names))},
        "metric_names": metric_names,
        "str_metric_names": [str(m) for m in learn.metrics],
        "orig_dls_train": orig_n_train,
        "orig_dls_valid": orig_n_valid,
        "new_valid": len(dls_aug.valid_ds),
    })

    log("C", "debug_val_submission:eval_track", "manual batch dice", {
        "track": name,
        **manual,
    })

    return {"track": name, "manual": manual, "results_aug": results_aug}


def geometry_on_val_sample(learn_fasc, learn_apo, fasc_root: Path, apo_root: Path) -> None:
    """H-E: run submission-style pipeline on a few val images with known masks."""
    fasc_img = fasc_root / "images"
    fasc_msk = fasc_root / "masks"
    fnames = pair_fnames(fasc_img, fasc_msk)[:30]
    pa_ok = fl_ok = mt_ok = 0
    n = 0
    for f in fnames:
        img_native = np.array(PILImage.create(fasc_img / f.name))
        if img_native.ndim == 3:
            img_native = img_native[..., 0]
        h, w = img_native.shape
        pil = open_image_pil(fasc_img / f.name)

        _, fasc_t, _ = learn_fasc.predict(pil)
        _, apo_t, _ = learn_apo.predict(pil)
        fasc_pred = tensor_to_mask(fasc_t)
        apo_pred = tensor_to_mask(apo_t)

        # upscale like submission
        fasc_up = np.array(
            PILImage.fromarray((fasc_pred * 255).astype(np.uint8)).resize((w, h), PILImage.NEAREST)
        ) > 0
        apo_up = np.array(
            PILImage.fromarray((apo_pred * 255).astype(np.uint8)).resize((w, h), PILImage.NEAREST)
        ) > 0

        apo_style = "region" if apo_up.mean() >= APO_REGION_THRESHOLD else "line"
        fpca = fascicle_pca(fasc_up.astype(np.uint8))
        n += 1
        if fpca is not None:
            fl_ok += 1
        if apo_up.sum() > 0:
            mt_ok += 1

        if n <= 3:
            log("E", "debug_val_submission:geometry", "val sample inference", {
                "file": f.name,
                "pred_fasc_cov_256": mask_coverage(fasc_pred),
                "pred_apo_cov_256": mask_coverage(apo_pred),
                "pred_fasc_cov_native": mask_coverage(fasc_up),
                "pred_apo_cov_native": mask_coverage(apo_up),
                "apo_style": apo_style,
                "fpca_ok": fpca is not None,
                "fl_px": fpca["length_px"] if fpca else None,
            })

    log("E", "debug_val_submission:geometry", "val geometry summary", {
        "n": n,
        "fl_ok_rate": fl_ok / max(1, n),
        "mt_ok_rate": mt_ok / max(1, n),
    })


def check_dice_metric_behavior() -> None:
    """H-B/C: manual dice on synthetic all-background predictions."""
    pred_cls = np.zeros((8, 8), dtype=np.int64)
    gt_cls = np.zeros((8, 8), dtype=np.int64)
    gt_cls[2:6, 2:6] = 1
    p1, g1 = pred_cls == 1, gt_cls == 1
    denom = p1.sum() + g1.sum()
    d = float((p1 & g1).sum() / denom) if denom else 0.0
    log("B", "debug_val_submission:check_dice", "synthetic all-bg pred", {
        "manual_dice": d,
        "targ_fg_frac": float((gt_cls == 1).mean()),
    })


def main() -> None:
    log("INIT", "debug_val_submission:main", "start", {
        "fasc_dataset": str(FASC_DATASET),
        "apo_dataset": str(APO_DATASET),
        "fasc_model": str(FASC_MODEL),
        "apo_model": str(APO_MODEL),
    })

    check_dice_metric_behavior()

    fasc_learn = load_learner(FASC_MODEL)
    apo_learn = load_learner(APO_MODEL)

    log("A", "debug_val_submission:main", "loaded learners", {
        "fasc_metrics": [str(m) for m in fasc_learn.metrics],
        "apo_metrics": [str(m) for m in apo_learn.metrics],
        "fasc_n_classes": getattr(fasc_learn, "n_out", None),
        "fasc_loss": str(fasc_learn.loss_func),
    })

    eval_track("fasc", FASC_DATASET, FASC_MODEL)
    eval_track("apo", APO_DATASET, APO_MODEL)

    geometry_on_val_sample(fasc_learn, apo_learn, FASC_DATASET, APO_DATASET)

    # H-A: check if retraining 1 batch improves pred fg
    img_dir = FASC_DATASET / "images"
    msk_dir = FASC_DATASET / "masks"
    fnames = pair_fnames(img_dir, msk_dir)[:50]
    dls = make_dls(img_dir, msk_dir, fnames, do_aug=False)
    mini = unet_learner(
        dls,
        resnet34,
        metrics=[Dice()],
        loss_func=CrossEntropyLossFlat(axis=1),
        self_attention=True,
    )
    mini.fine_tune(1)
    manual_after = analyze_batch_preds(mini, dls.valid, n_batches=5)
    log("A", "debug_val_submission:mini_train", "after 1 epoch on 50 fasc", manual_after)

    print("Debug complete. Logs:", LOG_PATH)


if __name__ == "__main__":
    main()
