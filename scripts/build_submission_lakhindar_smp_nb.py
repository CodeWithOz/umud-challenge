"""Generate Block20 submission notebook based on a public SMP U-Net++ approach.

The notebook trains fascicle and aponeurosis segmenters inside Kaggle, then
derives PA/FL/MT from the predicted masks. It writes a hidden-safe notebook
submission; no public-test CSV lookup is used.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks/submission-lakhindar-smp"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": src}


cells = [
    md(
        """# UMUD - Block20 SMP U-Net++ Geometry

GPU notebook adapted from the public EfficientNet-B7 U-Net++ baseline. It trains
both fascicle and aponeurosis segmenters from the competition masks, predicts
masks for the mounted test images, converts masks to PA/FL/MT geometry, smooths
within 5-frame groups, and writes `submission.csv`.

This is private-eligible because all predictions are computed from the images
available to the notebook at run time."""
    ),
    md("## Install dependencies"),
    code(
        """import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "albumentations",
        "ttach",
        "torch-optimizer",
        "segmentation-models-pytorch",
    ],
    check=True,
)
"""
    ),
    md("## Imports and configuration"),
    code(
        """from __future__ import annotations

import math
import os
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch_optimizer as optim
import ttach as tta

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setNumThreads(0)
warnings.filterwarnings("ignore")


class CFG:
    H = 512
    W = 768
    BATCH = 2
    ACCUM = 8
    EPOCHS = 30
    LR = 3e-4
    BACKBONE = "efficientnet-b7"
    PIXEL_TO_MM = 0.0881
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)

BASE_DIR = Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
FASC_IMG_DIR = BASE_DIR / "fasc_imgs_v1/fasc_images_new_model_v1"
FASC_MASK_DIR = BASE_DIR / "fasc_masks_v1/fasc_masks_new_model_v1"
APO_IMG_DIR = BASE_DIR / "apo_imgs_v1/apo_images_new_model_v1"
APO_MASK_DIR = BASE_DIR / "apo_masks_v1/apo_masks_new_model_v1"
TEST_DIR = BASE_DIR / "test_images_v2/test_set_v2"

print("device:", CFG.DEVICE)
print("torch:", torch.__version__)
"""
    ),
    md("## Dataset, loss, and geometry helpers"),
    code(
        """IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def match_mask(mask_dir: Path, image_path: Path) -> Path:
    direct = mask_dir / image_path.name
    if direct.exists():
        return direct
    matches = sorted(mask_dir.glob(f"{image_path.stem}.*"))
    if not matches:
        raise FileNotFoundError(f"No mask for {image_path.name}")
    return matches[0]


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(CFG.H, CFG.W),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(1, 64),
                hole_width_range=(1, 64),
                p=0.3,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.ElasticTransform(alpha=120, sigma=6.0, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.06, 0.06),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


class UltrasoundMaskDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, transform: A.Compose):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list_images(img_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = read_gray(img_path)
        mask = read_gray(match_mask(self.mask_dir, img_path))
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        aug = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)


class TestImageDataset(Dataset):
    def __init__(self, test_dir: Path):
        self.images = list_images(test_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        image = read_gray(path)
        orig_h, orig_w = image.shape
        resized = cv2.resize(image, (CFG.W, CFG.H))
        tensor = torch.tensor(((resized / 255.0).astype(np.float32) - 0.5) / 0.5).unsqueeze(0)
        return tensor, path.name, orig_h, orig_w


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.4, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_bce = bce_weight
        self.w_dice = 1.0 - bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        dice_score = (2.0 * (probs * targets).sum() + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return (self.w_bce * bce) + (self.w_dice * (1.0 - dice_score)), dice_score


def make_model(weights: str | None = "imagenet") -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name=CFG.BACKBONE,
        encoder_weights=weights,
        in_channels=1,
        classes=1,
    )


def extract_measurements_px(apo_mask: np.ndarray, fasc_mask: np.ndarray, orig_h: int, orig_w: int) -> dict:
    mid_y = orig_h // 2
    sup_mask = apo_mask.copy()
    deep_mask = apo_mask.copy()
    sup_mask[mid_y:, :] = 0
    deep_mask[:mid_y, :] = 0

    sup_pts = np.column_stack(np.where(sup_mask > 0))
    deep_pts = np.column_stack(np.where(deep_mask > 0))
    fasc_pts = np.column_stack(np.where(fasc_mask > 0))

    pa_deg = 15.0
    mt_px = 150.0
    status = "fallback"

    if len(sup_pts) > 50 and len(deep_pts) > 50:
        mt_px = abs(float(np.median(deep_pts[:, 0]) - np.median(sup_pts[:, 0])))
        status = "mt_ok"

    if len(fasc_pts) > 50 and len(deep_pts) > 50:
        try:
            [vx_f, vy_f, _, _] = cv2.fitLine(
                np.float32(np.flip(fasc_pts, axis=1)), cv2.DIST_L2, 0, 0.01, 0.01
            )
            [vx_d, vy_d, _, _] = cv2.fitLine(
                np.float32(np.flip(deep_pts, axis=1)), cv2.DIST_L2, 0, 0.01, 0.01
            )
            if vx_f[0] < 0:
                vx_f, vy_f = -vx_f, -vy_f
            if vx_d[0] < 0:
                vx_d, vy_d = -vx_d, -vy_d
            dot_product = np.clip((vx_f[0] * vx_d[0]) + (vy_f[0] * vy_d[0]), -1.0, 1.0)
            pa_deg = math.degrees(math.acos(float(dot_product)))
            if pa_deg > 90:
                pa_deg = 180 - pa_deg
            pa_deg = float(np.clip(pa_deg, 5.0, 45.0))
            status = status + "+pa_ok"
        except Exception:
            status = status + "+pa_fallback"

    fl_px = mt_px / (math.sin(math.radians(pa_deg)) + 1e-6)
    return {
        "pa_deg": pa_deg,
        "fl_px": float(fl_px),
        "mt_px": float(mt_px),
        "status": status,
    }


def px_to_raw_mm(row: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "pa_deg": float(row["pa_deg"]),
            "fl_mm": float(row["fl_px"] * CFG.PIXEL_TO_MM),
            "mt_mm": float(row["mt_px"] * CFG.PIXEL_TO_MM),
        }
    )
"""
    ),
    md("## Train fascicle and aponeurosis models"),
    code(
        """def train_target(name: str, img_dir: Path, mask_dir: Path, save_path: Path) -> None:
    ds = UltrasoundMaskDataset(img_dir, mask_dir, transform=train_transforms())
    loader = DataLoader(ds, batch_size=CFG.BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model = make_model(weights="imagenet").to(CFG.DEVICE)
    optimizer = optim.Ranger(
        model.parameters(),
        lr=CFG.LR,
        alpha=0.5,
        k=6,
        betas=(0.95, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)
    criterion = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))
    best_dice = -1.0
    print(f">>> Training {name}: n={len(ds)}, epochs={CFG.EPOCHS}, backbone={CFG.BACKBONE}")
    for epoch in range(CFG.EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"{name} {epoch + 1}/{CFG.EPOCHS}")
        for step, (imgs, masks) in enumerate(pbar, start=1):
            imgs = imgs.to(CFG.DEVICE, non_blocking=True)
            masks = masks.to(CFG.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(CFG.DEVICE == "cuda")):
                loss, dice = criterion(model(imgs), masks)
                loss = loss / CFG.ACCUM
            scaler.scale(loss).backward()
            if step % CFG.ACCUM == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach().cpu()) * CFG.ACCUM
            epoch_dice += float(dice.detach().cpu())
            pbar.set_postfix(loss=epoch_loss / step, dice=epoch_dice / step)
        scheduler.step()
        avg_dice = epoch_dice / max(1, len(loader))
        print(f"{name} epoch {epoch + 1}: train_dice={avg_dice:.4f}")
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), save_path)
            print(f"  saved {save_path} (best_dice={best_dice:.4f})")
    del model
    torch.cuda.empty_cache()


train_target("fascicle", FASC_IMG_DIR, FASC_MASK_DIR, Path("/kaggle/working/best_fascicle_model.pth"))
train_target("aponeurosis", APO_IMG_DIR, APO_MASK_DIR, Path("/kaggle/working/best_aponeurosis_model.pth"))
"""
    ),
    md("## Inference and smoothing"),
    code(
        """def load_trained(path: Path) -> nn.Module:
    model = make_model(weights=None)
    state = torch.load(path, map_location=CFG.DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(CFG.DEVICE)
    model.eval()
    return model


apo_model = load_trained(Path("/kaggle/working/best_aponeurosis_model.pth"))
fasc_model = load_trained(Path("/kaggle/working/best_fascicle_model.pth"))
transforms = tta.Compose([tta.HorizontalFlip(), tta.Multiply(factors=[0.9, 1.0, 1.1])])
tta_apo = tta.SegmentationTTAWrapper(apo_model, transforms, merge_mode="mean")
tta_fasc = tta.SegmentationTTAWrapper(fasc_model, transforms, merge_mode="mean")

loader = DataLoader(TestImageDataset(TEST_DIR), batch_size=CFG.BATCH, shuffle=False, num_workers=2)
rows = []
with torch.no_grad():
    for tensors, filenames, orig_hs, orig_ws in tqdm(loader, desc="test inference"):
        tensors = tensors.to(CFG.DEVICE)
        apo_probs = torch.sigmoid(tta_apo(tensors)).squeeze(1).detach().cpu().numpy()
        fasc_probs = torch.sigmoid(tta_fasc(tensors)).squeeze(1).detach().cpu().numpy()
        for i, image_id in enumerate(filenames):
            orig_h = int(orig_hs[i])
            orig_w = int(orig_ws[i])
            apo_mask = cv2.resize(
                (apo_probs[i] > 0.5).astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            fasc_mask = cv2.resize(
                (fasc_probs[i] > 0.5).astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            geom = extract_measurements_px(apo_mask, fasc_mask, orig_h, orig_w)
            geom.update(
                {
                    "image_id": image_id,
                    "apo_cov": float(apo_mask.mean()),
                    "fasc_cov": float(fasc_mask.mean()),
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                }
            )
            rows.append(geom)

debug = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
raw_mm = debug.apply(px_to_raw_mm, axis=1)
raw = pd.concat([debug[["image_id"]], raw_mm], axis=1)
raw[["pa_deg", "fl_mm", "mt_mm"]] = raw[["pa_deg", "fl_mm", "mt_mm"]].clip(
    lower=[5.0, 30.0, 10.0],
    upper=[45.0, 200.0, 50.0],
    axis=1,
)

smooth = raw.copy()
smooth["video_group"] = np.arange(len(smooth)) // 5
for cols in [["pa_deg"], ["fl_mm"], ["mt_mm"]]:
    col = cols[0]
    smoothed_parts = []
    for _, group in smooth.groupby("video_group", sort=False):
        vals = group[col].to_numpy(float)
        if len(vals) >= 5:
            vals = savgol_filter(vals, window_length=5, polyorder=2)
        elif len(vals) >= 3:
            vals = savgol_filter(vals, window_length=3, polyorder=2)
        smoothed_parts.extend(vals.tolist())
    smooth[col] = smoothed_parts
smooth = smooth.drop(columns=["video_group"])
smooth[["pa_deg", "fl_mm", "mt_mm"]] = smooth[["pa_deg", "fl_mm", "mt_mm"]].clip(
    lower=[5.0, 30.0, 10.0],
    upper=[45.0, 200.0, 50.0],
    axis=1,
)

debug.to_csv("/kaggle/working/submission_debug_px.csv", index=False)
raw.to_csv("/kaggle/working/submission_raw.csv", index=False)
smooth.to_csv("/kaggle/working/submission.csv", index=False)

print("debug rows:", len(debug), "raw NaNs:", int(raw.isna().sum().sum()), "smooth NaNs:", int(smooth.isna().sum().sum()))
display(debug[["pa_deg", "fl_px", "mt_px", "apo_cov", "fasc_cov"]].describe().round(3))
display(smooth[["pa_deg", "fl_mm", "mt_mm"]].describe().round(3))
display(smooth.head())
"""
    ),
]


def main() -> None:
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
    (OUT_DIR / "submission-lakhindar-smp.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-submission-lakhindar-smp",
        "title": "UMUD Submission Lakhindar SMP",
        "code_file": "submission-lakhindar-smp.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
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
