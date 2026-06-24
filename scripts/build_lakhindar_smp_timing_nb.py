"""Generate a Block20 timing benchmark notebook for the SMP U-Net++ approach.

The benchmark uses the same expensive ingredients as the full Block20 notebook
but caps sample counts and epochs so we can estimate runtime before committing
to a full hidden-safe submission run.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks/bench-lakhindar-smp-timing"


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
        """# UMUD - Block20 SMP Timing Benchmark

This notebook benchmarks the expensive Block20 ingredients before a full
submission run: 512x768 images, EfficientNet-B7 U-Net++, BCE+Dice loss, Ranger,
and the same train transforms. It trains one capped epoch for fascicle and
aponeurosis, then writes `timing_report.csv`.
"""
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

import os
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch_optimizer as optim

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setNumThreads(0)
warnings.filterwarnings("ignore")


class CFG:
    H = 512
    W = 768
    BATCH = 2
    ACCUM = 8
    EPOCHS = 1
    MAX_FASC = 80
    MAX_APO = 80
    LR = 3e-4
    BACKBONE = "efficientnet-b7"
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

print("device:", CFG.DEVICE)
print("cuda devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}:", torch.cuda.get_device_name(i))
print("torch:", torch.__version__)
"""
    ),
    md("## Dataset and model"),
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
    def __init__(self, img_dir: Path, mask_dir: Path):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = train_transforms()
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


def make_model() -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name=CFG.BACKBONE,
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
"""
    ),
    md("## Timed capped train"),
    code(
        """def timed_train(name: str, img_dir: Path, mask_dir: Path, max_items: int) -> dict:
    base_ds = UltrasoundMaskDataset(img_dir, mask_dir)
    n = min(max_items, len(base_ds))
    ds = Subset(base_ds, list(range(n)))
    loader = DataLoader(ds, batch_size=CFG.BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model = make_model().to(CFG.DEVICE)
    optimizer = optim.Ranger(
        model.parameters(),
        lr=CFG.LR,
        alpha=0.5,
        k=6,
        betas=(0.95, 0.999),
        weight_decay=1e-4,
    )
    criterion = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))
    optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_mem = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        start_mem = 0

    start = time.perf_counter()
    epoch_loss = 0.0
    epoch_dice = 0.0
    for step, (imgs, masks) in enumerate(tqdm(loader, desc=f"bench {name}"), start=1):
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0
    elapsed = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()
    return {
        "target": name,
        "backbone": CFG.BACKBONE,
        "height": CFG.H,
        "width": CFG.W,
        "batch": CFG.BATCH,
        "accum": CFG.ACCUM,
        "epochs": CFG.EPOCHS,
        "samples": n,
        "batches": len(loader),
        "train_sec": elapsed,
        "sec_per_pair_epoch": elapsed / max(1, n),
        "avg_loss": epoch_loss / max(1, len(loader)),
        "avg_dice": epoch_dice / max(1, len(loader)),
        "cuda_devices": torch.cuda.device_count(),
        "start_mem_bytes": int(start_mem),
        "peak_mem_bytes": int(peak_mem),
    }


rows = [
    timed_train("fascicle", FASC_IMG_DIR, FASC_MASK_DIR, CFG.MAX_FASC),
    timed_train("aponeurosis", APO_IMG_DIR, APO_MASK_DIR, CFG.MAX_APO),
]
report = pd.DataFrame(rows)
report["projected_full_30ep_sec"] = report.apply(
    lambda r: r["sec_per_pair_epoch"] * (2749 if r["target"] == "fascicle" else 1048) * 30,
    axis=1,
)
report["projected_full_30ep_hours"] = report["projected_full_30ep_sec"] / 3600.0
report.to_csv("/kaggle/working/timing_report.csv", index=False)
display(report)
print("projected total hours:", round(float(report["projected_full_30ep_hours"].sum()), 3))
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
    (OUT_DIR / "bench-lakhindar-smp-timing.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-bench-lakhindar-smp-timing",
        "title": "UMUD Bench Lakhindar SMP Timing",
        "code_file": "bench-lakhindar-smp-timing.ipynb",
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
