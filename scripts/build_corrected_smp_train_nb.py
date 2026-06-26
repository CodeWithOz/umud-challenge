"""Generate the Block28 corrected SMP train-only notebook (saves weights).

Corrected vs Block20:
- **Apo polarity fix**: normalize every mask so the thin minority structure is always
  foreground (`if fg-fraction > 0.5: invert`). ~40% of apo masks were inverted, which
  corrupted the Block20 apo model. Fascicle masks (~0.3% fg) are never inverted.
- **Higher resolution 640x960** (vs 512x768) so thin fascicles keep detail. Benchmark
  (Block27): peak 5.9 GB @ batch1, 0.49/0.46 sec/pair/epoch; batch 2 fits T4 easily.
- **Wall-clock guard**: each target stops after a time budget (saving the best-Dice
  checkpoint), so the run never exceeds the Kaggle session limit.

Train-only: outputs `best_aponeurosis_model.pth` + `best_fascicle_model.pth` for a
later fast inference notebook to mount (PCA geometry + calibrate + ensemble).
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks/train-corrected-smp"
KERNEL_ID = "ucheozoemena/umud-train-corrected-smp"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source.split("\n")]}


def code(source: str) -> dict:
    lines = source.split("\n")
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])}


INSTALL = """import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "albumentations", "torch-optimizer", "segmentation-models-pytorch"], check=True)
"""

SETUP = '''import os, random, time, warnings
from pathlib import Path
import cv2, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch_optimizer as optim

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"; cv2.setNumThreads(0); warnings.filterwarnings("ignore")


class CFG:
    H = 640
    W = 960
    BATCH = 2
    ACCUM = 8
    EPOCHS = 18
    LR = 3e-4
    BACKBONE = "efficientnet-b7"
    SEED = 42
    FASC_BUDGET_S = 5.0 * 3600    # wall-clock guard per target
    APO_BUDGET_S = 3.0 * 3600
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)

BASE_DIR = Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
FASC_IMG_DIR = BASE_DIR / "fasc_imgs_v1/fasc_images_new_model_v1"
FASC_MASK_DIR = BASE_DIR / "fasc_masks_v1/fasc_masks_new_model_v1"
APO_IMG_DIR = BASE_DIR / "apo_imgs_v1/apo_images_new_model_v1"
APO_MASK_DIR = BASE_DIR / "apo_masks_v1/apo_masks_new_model_v1"
print("device:", CFG.DEVICE, "torch:", torch.__version__)
'''

DATA = '''IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory):
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def match_mask(mask_dir, image_path):
    direct = mask_dir / image_path.name
    if direct.exists():
        return direct
    matches = sorted(mask_dir.glob(f"{image_path.stem}.*"))
    if not matches:
        raise FileNotFoundError(f"No mask for {image_path.name}")
    return matches[0]


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def train_transforms():
    return A.Compose([
        A.Resize(CFG.H, CFG.W),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(1, 64), hole_width_range=(1, 64), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ElasticTransform(alpha=120, sigma=6.0, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-15, 15), p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


class UltrasoundMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir; self.mask_dir = mask_dir
        self.transform = train_transforms(); self.images = list_images(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_gray(img_path)
        mask = read_gray(match_mask(self.mask_dir, img_path))
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        if mask.mean() > 0.5:            # polarity fix: aponeurosis is the thin minority structure
            mask = 1.0 - mask
        aug = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.4, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(); self.w_bce = bce_weight
        self.w_dice = 1.0 - bce_weight; self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).view(-1); targets = targets.view(-1)
        dice = (2.0 * (probs * targets).sum() + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return (self.w_bce * bce) + (self.w_dice * (1.0 - dice)), dice


def make_model():
    return smp.UnetPlusPlus(encoder_name=CFG.BACKBONE, encoder_weights="imagenet", in_channels=1, classes=1)
'''

TRAIN = '''def train_target(name, img_dir, mask_dir, save_path, budget_s):
    ds = UltrasoundMaskDataset(img_dir, mask_dir)
    loader = DataLoader(ds, batch_size=CFG.BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model = make_model().to(CFG.DEVICE)
    optimizer = optim.Ranger(model.parameters(), lr=CFG.LR, alpha=0.5, k=6, betas=(0.95, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)
    criterion = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))
    best_dice = -1.0
    t0 = time.perf_counter()
    print(f">>> Training {name}: n={len(ds)} epochs={CFG.EPOCHS} backbone={CFG.BACKBONE} res={CFG.H}x{CFG.W} budget={budget_s/3600:.1f}h", flush=True)
    for epoch in range(CFG.EPOCHS):
        model.train(); ep_dice = 0.0; nb = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, masks) in enumerate(tqdm(loader, desc=f"{name} {epoch+1}/{CFG.EPOCHS}"), start=1):
            imgs = imgs.to(CFG.DEVICE, non_blocking=True); masks = masks.to(CFG.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(CFG.DEVICE == "cuda")):
                loss, dice = criterion(model(imgs), masks); loss = loss / CFG.ACCUM
            scaler.scale(loss).backward()
            if step % CFG.ACCUM == 0 or step == len(loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            ep_dice += float(dice.detach().cpu()); nb += 1
        scheduler.step()
        ep_dice /= max(1, nb)
        elapsed = time.perf_counter() - t0
        if ep_dice > best_dice:
            best_dice = ep_dice
            torch.save(model.state_dict(), save_path)
            print(f"  [{name}] epoch {epoch+1} dice={ep_dice:.4f} *saved* elapsed={elapsed/3600:.2f}h", flush=True)
        else:
            print(f"  [{name}] epoch {epoch+1} dice={ep_dice:.4f} elapsed={elapsed/3600:.2f}h", flush=True)
        if elapsed > budget_s:
            print(f"  [{name}] wall-clock budget hit ({elapsed/3600:.2f}h) - stopping early", flush=True)
            break
    if best_dice < 0:   # ensure a file exists
        torch.save(model.state_dict(), save_path)
    del model
    torch.cuda.empty_cache()
    return {"target": name, "best_dice": best_dice, "elapsed_h": (time.perf_counter()-t0)/3600.0,
            "epochs_cfg": CFG.EPOCHS, "res": f"{CFG.H}x{CFG.W}", "n": len(ds)}


rows = []
# Fascicle first (the bottleneck / larger set), then aponeurosis.
rows.append(train_target("fascicle", FASC_IMG_DIR, FASC_MASK_DIR, "/kaggle/working/best_fascicle_model.pth", CFG.FASC_BUDGET_S))
rows.append(train_target("aponeurosis", APO_IMG_DIR, APO_MASK_DIR, "/kaggle/working/best_aponeurosis_model.pth", CFG.APO_BUDGET_S))
report = pd.DataFrame(rows)
report.to_csv("/kaggle/working/train_report.csv", index=False)
print(report.to_string(index=False))
print("saved:", os.path.exists("/kaggle/working/best_fascicle_model.pth"), os.path.exists("/kaggle/working/best_aponeurosis_model.pth"))
'''


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = [
        md("# UMUD - Block28 Corrected SMP Train (apo polarity fix + 640x960)\n\n"
           "Train-only. Saves `best_fascicle_model.pth` + `best_aponeurosis_model.pth` for a\n"
           "later fast inference notebook (PCA geometry + calibrate + ensemble). Wall-clock\n"
           "guard keeps the run under the Kaggle session limit (best-Dice checkpoint saved)."),
        md("## Install dependencies"), code(INSTALL),
        md("## Setup"), code(SETUP),
        md("## Dataset, loss, model (apo polarity fix in dataset)"), code(DATA),
        md("## Train both targets and save weights"), code(TRAIN),
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10.0"}},
          "cells": cells}
    (OUT_DIR / "train-corrected-smp.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": KERNEL_ID,
        "title": "UMUD Train Corrected SMP",
        "code_file": "train-corrected-smp.ipynb",
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
