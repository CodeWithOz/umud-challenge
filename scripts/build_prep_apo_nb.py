"""Generate notebooks/prep-apo-timing/prep-apo-timing.ipynb — Kaggle-native aligned apo dataset prep."""
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
        """# UMUD — Prepare Aligned Aponeurosis Dataset (Kaggle-native)

**CPU notebook** — same BirdCLEF gen-species pattern as fasc prep.

1. Read competition apo TIFFs from `/kaggle/input/competitions/...`
2. Stretch-align masks, resize to **256×256** (NEAREST masks)
3. Write PNG pairs + manifest to `/kaggle/working/upload/`
4. **`kaggle datasets create` / `version`** from inside this notebook

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42
PREP_RUN = 1  # 1=50, 2=200, 3=524 (50%), 4=1048 (full apo)

IMG_SIZE = 256
APO_FULL = 1048

PREP_PROFILES = {
    1: {
        "max_samples": 50,
        "dataset_id": "ucheozoemena/umud-aligned-apo-timing-50",
        "dataset_title": "UMUD Aligned Apo Timing 50",
        "version_msg": "AP1 timing: 50 apo pairs, 256px stretch-aligned",
        "zip_name": "umud_apo_timing_1",
    },
    2: {
        "max_samples": 200,
        "dataset_id": "ucheozoemena/umud-aligned-apo-timing-200",
        "dataset_title": "UMUD Aligned Apo Timing 200",
        "version_msg": "AP2 timing: 200 apo pairs, 256px stretch-aligned",
        "zip_name": "umud_apo_timing_2",
    },
    3: {
        "max_samples": APO_FULL // 2,
        "dataset_id": "ucheozoemena/umud-aligned-apo-timing-524",
        "dataset_title": "UMUD Aligned Apo Timing 524",
        "version_msg": "AP3 scaling check: 50% apo pairs (524), 256px stretch-aligned",
        "zip_name": "umud_apo_timing_3",
    },
    4: {
        "max_samples": APO_FULL,
        "dataset_id": "ucheozoemena/umud-aligned-apo-full",
        "dataset_title": "UMUD Aligned Apo Full",
        "version_msg": "Full apo prep: 1048 pairs, 256px stretch-aligned",
        "zip_name": "umud_apo_full",
    },
}

profile = PREP_PROFILES[PREP_RUN]
MAX_SAMPLES = profile["max_samples"]
DATASET_ID = profile["dataset_id"]
DATASET_TITLE = profile["dataset_title"]
VERSION_MSG = profile["version_msg"]
ZIP_NAME = profile["zip_name"]
print(f"PREP_RUN={PREP_RUN} | n<={MAX_SAMPLES} | dataset={DATASET_ID}")
"""
    )
)

cells.append(
    code(
        """from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

COMPETITION_DIR = Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
UPLOAD = Path("/kaggle/working/upload")
IMG_OUT = UPLOAD / "images"
MSK_OUT = UPLOAD / "masks"
MANIFEST_OUT = UPLOAD / "manifests"
TIMING_OUT = Path("/kaggle/working")

DIRS = {
    "apo_img": COMPETITION_DIR / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": COMPETITION_DIR / "apo_masks_v1/apo_masks_new_model_v1",
}
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


def load_gray(path: Path) -> np.ndarray:
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


def align_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask
    return (
        np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (target_w, target_h), Image.NEAREST
            )
        )
        > 0
    ).astype(np.uint8)


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    img_pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    msk_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return np.array(img_pil, dtype=np.uint8), (np.array(msk_pil) > 0).astype(np.uint8)


def subsample(fnames: list[str], max_n: int, seed: int) -> list[str]:
    if len(fnames) <= max_n:
        return fnames
    rng = random.Random(seed)
    return sorted(rng.sample(fnames, max_n))
"""
    )
)

cells.append(
    code(
        """if UPLOAD.exists():
    shutil.rmtree(UPLOAD)
UPLOAD.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)
MSK_OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT.mkdir(parents=True, exist_ok=True)

lookups = {k: build_lookup(v) for k, v in DIRS.items()}
print("Lookups:", {k: len(v) for k, v in lookups.items()})

t0 = time.perf_counter()
apo_common = sorted(set(lookups["apo_img"]) & set(lookups["apo_mask"]))
t_manifest = time.perf_counter()

targets = subsample(apo_common, MAX_SAMPLES, RANDOM_SEED)
print(f"Apo pairs: {len(apo_common)} | prep targets: {len(targets)}")
print(f"Manifest scan: {t_manifest - t0:.1f}s")
"""
    )
)

cells.append(
    code(
        """rows = []
t_prep = time.perf_counter()
for name in tqdm(targets, desc="prep pairs"):
    img = load_gray(lookups["apo_img"][name])
    mask = load_mask(lookups["apo_mask"][name])
    aligned = align_mask(mask, img.shape[0], img.shape[1])
    img_r, mask_r = resize_pair(img, aligned, IMG_SIZE)
    stem = Path(name).stem
    img_path = IMG_OUT / f"{stem}.png"
    msk_path = MSK_OUT / f"{stem}.png"
    Image.fromarray(img_r).save(img_path)
    Image.fromarray((mask_r * 255).astype(np.uint8)).save(msk_path)
    rows.append({"filename": name, "stem": stem, "img_h": img.shape[0], "img_w": img.shape[1]})
t_done = time.perf_counter()

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST_OUT / "train_apo_all.csv", index=False)

timing = pd.DataFrame(
    [
        {
            "prep_run": PREP_RUN,
            "n_pairs": len(targets),
            "img_size": IMG_SIZE,
            "manifest_sec": round(t_manifest - t0, 1),
            "transform_sec": round(t_done - t_prep, 1),
            "total_sec": round(t_done - t0, 1),
            "sec_per_pair": round((t_done - t_prep) / max(1, len(targets)), 3),
            "dataset_id": DATASET_ID,
        }
    ]
)
timing.to_csv(TIMING_OUT / "prep_timing.csv", index=False)
display(timing)
print(f"Wrote {len(targets)} pairs to {UPLOAD}")
"""
    )
)

cells.append(
    code(
        """import zipfile

ZIP_STAGING = Path("/kaggle/working/upload_staging")
ZIP_STAGING.mkdir(parents=True, exist_ok=True)
for old in ZIP_STAGING.glob("*.zip"):
    old.unlink()

archive = ZIP_STAGING / f"{ZIP_NAME}.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fp in UPLOAD.rglob("*"):
        if fp.is_file() and fp.name != "dataset-metadata.json":
            zf.write(fp, arcname=fp.relative_to(UPLOAD))

(ZIP_STAGING / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "title": DATASET_TITLE,
            "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        },
        indent=2,
    )
)

zips = sorted(ZIP_STAGING.glob("*.zip"))
total_mb = sum(p.stat().st_size for p in zips) / 1e6
print(f"Uploading {len(zips)} zip(s), {total_mb:.1f} MB")
for z in zips:
    print(f"  {z.name} ({z.stat().st_size / 1e6:.1f} MB)")

def upload_ok(result: subprocess.CompletedProcess) -> bool:
    out = (result.stdout or "") + (result.stderr or "")
    if "Your private Dataset is being created" in out:
        return True
    if "Dataset creation error" in out or "Upload failed" in out:
        return False
    if result.returncode != 0:
        return False
    if "Upload successful" in out:
        return True
    return False

print("Running: kaggle datasets version ...")
result = subprocess.run(
    ["kaggle", "datasets", "version", "-p", str(ZIP_STAGING), "-m", VERSION_MSG],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if not upload_ok(result):
    print("version failed — trying create ...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(ZIP_STAGING)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

if not upload_ok(result):
    raise RuntimeError("Kaggle dataset upload failed — check logs above")

print("Dataset published:", DATASET_ID)
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
    out = Path(__file__).resolve().parents[1] / "notebooks/prep-apo-timing"
    write_nb(out / "prep-apo-timing.ipynb")


if __name__ == "__main__":
    main()
