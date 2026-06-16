"""Generate notebooks/prep-apo-gray55/prep-apo-gray55.ipynb — gray55 bbox-fill apo prep."""
import json
from pathlib import Path

import pandas as pd


def load_exclude_apo_names() -> list[str]:
    csv_path = Path(__file__).resolve().parents[1] / "research/exclude_apo_mt_invalid.csv"
    return pd.read_csv(csv_path)["filename"].astype(str).tolist()


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


exclude_apo = load_exclude_apo_names()
exclude_literal = repr(exclude_apo)

cells: list[dict] = [
    md(
        """# UMUD — Prepare Gray55 Apo Dataset (Kaggle-native)

**CPU notebook** — same stretch-align + 256px resize as standard apo prep, but **before resize**:

1. Detect ultrasound ROI bbox (non-black content).
2. Fill everything **outside** the bbox with fixed gray **55** (RGB 55,55,55 → grayscale 55).
3. Stretch-align masks, resize to **256×256** (NEAREST masks).
4. Publish dataset via `kaggle datasets version/create`.

Train/inference match: model sees gray surround + ultrasound ROI only."""
    ),
    md("## Configuration"),
    code(
        f"""# --- Parameters you can change ---
RANDOM_SEED = 42
PREP_RUN = 4  # 1=50, 2=200, 3=524, 4=1044 full (MT-valid)

IMG_SIZE = 256
APO_FULL = 1044
EXCLUDE_APO_MT = {exclude_literal}

GRAY_FILL_VALUE = 55
ROI_THRESH = 5
ROI_PAD_PX = 10

PREP_PROFILES = {{
    1: {{
        "max_samples": 50,
        "dataset_id": "ucheozoemena/umud-aligned-apo-gray55-timing-50",
        "dataset_title": "UMUD Aligned Apo Gray55 Timing 50",
        "version_msg": "Gray55 apo timing: 50 pairs, bbox fill=55, 256px",
        "zip_name": "umud_apo_gray55_timing_1",
    }},
    2: {{
        "max_samples": 200,
        "dataset_id": "ucheozoemena/umud-aligned-apo-gray55-timing-200",
        "dataset_title": "UMUD Aligned Apo Gray55 Timing 200",
        "version_msg": "Gray55 apo timing: 200 pairs, bbox fill=55, 256px",
        "zip_name": "umud_apo_gray55_timing_2",
    }},
    3: {{
        "max_samples": APO_FULL // 2,
        "dataset_id": "ucheozoemena/umud-aligned-apo-gray55-timing-524",
        "dataset_title": "UMUD Aligned Apo Gray55 Timing 524",
        "version_msg": "Gray55 apo scaling: 524 pairs, bbox fill=55, 256px",
        "zip_name": "umud_apo_gray55_timing_3",
    }},
    4: {{
        "max_samples": APO_FULL,
        "dataset_id": "ucheozoemena/umud-aligned-apo-gray55-full",
        "dataset_title": "UMUD Aligned Apo Gray55 Full",
        "version_msg": "Full gray55 apo: 1044 MT-valid pairs, bbox fill=55, 256px",
        "zip_name": "umud_apo_gray55_full",
    }},
}}

profile = PREP_PROFILES[PREP_RUN]
MAX_SAMPLES = profile["max_samples"]
DATASET_ID = profile["dataset_id"]
DATASET_TITLE = profile["dataset_title"]
VERSION_MSG = profile["version_msg"]
ZIP_NAME = profile["zip_name"]
print(f"PREP_RUN={{PREP_RUN}} | n<={{MAX_SAMPLES}} | dataset={{DATASET_ID}}")
"""
    ),
    code(
        """from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

COMPETITION_DIR = Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
UPLOAD = Path("/kaggle/working/upload")
IMG_OUT = UPLOAD / "images"
MSK_OUT = UPLOAD / "masks"
MANIFEST_OUT = UPLOAD / "manifests"
QC_OUT = Path("/kaggle/working/qc_samples")
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


def find_roi_bbox(img_gray: np.ndarray, thr: float = ROI_THRESH, pad: int = ROI_PAD_PX):
    roi = (img_gray > thr).astype(np.uint8)
    if roi.sum() == 0:
        h, w = img_gray.shape
        return 0, h, 0, w
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    best = max(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h = stats[best, :4]
    y0 = max(0, y - pad)
    y1 = min(img_gray.shape[0], y + h + pad)
    x0 = max(0, x - pad)
    x1 = min(img_gray.shape[1], x + w + pad)
    return int(y0), int(y1), int(x0), int(x1)


def apply_gray55_fill(img_gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = find_roi_bbox(img_gray)
    y0, y1, x0, x1 = bbox
    out = np.full_like(img_gray, GRAY_FILL_VALUE)
    out[y0:y1, x0:x1] = img_gray[y0:y1, x0:x1]
    return out, bbox


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
    ),
    code(
        """if UPLOAD.exists():
    shutil.rmtree(UPLOAD)
UPLOAD.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)
MSK_OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT.mkdir(parents=True, exist_ok=True)
QC_OUT.mkdir(parents=True, exist_ok=True)

lookups = {k: build_lookup(v) for k, v in DIRS.items()}
print("Lookups:", {k: len(v) for k, v in lookups.items()})

t0 = time.perf_counter()
apo_all = sorted(set(lookups["apo_img"]) & set(lookups["apo_mask"]))
apo_common = [n for n in apo_all if n not in EXCLUDE_APO_MT]
t_manifest = time.perf_counter()

targets = subsample(apo_common, MAX_SAMPLES, RANDOM_SEED)
print(f"Apo pairs (raw): {len(apo_all)} | after exclude: {len(apo_common)} | prep targets: {len(targets)}")
"""
    ),
    code(
        """rows = []
t_prep = time.perf_counter()
for i, name in enumerate(tqdm(targets, desc="prep gray55 pairs")):
    img = load_gray(lookups["apo_img"][name])
    mask = load_mask(lookups["apo_mask"][name])
    img_g, bbox = apply_gray55_fill(img)
    aligned = align_mask(mask, img.shape[0], img.shape[1])
    img_r, mask_r = resize_pair(img_g, aligned, IMG_SIZE)
    stem = Path(name).stem
    Image.fromarray(img_r).save(IMG_OUT / f"{stem}.png")
    Image.fromarray((mask_r * 255).astype(np.uint8)).save(MSK_OUT / f"{stem}.png")
    rows.append({
        "filename": name,
        "stem": stem,
        "img_h": img.shape[0],
        "img_w": img.shape[1],
        "bbox_y0": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x0": bbox[2],
        "bbox_x1": bbox[3],
        "mask_cov_native": float(aligned.mean()),
        "mask_cov_256": float(mask_r.mean()),
    })
    if i < 6:
        side = np.concatenate([img, img_g], axis=1)
        Image.fromarray(side).save(QC_OUT / f"{stem}_raw_vs_gray55.png")
t_done = time.perf_counter()

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST_OUT / "train_apo_gray55.csv", index=False)

timing = pd.DataFrame(
    [
        {
            "prep_run": PREP_RUN,
            "n_pairs": len(targets),
            "img_size": IMG_SIZE,
            "gray_fill": GRAY_FILL_VALUE,
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
print(f"Wrote {len(targets)} gray55 pairs | QC samples in {QC_OUT}")
"""
    ),
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
    out = Path(__file__).resolve().parents[1] / "notebooks/prep-apo-gray55"
    write_nb(out / "prep-apo-gray55.ipynb")
    meta = {
        "id": "ucheozoemena/umud-prep-apo-gray55",
        "title": "UMUD Prep Apo Gray55",
        "code_file": "prep-apo-gray55.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
