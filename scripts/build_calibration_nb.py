"""Generate notebooks/calibration/calibration-phase-3.ipynb — mm calibration sprint on Kaggle."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_submission_nb as sub  # noqa: E402
from build_submission_nb import code, md


def _geometry_source() -> str:
    geom_cell = sub.cells[3]
    src = geom_cell["source"]
    return "".join(src) if isinstance(src, list) else src


cells: list[dict] = [
    md(
        """# UMUD — Calibration Sprint (Phase 3)

**CPU notebook** — gather evidence for pixel→mm calibration:

1. GT geometry on **train dual-track** pairs (fasc + apo masks, stretch-aligned)
2. TIFF metadata scan (spacing tags)
3. Resolution cohort stats (800×1200 vs 1080×1640)
4. Depth-scale strip heuristic (left margin tick spacing)

Outputs in `/kaggle/working/`:
- `calibration_train_geometry.csv`
- `calibration_tiff_tags.csv`
- `calibration_summary.json`
"""
    ),
    code(
        """from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

RANDOM_SEED = 42
MAX_TRAIN = None  # None = all dual-track pairs
REF_FL_MM = (30.0, 200.0)
REF_MT_MM = (10.0, 50.0)
SCALE_STRIP_COLS = 80

COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
"""
    ),
    code(_geometry_source()),
    code(
        """def build_lookups(root: Path, img_glob: str, mask_glob: str) -> tuple[dict[str, Path], dict[str, Path]]:
    imgs = {p.name: p for p in root.rglob(img_glob)}
    masks = {p.name: p for p in root.rglob(mask_glob)}
    return imgs, masks


def align_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mh, mw = mask.shape[:2]
    if mh == target_h and mw == target_w:
        return (mask > 0).astype(np.uint8)
    return cv2.resize((mask > 0).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def tiff_tag_summary(path: Path) -> dict:
    row = {"filename": path.name}
    try:
        with Image.open(path) as im:
            row["img_w"], row["img_h"] = im.size
            tags = getattr(im, "tag_v2", {}) or {}
            for k in (282, 283, 296, 270):
                if k in tags:
                    row[f"tag_{k}"] = tags.get(k)
    except Exception as exc:
        row["error"] = str(exc)
    return row


def depth_scale_mm_per_px(gray: np.ndarray, strip_cols: int = SCALE_STRIP_COLS) -> float | None:
    \"\"\"Heuristic: detect horizontal tick peaks in left depth strip; assume 1 cm spacing.\"\"\"
    h, w = gray.shape
    strip = gray[:, : min(strip_cols, w)]
    # emphasize horizontal edges in strip
    blur = cv2.GaussianBlur(strip, (5, 5), 0)
    edges = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    prof = np.abs(edges).mean(axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    thr = float(np.percentile(prof, 92))
    peaks = np.where(prof >= thr)[0]
    if len(peaks) < 4:
        return None
    # cluster adjacent peak rows
    clusters = []
    cur = [int(peaks[0])]
    for y in peaks[1:]:
        if y - cur[-1] <= 3:
            cur.append(int(y))
        else:
            clusters.append(int(np.mean(cur)))
            cur = [int(y)]
    clusters.append(int(np.mean(cur)))
    if len(clusters) < 3:
        return None
    gaps = np.diff(clusters)
    gap = float(np.median(gaps))
    if gap < 8 or gap > h / 3:
        return None
    # 1 cm between ticks -> 10 mm / gap_px
    return 10.0 / gap


fasc_img_dir = COMPETITION_DIR / "fasc_imgs_v1/fasc_images_new_model_v1"
fasc_mask_dir = COMPETITION_DIR / "fasc_masks_v1/fasc_masks_new_model_v1"
apo_img_dir = COMPETITION_DIR / "apo_imgs_v1/apo_images_new_model_v1"
apo_mask_dir = COMPETITION_DIR / "apo_masks_v1/apo_masks_new_model_v1"

fasc_img_lu = {p.name: p for p in fasc_img_dir.glob("*.tif")}
fasc_mask_lu = {p.name: p for p in fasc_mask_dir.glob("*.tif")}
apo_img_lu = {p.name: p for p in apo_img_dir.glob("*.tif")}
apo_mask_lu = {p.name: p for p in apo_mask_dir.glob("*.tif")}

common = sorted(set(fasc_img_lu) & set(fasc_mask_lu) & set(apo_img_lu) & set(apo_mask_lu))
if MAX_TRAIN:
    rng = random.Random(RANDOM_SEED)
    common = rng.sample(common, min(MAX_TRAIN, len(common)))
print(f"Dual-track train pairs: {len(common)}")
"""
    ),
    code(
        """rows = []
tag_rows = []
for name in tqdm(common, desc="train geometry"):
    img_path = fasc_img_lu[name]
    img = load_gray(img_path)
    h, w = img.shape
    fasc = align_mask(load_gray(fasc_mask_lu[name]), h, w)
    apo = align_mask(load_gray(apo_mask_lu[name]), h, w)
    style = tag_apo_style(float(apo.mean()))
    geo = derive_geometry(fasc, apo, style)
    scale_guess = depth_scale_mm_per_px(img)
    rows.append(
        {
            "filename": name,
            "img_h": h,
            "img_w": w,
            "apo_style": style,
            **{k: geo[k] for k in ["pa_deg", "fl_px", "mt_px", "geometry_path", "mt_fail_reason"]},
            "depth_scale_mm_per_px": scale_guess,
        }
    )
    tag_rows.append(tiff_tag_summary(img_path))

geom_df = pd.DataFrame(rows)
tags_df = pd.DataFrame(tag_rows)
geom_df.to_csv("/kaggle/working/calibration_train_geometry.csv", index=False)
tags_df.to_csv("/kaggle/working/calibration_tiff_tags.csv", index=False)

fl_mid = sum(REF_FL_MM) / 2
mt_mid = sum(REF_MT_MM) / 2
cohort = geom_df.groupby(["img_h", "img_w"]).agg(
    n=("filename", "count"),
    fl_px_med=("fl_px", "median"),
    mt_px_med=("mt_px", "median"),
    pa_med=("pa_deg", "median"),
    depth_scale_med=("depth_scale_mm_per_px", "median"),
).reset_index()

summary = {
    "n_train_pairs": int(len(geom_df)),
    "gt_fl_px_median": float(geom_df["fl_px"].median()),
    "gt_mt_px_median": float(geom_df["mt_px"].median()),
    "heuristic_mm_per_px_fl": fl_mid / float(geom_df["fl_px"].median()),
    "heuristic_mm_per_px_mt": mt_mid / float(geom_df["mt_px"].median()),
    "depth_scale_mm_per_px_median": float(geom_df["depth_scale_mm_per_px"].dropna().median())
    if geom_df["depth_scale_mm_per_px"].notna().any()
    else None,
    "depth_scale_n_detected": int(geom_df["depth_scale_mm_per_px"].notna().sum()),
    "cohort_table": cohort.to_dict(orient="records"),
    "tiff_spacing_tags_present": int(
        tags_df.filter(like="tag_").notna().any(axis=1).sum()
    ),
}
with open("/kaggle/working/calibration_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
display(cohort)
"""
    ),
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/calibration"
    out.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    (out / "calibration-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-calibration-phase-3",
        "title": "UMUD Calibration Phase 3",
        "code_file": "calibration-phase-3.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
