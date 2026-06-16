"""Generate notebooks/apo-roi-crop/apo-roi-crop-phase-3.ipynb — ROI crop inference.

Goal: Fix the apo-mask "letterbox saturation" failure mode by cropping away black gutters
before running apo segmentation, then inserting the predicted mask back into native size.

Outputs:
- /kaggle/working/apo_roi_crop_compare.csv
"""

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
        """# UMUD — Apo ROI Crop (Phase 3)

**GPU notebook** — compares MT geometry extraction using:

1. **Baseline**: run apo segmentation on the full resized image (no crop).
2. **ROI-crop**: detect the ultrasound region bbox (non-black threshold), run apo on the crop,
   then paste the predicted mask back into native coordinates before geometry.

Writes:
- `/kaggle/working/apo_roi_crop_compare.csv`
"""
    ),
    md("## Configuration"),
    code(
        """import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

IMG_SIZE = 256
ROI_THRESH = 5  # non-black threshold for bbox
ROI_PAD_PX = 10

N_GALLERY = 8
RANDOM_SEED = 42
MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_DIR = Path("/kaggle/working/figures/apo_roi_crop")
FIG_DIR.mkdir(parents=True, exist_ok=True)

APO_MODEL_PATH = Path(
    "/kaggle/input/notebooks/ucheozoemena/umud-train-apo-mounted-phase-3/apo_baseline.pkl"
)

COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
TEST_DIR = COMPETITION_DIR / "test_images_v2/test_set_v2"
"""
    ),
]

cells.append(code(_geometry_source()))

cells.extend(
    [
        code(
            """def overlay(img_gray: np.ndarray, mask: np.ndarray, color=APO_OVERLAY_COLOR, alpha=MASK_OVERLAY_ALPHA):
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    sel = mask.astype(bool)
    rgb[sel] = (1 - alpha) * rgb[sel] + alpha * tint[sel]
    return rgb.astype(np.uint8)


def find_roi_bbox(img_gray: np.ndarray, thr: float = ROI_THRESH, pad: int = ROI_PAD_PX):
    \"\"\"BBox of ultrasound content by non-black threshold + largest connected component.\"\"\"
    import cv2

    roi = (img_gray > thr).astype(np.uint8)
    if roi.sum() == 0:
        h, w = img_gray.shape
        return 0, h, 0, w

    # pick largest connected component to avoid speckle blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    # stats[0] is background
    best = max(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h = stats[best, :4]

    y0 = max(0, y - pad)
    y1 = min(img_gray.shape[0], y + h + pad)
    x0 = max(0, x - pad)
    x1 = min(img_gray.shape[1], x + w + pad)
    return int(y0), int(y1), int(x0), int(x1)
"""
        ),
        code(
            """assert APO_MODEL_PATH.exists(), f"Missing apo model: {APO_MODEL_PATH}"
assert TEST_DIR.exists(), f"Missing test dir: {TEST_DIR}"

apo_learn = load_learner(APO_MODEL_PATH)
test_paths = list_test_images(TEST_DIR)
print(f"Test images: {len(test_paths)}")
"""
        ),
        code(
            """def infer_apo_full(img_native: np.ndarray):
    h, w = img_native.shape
    pil = open_rgb_256(img_native)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)
    return apo_native


def infer_apo_roi_crop(img_native: np.ndarray):
    h, w = img_native.shape
    y0, y1, x0, x1 = find_roi_bbox(img_native)
    img_crop = img_native[y0:y1, x0:x1]
    pil = open_rgb_256(img_crop)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_crop = resize_mask_to(tensor_to_mask(apo_t), y1 - y0, x1 - x0)

    apo_native = np.zeros((h, w), dtype=np.uint8)
    apo_native[y0:y1, x0:x1] = apo_crop
    return apo_native, (y0, y1, x0, x1)


rows = []
for path in tqdm(test_paths, desc="roi-crop compare"):
    img_native = load_gray(path)
    h, w = img_native.shape

    apo_base = infer_apo_full(img_native)
    base_style = tag_apo_style(float(apo_base.mean()))
    base_geo = apo_geometry_from_mask(apo_base, base_style)

    apo_crop, bbox = infer_apo_roi_crop(img_native)
    crop_style = tag_apo_style(float(apo_crop.mean()))
    crop_geo = apo_geometry_from_mask(apo_crop, crop_style)

    rows.append(
        {
            "image_id": path.name,
            "res": f"{h}x{w}",
            "bbox": f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}",
            "pred_cov_base": float(apo_base.mean()),
            "pred_cov_crop": float(apo_crop.mean()),
            "style_base": base_style,
            "style_crop": crop_style,
            "mt_ok_base": bool(not np.isnan(base_geo["mt_px"])),
            "mt_ok_crop": bool(not np.isnan(crop_geo["mt_px"])),
            "mt_fail_reason_base": base_geo["mt_fail_reason"],
            "mt_fail_reason_crop": crop_geo["mt_fail_reason"],
            "mt_px_base": float(base_geo["mt_px"]) if not np.isnan(base_geo["mt_px"]) else np.nan,
            "mt_px_crop": float(crop_geo["mt_px"]) if not np.isnan(crop_geo["mt_px"]) else np.nan,
            "n_contours_base": base_geo.get("n_contours"),
            "n_contours_crop": crop_geo.get("n_contours"),
        }
    )

df = pd.DataFrame(rows)
out_csv = "/kaggle/working/apo_roi_crop_compare.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

print()
print("=== MT OK rates ===")
print("base mt_ok:", float(df.mt_ok_base.mean()))
print("crop mt_ok:", float(df.mt_ok_crop.mean()))

print()
print("=== Fail reason counts (base) ===")
print(df.loc[~df.mt_ok_base, "mt_fail_reason_base"].value_counts().to_dict())

print()
print("=== Fail reason counts (crop) ===")
print(df.loc[~df.mt_ok_crop, "mt_fail_reason_crop"].value_counts().to_dict())

fixed = df[(~df.mt_ok_base) & (df.mt_ok_crop)].copy()
still_fail = df[(~df.mt_ok_base) & (~df.mt_ok_crop)].copy()
print()
print(f"MT fixed count: {len(fixed)} (baseline NaN -> crop finite)")
print(f"Remaining NaN count: {len(still_fail)}")

# Gallery: show a few fixed cases if any
if len(fixed) > 0:
    fixed_ids = fixed.image_id.tolist()
    rng = random.Random(RANDOM_SEED)
    picks = rng.sample(fixed_ids, min(N_GALLERY, len(fixed_ids)))
    for i, image_id in enumerate(picks, start=1):
        p = TEST_DIR / image_id
        img_native = load_gray(p)
        apo_base = infer_apo_full(img_native)
        base_style = tag_apo_style(float(apo_base.mean()))
        base_geo = apo_geometry_from_mask(apo_base, base_style)

        apo_crop, bbox = infer_apo_roi_crop(img_native)
        crop_style = tag_apo_style(float(apo_crop.mean()))
        crop_geo = apo_geometry_from_mask(apo_crop, crop_style)

        inv_base = invert_mask(apo_base)
        inv_crop = invert_mask(apo_crop)

        fig, axes = plt.subplots(2, 5, figsize=(24, 8))
        axes = axes.reshape(-1)

        def panel(ax, title, m, overlay_mask=None):
            ax.imshow(img_native, cmap="gray")
            if overlay_mask is not None:
                ax.imshow(overlay(img_native, overlay_mask))
            else:
                ax.imshow(m, cmap="gray", vmin=0, vmax=1, alpha=0.6)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        # Row 0: baseline
        axes[0].imshow(img_native, cmap="gray"); axes[0].set_title("image", fontsize=9); axes[0].axis("off")
        axes[1].imshow(apo_base, cmap="gray", vmin=0, vmax=1); axes[1].set_title("base pred", fontsize=9); axes[1].axis("off")
        axes[2].imshow(inv_base, cmap="gray", vmin=0, vmax=1); axes[2].set_title("base inv", fontsize=9); axes[2].axis("off")
        axes[3].imshow(overlay(img_native, apo_base)); axes[3].set_title("base overlay", fontsize=9); axes[3].axis("off")
        axes[4].imshow(overlay(img_native, inv_base)); axes[4].set_title("base inv ov", fontsize=9); axes[4].axis("off")

        # Row 1: crop
        axes[5].imshow(img_native, cmap="gray"); axes[5].set_title("image", fontsize=9); axes[5].axis("off")
        axes[6].imshow(apo_crop, cmap="gray", vmin=0, vmax=1); axes[6].set_title("crop pred", fontsize=9); axes[6].axis("off")
        axes[7].imshow(inv_crop, cmap="gray", vmin=0, vmax=1); axes[7].set_title("crop inv", fontsize=9); axes[7].axis("off")
        axes[8].imshow(overlay(img_native, apo_crop)); axes[8].set_title("crop overlay", fontsize=9); axes[8].axis("off")
        axes[9].imshow(overlay(img_native, inv_crop)); axes[9].set_title("crop inv ov", fontsize=9); axes[9].axis("off")

        plt.suptitle(
            f"[fixed {i}/{len(picks)}] {image_id} bbox={bbox} "
            f"base mt={base_geo['mt_px'] if not np.isnan(base_geo['mt_px']) else 'NaN'} "
            f"crop mt={crop_geo['mt_px'] if not np.isnan(crop_geo['mt_px']) else 'NaN'} "
            f"styles base={base_style} crop={crop_style}",
            fontsize=11,
            y=1.02,
        )
        plt.tight_layout()
        fig_path = FIG_DIR / f"fixed_roi_{i:02d}_{image_id.replace('.tif','')}.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)
else:
    print("No MT-fixed cases found; still failures will dominate.")
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-roi-crop"
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
    (out / "apo-roi-crop-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-apo-roi-crop-phase-3",
                "title": "UMUD Apo ROI Crop Phase 3",
                "code_file": "apo-roi-crop-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": ["ucheozoemena/umud-train-apo-mounted-phase-3"],
                "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
                "model_sources": [],
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

