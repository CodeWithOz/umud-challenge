"""Generate notebooks/apo-contrast-fill/apo-contrast-fill-phase-3.ipynb — gray context contrast test.

Goal:
Test whether changing the *image context* (black gutters vs neutral gray) reduces the
letterbox saturation failure mode (pred_cov -> ~1.0 -> region invert -> no_contours).

Notebook does:
1) Baseline apo inference on raw test images.
2) Preprocess images by replacing non-ultrasound background (outside ROI bbox) with gray.
3) Run apo inference again on the preprocessed images.
4) Compare MT NaN/failure reasons between baseline and gray-context.
5) Show side-by-side raw vs gray-context images + predictions for a small gallery.
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
        """# UMUD — Apo Contrast Context Fill (Phase 3)

**GPU notebook** — tests your “contrast/context” hypothesis.

We take each test ultrasound image and:
1. Compute the ultrasound ROI bbox (non-black content).
2. Replace everything outside that bbox with a neutral **gray** value.
3. Run apo segmentation + Phase-2 geometry on both:
   - **Baseline**: raw image
   - **Gray-fill**: gray-context image

Outputs:
- `/kaggle/working/apo_contrast_fill_compare.csv`

The notebook also displays a small gallery of cases where you can visually inspect:
Raw image -> Gray-filled image -> Predicted mask(s) + overlays + inverted masks.
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
APO_REGION_THRESHOLD = 0.50

N_GALLERY_FAIL = 10
N_GALLERY_OK = 6
RANDOM_SEED = 42

ROI_THRESH = 5        # non-black threshold for bbox
ROI_PAD_PX = 10       # bbox padding

MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

# How to compute the neutral gray fill value.
# We take a dark-pixel percentile from the image, but clamp to a minimum.
GRAY_DARK_PERCENTILE = 10  # lower percentile => darker gray
GRAY_MIN_VALUE = 15        # avoid turning gutters into near-black

# Optional contrast stretch inside ROI (OFF by default; enabled only if you want).
DO_CONTRAST_STRETCH = False
P_LOW = 1
P_HIGH = 99

FIG_DIR = Path("/kaggle/working/figures/apo_contrast_fill")
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
    \"\"\"Ultrasound ROI bbox from non-black pixels.

    Uses connected components to suppress small speckle blobs.
    Returns (y0, y1, x0, x1) in native coordinates.
    \"\"\"
    import cv2

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


def compute_gray_fill_value(img_gray: np.ndarray):
    \"\"\"Pick a dark-but-not-black gray from the image background.\"\"\"
    flat = img_gray.reshape(-1)
    # Ignore exact zeros when possible (pure black borders).
    nonzero = flat[flat > 0]
    if len(nonzero) > 1000:
        dark_base = np.percentile(nonzero, GRAY_DARK_PERCENTILE)
    else:
        dark_base = np.percentile(flat, GRAY_DARK_PERCENTILE)
    return float(max(GRAY_MIN_VALUE, dark_base))


def maybe_contrast_stretch(img_gray: np.ndarray, y0: int, y1: int, x0: int, x1: int):
    if not DO_CONTRAST_STRETCH:
        return img_gray

    out = img_gray.astype(np.float32).copy()
    roi = out[y0:y1, x0:x1]
    lo = np.percentile(roi, P_LOW)
    hi = np.percentile(roi, P_HIGH)
    if hi <= lo + 1e-6:
        return img_gray

    # Linear rescale ROI only to [0,255], clipping to remove outliers.
    roi_clipped = np.clip(roi, lo, hi)
    roi_stretched = (roi_clipped - lo) / (hi - lo) * 255.0
    out[y0:y1, x0:x1] = roi_stretched
    return out.astype(np.uint8)


def preprocess_gray_fill(img_native: np.ndarray):
    \"\"\"Replace outside ROI bbox with a neutral gray, optionally stretch inside ROI.\"\"\"
    h, w = img_native.shape
    y0, y1, x0, x1 = find_roi_bbox(img_native)
    gray_val = compute_gray_fill_value(img_native)

    pre = img_native.copy().astype(np.float32)
    pre_mask = np.ones((h, w), dtype=bool)
    pre_mask[y0:y1, x0:x1] = False  # True outside ROI
    pre[pre_mask] = gray_val
    pre = pre.astype(np.uint8)
    pre = maybe_contrast_stretch(pre, y0, y1, x0, x1)
    return pre, (y0, y1, x0, x1), gray_val
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
            """rows = []

for path in tqdm(test_paths, desc="baseline + gray-fill infer"):
    img_native = load_gray(path)
    h, w = img_native.shape

    # Baseline
    pil = open_rgb_256(img_native)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)
    base_cov = float(apo_native.mean())
    base_style = tag_apo_style(base_cov)
    base_geo = apo_geometry_from_mask(apo_native, base_style)

    # Gray-fill preprocessing
    img_pre, bbox, gray_val = preprocess_gray_fill(img_native)
    pil_pre = open_rgb_256(img_pre)
    _, apo_t2, _ = apo_learn.predict(pil_pre)
    apo_pre_native = resize_mask_to(tensor_to_mask(apo_t2), h, w)
    pre_cov = float(apo_pre_native.mean())
    pre_style = tag_apo_style(pre_cov)
    pre_geo = apo_geometry_from_mask(apo_pre_native, pre_style)

    rows.append(
        {
            "image_id": path.name,
            "res": f"{h}x{w}",
            "bbox": f"{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}",
            "gray_val": float(gray_val),
            "base_pred_cov": base_cov,
            "pre_pred_cov": pre_cov,
            "base_style": base_style,
            "pre_style": pre_style,
            "base_mt_ok": bool(not np.isnan(base_geo["mt_px"])),
            "pre_mt_ok": bool(not np.isnan(pre_geo["mt_px"])),
            "base_mt_fail_reason": base_geo["mt_fail_reason"],
            "pre_mt_fail_reason": pre_geo["mt_fail_reason"],
        }
    )

df = pd.DataFrame(rows)
out_csv = "/kaggle/working/apo_contrast_fill_compare.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

print()
print("=== Overall ===")
print("base mt_ok rate:", float(df.base_mt_ok.mean()))
print("pre  mt_ok rate:", float(df.pre_mt_ok.mean()))

print()
print("=== base fail counts ===")
print(df.loc[~df.base_mt_ok, "base_mt_fail_reason"].value_counts().to_dict())

print()
print("=== pre fail counts ===")
print(df.loc[~df.pre_mt_ok, "pre_mt_fail_reason"].value_counts().to_dict())

fixed = df[(~df.base_mt_ok) & (df.pre_mt_ok)]
print()
print("MT-fixed count (baseline NaN -> gray finite):", len(fixed))
"""
        ),
        code(
            """# Gallery selection
fail_cases = df[df.base_mt_fail_reason == "no_contours"].copy()
ok_cases = df[df.base_mt_ok].copy()

print("no_contours baseline count:", len(fail_cases))
print("MT OK baseline count:", len(ok_cases))

rng = random.Random(RANDOM_SEED)
fail_pick = fail_cases.sample(min(N_GALLERY_FAIL, len(fail_cases)), random_state=RANDOM_SEED) if len(fail_cases) else fail_cases
ok_pick = ok_cases.sample(min(N_GALLERY_OK, len(ok_cases)), random_state=RANDOM_SEED) if len(ok_cases) else ok_cases

def show_one(image_id: str):
    path = TEST_DIR / image_id
    img_native = load_gray(path)
    h, w = img_native.shape

    # Baseline pred
    pil = open_rgb_256(img_native)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)
    base_cov = float(apo_native.mean())
    base_style = tag_apo_style(base_cov)
    base_geo = apo_geometry_from_mask(apo_native, base_style)

    # Preprocess pred
    img_pre, bbox, gray_val = preprocess_gray_fill(img_native)
    pil_pre = open_rgb_256(img_pre)
    _, apo_t2, _ = apo_learn.predict(pil_pre)
    apo_pre_native = resize_mask_to(tensor_to_mask(apo_t2), h, w)
    pre_cov = float(apo_pre_native.mean())
    pre_style = tag_apo_style(pre_cov)
    pre_geo = apo_geometry_from_mask(apo_pre_native, pre_style)

    inv_base = invert_mask(apo_native)
    inv_pre = invert_mask(apo_pre_native)

    fig, axes = plt.subplots(2, 6, figsize=(32, 8))
    axes = axes.reshape(-1)

    # Row 0: raw vs preprocessed + masks/overlays baseline
    axes[0].imshow(img_native, cmap="gray")
    axes[0].set_title("raw image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(img_pre, cmap="gray")
    axes[1].set_title(f"gray-fill image\\ngray_val={gray_val:.1f}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(apo_native, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"base pred mask\\n{base_style} cov={base_cov*100:.2f}%", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(inv_base, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"base inverted\\n{base_geo['mt_fail_reason']}", fontsize=10)
    axes[3].axis("off")

    axes[4].imshow(overlay(img_native, apo_native))
    axes[4].set_title("base overlay (pred)", fontsize=10)
    axes[4].axis("off")

    axes[5].imshow(overlay(img_native, inv_base))
    axes[5].set_title("base overlay (inv)", fontsize=10)
    axes[5].axis("off")

    # Row 1: masks/overlays after gray-fill
    axes[6].imshow(apo_pre_native, cmap="gray", vmin=0, vmax=1)
    axes[6].set_title(f"pre pred mask\\n{pre_style} cov={pre_cov*100:.2f}%", fontsize=10)
    axes[6].axis("off")

    axes[7].imshow(inv_pre, cmap="gray", vmin=0, vmax=1)
    axes[7].set_title(f"pre inverted\\n{pre_geo['mt_fail_reason']}", fontsize=10)
    axes[7].axis("off")

    axes[8].imshow(overlay(img_pre, apo_pre_native))
    axes[8].set_title("pre overlay (pred)", fontsize=10)
    axes[8].axis("off")

    axes[9].imshow(overlay(img_pre, inv_pre))
    axes[9].set_title("pre overlay (inv)", fontsize=10)
    axes[9].axis("off")

    # bbox diagnostic (create a fresh patch per axis)
    y0,y1,x0,x1 = bbox
    axes[10].imshow(img_native, cmap="gray")
    rect1 = plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='cyan', linewidth=2)
    axes[10].add_patch(rect1)
    axes[10].set_title("bbox on raw", fontsize=10)
    axes[10].axis("off")

    axes[11].imshow(img_pre, cmap="gray")
    rect2 = plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='cyan', linewidth=2)
    axes[11].add_patch(rect2)
    axes[11].set_title("bbox on gray-fill", fontsize=10)
    axes[11].axis("off")

    plt.suptitle(
        f"{image_id} | base mt_ok={base_geo['mt_px'] if not np.isnan(base_geo['mt_px']) else 'NaN'} "
        f"| pre mt_ok={pre_geo['mt_px'] if not np.isnan(pre_geo['mt_px']) else 'NaN'}",
        y=0.98,
        fontsize=12,
    )
    plt.tight_layout()
    return fig


print("\\n=== Gallery: baseline no_contours (fail) ===")
for i, image_id in enumerate(fail_pick.image_id.tolist(), start=1):
    print(f"[fail {i}] {image_id}")
    fig = show_one(image_id)
    fig_path = FIG_DIR / f"gallery_fail_{i:02d}_{image_id.replace('.tif','')}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)

print("\\n=== Gallery: baseline MT OK (success) ===")
for i, image_id in enumerate(ok_pick.image_id.tolist(), start=1):
    print(f"[ok {i}] {image_id}")
    fig = show_one(image_id)
    fig_path = FIG_DIR / f"gallery_ok_{i:02d}_{image_id.replace('.tif','')}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)

print("Done. Figures in:", FIG_DIR)
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-contrast-fill"
    out.mkdir(parents=True, exist_ok=True)
    nb_path = out / "apo-contrast-fill-phase-3.ipynb"

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    nb_path.write_text(json.dumps(nb, indent=1))

    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-apo-contrast-fill-phase-3",
                "title": "UMUD Apo Contrast Context Fill Phase 3",
                "code_file": "apo-contrast-fill-phase-3.ipynb",
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

