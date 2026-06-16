"""Generate notebooks/apo-contrast-fill-v3 — gray55 bbox pipeline + contrast stretch variant."""
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
        """# UMUD — Apo Gray55 Bbox Pipeline (Phase 3 v3)

**GPU notebook** — implements the refined context hypothesis:

1. Detect ultrasound ROI bbox (non-black content).
2. Replace everything **outside** the bbox with fixed gray **RGB (55, 55, 55)** → grayscale 55.
3. Run apo inference on the preprocessed image.
4. **Zero predicted mask pixels outside the bbox** before geometry (removes gutter noise).
5. Compare three pipelines on all test images:
   - **Baseline** (raw image, raw pred)
   - **Gray55+bbox** (gray fill + mask clip)
   - **Gray55+stretch+bbox** (gray fill + percentile contrast stretch inside ROI + mask clip)

Outputs:
- `/kaggle/working/apo_contrast_fill_compare.csv`
- Gallery figures under `/kaggle/working/figures/apo_contrast_fill/`
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

ROI_THRESH = 5
ROI_PAD_PX = 10

# Working-cohort gray from visual inspection (RGB 55,55,55 at full opacity)
GRAY_FILL_VALUE = 55

# Contrast stretch inside ROI only (percentile clip then linear rescale to 0..255)
P_LOW = 1
P_HIGH = 99

MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

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


def clip_mask_to_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    out = np.zeros_like(mask, dtype=np.uint8)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def contrast_stretch_roi(img_gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    out = img_gray.astype(np.float32).copy()
    roi = out[y0:y1, x0:x1]
    lo = np.percentile(roi, P_LOW)
    hi = np.percentile(roi, P_HIGH)
    if hi <= lo + 1e-6:
        return img_gray
    roi_clipped = np.clip(roi, lo, hi)
    out[y0:y1, x0:x1] = (roi_clipped - lo) / (hi - lo) * 255.0
    return out.astype(np.uint8)


def preprocess_gray55(img_native: np.ndarray, do_stretch: bool = False):
    \"\"\"Gray-fill outside bbox with fixed 55; optional contrast stretch inside ROI.\"\"\"
    h, w = img_native.shape
    bbox = find_roi_bbox(img_native)
    y0, y1, x0, x1 = bbox

    pre = img_native.copy()
    outside = np.ones((h, w), dtype=bool)
    outside[y0:y1, x0:x1] = False
    pre[outside] = GRAY_FILL_VALUE

    if do_stretch:
        pre = contrast_stretch_roi(pre, bbox)

    return pre, bbox


def infer_apo_on_image(img_gray: np.ndarray, bbox: tuple[int, int, int, int], clip_bbox: bool):
    h, w = img_gray.shape
    pil = open_rgb_256(img_gray)
    _, apo_t, _ = apo_learn.predict(pil)
    mask = resize_mask_to(tensor_to_mask(apo_t), h, w)
    if clip_bbox:
        mask = clip_mask_to_bbox(mask, bbox)
    style = tag_apo_style(float(mask.mean()))
    geo = apo_geometry_from_mask(mask, style)
    return mask, style, geo
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

for path in tqdm(test_paths, desc="compare pipelines"):
    img_native = load_gray(path)
    h, w = img_native.shape

    # Baseline
    bbox = find_roi_bbox(img_native)
    base_mask, base_style, base_geo = infer_apo_on_image(img_native, bbox, clip_bbox=False)

    # Gray55 + bbox mask clip
    img_g, bbox_g = preprocess_gray55(img_native, do_stretch=False)
    g_mask, g_style, g_geo = infer_apo_on_image(img_g, bbox_g, clip_bbox=True)

    # Gray55 + stretch + bbox mask clip
    img_s, bbox_s = preprocess_gray55(img_native, do_stretch=True)
    s_mask, s_style, s_geo = infer_apo_on_image(img_s, bbox_s, clip_bbox=True)

    rows.append(
        {
            "image_id": path.name,
            "res": f"{h}x{w}",
            "bbox": f"{bbox_g[0]}:{bbox_g[1]}:{bbox_g[2]}:{bbox_g[3]}",
            "base_pred_cov": float(base_mask.mean()),
            "gray_pred_cov": float(g_mask.mean()),
            "stretch_pred_cov": float(s_mask.mean()),
            "base_style": base_style,
            "gray_style": g_style,
            "stretch_style": s_style,
            "base_mt_ok": bool(not np.isnan(base_geo["mt_px"])),
            "gray_mt_ok": bool(not np.isnan(g_geo["mt_px"])),
            "stretch_mt_ok": bool(not np.isnan(s_geo["mt_px"])),
            "base_mt_fail_reason": base_geo["mt_fail_reason"],
            "gray_mt_fail_reason": g_geo["mt_fail_reason"],
            "stretch_mt_fail_reason": s_geo["mt_fail_reason"],
            "base_mt_px": float(base_geo["mt_px"]) if not np.isnan(base_geo["mt_px"]) else np.nan,
            "gray_mt_px": float(g_geo["mt_px"]) if not np.isnan(g_geo["mt_px"]) else np.nan,
            "stretch_mt_px": float(s_geo["mt_px"]) if not np.isnan(s_geo["mt_px"]) else np.nan,
        }
    )

df = pd.DataFrame(rows)
out_csv = "/kaggle/working/apo_contrast_fill_compare.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

print("\\n=== MT OK rates ===")
print("baseline:", float(df.base_mt_ok.mean()))
print("gray55+bbox:", float(df.gray_mt_ok.mean()))
print("gray55+stretch+bbox:", float(df.stretch_mt_ok.mean()))

print("\\nbaseline fail:", df.loc[~df.base_mt_ok, "base_mt_fail_reason"].value_counts().to_dict())
print("gray55+bbox fail:", df.loc[~df.gray_mt_ok, "gray_mt_fail_reason"].value_counts().to_dict())
print("stretch+bbox fail:", df.loc[~df.stretch_mt_ok, "stretch_mt_fail_reason"].value_counts().to_dict())

print("\\nMT-fixed vs baseline:")
print("gray55+bbox:", int(((~df.base_mt_ok) & (df.gray_mt_ok)).sum()))
print("stretch+bbox:", int(((~df.base_mt_ok) & (df.stretch_mt_ok)).sum()))
base_no = df[df.base_mt_fail_reason == "no_contours"]
print("no_contours fixed by gray55+bbox:", int(((~base_no.base_mt_ok) & (base_no.gray_mt_ok)).sum()), "/", len(base_no))
print("no_contours fixed by stretch+bbox:", int(((~base_no.base_mt_ok) & (base_no.stretch_mt_ok)).sum()), "/", len(base_no))
"""
        ),
        code(
            """fail_cases = df[df.base_mt_fail_reason == "no_contours"].copy()
ok_cases = df[df.base_mt_ok].copy()

fail_pick = fail_cases.sample(min(N_GALLERY_FAIL, len(fail_cases)), random_state=RANDOM_SEED) if len(fail_cases) else fail_cases
ok_pick = ok_cases.sample(min(N_GALLERY_OK, len(ok_cases)), random_state=RANDOM_SEED) if len(ok_cases) else ok_cases


def show_one(image_id: str):
    path = TEST_DIR / image_id
    img_native = load_gray(path)
    bbox = find_roi_bbox(img_native)

    img_g, _ = preprocess_gray55(img_native, do_stretch=False)
    img_s, _ = preprocess_gray55(img_native, do_stretch=True)

    base_mask, _, base_geo = infer_apo_on_image(img_native, bbox, clip_bbox=False)
    g_mask, _, g_geo = infer_apo_on_image(img_g, bbox, clip_bbox=True)
    s_mask, _, s_geo = infer_apo_on_image(img_s, bbox, clip_bbox=True)

    # noise outside bbox on raw pred
    outside_noise = int(base_mask.sum() - clip_mask_to_bbox(base_mask, bbox).sum())

    fig, axes = plt.subplots(2, 5, figsize=(28, 9))

    # Row 0: images + bbox
    axes[0, 0].imshow(img_native, cmap="gray")
    axes[0, 0].set_title("raw", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_g, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"gray55 outside bbox\\nfill={GRAY_FILL_VALUE}", fontsize=9)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_s, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("gray55 + stretch", fontsize=9)
    axes[0, 2].axis("off")

    for ax, img, title in zip(axes[0, 3:], [img_native, img_g], ["bbox on raw", "bbox on gray55"]):
        ax.imshow(img, cmap="gray")
        y0, y1, x0, x1 = bbox
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 1: preds
    axes[1, 0].imshow(base_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"base pred\\n{base_geo['mt_fail_reason']}", fontsize=9)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(g_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"gray55+bbox\\n{g_geo['mt_fail_reason']}", fontsize=9)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(s_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title(f"stretch+bbox\\n{s_geo['mt_fail_reason']}", fontsize=9)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(overlay(img_native, base_mask))
    axes[1, 3].set_title(f"base ov\\noutside noise px={outside_noise}", fontsize=9)
    axes[1, 3].axis("off")

    axes[1, 4].imshow(overlay(img_g, g_mask))
    axes[1, 4].set_title("gray55 ov", fontsize=9)
    axes[1, 4].axis("off")

    plt.suptitle(
        f"{image_id} | base mt={base_geo['mt_px'] if not np.isnan(base_geo['mt_px']) else 'NaN'} "
        f"| gray mt={g_geo['mt_px'] if not np.isnan(g_geo['mt_px']) else 'NaN'} "
        f"| stretch mt={s_geo['mt_px'] if not np.isnan(s_geo['mt_px']) else 'NaN'}",
        y=1.02,
        fontsize=11,
    )
    plt.tight_layout()
    return fig


print("\\n=== Gallery: baseline no_contours ===")
for i, image_id in enumerate(fail_pick.image_id.tolist(), start=1):
    print(f"[fail {i}] {image_id}")
    fig = show_one(image_id)
    out = FIG_DIR / f"gallery_fail_{i:02d}_{Path(image_id).stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

print("\\n=== Gallery: baseline MT OK ===")
for i, image_id in enumerate(ok_pick.image_id.tolist(), start=1):
    print(f"[ok {i}] {image_id}")
    fig = show_one(image_id)
    out = FIG_DIR / f"gallery_ok_{i:02d}_{Path(image_id).stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

print("Done. Figures in:", FIG_DIR)
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-contrast-fill-v3"
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
                "id": "ucheozoemena/umud-apo-contrast-fill-v3-phase-3",
                "title": "UMUD Apo Gray55 Bbox Pipeline Phase 3 v3",
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
