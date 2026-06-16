"""Generate notebooks/apo-geometry-guard/apo-geometry-guard-phase-3.ipynb — geometry guard.

Goal: When predicted apo masks saturate (coverage near 1.0), skip region inversion behavior
that leads to empty "effective region" masks, and instead use a boundary/line geometry guard.

Outputs:
- /kaggle/working/apo_geometry_guard_compare.csv
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
        """# UMUD — Apo Geometry Guard (Phase 3)

**GPU notebook** — compares MT geometry extraction using:

1. **Baseline geometry**: use predicted mask style (`tag_apo_style`) and the normal geometry pipeline.
2. **Guarded geometry**: if predicted apo coverage is extremely high, derive a boundary mask
   (mask minus eroded mask) and compute geometry from that boundary as a `line` mask.

Writes:
- `/kaggle/working/apo_geometry_guard_compare.csv`
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
PRED_COV_GUARD_THRESH = 0.95

ERODE_KERNEL = 5
N_GALLERY = 8
RANDOM_SEED = 42

MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_DIR = Path("/kaggle/working/figures/apo_geometry_guard")
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


def boundary_from_mask(mask01: np.ndarray, erode_kernel: int = ERODE_KERNEL):
    import cv2

    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel))
    er = cv2.erode(m, k, iterations=1)
    b = (m.astype(np.int16) - er.astype(np.int16)) > 0
    return b.astype(np.uint8)
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
for path in tqdm(test_paths, desc="geometry guard compare"):
    img_native = load_gray(path)
    h, w = img_native.shape

    pil = open_rgb_256(img_native)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)

    pred_cov = float(apo_native.mean())
    style_base = tag_apo_style(pred_cov)
    base_geo = apo_geometry_from_mask(apo_native, style_base)

    guard_applied = pred_cov >= PRED_COV_GUARD_THRESH
    if guard_applied:
        bmask = boundary_from_mask(apo_native, erode_kernel=ERODE_KERNEL)
        style_guard = "line"
        guard_geo = apo_geometry_from_mask(bmask, style_guard)
    else:
        bmask = apo_native
        style_guard = style_base
        guard_geo = apo_geometry_from_mask(apo_native, style_guard)

    rows.append(
        {
            "image_id": path.name,
            "res": f"{h}x{w}",
            "pred_cov": pred_cov,
            "style_base": style_base,
            "style_guard": style_guard,
            "guard_applied": bool(guard_applied),
            "mt_ok_base": bool(not np.isnan(base_geo["mt_px"])),
            "mt_ok_guard": bool(not np.isnan(guard_geo["mt_px"])),
            "mt_fail_reason_base": base_geo["mt_fail_reason"],
            "mt_fail_reason_guard": guard_geo["mt_fail_reason"],
            "mt_px_base": float(base_geo["mt_px"]) if not np.isnan(base_geo["mt_px"]) else np.nan,
            "mt_px_guard": float(guard_geo["mt_px"]) if not np.isnan(guard_geo["mt_px"]) else np.nan,
            "n_contours_base": base_geo.get("n_contours"),
            "n_contours_guard": guard_geo.get("n_contours"),
        }
    )

df = pd.DataFrame(rows)
out_csv = "/kaggle/working/apo_geometry_guard_compare.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

print()
print("=== MT OK rates ===")
print("base mt_ok:", float(df.mt_ok_base.mean()))
print("guard mt_ok:", float(df.mt_ok_guard.mean()))

print()
print("=== Fail reason counts (base) ===")
print(df.loc[~df.mt_ok_base, "mt_fail_reason_base"].value_counts().to_dict())

print()
print("=== Fail reason counts (guard) ===")
print(df.loc[~df.mt_ok_guard, "mt_fail_reason_guard"].value_counts().to_dict())

fixed = df[(~df.mt_ok_base) & (df.mt_ok_guard)].copy()
print()
print(f"MT fixed count: {len(fixed)}")

# Gallery: show a few fixed cases
if len(fixed) > 0:
    ids = fixed.image_id.tolist()
    rng = random.Random(RANDOM_SEED)
    picks = rng.sample(ids, min(N_GALLERY, len(ids)))
    for i, image_id in enumerate(picks, start=1):
        p = TEST_DIR / image_id
        img_native = load_gray(p)
        h, w = img_native.shape

        pil = open_rgb_256(img_native)
        _, apo_t, _ = apo_learn.predict(pil)
        apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)

        pred_cov = float(apo_native.mean())
        style_base = tag_apo_style(pred_cov)
        base_geo = apo_geometry_from_mask(apo_native, style_base)

        guard_applied = pred_cov >= PRED_COV_GUARD_THRESH
        if guard_applied:
            bmask = boundary_from_mask(apo_native, erode_kernel=ERODE_KERNEL)
            guard_geo = apo_geometry_from_mask(bmask, "line")
        else:
            bmask = apo_native
            guard_geo = apo_geometry_from_mask(apo_native, style_base)

        inv_base = invert_mask(apo_native)  # for visual comparison only
        inv_guard = invert_mask(bmask)

        fig, axes = plt.subplots(1, 6, figsize=(28, 4.8))
        axes = axes.reshape(-1)

        # image
        axes[0].imshow(img_native, cmap="gray")
        axes[0].set_title("image", fontsize=9)
        axes[0].axis("off")

        # base
        axes[1].imshow(apo_native, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"base pred\\n{style_base}", fontsize=9)
        axes[1].axis("off")
        axes[2].imshow(inv_base, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("base inv", fontsize=9)
        axes[2].axis("off")
        axes[3].imshow(overlay(img_native, apo_native))
        axes[3].set_title("base overlay", fontsize=9)
        axes[3].axis("off")

        # guard
        axes[4].imshow(bmask, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title(f"guard mask\\n{style_base if not guard_applied else 'line'}", fontsize=9)
        axes[4].axis("off")
        axes[5].imshow(overlay(img_native, bmask))
        axes[5].set_title("guard overlay", fontsize=9)
        axes[5].axis("off")

        plt.suptitle(
            f"[fixed {i}/{len(picks)}] {image_id} pred_cov={pred_cov:.3f} "
            f"base mt={base_geo['mt_px'] if not np.isnan(base_geo['mt_px']) else 'NaN'} "
            f"guard mt={guard_geo['mt_px'] if not np.isnan(guard_geo['mt_px']) else 'NaN'}",
            fontsize=11,
            y=1.02,
        )
        plt.tight_layout()
        fig_path = FIG_DIR / f"fixed_guard_{i:02d}_{image_id.replace('.tif','')}.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)
else:
    print("No MT-fixed cases found; guard likely didn't rescue any.")
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-geometry-guard"
    out.mkdir(parents=True, exist_ok=True)
    nb_path = out / "apo-geometry-guard-phase-3.ipynb"

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
                "id": "ucheozoemena/umud-apo-geometry-guard-phase-3",
                "title": "UMUD Apo Geometry Guard Phase 3",
                "code_file": "apo-geometry-guard-phase-3.ipynb",
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

