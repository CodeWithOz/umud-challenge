"""Generate notebooks/submission/submission-phase-3.ipynb — segment-then-measure inference + submission CSV."""
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
        """# UMUD — Submission (Phase 3 Baseline)

**GPU notebook** — segment-then-measure pipeline for test images:

1. Load **fasc** + **apo** fastai learners (train kernel outputs)
2. Predict masks on each test `.tif` (256px inference, upscale masks to native size)
3. Derive **PA / FL / MT** via Phase 2 geometry (pixels)
4. Apply **`MM_PER_PIXEL`** to convert FL/MT to mm (set before first scored submit)
5. Write `submission.csv` (semicolon-separated)

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        """from pathlib import Path

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50

# Pixel → mm scale (Option C). Replace before first leaderboard submit.
MM_PER_PIXEL = 1.0  # placeholder — hunt calibration in Phase 3 work item 5

FASC_MODEL_PATH = Path(
    "/kaggle/input/notebooks/ucheozoemena/umud-train-mounted-phase-3/fasc_baseline.pkl"
)
APO_MODEL_PATH = Path(
    "/kaggle/input/notebooks/ucheozoemena/umud-train-apo-mounted-phase-3/apo_baseline.pkl"
)

COMPETITION_DIR = Path(
    "/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"
)
TEST_DIR = COMPETITION_DIR / "test_images_v2/test_set_v2"
SAMPLE_SUBMISSION = COMPETITION_DIR / "sample_submission.csv"
"""
    )
)

cells.append(
    code(
        """from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from fastai.vision.all import PILImage, load_learner
from PIL import Image
from tqdm.auto import tqdm

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def list_test_images(directory: Path) -> list[Path]:
    files = [
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    ]
    # Prefer .tif when both .tif and .png exist for same stem
    by_stem: dict[str, Path] = {}
    for p in sorted(files):
        stem = p.stem
        if stem not in by_stem or p.suffix.lower() in {".tif", ".tiff"}:
            by_stem[stem] = p
    return sorted(by_stem.values(), key=lambda p: p.name)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr.astype(np.uint8)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def resize_mask_to(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return (mask > 0).astype(np.uint8)
    src = (mask > 0).astype(np.uint8) * 255
    out = Image.fromarray(src).resize((target_w, target_h), Image.NEAREST)
    return (np.array(out) > 0).astype(np.uint8)


def tensor_to_mask(pred) -> np.ndarray:
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    arr = np.asarray(pred)
    if arr.ndim == 3:
        arr = arr.argmax(axis=0)
    return (arr > 0).astype(np.uint8)


def open_rgb_256(img_native: np.ndarray) -> PILImage:
    small = resize_image(img_native, IMG_SIZE)
    rgb = np.stack([small, small, small], axis=-1).astype(np.uint8)
    return PILImage.create(rgb)


def tag_apo_style(coverage: float) -> str:
    return "region" if coverage >= APO_REGION_THRESHOLD else "line"


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return (1 - mask).astype(np.uint8)


def effective_apo_mask(mask: np.ndarray, style: str) -> tuple[np.ndarray, str]:
    if style == "region":
        return invert_mask(mask), "inverted_region"
    return mask, "raw_line"


def find_apo_contours(mask: np.ndarray, min_area_frac: float = 0.0003) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = mask.size * min_area_frac
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    big.sort(key=lambda c: cv2.boundingRect(c)[1])
    return big


def pick_superficial_deep(contours: list[np.ndarray], min_sep_px: int = 15):
    if len(contours) < 2:
        return None, None, len(contours)
    sup = contours[0]
    _, y0, _, _ = cv2.boundingRect(sup)
    deep = None
    for c in contours[1:]:
        _, y1, _, _ = cv2.boundingRect(c)
        if y1 >= y0 + min_sep_px:
            deep = c
            break
    if deep is None:
        deep = contours[min(2, len(contours) - 1)]
    return sup, deep, len(contours)


def edge_polyline(contour: np.ndarray, which: str = "bottom", n_bins: int = 60):
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return np.array([]), np.array([])
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    if x_max <= x_min:
        return pts[:, 0].astype(float), pts[:, 1].astype(float)
    edges = np.linspace(x_min, x_max, n_bins + 1)
    xs_out, ys_out = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = pts[(pts[:, 0] >= lo) & (pts[:, 0] < hi)]
        if len(in_bin) == 0:
            continue
        y = in_bin[:, 1].max() if which == "bottom" else in_bin[:, 1].min()
        xs_out.append((lo + hi) / 2.0)
        ys_out.append(float(y))
    return np.array(xs_out), np.array(ys_out)


def fit_line(xs: np.ndarray, ys: np.ndarray):
    if len(xs) < 2:
        return None
    return np.poly1d(np.polyfit(xs, ys, 1))


def line_angle_deg(line) -> float:
    return float(np.degrees(np.arctan(line[1]))) if line is not None else np.nan


def mt_from_apo_edges(sup_line, deep_line, x_left: float, x_right: float) -> float:
    if sup_line is None or deep_line is None or x_right <= x_left:
        return np.nan
    thirds = [x_left + (x_right - x_left) * t for t in (1 / 6, 3 / 6, 5 / 6)]
    dists = [abs(float(deep_line(x) - sup_line(x))) for x in thirds]
    return float(np.mean(dists))


def fascicle_pca(mask: np.ndarray) -> dict | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    return {
        "length_px": float(projections.max() - projections.min()),
        "angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }


def acute_angle_deg(a1: float, a2: float) -> float:
    d = abs(a1 - a2) % 180.0
    return float(min(d, 180.0 - d))


def apo_geometry_from_mask(apo_mask: np.ndarray, style: str) -> dict:
    eff, method = effective_apo_mask(apo_mask, style)
    contours = find_apo_contours(eff)
    sup_c, deep_c, n_contours = pick_superficial_deep(contours)
    out = {
        "apo_method": method,
        "n_contours": n_contours,
        "mt_px": np.nan,
        "deep_angle_deg": np.nan,
        "sup_line": None,
        "deep_line": None,
        "sup_xs": None,
        "sup_ys": None,
        "deep_xs": None,
        "deep_ys": None,
    }
    if sup_c is None or deep_c is None:
        return out
    sup_x, sup_y = edge_polyline(sup_c, which="bottom")
    deep_x, deep_y = edge_polyline(deep_c, which="top")
    sup_line = fit_line(sup_x, sup_y)
    deep_line = fit_line(deep_x, deep_y)
    out.update(sup_line=sup_line, deep_line=deep_line, sup_xs=sup_x, sup_ys=sup_y, deep_xs=deep_x, deep_ys=deep_y)
    if sup_line and deep_line and len(sup_x) and len(deep_x):
        x_left = max(sup_x.min(), deep_x.min())
        x_right = min(sup_x.max(), deep_x.max())
        out["mt_px"] = mt_from_apo_edges(sup_line, deep_line, x_left, x_right)
        out["deep_angle_deg"] = line_angle_deg(deep_line)
    return out


def derive_geometry(fasc_mask: np.ndarray, apo_mask: np.ndarray, apo_style: str) -> dict:
    apo = apo_geometry_from_mask(apo_mask, apo_style)
    fpca = fascicle_pca(fasc_mask)
    out = {
        "pa_deg": np.nan,
        "fl_px": np.nan,
        "mt_px": apo["mt_px"],
    }
    if fpca is not None:
        out["fl_px"] = fpca["length_px"]
        ref = apo["deep_angle_deg"] if not np.isnan(apo["deep_angle_deg"]) else 0.0
        out["pa_deg"] = acute_angle_deg(fpca["angle_deg"], ref)
    return out
"""
    )
)

cells.append(
    code(
        """assert FASC_MODEL_PATH.exists(), f"Missing fasc model: {FASC_MODEL_PATH}"
assert APO_MODEL_PATH.exists(), f"Missing apo model: {APO_MODEL_PATH}"
assert TEST_DIR.exists(), f"Missing test dir: {TEST_DIR}"

fasc_learn = load_learner(FASC_MODEL_PATH)
apo_learn = load_learner(APO_MODEL_PATH)

test_paths = list_test_images(TEST_DIR)
print(f"Test images: {len(test_paths)}")
"""
    )
)

cells.append(
    code(
        """rows = []
for path in tqdm(test_paths, desc="infer test"):
    img_native = load_gray(path)
    h, w = img_native.shape
    pil = open_rgb_256(img_native)

    _, fasc_t, _ = fasc_learn.predict(pil)
    _, apo_t, _ = apo_learn.predict(pil)
    fasc_native = resize_mask_to(tensor_to_mask(fasc_t), h, w)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)

    apo_style = tag_apo_style(float(apo_native.mean()))
    geo = derive_geometry(fasc_native, apo_native, apo_style)

    pa = geo["pa_deg"]
    fl_mm = geo["fl_px"] * MM_PER_PIXEL if not np.isnan(geo["fl_px"]) else np.nan
    mt_mm = geo["mt_px"] * MM_PER_PIXEL if not np.isnan(geo["mt_px"]) else np.nan

    rows.append(
        {
            "image_id": path.stem,
            "pa_deg": pa,
            "fl_mm": fl_mm,
            "mt_mm": mt_mm,
            "apo_style": apo_style,
            "fl_px": geo["fl_px"],
            "mt_px": geo["mt_px"],
        }
    )

pred_df = pd.DataFrame(rows)
display(pred_df.head())
print("NaN rates:", pred_df[["pa_deg", "fl_mm", "mt_mm"]].isna().mean().round(4).to_dict())
"""
    )
)

cells.append(
    code(
        """if SAMPLE_SUBMISSION.exists():
    template = pd.read_csv(SAMPLE_SUBMISSION, sep=";")
    template_ids = template["image_id"].astype(str)
    pred_lookup = pred_df.set_index("image_id")
    missing = [i for i in template_ids if i not in pred_lookup.index]
    if missing:
        print(f"Warning: {len(missing)} template ids missing predictions (first 5): {missing[:5]}")
    submit = template.copy()
    for col_src, col_dst in [("pa_deg", "pa_deg"), ("fl_mm", "fl_mm"), ("mt_mm", "mt_mm")]:
        submit[col_dst] = submit["image_id"].map(pred_lookup[col_src])
else:
    submit = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()

out_path = Path("/kaggle/working/submission.csv")
submit.to_csv(out_path, sep=";", index=False)
pred_df.to_csv("/kaggle/working/submission_debug.csv", index=False)
print(f"Wrote {out_path} ({len(submit)} rows)")
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
    out = Path(__file__).resolve().parents[1] / "notebooks/submission"
    write_nb(out / "submission-phase-3.ipynb")


if __name__ == "__main__":
    main()
