"""Generate notebooks/apo-gray55-line-eval — MT eval of gray55+line-trained apo model."""
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
        """# UMUD — Gray55+Line Apo Model Eval (Phase 3 micro)

**GPU notebook** — compare **baseline apo** vs **gray55+line-trained apo** (region GT converted to dual-line targets).

Inference: gray55 outside bbox + mask clip to bbox.

Outputs:
- `/kaggle/working/apo_gray55_line_eval.csv`
- `/kaggle/working/apo_gray55_line_eval_summary.json`
"""
    ),
    md("## Configuration"),
    code(
        """from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
GRAY_FILL_VALUE = 55
ROI_THRESH = 5
ROI_PAD_PX = 10


def resolve_pkl(preferred: list[Path], filename: str) -> Path:
    for p in preferred:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")


BASELINE_MODEL = resolve_pkl(
    [Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-mounted-phase-3/apo_baseline.pkl")],
    "apo_baseline.pkl",
)
LINE_MODEL = resolve_pkl(
    [Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-gray55-phase-3/apo_gray55_line_baseline.pkl")],
    "apo_gray55_line_baseline.pkl",
)
print("Baseline model:", BASELINE_MODEL)
print("Gray55+line model:", LINE_MODEL)

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
            """import cv2
from fastai.vision.all import load_learner

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


def preprocess_gray55(img_native: np.ndarray):
    bbox = find_roi_bbox(img_native)
    y0, y1, x0, x1 = bbox
    pre = img_native.copy()
    outside = np.ones(img_native.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False
    pre[outside] = GRAY_FILL_VALUE
    return pre, bbox


def clip_mask_to_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    out = np.zeros_like(mask, dtype=np.uint8)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def infer_apo(learn, img_gray: np.ndarray, bbox, clip_bbox: bool = True):
    h, w = img_gray.shape
    pil = open_rgb_256(img_gray)
    _, apo_t, _ = learn.predict(pil)
    mask = resize_mask_to(tensor_to_mask(apo_t), h, w)
    if clip_bbox:
        mask = clip_mask_to_bbox(mask, bbox)
    style = tag_apo_style(float(mask.mean()))
    geo = apo_geometry_from_mask(mask, style)
    return mask, style, geo, float(mask.mean())

base_learn = load_learner(BASELINE_MODEL)
line_learn = load_learner(LINE_MODEL)
print("Models loaded")
"""
        ),
        code(
            """rows = []
for p in tqdm(sorted(TEST_DIR.glob("*.tif")), desc="eval"):
    with __import__("PIL").Image.open(p) as im:
        arr = np.array(im)
    img = arr.mean(axis=-1).astype(np.uint8) if arr.ndim == 3 else arr.astype(np.uint8)
    h, w = img.shape
    img_g, bbox = preprocess_gray55(img)

    b_mask, b_style, b_geo, b_cov = infer_apo(base_learn, img_g, bbox)
    l_mask, l_style, l_geo, l_cov = infer_apo(line_learn, img_g, bbox)

    rows.append({
        "image_id": p.name,
        "res": f"{h}x{w}",
        "base_pred_cov": b_cov,
        "line_model_pred_cov": l_cov,
        "base_style": b_style,
        "line_model_style": l_style,
        "base_mt_ok": bool(np.isfinite(b_geo["mt_px"])),
        "line_model_mt_ok": bool(np.isfinite(l_geo["mt_px"])),
        "base_mt_fail_reason": b_geo["mt_fail_reason"],
        "line_model_mt_fail_reason": l_geo["mt_fail_reason"],
    })

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/apo_gray55_line_eval.csv", index=False)

summary = {
    "n_test": len(df),
    "baseline_gray55_infer_mt_ok_mean": float(df.base_mt_ok.mean()),
    "line_model_gray55_infer_mt_ok_mean": float(df.line_model_mt_ok.mean()),
    "baseline_fail_counts": df.loc[~df.base_mt_ok, "base_mt_fail_reason"].value_counts().to_dict(),
    "line_model_fail_counts": df.loc[~df.line_model_mt_ok, "line_model_mt_fail_reason"].value_counts().to_dict(),
    "fixed_by_line_model": int(((~df.base_mt_ok) & (df.line_model_mt_ok)).sum()),
    "broken_by_line_model": int((df.base_mt_ok & ~df.line_model_mt_ok).sum()),
    "single_contour_baseline": int((df.base_mt_fail_reason == "single_contour").sum()),
    "single_contour_line_model": int((df.line_model_mt_fail_reason == "single_contour").sum()),
    "no_contours_baseline": int((df.base_mt_fail_reason == "no_contours").sum()),
    "no_contours_line_model": int((df.line_model_mt_fail_reason == "no_contours").sum()),
}
with open("/kaggle/working/apo_gray55_line_eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-gray55-line-eval"
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
    (out / "apo-gray55-line-eval-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": "ucheozoemena/umud-apo-gray55-line-eval-phase-3",
        "title": "UMUD Apo Gray55 Line Eval Phase 3",
        "code_file": "apo-gray55-line-eval-phase-3.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [
            "ucheozoemena/umud-train-apo-mounted-phase-3",
            "ucheozoemena/umud-train-apo-gray55-phase-3",
        ],
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
