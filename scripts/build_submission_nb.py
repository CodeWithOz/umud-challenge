"""Generate notebooks/submission/submission-phase-3.ipynb — segment-then-measure inference + submission CSV."""
import json
from pathlib import Path

# Production 5ep r50 (Block 6c — score 1.873 beats r34 1.913)
BUILD_APO_MODEL_FILE = "apo_gray55_line_200_r50.pkl"
BUILD_SUBMISSION_LABEL = "Phase 4 production — 200-tier apo r50 5ep + MM=0.075"
BUILD_MM_PER_PIXEL = 0.075


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
        f"""# UMUD — Submission ({BUILD_SUBMISSION_LABEL})

**GPU notebook** — segment-then-measure pipeline for test images:

1. Load **fasc** + **gray55+line apo** fastai learners
2. Apo inference: gray55 outside ROI bbox + mask clip; fasc on raw image
3. Derive **PA / FL / MT** via horizontality+parallelism contour pairing
4. Apply **`MM_PER_PIXEL`** to convert FL/MT to mm (production: **0.075**)
5. Write `submission.csv` (comma-separated, 309 rows)

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        f"""from pathlib import Path

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
GRAY_FILL_VALUE = 55
ROI_THRESH = 5
ROI_PAD_PX = 10
TOP_K_CANDIDATES = 8
MIN_SEP_PX = 15

# Production calibration (Block 1 bracket + Block 3 confirm).
MM_PER_PIXEL = {BUILD_MM_PER_PIXEL}
APO_MODEL_FILE = "{BUILD_APO_MODEL_FILE}"


def resolve_pkl(preferred: list[Path], filename: str) -> Path:
    for p in preferred:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find {{filename}} under /kaggle/input")


FASC_MODEL_PATH = resolve_pkl(
    [Path("/kaggle/input/notebooks/ucheozoemena/umud-train-mounted-phase-3/fasc_baseline.pkl")],
    "fasc_baseline.pkl",
)
APO_MODEL_PATH = resolve_pkl(
    [Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-gray55-phase-3") / APO_MODEL_FILE],
    APO_MODEL_FILE,
)
print("Fasc model:", FASC_MODEL_PATH)
print("Apo model:", APO_MODEL_PATH)

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


def contour_feats(c: np.ndarray) -> dict:
    x, y, w, h = cv2.boundingRect(c)
    pts = c.reshape(-1, 2)
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    area = float(cv2.contourArea(c))
    return {"area": area, "x_span": x_span, "y_top": y, "y_bot": y + h, "w": w, "h": h}


def x_overlap_from_contours(sup_c: np.ndarray, deep_c: np.ndarray) -> float:
    sup_x, _ = edge_polyline(sup_c, which="bottom")
    deep_x, _ = edge_polyline(deep_c, which="top")
    if len(sup_x) == 0 or len(deep_x) == 0:
        return 0.0
    return max(0.0, min(float(sup_x.max()), float(deep_x.max())) - max(float(sup_x.min()), float(deep_x.min())))


def edge_angle_from_horizontal(contour: np.ndarray, which: str):
    xs, ys = edge_polyline(contour, which=which)
    if len(xs) < 2:
        return None
    line = fit_line(xs, ys)
    if line is None:
        return None
    ang = abs(float(np.degrees(np.arctan(line[1])))) % 180.0
    return float(min(ang, 180.0 - ang))


def horizontality_factor(angle_deg) -> float:
    if angle_deg is None:
        return 0.0
    return float(np.cos(np.radians(angle_deg)) ** 2)


def parallelism_factor(sup_ang, deep_ang) -> float:
    if sup_ang is None or deep_ang is None:
        return 0.0
    d = abs(sup_ang - deep_ang) % 180.0
    d = min(d, 180.0 - d)
    return float(np.cos(np.radians(d)) ** 2)


def pick_best_pair_xspan(contours: list[np.ndarray], min_sep_px: int = MIN_SEP_PX, top_k: int = TOP_K_CANDIDATES):
    if len(contours) < 2:
        return None, None
    ranked = sorted(contours, key=lambda c: contour_feats(c)["x_span"], reverse=True)
    candidates = ranked[: min(top_k, len(ranked))]
    best_pair = None
    best_overlap = -1.0
    for i, ci in enumerate(candidates):
        for cj in candidates[i + 1 :]:
            fi, fj = contour_feats(ci), contour_feats(cj)
            if fi["y_top"] <= fj["y_top"]:
                sup_c, deep_c = ci, cj
                fs, fd = fi, fj
            else:
                sup_c, deep_c = cj, ci
                fs, fd = fj, fi
            if fd["y_top"] < fs["y_top"] + min_sep_px:
                continue
            overlap = x_overlap_from_contours(sup_c, deep_c)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (sup_c, deep_c)
    if best_pair is not None:
        return best_pair
    if len(candidates) >= 2:
        top2 = candidates[:2]
        top2.sort(key=lambda c: contour_feats(c)["y_top"])
        return top2[0], top2[1]
    return None, None


def pick_horiz_parallel(contours: list[np.ndarray], min_sep_px: int = MIN_SEP_PX, top_k: int = TOP_K_CANDIDATES):
    if len(contours) < 2:
        return None, None
    ranked = sorted(
        contours,
        key=lambda c: max(
            contour_feats(c)["x_span"] * horizontality_factor(edge_angle_from_horizontal(c, "bottom")),
            contour_feats(c)["x_span"] * horizontality_factor(edge_angle_from_horizontal(c, "top")),
        ),
        reverse=True,
    )
    candidates = ranked[: min(top_k, len(ranked))]
    best_pair = None
    best_score = -1.0
    for i, ci in enumerate(candidates):
        for cj in candidates[i + 1 :]:
            fi, fj = contour_feats(ci), contour_feats(cj)
            if fi["y_top"] <= fj["y_top"]:
                sup_c, deep_c = ci, cj
                fs, fd = fi, fj
            else:
                sup_c, deep_c = cj, ci
                fs, fd = fj, fi
            if fd["y_top"] < fs["y_top"] + min_sep_px:
                continue
            overlap = x_overlap_from_contours(sup_c, deep_c)
            if overlap <= 0:
                continue
            sup_ang = edge_angle_from_horizontal(sup_c, "bottom")
            deep_ang = edge_angle_from_horizontal(deep_c, "top")
            score = (
                overlap
                * horizontality_factor(sup_ang)
                * horizontality_factor(deep_ang)
                * parallelism_factor(sup_ang, deep_ang)
            )
            if score > best_score:
                best_score = score
                best_pair = (sup_c, deep_c)
    if best_pair is not None:
        return best_pair
    return pick_best_pair_xspan(contours, min_sep_px=min_sep_px, top_k=top_k)


def pick_superficial_deep(contours: list[np.ndarray], min_sep_px: int = 15):
    sup_c, deep_c = pick_horiz_parallel(contours, min_sep_px=min_sep_px, top_k=TOP_K_CANDIDATES)
    return sup_c, deep_c, len(contours)


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


def apo_geometry_attempt(apo_mask: np.ndarray, style: str) -> dict:
    eff, method = effective_apo_mask(apo_mask, style)
    contours = find_apo_contours(eff)
    sup_c, deep_c, n_contours = pick_superficial_deep(contours)
    out = {
        "apo_method": method,
        "n_contours": n_contours,
        "mt_px": np.nan,
        "deep_angle_deg": np.nan,
        "mt_fail_reason": "ok",
        "sup_line": None,
        "deep_line": None,
        "sup_xs": None,
        "sup_ys": None,
        "deep_xs": None,
        "deep_ys": None,
    }
    if len(contours) == 0:
        out["mt_fail_reason"] = "no_contours"
        return out
    if n_contours < 2 or sup_c is None or deep_c is None:
        out["mt_fail_reason"] = "single_contour"
        return out
    sup_x, sup_y = edge_polyline(sup_c, which="bottom")
    deep_x, deep_y = edge_polyline(deep_c, which="top")
    sup_line = fit_line(sup_x, sup_y)
    deep_line = fit_line(deep_x, deep_y)
    out.update(sup_line=sup_line, deep_line=deep_line, sup_xs=sup_x, sup_ys=sup_y, deep_xs=deep_x, deep_ys=deep_y)
    if sup_line is None or deep_line is None:
        out["mt_fail_reason"] = "line_fit_fail"
        return out
    if len(sup_x) == 0 or len(deep_x) == 0:
        out["mt_fail_reason"] = "empty_edge_polyline"
        return out
    x_left = max(sup_x.min(), deep_x.min())
    x_right = min(sup_x.max(), deep_x.max())
    if x_right <= x_left:
        out["mt_fail_reason"] = "no_x_overlap"
        return out
    out["mt_px"] = mt_from_apo_edges(sup_line, deep_line, x_left, x_right)
    out["deep_angle_deg"] = line_angle_deg(deep_line)
    if np.isnan(out["mt_px"]):
        out["mt_fail_reason"] = "mt_compute_nan"
    return out


def apo_geometry_from_mask(apo_mask: np.ndarray, style: str) -> dict:
    primary = apo_geometry_attempt(apo_mask, style)
    primary["geometry_path"] = primary["apo_method"]
    if not np.isnan(primary["mt_px"]):
        return primary
    if style == "region":
        fallback = apo_geometry_attempt(apo_mask, "line")
        if not np.isnan(fallback["mt_px"]):
            fallback["apo_method"] = f"{primary['apo_method']}+fallback_line"
            fallback["geometry_path"] = "fallback_line"
            fallback["mt_fail_reason_primary"] = primary["mt_fail_reason"]
            return fallback
    primary["mt_fail_reason_primary"] = primary["mt_fail_reason"]
    return primary


def derive_geometry(fasc_mask: np.ndarray, apo_mask: np.ndarray, apo_style: str) -> dict:
    apo = apo_geometry_from_mask(apo_mask, apo_style)
    fpca = fascicle_pca(fasc_mask)
    out = {
        "pa_deg": np.nan,
        "fl_px": np.nan,
        "mt_px": apo["mt_px"],
        "apo_method": apo.get("apo_method"),
        "geometry_path": apo.get("geometry_path"),
        "n_contours": apo.get("n_contours"),
        "mt_fail_reason": apo.get("mt_fail_reason"),
        "mt_fail_reason_primary": apo.get("mt_fail_reason_primary"),
        "apo_cov": float(apo_mask.mean()),
        "apo_fg_pixels": int(apo_mask.sum()),
        "fasc_cov": float(fasc_mask.mean()),
        "fasc_fg_pixels": int(fasc_mask.sum()),
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
    img_gray55, bbox = preprocess_gray55(img_native)

    pil_fasc = open_rgb_256(img_native)
    pil_apo = open_rgb_256(img_gray55)

    _, fasc_t, _ = fasc_learn.predict(pil_fasc)
    _, apo_t, _ = apo_learn.predict(pil_apo)
    fasc_native = resize_mask_to(tensor_to_mask(fasc_t), h, w)
    apo_native = clip_mask_to_bbox(resize_mask_to(tensor_to_mask(apo_t), h, w), bbox)

    apo_style = tag_apo_style(float(apo_native.mean()))
    geo = derive_geometry(fasc_native, apo_native, apo_style)

    pa = geo["pa_deg"]
    fl_mm = geo["fl_px"] * MM_PER_PIXEL if not np.isnan(geo["fl_px"]) else np.nan
    mt_mm = geo["mt_px"] * MM_PER_PIXEL if not np.isnan(geo["mt_px"]) else np.nan

    rows.append(
        {
            "image_id": path.name,
            "pa_deg": pa,
            "fl_mm": fl_mm,
            "mt_mm": mt_mm,
            "apo_style": apo_style,
            "fl_px": geo["fl_px"],
            "mt_px": geo["mt_px"],
            "apo_cov": geo["apo_cov"],
            "apo_fg_pixels": geo["apo_fg_pixels"],
            "fasc_cov": geo["fasc_cov"],
            "n_contours": geo["n_contours"],
            "geometry_path": geo["geometry_path"],
            "mt_fail_reason": geo["mt_fail_reason"],
            "mt_fail_reason_primary": geo.get("mt_fail_reason_primary"),
            "img_h": h,
            "img_w": w,
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
        """def read_sample_submission(path: Path) -> pd.DataFrame:
    for sep in (",", ";"):
        try:
            df = pd.read_csv(path, sep=sep)
            if {"image_id", "pa_deg", "fl_mm", "mt_mm"}.issubset(df.columns):
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python")


pred_lookup = pred_df.set_index("image_id")

if SAMPLE_SUBMISSION.exists():
    template = read_sample_submission(SAMPLE_SUBMISSION)
    template_ids = template["image_id"].astype(str)
    missing = [i for i in template_ids if i not in pred_lookup.index]
    if missing:
        print(f"Warning: {len(missing)} template ids missing predictions (first 5): {missing[:5]}")
    if len(template_ids) >= max(10, len(pred_df) // 2):
        submit = template[["image_id"]].copy()
        for col in ["pa_deg", "fl_mm", "mt_mm"]:
            submit[col] = submit["image_id"].map(pred_lookup[col])
    else:
        print(
            f"Template has only {len(template_ids)} rows; "
            f"writing {len(pred_df)} predictions instead"
        )
        submit = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
else:
    submit = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")

out_path = Path("/kaggle/working/submission.csv")
submit.to_csv(out_path, index=False)
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
