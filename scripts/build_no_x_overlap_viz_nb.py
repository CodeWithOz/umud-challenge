"""Generate notebooks/no-x-overlap-viz — visual QC for apo MT no_x_overlap failures."""
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
        """# UMUD — Apo MT Visual QC: `no_x_overlap` (Phase 3)

**GPU notebook** — side-by-side galleries for test images where apo geometry finds **two contours** but **no shared horizontal span** between superficial and deep edge lines (`mt_fail_reason=no_x_overlap`).

Uses the **gray55+line** apo model with standard inference preprocessing:
1. Gray-fill outside ROI bbox (RGB 55)
2. Predict mask; clip mask to bbox
3. Run geometry

Each case uses a 6-panel layout:

| # | Panel |
|---|--------|
| 1 | Raw test image + ROI bbox |
| 2 | Gray55-preprocessed image |
| 3 | Predicted apo mask (bbox-clipped) |
| 4 | Orange overlay — mask on gray55 image |
| 5 | **X-overlap** — fitted sup (cyan) / deep (magenta) edges + shaded overlap band (green) or gap marker (red) |
| 6 | 1D x-range bars — superficial vs deep horizontal spans |

Overlays: Phase 0/1 style (`MASK_OVERLAY_ALPHA=0.55`, orange `(255,140,0)`).

Outputs:
- `/kaggle/working/no_x_overlap_scan.csv`
- Figures under `/kaggle/working/figures/no_x_overlap/`"""
    ),
    md("## Configuration"),
    code(
        """import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
GRAY_FILL_VALUE = 55
ROI_THRESH = 5
ROI_PAD_PX = 10

N_SHOW = None  # None = show all no_x_overlap cases; set e.g. 20 for a random sample
RANDOM_SEED = 42

MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_ROOT = Path("/kaggle/working/figures/no_x_overlap")
FIG_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_pkl(preferred: list[Path], filename: str) -> Path:
    for p in preferred:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")


LINE_MODEL = resolve_pkl(
    [Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-gray55-phase-3/apo_gray55_line_baseline.pkl")],
    "apo_gray55_line_baseline.pkl",
)
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


def overlay(img_gray: np.ndarray, mask: np.ndarray, color=APO_OVERLAY_COLOR, alpha=MASK_OVERLAY_ALPHA):
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    sel = mask.astype(bool)
    rgb[sel] = (1 - alpha) * rgb[sel] + alpha * tint[sel]
    return rgb.astype(np.uint8)


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


def x_overlap_stats(geo: dict) -> dict:
    sup_x = geo.get("sup_xs")
    deep_x = geo.get("deep_xs")
    if sup_x is None or deep_x is None or len(sup_x) == 0 or len(deep_x) == 0:
        return {
            "sup_xmin": np.nan,
            "sup_xmax": np.nan,
            "deep_xmin": np.nan,
            "deep_xmax": np.nan,
            "overlap_px": 0.0,
            "gap_px": np.nan,
        }
    sup_lo, sup_hi = float(sup_x.min()), float(sup_x.max())
    deep_lo, deep_hi = float(deep_x.min()), float(deep_x.max())
    x_left = max(sup_lo, deep_lo)
    x_right = min(sup_hi, deep_hi)
    overlap = max(0.0, x_right - x_left)
    if overlap > 0:
        gap = 0.0
    else:
        gap = max(sup_lo, deep_lo) - min(sup_hi, deep_hi)
    return {
        "sup_xmin": sup_lo,
        "sup_xmax": sup_hi,
        "deep_xmin": deep_lo,
        "deep_xmax": deep_hi,
        "overlap_px": overlap,
        "gap_px": gap,
    }


def draw_apo_edges(ax, geo: dict):
    if geo.get("sup_xs") is not None and len(geo["sup_xs"]):
        ax.scatter(geo["sup_xs"], geo["sup_ys"], s=4, c="cyan", label="sup edge")
        xs = np.linspace(geo["sup_xs"].min(), geo["sup_xs"].max(), 50)
        if geo.get("sup_line") is not None:
            ax.plot(xs, geo["sup_line"](xs), c="cyan", lw=2)
    if geo.get("deep_xs") is not None and len(geo["deep_xs"]):
        ax.scatter(geo["deep_xs"], geo["deep_ys"], s=4, c="magenta", label="deep edge")
        xs = np.linspace(geo["deep_xs"].min(), geo["deep_xs"].max(), 50)
        if geo.get("deep_line") is not None:
            ax.plot(xs, geo["deep_line"](xs), c="magenta", lw=2)
    ax.legend(fontsize=6, loc="upper right")


def draw_overlap_band(ax, geo: dict, img_h: int):
    stats = x_overlap_stats(geo)
    sup_lo, sup_hi = stats["sup_xmin"], stats["sup_xmax"]
    deep_lo, deep_hi = stats["deep_xmin"], stats["deep_xmax"]
    if not np.isfinite(sup_lo):
        return stats
    x_left = max(sup_lo, deep_lo)
    x_right = min(sup_hi, deep_hi)
    if x_right > x_left:
        ax.axvspan(x_left, x_right, color="lime", alpha=0.25, label=f"overlap {x_right - x_left:.0f}px")
    else:
        mid = (max(sup_hi, deep_hi) + min(sup_lo, deep_lo)) / 2.0
        ax.axvline(mid, color="red", lw=2, linestyle="--", label=f"gap {stats['gap_px']:.0f}px")
    ax.set_xlim(0, ax.get_xlim()[1])
    return stats


def x_range_panel(ax, stats: dict, img_w: int):
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.75, 0.25])
    ax.set_yticklabels(["sup", "deep"], fontsize=8)
    ax.set_xlabel("x (px)", fontsize=8)
    if np.isfinite(stats["sup_xmin"]):
        ax.barh(0.75, stats["sup_xmax"] - stats["sup_xmin"], left=stats["sup_xmin"], height=0.2, color="cyan", alpha=0.8)
    if np.isfinite(stats["deep_xmin"]):
        ax.barh(0.25, stats["deep_xmax"] - stats["deep_xmin"], left=stats["deep_xmin"], height=0.2, color="magenta", alpha=0.8)
    if stats["overlap_px"] > 0:
        ax.axvspan(
            max(stats["sup_xmin"], stats["deep_xmin"]),
            min(stats["sup_xmax"], stats["deep_xmax"]),
            color="lime",
            alpha=0.2,
        )
    else:
        ax.text(img_w * 0.5, 0.5, f"no overlap\\ngap={stats['gap_px']:.0f}px", ha="center", va="center", color="red", fontsize=9)
    ax.set_title("x-ranges", fontsize=9)


def no_x_overlap_panel(case: dict, idx: int, fig_dir: Path):
    img = case["img_raw"]
    img_g = case["img_gray55"]
    pred_mask = case["pred_mask"]
    bbox = case["bbox"]
    geo = case["geo"]
    stats = case["x_stats"]
    y0, y1, x0, x1 = bbox
    cov = float(pred_mask.mean()) * 100

    fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))

    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
    axes[0].set_title("raw + bbox", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img_g, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"gray55 (fill={GRAY_FILL_VALUE})", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"pred mask\\ncov={cov:.2f}% n_ct={geo['n_contours']}", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(overlay(img_g, pred_mask))
    axes[3].set_title("overlay", fontsize=9)
    axes[3].axis("off")

    axes[4].imshow(img_g, cmap="gray")
    draw_apo_edges(axes[4], geo)
    draw_overlap_band(axes[4], geo, img.shape[0])
    axes[4].set_title("edges + overlap band", fontsize=9)
    axes[4].axis("off")

    x_range_panel(axes[5], stats, img.shape[1])

    plt.suptitle(
        f"[{idx}] {case['image_id']}  style={case['apo_style']}  fail={geo['mt_fail_reason']}  "
        f"sup_x=[{stats['sup_xmin']:.0f},{stats['sup_xmax']:.0f}]  "
        f"deep_x=[{stats['deep_xmin']:.0f},{stats['deep_xmax']:.0f}]  "
        f"overlap={stats['overlap_px']:.0f}px gap={stats['gap_px']:.0f}px",
        y=1.05,
        fontsize=9,
    )
    plt.tight_layout()
    stem = Path(case["image_id"]).stem
    out = fig_dir / f"no_x_overlap_{idx:03d}_{stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out
"""
        ),
        code(
            """assert LINE_MODEL.exists(), f"Missing model: {LINE_MODEL}"
assert TEST_DIR.exists(), f"Missing test dir: {TEST_DIR}"

line_learn = load_learner(LINE_MODEL)
test_paths = list_test_images(TEST_DIR)
print(f"Test images: {len(test_paths)}")
"""
        ),
        code(
            """all_cases = []

for path in tqdm(test_paths, desc="scan gray55+line predictions"):
    img_native = load_gray(path)
    h, w = img_native.shape
    img_g, bbox = preprocess_gray55(img_native)

    pil = open_rgb_256(img_g)
    _, apo_t, _ = line_learn.predict(pil)
    pred_mask = resize_mask_to(tensor_to_mask(apo_t), h, w)
    pred_mask = clip_mask_to_bbox(pred_mask, bbox)

    apo_style = tag_apo_style(float(pred_mask.mean()))
    geo = apo_geometry_from_mask(pred_mask, apo_style)
    x_stats = x_overlap_stats(geo)

    all_cases.append(
        {
            "image_id": path.name,
            "img_raw": img_native,
            "img_gray55": img_g,
            "pred_mask": pred_mask,
            "bbox": bbox,
            "apo_style": apo_style,
            "apo_cov": float(pred_mask.mean()),
            "n_contours": geo["n_contours"],
            "mt_px": geo["mt_px"],
            "mt_ok": not np.isnan(geo["mt_px"]),
            "mt_fail_reason": geo["mt_fail_reason"],
            "geo": geo,
            "x_stats": x_stats,
            "img_h": h,
            "img_w": w,
        }
    )

scan_rows = []
for c in all_cases:
    row = {
        "image_id": c["image_id"],
        "res": f"{c['img_h']}x{c['img_w']}",
        "apo_style": c["apo_style"],
        "apo_cov": c["apo_cov"],
        "n_contours": c["n_contours"],
        "mt_ok": c["mt_ok"],
        "mt_fail_reason": c["mt_fail_reason"],
        **{f"x_{k}": v for k, v in c["x_stats"].items()},
    }
    scan_rows.append(row)

scan_df = pd.DataFrame(scan_rows)
scan_df.to_csv("/kaggle/working/no_x_overlap_scan.csv", index=False)

fail_cases = [c for c in all_cases if c["mt_fail_reason"] == "no_x_overlap"]
print(f"Total test images: {len(all_cases)}")
print(f"no_x_overlap failures: {len(fail_cases)}")
print(f"MT OK: {sum(c['mt_ok'] for c in all_cases)}")
print()
print("All failure reasons:")
print(scan_df.loc[~scan_df.mt_ok, "mt_fail_reason"].value_counts().to_string())
print()
if len(fail_cases):
    print("no_x_overlap gap_px stats:")
    gaps = [c["x_stats"]["gap_px"] for c in fail_cases]
    print(pd.Series(gaps).describe().round(1))
    print()
    print("no_x_overlap n_contours:")
    print(pd.Series([c["n_contours"] for c in fail_cases]).value_counts().to_string())
"""
        ),
        md("## Gallery — `no_x_overlap` cases (gray55+line model)"),
        code(
            """if not fail_cases:
    print("No no_x_overlap cases found — nothing to plot.")
else:
    show_cases = fail_cases
    if N_SHOW is not None and len(fail_cases) > N_SHOW:
        rng = random.Random(RANDOM_SEED)
        show_cases = rng.sample(fail_cases, N_SHOW)
        print(f"Showing random sample of {N_SHOW} / {len(fail_cases)} cases (seed={RANDOM_SEED})")
    else:
        print(f"Showing all {len(show_cases)} no_x_overlap cases")

    saved = []
    for i, case in enumerate(show_cases, start=1):
        out = no_x_overlap_panel(case, i, FIG_ROOT)
        saved.append(str(out))

    print(f"\\nSaved {len(saved)} figures under {FIG_ROOT}")
    for p in saved[:5]:
        print(" ", p)
    if len(saved) > 5:
        print(f"  ... and {len(saved) - 5} more")
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/no-x-overlap-viz"
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
    (out / "no-x-overlap-viz-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-no-x-overlap-viz-phase-3",
                "title": "UMUD No X Overlap Viz Phase 3",
                "code_file": "no-x-overlap-viz-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": [
                    "ucheozoemena/umud-train-apo-gray55-phase-3",
                ],
                "competition_sources": [
                    "umud-challenge-muscle-architecture-in-ultrasound-data"
                ],
                "model_sources": [],
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9",
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
