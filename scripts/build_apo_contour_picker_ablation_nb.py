"""Generate notebooks/apo-contour-picker-ablation — legacy vs x-span pair picker."""
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
        """# UMUD — Apo Contour Picker Ablation (Phase 3)

**GPU notebook** — compare two contour-pair strategies on gray55+line apo predictions:

| Picker | Rule |
|--------|------|
| **legacy** | Sort contours top→bottom; sup = topmost; deep = first separated ≥15px below (current submission) |
| **xspan_pair** | Top-K candidates by horizontal span; pick pair with **maximum x-overlap** and vertical separation |

Inference: gray55 outside bbox + mask clip (unchanged).

Outputs:
- `/kaggle/working/contour_picker_ablation.csv`
- `/kaggle/working/contour_picker_ablation_summary.json`
- Figures under `/kaggle/working/figures/contour_picker_ablation/`"""
    ),
    md("## Configuration"),
    code(
        """import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
GRAY_FILL_VALUE = 55
ROI_THRESH = 5
ROI_PAD_PX = 10

TOP_K_CANDIDATES = 8
MIN_SEP_PX = 15

N_SHOW_RESCUED = None  # None = all legacy no_x_overlap rescued by xspan_pair
N_SHOW_STILL_FAIL = 12
RANDOM_SEED = 42

MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_ROOT = Path("/kaggle/working/figures/contour_picker_ablation")
FIG_RESCUED = FIG_ROOT / "rescued"
FIG_STILL_FAIL = FIG_ROOT / "still_fail"
FIG_RESCUED.mkdir(parents=True, exist_ok=True)
FIG_STILL_FAIL.mkdir(parents=True, exist_ok=True)


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


def contour_feats(c: np.ndarray) -> dict:
    x, y, w, h = cv2.boundingRect(c)
    pts = c.reshape(-1, 2)
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    area = float(cv2.contourArea(c))
    return {
        "area": area,
        "x_span": x_span,
        "y_top": y,
        "y_bot": y + h,
        "w": w,
        "h": h,
        "score": x_span * float(np.sqrt(max(area, 1.0))),
    }


def x_overlap_from_contours(sup_c: np.ndarray, deep_c: np.ndarray) -> float:
    sup_x, _ = edge_polyline(sup_c, which="bottom")
    deep_x, _ = edge_polyline(deep_c, which="top")
    if len(sup_x) == 0 or len(deep_x) == 0:
        return 0.0
    return max(0.0, min(float(sup_x.max()), float(deep_x.max())) - max(float(sup_x.min()), float(deep_x.min())))


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


def apo_geometry_with_picker(apo_mask: np.ndarray, style: str, picker: str = "legacy") -> dict:
    eff, method = effective_apo_mask(apo_mask, style)
    contours = find_apo_contours(eff)
    if picker == "legacy":
        sup_c, deep_c, n_contours = pick_superficial_deep(contours)
    else:
        sup_c, deep_c = pick_best_pair_xspan(contours)
        n_contours = len(contours)

    out = {
        "apo_method": method,
        "picker": picker,
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
    gap = 0.0 if overlap > 0 else max(sup_lo, deep_lo) - min(sup_hi, deep_hi)
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
        ax.scatter(geo["sup_xs"], geo["sup_ys"], s=4, c="cyan", label="sup")
        xs = np.linspace(geo["sup_xs"].min(), geo["sup_xs"].max(), 50)
        if geo.get("sup_line") is not None:
            ax.plot(xs, geo["sup_line"](xs), c="cyan", lw=2)
    if geo.get("deep_xs") is not None and len(geo["deep_xs"]):
        ax.scatter(geo["deep_xs"], geo["deep_ys"], s=4, c="magenta", label="deep")
        xs = np.linspace(geo["deep_xs"].min(), geo["deep_xs"].max(), 50)
        if geo.get("deep_line") is not None:
            ax.plot(xs, geo["deep_line"](xs), c="magenta", lw=2)
    ax.legend(fontsize=6, loc="upper right")


def draw_overlap_band(ax, geo: dict):
    stats = x_overlap_stats(geo)
    if not np.isfinite(stats["sup_xmin"]):
        return stats
    x_left = max(stats["sup_xmin"], stats["deep_xmin"])
    x_right = min(stats["sup_xmax"], stats["deep_xmax"])
    if x_right > x_left:
        ax.axvspan(x_left, x_right, color="lime", alpha=0.25)
    else:
        mid = (max(stats["sup_xmax"], stats["deep_xmax"]) + min(stats["sup_xmin"], stats["deep_xmin"])) / 2.0
        ax.axvline(mid, color="red", lw=2, linestyle="--")
    return stats


def x_range_panel(ax, stats: dict, img_w: int, title: str):
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.75, 0.25])
    ax.set_yticklabels(["sup", "deep"], fontsize=8)
    if np.isfinite(stats["sup_xmin"]):
        ax.barh(0.75, stats["sup_xmax"] - stats["sup_xmin"], left=stats["sup_xmin"], height=0.2, color="cyan", alpha=0.8)
    if np.isfinite(stats["deep_xmin"]):
        ax.barh(0.25, stats["deep_xmax"] - stats["deep_xmin"], left=stats["deep_xmin"], height=0.2, color="magenta", alpha=0.8)
    if stats["overlap_px"] > 0:
        ax.axvspan(max(stats["sup_xmin"], stats["deep_xmin"]), min(stats["sup_xmax"], stats["deep_xmax"]), color="lime", alpha=0.2)
    else:
        ax.text(img_w * 0.5, 0.5, f"gap={stats['gap_px']:.0f}px", ha="center", va="center", color="red", fontsize=8)
    ax.set_title(title, fontsize=9)


def compare_panel(case: dict, idx: int, group: str, fig_dir: Path):
    img = case["img_raw"]
    img_g = case["img_gray55"]
    mask = case["pred_mask"]
    bbox = case["bbox"]
    legacy = case["legacy_geo"]
    alt = case["alt_geo"]
    legacy_stats = case["legacy_stats"]
    alt_stats = case["alt_stats"]
    y0, y1, x0, x1 = bbox

    fig, axes = plt.subplots(2, 4, figsize=(22, 8))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
    axes[0, 0].set_title("raw + bbox", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_g, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("gray55", fontsize=9)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"pred mask\\nn_ct={legacy['n_contours']}", fontsize=9)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(overlay(img_g, mask))
    axes[0, 3].set_title("overlay", fontsize=9)
    axes[0, 3].axis("off")

    axes[1, 0].imshow(img_g, cmap="gray")
    draw_apo_edges(axes[1, 0], legacy)
    draw_overlap_band(axes[1, 0], legacy)
    axes[1, 0].set_title(f"legacy: {legacy['mt_fail_reason']}", fontsize=9)
    axes[1, 0].axis("off")

    x_range_panel(axes[1, 1], legacy_stats, img.shape[1], "legacy x-ranges")

    axes[1, 2].imshow(img_g, cmap="gray")
    draw_apo_edges(axes[1, 2], alt)
    draw_overlap_band(axes[1, 2], alt)
    mt_lbl = f"{alt['mt_px']:.0f}px" if np.isfinite(alt["mt_px"]) else "NaN"
    axes[1, 2].set_title(f"xspan_pair: {alt['mt_fail_reason']} mt={mt_lbl}", fontsize=9)
    axes[1, 2].axis("off")

    x_range_panel(axes[1, 3], alt_stats, img.shape[1], "xspan_pair x-ranges")

    plt.suptitle(
        f"[{group} {idx}] {case['image_id']}  legacy_overlap={legacy_stats['overlap_px']:.0f}px  "
        f"alt_overlap={alt_stats['overlap_px']:.0f}px",
        y=1.02,
        fontsize=10,
    )
    plt.tight_layout()
    stem = Path(case["image_id"]).stem
    out = fig_dir / f"{group}_{idx:03d}_{stem}.png"
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
            """rows = []
cases = []

for path in tqdm(test_paths, desc="ablation scan"):
    img_native = load_gray(path)
    h, w = img_native.shape
    img_g, bbox = preprocess_gray55(img_native)

    _, apo_t, _ = line_learn.predict(open_rgb_256(img_g))
    pred_mask = clip_mask_to_bbox(resize_mask_to(tensor_to_mask(apo_t), h, w), bbox)
    style = tag_apo_style(float(pred_mask.mean()))

    legacy_geo = apo_geometry_with_picker(pred_mask, style, picker="legacy")
    alt_geo = apo_geometry_with_picker(pred_mask, style, picker="xspan_pair")
    legacy_stats = x_overlap_stats(legacy_geo)
    alt_stats = x_overlap_stats(alt_geo)

    legacy_ok = bool(np.isfinite(legacy_geo["mt_px"]))
    alt_ok = bool(np.isfinite(alt_geo["mt_px"]))

    rows.append(
        {
            "image_id": path.name,
            "res": f"{h}x{w}",
            "apo_style": style,
            "apo_cov": float(pred_mask.mean()),
            "n_contours": legacy_geo["n_contours"],
            "legacy_mt_ok": legacy_ok,
            "legacy_mt_fail_reason": legacy_geo["mt_fail_reason"],
            "legacy_mt_px": float(legacy_geo["mt_px"]) if legacy_ok else np.nan,
            "legacy_overlap_px": legacy_stats["overlap_px"],
            "legacy_gap_px": legacy_stats["gap_px"],
            "alt_mt_ok": alt_ok,
            "alt_mt_fail_reason": alt_geo["mt_fail_reason"],
            "alt_mt_px": float(alt_geo["mt_px"]) if alt_ok else np.nan,
            "alt_overlap_px": alt_stats["overlap_px"],
            "alt_gap_px": alt_stats["gap_px"],
            "rescued": bool((not legacy_ok) and alt_ok),
            "broken": bool(legacy_ok and (not alt_ok)),
        }
    )

    cases.append(
        {
            "image_id": path.name,
            "img_raw": img_native,
            "img_gray55": img_g,
            "pred_mask": pred_mask,
            "bbox": bbox,
            "legacy_geo": legacy_geo,
            "alt_geo": alt_geo,
            "legacy_stats": legacy_stats,
            "alt_stats": alt_stats,
            "legacy_ok": legacy_ok,
            "alt_ok": alt_ok,
        }
    )

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/contour_picker_ablation.csv", index=False)

legacy_no_x = (df.legacy_mt_fail_reason == "no_x_overlap").sum()
rescued_from_no_x = int(((df.legacy_mt_fail_reason == "no_x_overlap") & df.alt_mt_ok).sum())
still_no_x = int(((df.legacy_mt_fail_reason == "no_x_overlap") & (df.alt_mt_fail_reason == "no_x_overlap")).sum())

summary = {
    "n_test": int(len(df)),
    "legacy_mt_ok_rate": float(df.legacy_mt_ok.mean()),
    "alt_mt_ok_rate": float(df.alt_mt_ok.mean()),
    "rescued_total": int(df.rescued.sum()),
    "broken_total": int(df.broken.sum()),
    "legacy_fail_counts": df.loc[~df.legacy_mt_ok, "legacy_mt_fail_reason"].value_counts().to_dict(),
    "alt_fail_counts": df.loc[~df.alt_mt_ok, "alt_mt_fail_reason"].value_counts().to_dict(),
    "legacy_no_x_overlap": int(legacy_no_x),
    "rescued_from_no_x_overlap": rescued_from_no_x,
    "still_no_x_overlap_both": still_no_x,
    "picker": {"top_k": TOP_K_CANDIDATES, "min_sep_px": MIN_SEP_PX},
}
with open("/kaggle/working/contour_picker_ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))

rescued_cases = [c for c in cases if (not c["legacy_ok"]) and c["alt_ok"]]
legacy_no_x_cases = [c for c in cases if c["legacy_geo"]["mt_fail_reason"] == "no_x_overlap"]
still_fail_cases = [c for c in cases if (not c["legacy_ok"]) and (not c["alt_ok"])]
print(f"\\nrescued_cases={len(rescued_cases)} legacy_no_x_cases={len(legacy_no_x_cases)} still_fail={len(still_fail_cases)}")
"""
        ),
        md("## Gallery A — rescued (legacy fail → xspan_pair MT OK)"),
        code(
            """show_rescued = rescued_cases
if N_SHOW_RESCUED is not None and len(rescued_cases) > N_SHOW_RESCUED:
    rng = random.Random(RANDOM_SEED)
    show_rescued = rng.sample(rescued_cases, N_SHOW_RESCUED)

saved_rescued = []
for i, case in enumerate(show_rescued, start=1):
    out = compare_panel(case, i, "rescued", FIG_RESCUED)
    saved_rescued.append(str(out))
print(f"Saved {len(saved_rescued)} rescued figures under {FIG_RESCUED}")
"""
        ),
        md("## Gallery B — still failing both pickers (sample)"),
        code(
            """rng = random.Random(RANDOM_SEED + 1)
show_still = still_fail_cases
if len(still_fail_cases) > N_SHOW_STILL_FAIL:
    show_still = rng.sample(still_fail_cases, N_SHOW_STILL_FAIL)

saved_still = []
for i, case in enumerate(show_still, start=1):
    out = compare_panel(case, i, "still_fail", FIG_STILL_FAIL)
    saved_still.append(str(out))
print(f"Saved {len(saved_still)} still-fail figures under {FIG_STILL_FAIL}")
"""
        ),
        md("## Gallery C — legacy `no_x_overlap` only (all)"),
        code(
            """saved_no_x = []
for i, case in enumerate(legacy_no_x_cases, start=1):
    out = compare_panel(case, i, "legacy_no_x", FIG_ROOT)
    saved_no_x.append(str(out))
print(f"Saved {len(saved_no_x)} legacy no_x_overlap comparison figures under {FIG_ROOT}")
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-contour-picker-ablation"
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
    (out / "apo-contour-picker-ablation-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-apo-contour-picker-ablation-phase-3",
                "title": "UMUD Apo Contour Picker Ablation Phase 3",
                "code_file": "apo-contour-picker-ablation-phase-3.ipynb",
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
