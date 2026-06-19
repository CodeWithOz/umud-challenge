"""Generate notebooks/block4-mt-fail-viz — dual-model MT-fail visual QC (200 vs 524 tier)."""
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
        """# UMUD — Block 4 MT-Fail Visual QC (524 vs 200 tier)

**GPU notebook** — side-by-side overlays for test images where the **524-tier** apo model fails MT geometry (`mt_fail_reason != ok`), compared against the **200-tier** model on the same image.

Block 4 submission (524-tier + `MM_PER_PIXEL=0.075`) had **62 MT NaN** (~20%): `single_contour` 29, `no_contours` 22, `no_x_overlap` 11.

**Inputs:**
- `umud-train-apo-gray55-phase-3` kernel output → `apo_gray55_line_524.pkl` (latest train run)
- `umud-apo-line-model-200` dataset → `apo_gray55_line_200.pkl` (train v9 / TRAIN_RUN=7; not kept on latest train output)

Each case uses an **8-panel** layout (2×4):

| Row | Panels |
|-----|--------|
| 1 | Raw + ROI bbox · 200-tier apo overlay · 524-tier apo overlay · mask XOR diff |
| 2 | 200 mask · 524 mask · 524 geometry debug · summary (both tiers) |

Outputs:
- `/kaggle/working/block4_mt_fail_scan.csv` — all test images, both tiers
- `/kaggle/working/block4_mt_fail_regressions.csv` — 200 OK but 524 fail
- Figures under `/kaggle/working/figures/block4_mt_fail/{reason}/`"""
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
TOP_K_CANDIDATES = 8
MIN_SEP_PX = 15

N_SHOW = None  # None = all 524 MT-fail cases; set e.g. 15 for random sample per reason
RANDOM_SEED = 42

MASK_OVERLAY_ALPHA = 0.55
COLOR_200 = (0, 200, 255)    # cyan — production tier
COLOR_524 = (255, 140, 0)    # orange — block 4 tier
COLOR_DIFF = (255, 0, 255)   # magenta — mask disagreement

FIG_ROOT = Path("/kaggle/working/figures/block4_mt_fail")
FIG_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_pkl(filename: str, preferred: list[Path] | None = None) -> Path:
    for p in preferred or []:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        print(f"Found {len(hits)} mount(s) for {filename}; using {hits[0]}")
        return hits[0]
    raise FileNotFoundError(
        f"Could not find {filename} under /kaggle/input. "
        "Mount umud-apo-line-model-200 (200-tier) and umud-train-apo-gray55-phase-3 (524-tier)."
    )


MODEL_200 = resolve_pkl(
    "apo_gray55_line_200.pkl",
    preferred=[
        Path("/kaggle/input/datasets/ucheozoemena/umud-apo-line-model-200/apo_gray55_line_200.pkl"),
    ],
)
MODEL_524 = resolve_pkl(
    "apo_gray55_line_524.pkl",
    preferred=[
        Path("/kaggle/input/notebooks/ucheozoemena/umud-train-apo-gray55-phase-3/apo_gray55_line_524.pkl"),
    ],
)
print("200-tier apo:", MODEL_200)
print("524-tier apo:", MODEL_524)

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
from tqdm.auto import tqdm


def overlay(img_gray: np.ndarray, mask: np.ndarray, color, alpha=MASK_OVERLAY_ALPHA):
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    sel = mask.astype(bool)
    rgb[sel] = (1 - alpha) * rgb[sel] + alpha * tint[sel]
    return rgb.astype(np.uint8)


def mask_diff_panel(img_gray: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray):
    xor = (mask_a.astype(bool) ^ mask_b.astype(bool)).astype(np.uint8)
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = COLOR_DIFF
    sel = xor.astype(bool)
    rgb[sel] = (1 - MASK_OVERLAY_ALPHA) * rgb[sel] + MASK_OVERLAY_ALPHA * tint[sel]
    return rgb.astype(np.uint8), float(xor.mean()) * 100


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


def draw_overlap_band(ax, geo: dict):
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
    return stats


def geometry_debug_panel(ax, img_g: np.ndarray, pred_mask: np.ndarray, geo: dict):
    reason = geo["mt_fail_reason"]
    ax.imshow(img_g, cmap="gray")
    if reason == "no_contours":
        inv = invert_mask(pred_mask)
        ax.imshow(inv, cmap="Reds", alpha=0.35)
        ax.set_title(f"524: invert overlay\\nn_ct={geo['n_contours']}", fontsize=9)
    elif reason == "single_contour":
        ax.imshow(pred_mask, cmap="Greens", alpha=0.35)
        ax.set_title(f"524: single contour\\nn_ct={geo['n_contours']}", fontsize=9)
    elif reason == "no_x_overlap":
        draw_apo_edges(ax, geo)
        draw_overlap_band(ax, geo)
        ax.set_title("524: edges + overlap", fontsize=9)
    else:
        ax.set_title(f"524: fail={reason}", fontsize=9)
    ax.axis("off")


def summary_panel(ax, case: dict):
    ax.axis("off")
    g200, g524 = case["geo_200"], case["geo_524"]
    lines = [
        f"524 fail: {g524['mt_fail_reason']}",
        f"200 mt_ok: {case['mt_ok_200']}",
        f"524 mt_ok: {case['mt_ok_524']}",
        f"regression: {case['is_regression']}",
        "",
        f"200 cov: {case['cov_200']*100:.3f}%  n_ct={g200['n_contours']}",
        f"524 cov: {case['cov_524']*100:.3f}%  n_ct={g524['n_contours']}",
        f"mask xor: {case['xor_pct']:.3f}%",
        "",
        f"res: {case['img_h']}x{case['img_w']}",
        f"200 style: {case['style_200']}",
        f"524 style: {case['style_524']}",
    ]
    if g524["mt_fail_reason"] == "no_x_overlap":
        xs = case["x_stats_524"]
        lines.append(f"overlap_px: {xs['overlap_px']:.0f}  gap: {xs.get('gap_px', np.nan):.0f}")
    ax.text(0.05, 0.95, "\\n".join(lines), va="top", fontsize=9, family="monospace")
    title = "REGRESSION" if case["is_regression"] else "summary"
    color = "red" if case["is_regression"] else "black"
    ax.set_title(title, fontsize=10, color=color)


def infer_apo_mask(learn, img_native: np.ndarray):
    h, w = img_native.shape
    img_g, bbox = preprocess_gray55(img_native)
    pil = open_rgb_256(img_g)
    _, apo_t, _ = learn.predict(pil)
    pred = clip_mask_to_bbox(resize_mask_to(tensor_to_mask(apo_t), h, w), bbox)
    style = tag_apo_style(float(pred.mean()))
    geo = apo_geometry_from_mask(pred, style)
    return {
        "img_gray55": img_g,
        "bbox": bbox,
        "pred_mask": pred,
        "apo_style": style,
        "apo_cov": float(pred.mean()),
        "geo": geo,
        "mt_ok": not np.isnan(geo["mt_px"]),
        "mt_fail_reason": geo["mt_fail_reason"],
        "x_stats": x_overlap_stats(geo),
    }


def compare_fail_panel(case: dict, idx: int, fig_dir: Path):
    img = case["img_raw"]
    img_g = case["img_gray55"]
    m200 = case["mask_200"]
    m524 = case["mask_524"]
    bbox = case["bbox"]
    y0, y1, x0, x1 = bbox
    reason = case["geo_524"]["mt_fail_reason"]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    ax = axes.ravel()

    ax[0].imshow(img, cmap="gray")
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
    ax[0].set_title("raw + bbox", fontsize=9)
    ax[0].axis("off")

    ax[1].imshow(overlay(img_g, m200, COLOR_200))
    ax[1].set_title(f"200-tier overlay\\ncov={case['cov_200']*100:.2f}%", fontsize=9)
    ax[1].axis("off")

    ax[2].imshow(overlay(img_g, m524, COLOR_524))
    ax[2].set_title(f"524-tier overlay\\ncov={case['cov_524']*100:.2f}%", fontsize=9)
    ax[2].axis("off")

    diff_rgb, xor_pct = mask_diff_panel(img_g, m200, m524)
    ax[3].imshow(diff_rgb)
    ax[3].set_title(f"mask XOR diff\\n{xor_pct:.2f}% pixels", fontsize=9)
    ax[3].axis("off")

    ax[4].imshow(m200, cmap="gray", vmin=0, vmax=1)
    ax[4].set_title(f"200 mask  n_ct={case['geo_200']['n_contours']}", fontsize=9)
    ax[4].axis("off")

    ax[5].imshow(m524, cmap="gray", vmin=0, vmax=1)
    ax[5].set_title(f"524 mask  n_ct={case['geo_524']['n_contours']}", fontsize=9)
    ax[5].axis("off")

    geometry_debug_panel(ax[6], img_g, m524, case["geo_524"])
    summary_panel(ax[7], case)

    reg_tag = " [REGRESSION]" if case["is_regression"] else ""
    plt.suptitle(
        f"[{idx}] {case['image_id']}  res={case['img_h']}x{case['img_w']}  "
        f"524_fail={reason}{reg_tag}",
        y=1.02,
        fontsize=10,
    )
    plt.tight_layout()
    stem = Path(case["image_id"]).stem
    out = fig_dir / f"block4_fail_{idx:03d}_{reason}_{stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out
"""
        ),
        code(
            """assert MODEL_200.exists(), MODEL_200
assert MODEL_524.exists(), MODEL_524
assert TEST_DIR.exists(), TEST_DIR

learn_200 = load_learner(MODEL_200)
learn_524 = load_learner(MODEL_524)
test_paths = list_test_images(TEST_DIR)
print(f"Test images: {len(test_paths)}")
"""
        ),
        code(
            """all_rows = []
fail_cases = []

for path in tqdm(test_paths, desc="dual-model scan"):
    img_native = load_gray(path)
    h, w = img_native.shape

    r200 = infer_apo_mask(learn_200, img_native)
    r524 = infer_apo_mask(learn_524, img_native)

    is_regression = r200["mt_ok"] and not r524["mt_ok"]
    xor = (r200["pred_mask"].astype(bool) ^ r524["pred_mask"].astype(bool)).astype(np.uint8)
    xor_pct = float(xor.mean()) * 100

    row = {
        "image_id": path.name,
        "res": f"{h}x{w}",
        "cov_200": r200["apo_cov"],
        "cov_524": r524["apo_cov"],
        "style_200": r200["apo_style"],
        "style_524": r524["apo_style"],
        "n_ct_200": r200["geo"]["n_contours"],
        "n_ct_524": r524["geo"]["n_contours"],
        "mt_ok_200": r200["mt_ok"],
        "mt_ok_524": r524["mt_ok"],
        "mt_fail_200": r200["mt_fail_reason"],
        "mt_fail_524": r524["mt_fail_reason"],
        "xor_pct": xor_pct,
        "is_regression": is_regression,
    }
    all_rows.append(row)

    if not r524["mt_ok"]:
        fail_cases.append(
            {
                "image_id": path.name,
                "img_raw": img_native,
                "img_gray55": r200["img_gray55"],
                "bbox": r200["bbox"],
                "mask_200": r200["pred_mask"],
                "mask_524": r524["pred_mask"],
                "cov_200": r200["apo_cov"],
                "cov_524": r524["apo_cov"],
                "style_200": r200["apo_style"],
                "style_524": r524["apo_style"],
                "geo_200": r200["geo"],
                "geo_524": r524["geo"],
                "mt_ok_200": r200["mt_ok"],
                "mt_ok_524": r524["mt_ok"],
                "x_stats_524": r524["x_stats"],
                "xor_pct": xor_pct,
                "is_regression": is_regression,
                "img_h": h,
                "img_w": w,
            }
        )

scan_df = pd.DataFrame(all_rows)
scan_df.to_csv("/kaggle/working/block4_mt_fail_scan.csv", index=False)

reg_df = scan_df.loc[scan_df["is_regression"]].copy()
reg_df.to_csv("/kaggle/working/block4_mt_fail_regressions.csv", index=False)

print(f"Total test images: {len(scan_df)}")
print(f"524 MT failures: {len(fail_cases)}")
print(f"200 MT OK: {scan_df.mt_ok_200.sum()}")
print(f"524 MT OK: {scan_df.mt_ok_524.sum()}")
print(f"Regressions (200 OK, 524 fail): {len(reg_df)}")
print()
print("524 failure reasons:")
print(scan_df.loc[~scan_df.mt_ok_524, "mt_fail_524"].value_counts().to_string())
print()
print("Resolution cohorts among 524 failures:")
fail_res = scan_df.loc[~scan_df.mt_ok_524, "res"].value_counts()
print(fail_res.to_string())
"""
        ),
        md("## Gallery — 524-tier MT failures (vs 200-tier)"),
        code(
            """if not fail_cases:
    print("No 524-tier MT failures found.")
else:
    by_reason: dict[str, list] = {}
    for c in fail_cases:
        by_reason.setdefault(c["geo_524"]["mt_fail_reason"], []).append(c)

    saved = []
    idx = 0
    for reason in sorted(by_reason.keys()):
        cases = by_reason[reason]
        show = cases
        if N_SHOW is not None and len(cases) > N_SHOW:
            rng = random.Random(RANDOM_SEED)
            show = rng.sample(cases, N_SHOW)
            print(f"{reason}: sample {N_SHOW}/{len(cases)}")
        else:
            print(f"{reason}: all {len(cases)} cases")
        fig_dir = FIG_ROOT / reason
        fig_dir.mkdir(parents=True, exist_ok=True)
        for case in show:
            idx += 1
            out = compare_fail_panel(case, idx, fig_dir)
            saved.append(str(out))

    print(f"\\nSaved {len(saved)} figures under {FIG_ROOT}")
"""
        ),
        md("## Regression subset — 200 OK but 524 fail"),
        code(
            """reg_cases = [c for c in fail_cases if c["is_regression"]]
print(f"Regression cases: {len(reg_cases)}")
if reg_cases:
    reg_dir = FIG_ROOT / "_regressions_only"
    reg_dir.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(reg_cases, 1):
        out = compare_fail_panel(case, i, reg_dir)
        print(out)
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/block4-mt-fail-viz"
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
    (out / "block4-mt-fail-viz-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-block4-mt-fail-viz-phase-3",
                "title": "UMUD Block4 MT Fail Viz Phase 3",
                "code_file": "block4-mt-fail-viz-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [
                    "ucheozoemena/umud-apo-line-model-200",
                ],
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
