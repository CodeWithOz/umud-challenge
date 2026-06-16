"""Generate notebooks/no-contour-viz/no-contour-viz-phase-3.ipynb — apo MT fail vs OK galleries."""
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
        """# UMUD — Apo MT Visual QC: Failures vs Successes (Phase 3)

**GPU notebook** — side-by-side galleries of predicted aponeurosis masks:

1. **20 `no_contours` failures** — region path → inverted mask empty (MT NaN)
2. **20 MT-OK successes** — geometry yields finite `mt_px` (no NaN)

Each case uses the same 5-panel layout:

| # | Panel |
|---|--------|
| 1 | Raw test image |
| 2 | Predicted apo mask (native size) |
| 3 | Inverted mask |
| 4 | Orange overlay — predicted mask on image |
| 5 | Orange overlay — inverted mask on image |

Overlays: Phase 0/1 style (`MASK_OVERLAY_ALPHA=0.55`, orange `(255,140,0)`, no `cmap="gray"` on RGB)."""
    ),
    md("## Configuration"),
    code(
        """import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
N_SHOW_FAIL = 20
N_SHOW_OK = 20
RANDOM_SEED = 42
MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_ROOT = Path("/kaggle/working/figures")
FIG_FAIL = FIG_ROOT / "no_contours"
FIG_OK = FIG_ROOT / "mt_ok"
FIG_FAIL.mkdir(parents=True, exist_ok=True)
FIG_OK.mkdir(parents=True, exist_ok=True)

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
    \"\"\"Phase 0/1 colored mask overlay — masks already native-sized (no align step).\"\"\"
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    color_arr = np.zeros_like(rgb)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color_arr[m]
    return rgb.astype(np.uint8)


def apo_panel(case: dict, idx: int, group: str, fig_dir: Path):
    img = case["img"]
    pred_mask = case["pred_mask"]
    inv_mask = invert_mask(pred_mask)
    cov = float(pred_mask.mean()) * 100
    inv_cov = float(inv_mask.mean()) * 100
    n_fg = int(pred_mask.sum())
    n_inv = int(inv_mask.sum())
    mt_lbl = f"{case['mt_px']:.1f}px" if not np.isnan(case["mt_px"]) else "NaN"

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    panels = [
        ("image", lambda ax: ax.imshow(img, cmap="gray")),
        ("predicted mask", lambda ax: ax.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)),
        ("inverted mask", lambda ax: ax.imshow(inv_mask, cmap="gray", vmin=0, vmax=1)),
        ("overlay (pred)", lambda ax: ax.imshow(overlay(img, pred_mask))),
        ("overlay (inverted)", lambda ax: ax.imshow(overlay(img, inv_mask))),
    ]
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle(
        f"[{group} {idx}] {case['image_id']}  style={case['apo_style']}  "
        f"pred_cov={cov:.3f}% ({n_fg}px)  inv_cov={inv_cov:.3f}% ({n_inv}px)  "
        f"mt={mt_lbl}  fail={case['mt_fail_reason']}",
        y=1.04,
        fontsize=9,
    )
    plt.tight_layout()
    stem = Path(case["image_id"]).stem
    out = fig_dir / f"{group}_{idx:02d}_{stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return out
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
            """all_cases = []

for path in tqdm(test_paths, desc="scan test predictions"):
    img_native = load_gray(path)
    h, w = img_native.shape
    pil = open_rgb_256(img_native)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)
    apo_style = tag_apo_style(float(apo_native.mean()))
    apo_geo = apo_geometry_from_mask(apo_native, apo_style)
    all_cases.append(
        {
            "image_id": path.name,
            "img": img_native,
            "pred_mask": apo_native,
            "apo_style": apo_style,
            "apo_cov": float(apo_native.mean()),
            "apo_fg_pixels": int(apo_native.sum()),
            "n_contours": apo_geo["n_contours"],
            "mt_px": apo_geo["mt_px"],
            "mt_ok": not np.isnan(apo_geo["mt_px"]),
            "mt_fail_reason": apo_geo["mt_fail_reason"],
            "geometry_path": apo_geo.get("geometry_path"),
            "img_h": h,
            "img_w": w,
        }
    )

scan_df = pd.DataFrame(
    [
        {
            "image_id": c["image_id"],
            "apo_style": c["apo_style"],
            "apo_cov": c["apo_cov"],
            "apo_fg_pixels": c["apo_fg_pixels"],
            "mt_ok": c["mt_ok"],
            "mt_fail_reason": c["mt_fail_reason"],
            "n_contours": c["n_contours"],
            "res": f"{c['img_h']}x{c['img_w']}",
        }
        for c in all_cases
    ]
)
scan_df.to_csv("/kaggle/working/apo_scan_summary.csv", index=False)

fail_cases = [c for c in all_cases if c["mt_fail_reason"] == "no_contours"]
ok_cases = [c for c in all_cases if c["mt_ok"]]

print(f"Total: {len(all_cases)}")
print(f"no_contours failures: {len(fail_cases)}")
print(f"MT OK: {len(ok_cases)}")
print(f"MT NaN (all reasons): {(~scan_df.mt_ok).sum()}")
print()
print("Failure reasons:")
print(scan_df.loc[~scan_df.mt_ok, "mt_fail_reason"].value_counts().to_string())
print()
print("no_contours pred_cov stats:")
print(scan_df.loc[scan_df.mt_fail_reason == "no_contours", "apo_cov"].describe().round(4))
print()
print("MT OK pred_cov stats:")
print(scan_df.loc[scan_df.mt_ok, "apo_cov"].describe().round(4))
print()
print("MT OK by style:")
print(scan_df.loc[scan_df.mt_ok, "apo_style"].value_counts().to_string())

assert len(fail_cases) >= N_SHOW_FAIL, f"Need >= {N_SHOW_FAIL} failures, got {len(fail_cases)}"
assert len(ok_cases) >= N_SHOW_OK, f"Need >= {N_SHOW_OK} MT-OK cases, got {len(ok_cases)}"

rng = random.Random(RANDOM_SEED)
fail_sample = rng.sample(fail_cases, N_SHOW_FAIL)
ok_sample = rng.sample(ok_cases, N_SHOW_OK)

print(f"\\nFail sample (seed={RANDOM_SEED}):")
for row in fail_sample[:5]:
    print(f"  {row['image_id']} cov={row['apo_cov']:.4f} res={row['img_h']}x{row['img_w']}")
print(f"\\nOK sample (seed={RANDOM_SEED}):")
for row in ok_sample[:5]:
    print(f"  {row['image_id']} cov={row['apo_cov']:.4f} mt={row['mt_px']:.1f}px style={row['apo_style']}")
"""
        ),
        md("## Gallery A — `no_contours` failures (MT NaN)"),
        code(
            """saved_fail = []
for i, case in enumerate(fail_sample, start=1):
    out = apo_panel(case, i, "fail", FIG_FAIL)
    saved_fail.append(str(out))

print(f"Saved {len(saved_fail)} failure figures under {FIG_FAIL}")
"""
        ),
        md("## Gallery B — MT-OK successes (finite `mt_px`)"),
        code(
            """saved_ok = []
for i, case in enumerate(ok_sample, start=1):
    out = apo_panel(case, i, "ok", FIG_OK)
    saved_ok.append(str(out))

print(f"Saved {len(saved_ok)} success figures under {FIG_OK}")
for p in saved_ok:
    print(p)
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/no-contour-viz"
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
    (out / "no-contour-viz-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-no-contour-viz-phase-3",
                "title": "UMUD No Contour Viz Phase 3",
                "code_file": "no-contour-viz-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": [
                    "ucheozoemena/umud-train-apo-mounted-phase-3",
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
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
