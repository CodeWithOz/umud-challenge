"""Generate notebooks/no-contour-viz/no-contour-viz-phase-3.ipynb — MT no_contours gallery."""
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
        """# UMUD — MT `no_contours` Visual QC (Phase 3)

**GPU notebook** — sample predicted aponeurosis masks that fail MT geometry with `no_contours` (region style → inverted blob empty).

For each of **20** cases, panels:

| # | Panel |
|---|--------|
| 1 | Raw test image |
| 2 | Predicted apo mask (native size) |
| 3 | Inverted mask (region geometry path) |
| 4 | Orange overlay — predicted mask on image |
| 5 | Orange overlay — inverted mask on image |

Overlays use Phase 0/1 style (`MASK_OVERLAY_ALPHA=0.55`, orange `(255,140,0)`, **no** `cmap="gray"` on RGB)."""
    ),
    md("## Configuration"),
    code(
        """import random
from pathlib import Path

import matplotlib.pyplot as plt

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
N_SHOW = 20
RANDOM_SEED = 42
MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_DIR = Path("/kaggle/working/figures/no_contours")
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
    \"\"\"Phase 0/1 colored mask overlay — masks already native-sized (no align step).\"\"\"
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    color_arr = np.zeros_like(rgb)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color_arr[m]
    return rgb.astype(np.uint8)


def no_contour_panel(img: np.ndarray, pred_mask: np.ndarray, name: str, idx: int):
    inv_mask = invert_mask(pred_mask)
    cov = float(pred_mask.mean()) * 100
    inv_cov = float(inv_mask.mean()) * 100
    n_fg = int(pred_mask.sum())
    n_inv = int(inv_mask.sum())

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
        f"[{idx}] {name}  pred_cov={cov:.3f}% ({n_fg}px)  inv_cov={inv_cov:.3f}% ({n_inv}px)",
        y=1.03,
        fontsize=10,
    )
    plt.tight_layout()
    out = FIG_DIR / f"no_contour_{idx:02d}_{name.replace('.tif', '')}.png"
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
            """no_contour_cases = []

for path in tqdm(test_paths, desc="scan test for no_contours"):
    img_native = load_gray(path)
    h, w = img_native.shape
    pil = open_rgb_256(img_native)
    _, apo_t, _ = apo_learn.predict(pil)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)
    apo_style = tag_apo_style(float(apo_native.mean()))
    apo_geo = apo_geometry_from_mask(apo_native, apo_style)
    if apo_geo["mt_fail_reason"] != "no_contours":
        continue
    no_contour_cases.append(
        {
            "image_id": path.name,
            "img": img_native,
            "pred_mask": apo_native,
            "apo_cov": float(apo_native.mean()),
            "apo_fg_pixels": int(apo_native.sum()),
            "n_contours": apo_geo["n_contours"],
        }
    )

print(f"no_contours cases: {len(no_contour_cases)}")
assert len(no_contour_cases) >= N_SHOW, f"Expected >= {N_SHOW}, got {len(no_contour_cases)}"

rng = random.Random(RANDOM_SEED)
sample = rng.sample(no_contour_cases, N_SHOW)
print(f"Showing {N_SHOW} samples (seed={RANDOM_SEED})")
for row in sample[:5]:
    print(f"  {row['image_id']} cov={row['apo_cov']:.4f} fg={row['apo_fg_pixels']}")
"""
        ),
        code(
            """saved = []
for i, case in enumerate(sample, start=1):
    out = no_contour_panel(case["img"], case["pred_mask"], case["image_id"], i)
    saved.append(str(out))

print(f"Saved {len(saved)} figures under {FIG_DIR}")
for p in saved:
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
