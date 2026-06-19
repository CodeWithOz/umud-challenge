"""Generate notebooks/block5-geom-ablation — full-309 picker ablation on 200-tier apo (Block 5)."""
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
        """# UMUD — Block 5 Geometry Ablation (200-tier apo, full 309)

**GPU notebook** — compare contour-pair pickers on **all 309 test images** using production **`apo_gray55_line_200.pkl`**:

| Picker | Rule |
|--------|------|
| **top_bottom** | Sort contours top→bottom; sup = topmost; deep = first ≥15px below (original DLTrack rule) |
| **xspan_pair** | Top-K by x-span; max x-overlap pair |
| **horiz_parallel** | Production — horizontality × parallelism score (submission default) |

Outputs:
- `/kaggle/working/block5_geom_ablation.csv`
- `/kaggle/working/block5_geom_ablation_summary.json`"""
    ),
    md("## Configuration"),
    code(
        """import json
from pathlib import Path

import pandas as pd

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50
GRAY_FILL_VALUE = 55
ROI_THRESH = 5
ROI_PAD_PX = 10
TOP_K_CANDIDATES = 8
MIN_SEP_PX = 15

PICKERS = ("top_bottom", "xspan_pair", "horiz_parallel")


def resolve_pkl(filename: str, preferred: list[Path] | None = None) -> Path:
    for p in preferred or []:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(filename))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find {filename} under /kaggle/input")


MODEL_200 = resolve_pkl(
    "apo_gray55_line_200.pkl",
    preferred=[
        Path("/kaggle/input/datasets/ucheozoemena/umud-apo-line-model-200/apo_gray55_line_200.pkl"),
    ],
)
print("200-tier apo:", MODEL_200)

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
            """from fastai.vision.all import load_learner
from tqdm.auto import tqdm


def pick_top_bottom(contours: list[np.ndarray], min_sep_px: int = MIN_SEP_PX):
    if len(contours) < 2:
        return None, None
    ranked = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    sup = ranked[0]
    y0 = cv2.boundingRect(sup)[1]
    deep = None
    for c in ranked[1:]:
        if cv2.boundingRect(c)[1] >= y0 + min_sep_px:
            deep = c
            break
    if deep is None:
        deep = ranked[min(1, len(ranked) - 1)]
    return sup, deep


def apo_geometry_with_picker(apo_mask: np.ndarray, style: str, picker: str) -> dict:
    eff, method = effective_apo_mask(apo_mask, style)
    contours = find_apo_contours(eff)
    n_contours = len(contours)

    if picker == "top_bottom":
        sup_c, deep_c = pick_top_bottom(contours)
    elif picker == "xspan_pair":
        sup_c, deep_c = pick_best_pair_xspan(contours)
    elif picker == "horiz_parallel":
        sup_c, deep_c = pick_horiz_parallel(contours)
    else:
        raise ValueError(picker)

    out = {
        "apo_method": method,
        "picker": picker,
        "n_contours": n_contours,
        "mt_px": np.nan,
        "deep_angle_deg": np.nan,
        "mt_fail_reason": "ok",
    }
    if n_contours == 0:
        out["mt_fail_reason"] = "no_contours"
        return out
    if n_contours < 2 or sup_c is None or deep_c is None:
        out["mt_fail_reason"] = "single_contour"
        return out

    sup_x, sup_y = edge_polyline(sup_c, which="bottom")
    deep_x, deep_y = edge_polyline(deep_c, which="top")
    sup_line = fit_line(sup_x, sup_y)
    deep_line = fit_line(deep_x, deep_y)
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
"""
        ),
        code(
            """assert MODEL_200.exists(), MODEL_200
assert TEST_DIR.exists(), TEST_DIR

learn = load_learner(MODEL_200)
test_paths = list_test_images(TEST_DIR)
print(f"Test images: {len(test_paths)}")
"""
        ),
        code(
            """rows = []

for path in tqdm(test_paths, desc="block5 picker scan"):
    img_native = load_gray(path)
    h, w = img_native.shape
    img_g, bbox = preprocess_gray55(img_native)
    _, apo_t, _ = learn.predict(open_rgb_256(img_g))
    pred_mask = clip_mask_to_bbox(resize_mask_to(tensor_to_mask(apo_t), h, w), bbox)
    style = tag_apo_style(float(pred_mask.mean()))

    row = {"image_id": path.name, "res": f"{h}x{w}", "apo_style": style, "apo_cov": float(pred_mask.mean())}
    for picker in PICKERS:
        geo = apo_geometry_with_picker(pred_mask, style, picker)
        ok = bool(np.isfinite(geo["mt_px"]))
        row[f"{picker}_mt_ok"] = ok
        row[f"{picker}_fail"] = geo["mt_fail_reason"]
        row[f"{picker}_mt_px"] = geo["mt_px"]
        row[f"{picker}_n_ct"] = geo["n_contours"]
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/block5_geom_ablation.csv", index=False)

n = len(df)
summary = {
    "n_test": n,
    "apo_model": "apo_gray55_line_200.pkl",
    "pickers": list(PICKERS),
}
for picker in PICKERS:
    ok_col = f"{picker}_mt_ok"
    summary[f"{picker}_mt_ok"] = int(df[ok_col].sum())
    summary[f"{picker}_mt_ok_pct"] = round(100.0 * df[ok_col].mean(), 2)
    fails = df.loc[~df[ok_col], f"{picker}_fail"].value_counts().to_dict()
    summary[f"{picker}_fail_counts"] = fails

# Pairwise deltas vs production (horiz_parallel)
prod = "horiz_parallel"
for alt in ("top_bottom", "xspan_pair"):
    rescued = int(((~df[f"{alt}_mt_ok"]) & df[f"{prod}_mt_ok"]).sum())
    broken = int(((df[f"{alt}_mt_ok"]) & (~df[f"{prod}_mt_ok"])).sum())
    both_ok = df[f"{alt}_mt_ok"] & df[f"{prod}_mt_ok"]
    mt_diff = (df.loc[both_ok, f"{alt}_mt_px"] - df.loc[both_ok, f"{prod}_mt_px"]).abs()
    summary[f"{alt}_vs_{prod}"] = {
        "rescued_by_prod": rescued,
        "broken_by_prod": broken,
        "both_ok_n": int(both_ok.sum()),
        "mt_px_median_abs_diff": float(mt_diff.median()) if len(mt_diff) else None,
        "mt_px_max_abs_diff": float(mt_diff.max()) if len(mt_diff) else None,
    }

with open("/kaggle/working/block5_geom_ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/block5-geom-ablation"
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
    (out / "block5-geom-ablation-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-block5-geom-ablation-phase-3",
                "title": "UMUD Block5 Geom Ablation Phase 3",
                "code_file": "block5-geom-ablation-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": ["ucheozoemena/umud-apo-line-model-200"],
                "kernel_sources": [],
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
