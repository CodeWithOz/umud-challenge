"""Generate notebooks/mt-diagnosis/mt-diagnosis-phase-3.ipynb — MT NaN root-cause report on test preds."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_submission_nb as sub  # noqa: E402

from build_submission_nb import code, md


def _geometry_source() -> str:
    return sub.cells[2]["source"] if isinstance(sub.cells[2]["source"], list) else [sub.cells[2]["source"]]


cells: list[dict] = [
    md(
        """# UMUD — MT NaN Diagnosis (Phase 3)

**GPU notebook** — runs test inference with the same geometry as submission (including **region→line fallback**), then writes per-image diagnostics:

- `mt_diagnosis.csv` — full per-test breakdown
- `mt_diagnosis_summary.json` — aggregate failure counts

Use this to verify A1 before/after submission fixes."""
    ),
    md("## Configuration"),
    code(
        """from pathlib import Path

IMG_SIZE = 256
APO_REGION_THRESHOLD = 0.50

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
"""
    ),
]

# geometry + helpers cell from submission builder
geom_src = "".join(_geometry_source())
cells.append(code(geom_src))

cells.extend(
    [
        code(
            """import json

assert FASC_MODEL_PATH.exists(), f"Missing fasc model: {FASC_MODEL_PATH}"
assert APO_MODEL_PATH.exists(), f"Missing apo model: {APO_MODEL_PATH}"
assert TEST_DIR.exists(), f"Missing test dir: {TEST_DIR}"

fasc_learn = load_learner(FASC_MODEL_PATH)
apo_learn = load_learner(APO_MODEL_PATH)
test_paths = list_test_images(TEST_DIR)
print(f"Test images: {len(test_paths)}")
"""
        ),
        code(
            """rows = []
for path in tqdm(test_paths, desc="diagnose test"):
    img_native = load_gray(path)
    h, w = img_native.shape
    pil = open_rgb_256(img_native)
    _, fasc_t, _ = fasc_learn.predict(pil)
    _, apo_t, _ = apo_learn.predict(pil)
    fasc_native = resize_mask_to(tensor_to_mask(fasc_t), h, w)
    apo_native = resize_mask_to(tensor_to_mask(apo_t), h, w)
    apo_style = tag_apo_style(float(apo_native.mean()))
    geo = derive_geometry(fasc_native, apo_native, apo_style)
    rows.append({
        "image_id": path.name,
        "img_h": h,
        "img_w": w,
        "res": f"{h}x{w}",
        "apo_style": apo_style,
        "apo_cov": geo["apo_cov"],
        "apo_fg_pixels": geo["apo_fg_pixels"],
        "fasc_cov": geo["fasc_cov"],
        "fasc_fg_pixels": geo["fasc_fg_pixels"],
        "n_contours": geo["n_contours"],
        "geometry_path": geo["geometry_path"],
        "mt_fail_reason": geo["mt_fail_reason"],
        "mt_fail_reason_primary": geo.get("mt_fail_reason_primary"),
        "mt_px": geo["mt_px"],
        "mt_ok": not np.isnan(geo["mt_px"]),
        "pa_deg": geo["pa_deg"],
        "fl_px": geo["fl_px"],
    })

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/mt_diagnosis.csv", index=False)

summary = {
    "n_test": len(df),
    "mt_nan_rate": float((~df.mt_ok).mean()),
    "by_apo_style": df.groupby("apo_style").mt_ok.mean().to_dict(),
    "by_geometry_path": df.groupby("geometry_path").mt_ok.mean().to_dict(),
    "fail_reason_counts": df.loc[~df.mt_ok, "mt_fail_reason"].value_counts().to_dict(),
    "fail_reason_x_style": pd.crosstab(df.apo_style, df.mt_fail_reason).to_dict(),
    "fallback_line_rescues": int((df.geometry_path == "fallback_line").sum()),
}
Path("/kaggle/working/mt_diagnosis_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
display(df.loc[~df.mt_ok].head(20))
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/mt-diagnosis"
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
    (out / "mt-diagnosis-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-mt-diagnosis-phase-3",
                "title": "UMUD MT Diagnosis Phase 3",
                "code_file": "mt-diagnosis-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": [
                    "ucheozoemena/umud-train-mounted-phase-3",
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
