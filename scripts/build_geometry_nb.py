"""Generate geometry notebooks: Kaggle (geometry-phase-2.ipynb) and local (geometry-phase-2-local.ipynb)."""
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


def intro_kaggle() -> dict:
    return md(
        """# UMUD Challenge — Geometry & Calibration (Phase 2)

**Kaggle notebook** — runs on Kaggle CPU with competition data via `kagglehub`.

This notebook turns aligned masks into muscle-architecture measurements and exports clean training manifests.

### How to read this notebook

Each section has a markdown intro explaining *why* the next code cell exists. Functions include docstrings; tricky lines have inline comments.

### What Phase 2 produces (written to `/kaggle/working/`)

| Output | Purpose |
|--------|---------|
| `train_fasc_clean.csv` | Fascicle pairs safe for training |
| `train_apo_all.csv` | Aponeurosis pairs + region/line style tag |
| `geometry_sample.csv` | Prototype PA / FL / MT in **pixels** |
| `figures/*.png` | QC panels (download from Output tab) |

> For local runs with `data/umud-challenge/`, use **`geometry-phase-2-local.ipynb`** instead — same analysis, local paths.

> Edit *Configuration*, then re-run from there downward."""
    )


def intro_local() -> dict:
    return md(
        """# UMUD Challenge — Geometry & Calibration (Phase 2, Local)

**Local notebook** — runs in this repo with extracted competition data under `data/umud-challenge/`.

Same analysis as the Kaggle notebook (`geometry-phase-2.ipynb`) but with local paths and outputs under `tmp/geometry-local-output/`. **Not pushed to Kaggle.**

### Prerequisites

```bash
# From repo root — competition zip already extracted to data/umud-challenge/
uv sync
uv run python scripts/build_geometry_nb.py   # regenerate both notebooks
uv run jupyter nbconvert --execute notebooks/geometry/geometry-phase-2-local.ipynb
```

QC PNGs: `tmp/geometry-local-output/figures/`

> Edit *Configuration*, then re-run from there downward."""
    )


def paths_kaggle() -> dict:
    return md(
        """## Paths and file discovery (Kaggle)

Competition data loads via **`kagglehub.competition_download`** (same as Phase 0/1 audit).

Outputs go to **`/kaggle/working/`** (CSVs + `figures/`). Download from the kernel Output tab after the run completes."""
    )


def paths_local() -> dict:
    return md(
        """## Paths and file discovery (Local)

Data is read from **`data/umud-challenge/`** at the repo root (gitignored). No Kaggle download or credentials needed.

Outputs go to **`tmp/geometry-local-output/`** (also gitignored)."""
    )


def code_paths_kaggle() -> dict:
    return code(
        """from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import kagglehub

COMPETITION_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
DATA_ROOT = Path(kagglehub.competition_download(COMPETITION_SLUG))
print("Competition dir:", DATA_ROOT)

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
print("Output dir:", OUT)


def finish_fig(fig, path: Path | None = None):
    \"\"\"Save figure to Kaggle working dir and display inline in the notebook.\"\"\"
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


DIRS = {
    "apo_img": DATA_ROOT / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": DATA_ROOT / "apo_masks_v1/apo_masks_new_model_v1",
    "fasc_img": DATA_ROOT / "fasc_imgs_v1/fasc_images_new_model_v1",
    "fasc_mask": DATA_ROOT / "fasc_masks_v1/fasc_masks_new_model_v1",
    "test": DATA_ROOT / "test_images_v2/test_set_v2",
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


lookups = {k: build_lookup(v) for k, v in DIRS.items()}
display(pd.DataFrame([{"key": k, "n_files": len(v)} for k, v in lookups.items()]))
"""
    )


def code_paths_local() -> dict:
    return code(
        """from pathlib import Path
import random

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-friendly for nbconvert / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def find_repo_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "pyproject.toml").exists():
            return base.resolve()
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
DATA_ROOT = REPO_ROOT / "data/umud-challenge"
if not DATA_ROOT.joinpath("apo_imgs_v1").exists():
    raise FileNotFoundError(
        f"Local data not found at {DATA_ROOT}. "
        "Download and extract the competition zip — see README.md."
    )
print("Competition dir:", DATA_ROOT)

OUT = REPO_ROOT / "tmp/geometry-local-output"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
print("Output dir:", OUT)


def finish_fig(fig, path: Path | None = None):
    \"\"\"Save figure to tmp/ and close (no inline display in headless runs).\"\"\"
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


DIRS = {
    "apo_img": DATA_ROOT / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": DATA_ROOT / "apo_masks_v1/apo_masks_new_model_v1",
    "fasc_img": DATA_ROOT / "fasc_imgs_v1/fasc_images_new_model_v1",
    "fasc_mask": DATA_ROOT / "fasc_masks_v1/fasc_masks_new_model_v1",
    "test": DATA_ROOT / "test_images_v2/test_set_v2",
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


lookups = {k: build_lookup(v) for k, v in DIRS.items()}
display(pd.DataFrame([{"key": k, "n_files": len(v)} for k, v in lookups.items()]))
"""
    )


def export_kaggle() -> dict:
    return md(
        """## Export artifacts

CSV columns exclude internal visualization caches. PNGs are under `/kaggle/working/figures/` — download from the kernel **Output** tab."""
    )


def export_local() -> dict:
    return md(
        """## Export artifacts

CSV columns exclude internal visualization caches. PNGs are under `tmp/geometry-local-output/figures/`."""
    )


def shared_cells() -> list[dict]:
    """Cells identical on Kaggle and local (after environment-specific paths cell)."""
    cells: list[dict] = []

    cells.append(
        md(
            """## Competition measurement protocol (from Data tab + DLTrack)

| Target | Definition (competition) | Notes for this notebook |
|--------|--------------------------|-------------------------|
| **PA** (`pa_deg`) | Angle between fascicle and **deep aponeurosis** | Degrees; no mm calibration needed |
| **FL** (`fl_px` here) | Length along fascicle between aponeuroses | Fascicle PCA span in px for now |
| **MT** (`mt_px` here) | Perpendicular distance superficial↔deep at **three** x positions (manual protocol) | Mean of three samples |

**Aponeurosis geometry:** DLTrack fits **local linear edges** from mask contours — not fixed horizontal lines.

**3+ aponeuroses in mask:** sort contours top→bottom; superficial = top; deep = next separated (DLTrack rule).

**Region vs line masks:** Line → raw mask. Region → **invert** then same contour pipeline."""
        )
    )

    cells.append(md("""## Configuration"""))

    cells.append(
        code(
            """# --- Parameters you can change ---
RANDOM_SEED = 42
N_GEOMETRY_SAMPLE = 200
N_APO_GALLERY_PER_STYLE = 4
N_FASC_STRETCH_CHECK = 6
APO_REGION_THRESHOLD = 0.50
FASC_NEAR_EMPTY_THRESHOLD = 0.0005
DEFAULT_ALIGN_MODE = "stretch"
MASK_OVERLAY_ALPHA = 0.55
REF_PA_DEG = (5, 45)
REF_FL_MM = (30, 200)
REF_MT_MM = (10, 50)
FL_BIN_LOW_MAX = 900
FL_BIN_HIGH_MIN = 900
"""
        )
    )

    cells.append(
        md(
            """## Alignment utilities (ported from Phase 0/1)

When image and mask shapes differ (~60–70% of pairs), we **stretch** the mask to image size before geometry."""
        )
    )

    cells.append(
        code(
            """def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def mask_coverage(mask: np.ndarray) -> float:
    return float(mask.mean())


def place_mask_center(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    mh, mw = mask.shape
    y0 = max(0, (target_h - mh) // 2)
    x0 = max(0, (target_w - mw) // 2)
    y1 = min(target_h, y0 + mh)
    x1 = min(target_w, x0 + mw)
    mask_row_start = max(0, (mh - target_h) // 2)
    mask_col_start = max(0, (mw - target_w) // 2)
    canvas[y0:y1, x0:x1] = mask[
        mask_row_start : mask_row_start + (y1 - y0),
        mask_col_start : mask_col_start + (x1 - x0),
    ]
    return canvas


def scale_mask_uniform(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    mh, mw = mask.shape
    scale = min(target_h / mh, target_w / mw)
    new_h = max(1, int(round(mh * scale)))
    new_w = max(1, int(round(mw * scale)))
    resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    ) > 0
    return place_mask_center(resized.astype(np.uint8), target_h, target_w)


def align_mask(mask: np.ndarray, target_h: int, target_w: int, mode: str = DEFAULT_ALIGN_MODE) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask
    if mode == "stretch":
        return (
            np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (target_w, target_h), Image.NEAREST
                )
            )
            > 0
        ).astype(np.uint8)
    if mode == "center":
        return place_mask_center(mask, target_h, target_w)
    if mode == "scale":
        return scale_mask_uniform(mask, target_h, target_w)
    raise ValueError(f"Unknown mode: {mode}")


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return (1 - mask).astype(np.uint8)


def tag_apo_style(coverage: float) -> str:
    return "region" if coverage >= APO_REGION_THRESHOLD else "line"
"""
        )
    )

    cells.append(
        md(
            """## Clean training manifests

Exclude empty / near-empty fascicle masks (Phase 0/1 thresholds).

### Why dual-track (1040) < apo pairs (1048)?

Dual-track = filenames in **both** apo set **and** clean fasc set. **Eight** apo images have fasc masks on the exclude list — listed in the next cell. Not a data bug."""
        )
    )

    cells.append(
        code(
            """def mask_coverage_from_path(path: Path) -> float:
    return mask_coverage(load_mask(path))


print("Scanning all fascicle pairs for empty / near-empty masks (may take ~1–2 min)...")
empty_rows, near_empty_rows = [], []
fasc_common = sorted(set(lookups["fasc_img"]) & set(lookups["fasc_mask"]))
for name in fasc_common:
    cov = mask_coverage_from_path(lookups["fasc_mask"][name])
    if cov <= 0.0:
        empty_rows.append({"filename": name, "mask_coverage": cov, "reason": "empty"})
    elif cov < FASC_NEAR_EMPTY_THRESHOLD:
        near_empty_rows.append({"filename": name, "mask_coverage": cov, "reason": "near_empty"})

exclude_fasc = pd.DataFrame(empty_rows + near_empty_rows)
exclude_names = set(exclude_fasc["filename"]) if len(exclude_fasc) else set()
clean_fasc_names = [n for n in fasc_common if n not in exclude_names]

train_fasc_clean = pd.DataFrame({"filename": clean_fasc_names})
train_apo_all = pd.DataFrame({"filename": sorted(set(lookups["apo_img"]) & set(lookups["apo_mask"]))})

print(f"Fasc pairs total: {len(fasc_common)}")
print(f"Exclude: {len(exclude_names)} | Clean fasc: {len(clean_fasc_names)}")
print(f"Apo pairs: {len(train_apo_all)}")

dual_track = sorted(set(train_apo_all["filename"]) & set(train_fasc_clean["filename"]))
apo_missing_clean_fasc = sorted(set(train_apo_all["filename"]) - set(train_fasc_clean["filename"]))

print(f"Dual-track (apo + clean fasc): {len(dual_track)}")
print(f"Apo-only (fasc excluded): {len(apo_missing_clean_fasc)}")
if apo_missing_clean_fasc:
    display(exclude_fasc[exclude_fasc.filename.isin(apo_missing_clean_fasc)])
"""
        )
    )

    cells.append(md("""## Apo mask style census"""))

    cells.append(
        code(
            """apo_style_rows = []
for name in train_apo_all["filename"]:
    cov = mask_coverage_from_path(lookups["apo_mask"][name])
    apo_style_rows.append(
        {"filename": name, "mask_coverage": cov, "mask_style": tag_apo_style(cov)}
    )
apo_styles = pd.DataFrame(apo_style_rows)
train_apo_all = train_apo_all.merge(apo_styles, on="filename", how="left")
display(apo_styles["mask_style"].value_counts().to_frame("count"))
"""
        )
    )

    cells.append(
        md(
            """## Calibration hunt (TIFF metadata)

Option C: pixels now, mm before submit. Scans TIFF tags for spacing hints."""
        )
    )

    cells.append(
        code(
            """CALIBRATION_KEYS = (
    "XResolution", "YResolution", "ResolutionUnit",
    "PixelSpacing", "Spacing", "PhysicalDeltaX", "PhysicalDeltaY",
)


def tiff_tag_summary(path: Path) -> dict:
    out = {"filename": path.name}
    with Image.open(path) as im:
        tags = {k: v for k, v in im.tag_v2.items()} if hasattr(im, "tag_v2") else {}
        for key in CALIBRATION_KEYS:
            if key in tags:
                out[key] = tags[key]
        out["size"] = im.size
    return out


rng = random.Random(RANDOM_SEED)
calib_sample = rng.sample(sorted(lookups["fasc_img"].values()), min(40, len(lookups["fasc_img"])))
calib_df = pd.DataFrame([tiff_tag_summary(p) for p in calib_sample])
interesting = calib_df[[c for c in CALIBRATION_KEYS if c in calib_df.columns]].notna().any(axis=1)
print("Calibration-related tags found:", int(interesting.sum()), "of", len(calib_df))
display(calib_df.head(5))
"""
        )
    )

    cells.append(
        md(
            """## Geometry core (contour-based, DLTrack-inspired)

1. Effective apo mask (raw line or inverted region)
2. OpenCV external contours, sorted top→bottom
3. Superficial = top contour bottom edge; deep = next contour top edge
4. Linear fit per edge; MT = mean perpendicular distance at 3 x positions
5. Fascicle PCA → `fl_px`, PA vs deep apo slope"""
        )
    )

    cells.append(
        code(
            """def effective_apo_mask(mask: np.ndarray, style: str) -> tuple[np.ndarray, str]:
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


def derive_geometry(fasc_mask, apo_mask, apo_style, target_shape):
    h, w = target_shape
    fasc_al = align_mask(fasc_mask, h, w)
    apo_al = align_mask(apo_mask, h, w)
    apo = apo_geometry_from_mask(apo_al, apo_style)
    fpca = fascicle_pca(fasc_al)
    out = {
        "apo_method": apo["apo_method"],
        "n_apo_contours": apo["n_contours"],
        "fasc_pixels": int(fasc_al.sum()),
        "pa_deg": np.nan,
        "fl_px": np.nan,
        "mt_px": apo["mt_px"],
        "fasc_angle_deg": np.nan,
        "deep_apo_angle_deg": apo["deep_angle_deg"],
    }
    if fpca is not None:
        out["fasc_angle_deg"] = fpca["angle_deg"]
        out["fl_px"] = fpca["length_px"]
        ref = apo["deep_angle_deg"] if not np.isnan(apo["deep_angle_deg"]) else 0.0
        out["pa_deg"] = acute_angle_deg(fpca["angle_deg"], ref)
    out["_apo_vis"] = apo
    return out
"""
        )
    )

    cells.append(
        md(
            """## Derive geometry on dual-track sample

`NaN` = could not compute (missing contours etc.), **not** zero. Inspect failures; do not auto-drop from training."""
        )
    )

    cells.append(
        code(
            """rng = random.Random(RANDOM_SEED)
geo_names = dual_track if len(dual_track) <= N_GEOMETRY_SAMPLE else rng.sample(dual_track, N_GEOMETRY_SAMPLE)

geo_rows = []
for name in geo_names:
    img = load_gray(lookups["fasc_img"][name])
    style = train_apo_all.loc[train_apo_all.filename == name, "mask_style"].iloc[0]
    fasc_mask = load_mask(lookups["fasc_mask"][name])
    apo_mask = load_mask(lookups["apo_mask"][name])
    metrics = derive_geometry(fasc_mask, apo_mask, style, img.shape)
    metrics.pop("_apo_vis")
    geo_rows.append(
        {
            "filename": name,
            "img_h": img.shape[0],
            "img_w": img.shape[1],
            "mask_style": style,
            "same_shape_fasc": fasc_mask.shape == img.shape,
            **metrics,
        }
    )

geometry_df = pd.DataFrame(geo_rows)
print(f"Derived geometry for {len(geometry_df)} dual-track images")
display(geometry_df.head(8))
print("NaN rates:", geometry_df[["pa_deg", "fl_px", "mt_px"]].isna().mean().round(4).to_dict())
if geometry_df["mt_px"].isna().any():
    display(geometry_df.loc[geometry_df.mt_px.isna(), ["filename", "mask_style", "n_apo_contours"]])
"""
        )
    )

    cells.append(
        md(
            """### Distribution histograms

**Y-axis:** count of **images** per bin (not percent). **PA:** green dashed = ref min 5°, red dashed = ref max 45°."""
        )
    )

    cells.append(
        code(
            """fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title, show_ref in zip(
    axes,
    ["pa_deg", "fl_px", "mt_px"],
    ["Pennation angle (deg)", "Fascicle length (px)", "Muscle thickness (px)"],
    [True, False, False],
):
    vals = geometry_df[col].dropna()
    ax.hist(vals, bins=30, color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Count (images)")
    if show_ref:
        ax.axvline(REF_PA_DEG[0], color="green", ls="--", lw=1.5, label=f"ref min {REF_PA_DEG[0]}°")
        ax.axvline(REF_PA_DEG[1], color="red", ls="--", lw=1.5, label=f"ref max {REF_PA_DEG[1]}°")
        ax.legend(fontsize=8)
plt.tight_layout()
finish_fig(fig, FIG_DIR / "histograms_pa_fl_mt.png")
"""
        )
    )

    cells.append(
        md(
            """### FL bimodality — across images, not inside one image

Each image has **one** `fl_px`. Two histogram peaks = two **image cohorts** (usually resolution), not two fascicles per image."""
        )
    )

    cells.append(
        code(
            """geometry_df["fl_bin"] = np.where(
    geometry_df["fl_px"] < FL_BIN_LOW_MAX, "low (<900 px)", "high (>=900 px)"
)
display(geometry_df["fl_bin"].value_counts().to_frame("count"))
display(pd.crosstab(geometry_df["fl_bin"], geometry_df["mask_style"]))
display(geometry_df.groupby("fl_bin")[["img_h", "img_w"]].agg(["mean", "min", "max"]).round(1))
"""
        )
    )

    cells.append(
        md(
            """## Apo overlay QC (contour edges)

| Panel | Content |
|-------|---------|
| 1–4 | Image, raw, inverted, effective mask |
| 5 | Fitted edges: **cyan** = superficial, **magenta** = deep |

Saved as `figures/apo_qc_<filename>.png`."""
        )
    )

    cells.append(
        code(
            """def draw_apo_edges(ax, apo_vis: dict):
    if apo_vis.get("sup_xs") is not None and len(apo_vis["sup_xs"]):
        ax.scatter(apo_vis["sup_xs"], apo_vis["sup_ys"], s=4, c="cyan", label="sup edge")
        xs = np.linspace(apo_vis["sup_xs"].min(), apo_vis["sup_xs"].max(), 50)
        ax.plot(xs, apo_vis["sup_line"](xs), c="cyan", lw=2)
    if apo_vis.get("deep_xs") is not None and len(apo_vis["deep_xs"]):
        ax.scatter(apo_vis["deep_xs"], apo_vis["deep_ys"], s=4, c="magenta", label="deep edge")
        xs = np.linspace(apo_vis["deep_xs"].min(), apo_vis["deep_xs"].max(), 50)
        ax.plot(xs, apo_vis["deep_line"](xs), c="magenta", lw=2)
    ax.legend(fontsize=6, loc="upper right")


def mask_overlay_rgb(img, mask, color=(255, 140, 0)):
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    sel = mask.astype(bool)
    rgb[sel] = (1 - MASK_OVERLAY_ALPHA) * rgb[sel] + MASK_OVERLAY_ALPHA * tint[sel]
    return rgb.astype(np.uint8)


def apo_qc_panel(name: str):
    img = load_gray(lookups["apo_img"][name])
    raw = align_mask(load_mask(lookups["apo_mask"][name]), *img.shape)
    style = tag_apo_style(raw.mean())
    inv = invert_mask(raw)
    eff, method = effective_apo_mask(raw, style)
    apo_vis = apo_geometry_from_mask(raw, style)
    mt_lbl = f"{apo_vis['mt_px']:.0f}px" if not np.isnan(apo_vis["mt_px"]) else "NaN"
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, title, m, edges in zip(
        axes,
        ["image", f"raw ({style})", "inverted", f"effective ({method})", f"edges mt={mt_lbl}"],
        [None, raw, inv, eff, eff],
        [False, False, False, False, True],
    ):
        if m is None:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(mask_overlay_rgb(img, m))
            if edges:
                draw_apo_edges(ax, apo_vis)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.suptitle(f"{name} coverage={raw.mean():.3f}", y=1.02, fontsize=10)
    plt.tight_layout()
    finish_fig(fig, FIG_DIR / f"apo_qc_{name.replace('.tif', '')}.png")


rng = random.Random(RANDOM_SEED + 1)
for style in ("region", "line"):
    pool = apo_styles[apo_styles.mask_style == style]["filename"].tolist()
    for name in rng.sample(pool, min(N_APO_GALLERY_PER_STYLE, len(pool))):
        print(f"Apo {style}:", name)
        apo_qc_panel(name)
"""
        )
    )

    cells.append(md("""## Fasc stretch validation"""))

    cells.append(
        code(
            """def fasc_stretch_panel(name: str):
    img = load_gray(lookups["fasc_img"][name])
    mask = load_mask(lookups["fasc_mask"][name])
    if mask.shape == img.shape:
        return
    aligned = align_mask(mask, *img.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"image {img.shape}")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"raw {mask.shape}")
    axes[2].imshow(mask_overlay_rgb(img, aligned, color=(0, 200, 80)))
    axes[2].set_title("stretch overlay")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(name, y=1.02)
    plt.tight_layout()
    finish_fig(fig, FIG_DIR / f"fasc_stretch_{name.replace('.tif', '')}.png")


mismatch = [
    n for n in clean_fasc_names
    if load_mask(lookups["fasc_mask"][n]).shape != load_gray(lookups["fasc_img"][n]).shape
]
picks = random.Random(RANDOM_SEED + 2).sample(mismatch, min(N_FASC_STRETCH_CHECK, len(mismatch)))
print(f"Showing {len(picks)} of {len(mismatch)} fasc mismatches")
for name in picks:
    fasc_stretch_panel(name)
"""
        )
    )

    return cells


def export_code_cell() -> dict:
    return code(
        """export_cols = [c for c in geometry_df.columns if not c.startswith("_")]
geometry_df[export_cols].to_csv(OUT / "geometry_sample.csv", index=False)
train_fasc_clean.to_csv(OUT / "train_fasc_clean.csv", index=False)
train_apo_all.to_csv(OUT / "train_apo_all.csv", index=False)
if len(exclude_fasc):
    exclude_fasc.to_csv(OUT / "exclude_fasc_masks.csv", index=False)
apo_styles.to_csv(OUT / "apo_mask_styles.csv", index=False)
calib_df.to_csv(OUT / "tiff_calibration_sample.csv", index=False)
print("Wrote to", OUT.resolve())
for p in sorted(OUT.glob("**/*")):
    if p.is_file():
        print(" ", p.relative_to(OUT))
"""
    )


def build_notebook(intro_cell, paths_md, paths_code, export_md) -> list[dict]:
    shared = shared_cells()
    return [
        intro_cell,
        shared[0],
        shared[1],
        shared[2],
        paths_md,
        paths_code,
        *shared[3:],
        export_md,
        export_code_cell(),
    ]


def write_nb(cells: list[dict], path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path} ({len(cells)} cells)")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "notebooks/geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle_cells = build_notebook(intro_kaggle(), paths_kaggle(), code_paths_kaggle(), export_kaggle())
    local_cells = build_notebook(intro_local(), paths_local(), code_paths_local(), export_local())

    write_nb(kaggle_cells, out_dir / "geometry-phase-2.ipynb")
    write_nb(local_cells, out_dir / "geometry-phase-2-local.ipynb")


if __name__ == "__main__":
    main()
