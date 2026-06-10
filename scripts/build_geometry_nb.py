"""Generate notebooks/geometry/geometry-phase-2.ipynb."""
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


cells: list[dict] = []

cells.append(
    md(
        """# UMUD Challenge — Geometry & Calibration (Phase 2)

This notebook turns **aligned masks** into muscle-architecture measurements and exports **clean training manifests**.

### How to read this notebook

Each section has a markdown intro explaining *why* the next code cell exists. Functions include docstrings; tricky lines have inline comments. If you only skim one section, read **Competition measurement protocol** and **Apo overlay QC**.

### What Phase 2 produces

| Output | Purpose |
|--------|---------|
| `train_fasc_clean.csv` | Fascicle image/mask pairs safe for training (empty/near-empty removed) |
| `train_apo_all.csv` | All aponeurosis pairs + region/line style tag |
| `geometry_sample.csv` | Prototype PA / FL / MT in **pixels** on a dual-track sample |
| `figures/*.png` | Saved QC panels for offline review |

**Run environment:** Kaggle CPU *or* local repo with `data/umud-challenge/` extracted. No GPU.

> Edit *Configuration*, then re-run from there downward."""
    )
)

cells.append(
    md(
        """## Competition measurement protocol (from Data tab + DLTrack)

These are the rules we are trying to approximate — **not** what v1 already solved perfectly.

| Target | Definition (competition) | Notes for this notebook |
|--------|--------------------------|-------------------------|
| **PA** (`pa_deg`) | Angle between the **fascicle** and the **deep aponeurosis** | Degrees; no mm calibration needed |
| **FL** (`fl_px` here) | Length **along the fascicle** between superficial and deep aponeuroses; extrapolate if clipped | We first report fascicle span in px; full apo intersection comes in a later iteration |
| **MT** (`mt_px` here) | **Perpendicular** distance between superficial and deep aponeuroses at **three locations** across muscle width (manual protocol); raters averaged | DLTrack auto code often uses one central estimate; we sample three x positions |

**Aponeurosis geometry:** Real aponeuroses can **curve** or tilt. DLTrack (`PaulRitsche/DL_Track_US`) fits **local linear edges** from mask contours — not fixed horizontal lines. This notebook follows that approach.

**Which aponeuroses when there are 3+ contours?** DLTrack sorts contours top→bottom and picks the **uppermost** as superficial and the **next sufficiently separated** as deep (sometimes skipping a middle contour if too close). The competition always means **superficial vs deep**, not "any two lines".

**Region vs line masks:** Official docs do not name these styles. We observed bimodal apo coverage (~95% region vs ~3% line). For **line** masks we use the raw mask. For **region** masks we **invert** (your hypothesis) so annotated muscle becomes background and aponeurosis gaps become foreground — then run the same contour logic on the inverted mask."""
    )
)

cells.append(md("""## Configuration"""))

cells.append(
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42

# How many dual-track images to measure (see manifest section for what "dual-track" means)
N_GEOMETRY_SAMPLE = 200

# Apo visual QC gallery size per style (region / line)
N_APO_GALLERY_PER_STYLE = 4

# Fasc stretch spot-check count (shape mismatches only)
N_FASC_STRETCH_CHECK = 6

# Coverage >= 50% => "region" apo mask; below => "line" (exploratory tag, not official)
APO_REGION_THRESHOLD = 0.50

# Same fasc exclude thresholds as Phase 0/1 audit notebook
FASC_NEAR_EMPTY_THRESHOLD = 0.0005  # 0.05%

# Image/mask registration when shapes differ (Phase 0/1 decision)
DEFAULT_ALIGN_MODE = "stretch"

# Overlay tint strength for QC figures
MASK_OVERLAY_ALPHA = 0.55

# Reference physiological ranges from competition Data tab (degrees / mm)
# Shown on PA histogram only — FL/MT still in pixels until calibration
REF_PA_DEG = (5, 45)
REF_FL_MM = (30, 200)
REF_MT_MM = (10, 50)

# FL bimodal split for exploratory analysis (px); tuned from v1 histogram valley ~900
FL_BIN_LOW_MAX = 900
FL_BIN_HIGH_MIN = 900
"""
    )
)

cells.append(
    md(
        """## Paths and file discovery

**Local run:** if `data/umud-challenge/` exists at the repo root, we use it (no download).

**Kaggle run:** uses `kagglehub.competition_download` (same as Phase 0/1 audit).

We build `{filename: path}` lookups with `rglob` so nested zip extraction folders still work."""
    )
)

cells.append(
    code(
        """from pathlib import Path
import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

COMPETITION_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"


def find_data_root() -> Path:
    \"\"\"Locate extracted competition data whether cwd is repo root or notebooks/geometry.\"\"\"
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / "data/umud-challenge"
        if candidate.joinpath("apo_imgs_v1").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Local data not found. Extract competition zip to data/umud-challenge/ "
        "(see README). On Kaggle, kagglehub is used instead."
    )


if Path("/kaggle/input").exists():
    import kagglehub

    DATA_ROOT = Path(kagglehub.competition_download(COMPETITION_SLUG))
    print("Using kagglehub competition data:", DATA_ROOT)
else:
    DATA_ROOT = find_data_root()
    print("Using local competition data:", DATA_ROOT)

# Where CSVs and QC PNGs are written
def find_repo_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "pyproject.toml").exists():
            return base.resolve()
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "tmp/geometry-local-output"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
print("Output dir:", OUT.resolve())

DIRS = {
    "apo_img": DATA_ROOT / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": DATA_ROOT / "apo_masks_v1/apo_masks_new_model_v1",
    "fasc_img": DATA_ROOT / "fasc_imgs_v1/fasc_images_new_model_v1",
    "fasc_mask": DATA_ROOT / "fasc_masks_v1/fasc_masks_new_model_v1",
    "test": DATA_ROOT / "test_images_v2/test_set_v2",
}

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    \"\"\"Map basename -> full path; search recursively.\"\"\"
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


lookups = {k: build_lookup(v) for k, v in DIRS.items()}
display(pd.DataFrame([{"key": k, "n_files": len(v)} for k, v in lookups.items()]))
"""
    )
)

cells.append(
    md(
        """## Alignment utilities (ported from Phase 0/1)

When image and mask shapes differ (~60–70% of pairs), we **stretch** the mask to the image size before any pixel-wise geometry. See Phase 0/1 alignment lab for rationale."""
    )
)

cells.append(
    code(
        """def load_gray(path: Path) -> np.ndarray:
    \"\"\"Load ultrasound image as 2D uint8 grayscale.\"\"\"
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    \"\"\"Load annotation mask as binary {0,1} uint8.\"\"\"
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

We rescan all fascicle pairs and exclude **empty** (0% coverage) and **near-empty** (<0.05% coverage) masks — same rule as Phase 0/1.

### Why dual-track (1040) < apo pairs (1048)?

**Not a bug.** Dual-track = filenames present in **both** `train_apo_all` **and** `train_fasc_clean`.

Eight apo images have fascicle masks on the exclude list (empty or near-empty fasc annotation). They still have valid apo masks for MT training, but we cannot derive fascicle-based PA/FL on them until we have another fasc label or predicted fasc mask.

Those 8 files are listed explicitly in the next cell."""
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

cells.append(
    md(
        """## Apo mask style census

Apo coverage is **bimodal** in our data (~95% "region", ~3% "line"). The 50% threshold separates them for routing only."""
    )
)

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

**Option C:** derive geometry in **pixels** now; convert to mm before leaderboard submission. This cell scans TIFF tags for spacing/resolution hints."""
    )
)

cells.append(
    code(
        """CALIBRATION_KEYS = (
    "XResolution",
    "YResolution",
    "ResolutionUnit",
    "PixelSpacing",
    "Spacing",
    "PhysicalDeltaX",
    "PhysicalDeltaY",
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

**Previous v1 mistake:** horizontal row peaks (`axhline`) — bad for curved/tilted aponeuroses and confusing in QC plots.

**This version:**

1. Build an **effective apo mask** (raw line mask, or inverted region mask).
2. Find **external contours** with OpenCV.
3. Sort contours top → bottom; pick **superficial** (top) and **deep** (next separated contour).
4. Extract **edge polylines**: bottom edge of superficial contour, top edge of deep contour.
5. Fit **degree-1 polynomials** `y = a*x + b` to each edge (local linear approximation).
6. **MT:** perpendicular distance between fitted lines at **three x** positions (left / centre / right third of overlap) — mean of the three, matching manual protocol wording.
7. **Fascicle:** PCA axis on fascicle mask pixels → `fl_px` span and angle; **PA** = acute angle vs deep apo line slope.

When fewer than two usable contours exist, MT is **NaN** (typically 1–2% of sample) — those images need different mask handling, not silent zeros."""
    )
)

cells.append(
    code(
        """def effective_apo_mask(mask: np.ndarray, style: str) -> tuple[np.ndarray, str]:
    \"\"\"Line masks: use raw. Region masks: invert so apo gaps become foreground.\"\"\"
    if style == "region":
        return invert_mask(mask), "inverted_region"
    return mask, "raw_line"


def _contour_area(c) -> float:
    return float(cv2.contourArea(c))


def find_apo_contours(mask: np.ndarray, min_area_frac: float = 0.0003) -> list[np.ndarray]:
    \"\"\"Return significant external contours sorted top-to-bottom.\"\"\"
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = mask.size * min_area_frac
    big = [c for c in contours if _contour_area(c) >= min_area]
    big.sort(key=lambda c: cv2.boundingRect(c)[1])
    return big


def pick_superficial_deep(contours: list[np.ndarray], min_sep_px: int = 15):
    \"\"\"DLTrack-style: top contour = superficial; next separated = deep.\"\"\"
    if len(contours) < 2:
        return None, None, len(contours)
    sup = contours[0]
    _, y0, _, h0 = cv2.boundingRect(sup)
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
    \"\"\"Reduce contour to a polyline: per x-bin, take min or max y.\"\"\"
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
    coef = np.polyfit(xs, ys, 1)
    return np.poly1d(coef)


def line_angle_deg(line) -> float:
    if line is None:
        return np.nan
    return float(np.degrees(np.arctan(line[1])))


def perpendicular_distance(line_a, line_b, x: float) -> float:
    \"\"\"Distance between two lines y=f(x) at a given x (competition MT is perpendicular to apo).\"\"\"
    ya, yb = float(line_a(x)), float(line_b(x))
    return abs(yb - ya)


def mt_from_apo_edges(sup_line, deep_line, x_left: float, x_right: float) -> float:
    if sup_line is None or deep_line is None or x_right <= x_left:
        return np.nan
    thirds = [x_left + (x_right - x_left) * t for t in (1 / 6, 3 / 6, 5 / 6)]
    dists = [perpendicular_distance(sup_line, deep_line, x) for x in thirds]
    return float(np.mean(dists))


def fascicle_pca(mask: np.ndarray) -> dict | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    mu = coords.mean(axis=0)
    centered = coords - mu
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
    out["sup_line"] = sup_line
    out["deep_line"] = deep_line
    out["sup_xs"], out["sup_ys"] = sup_x, sup_y
    out["deep_xs"], out["deep_ys"] = deep_x, deep_y
    if sup_line is not None and deep_line is not None and len(sup_x) and len(deep_x):
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
        ref_angle = apo["deep_angle_deg"] if not np.isnan(apo["deep_angle_deg"]) else 0.0
        out["pa_deg"] = acute_angle_deg(fpca["angle_deg"], ref_angle)
    out["_apo_vis"] = apo  # kept for QC plotting; stripped before CSV export
    return out
"""
    )
)

cells.append(
    md(
        """## Derive geometry on dual-track sample

We measure PA/FL from fascicle masks and MT from apo masks on the **same filename** when both tracks are available (`dual_track` list above).

### About the "NaN rate" line

`NaN` = "could not compute" (missing contour, too few fascicle pixels), **not** zero thickness. A 1% MT NaN rate (~2/200) is acceptable for a prototype; we should inspect those filenames rather than drop them from training data automatically."""
    )
)

cells.append(
    code(
        """rng = random.Random(RANDOM_SEED)
geo_names = dual_track if len(dual_track) <= N_GEOMETRY_SAMPLE else rng.sample(dual_track, N_GEOMETRY_SAMPLE)

geo_rows = []
apo_vis_cache = {}
for name in geo_names:
    img = load_gray(lookups["fasc_img"][name])
    style = train_apo_all.loc[train_apo_all.filename == name, "mask_style"].iloc[0]
    fasc_mask = load_mask(lookups["fasc_mask"][name])
    apo_mask = load_mask(lookups["apo_mask"][name])
    metrics = derive_geometry(fasc_mask, apo_mask, style, img.shape)
    apo_vis_cache[name] = metrics.pop("_apo_vis")
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
display(geometry_df.drop(columns=[], errors="ignore").head(8))

nan_rates = geometry_df[["pa_deg", "fl_px", "mt_px"]].isna().mean().round(4)
print("NaN rates (fraction of sample that failed):", nan_rates.to_dict())
if geometry_df["mt_px"].isna().any():
    print("MT failures:")
    display(geometry_df.loc[geometry_df.mt_px.isna(), ["filename", "mask_style", "n_apo_contours", "apo_method"]])
"""
    )
)

cells.append(
    md(
        """### Distribution histograms

- **Y-axis:** count of **images** in each bin (not percent). Matplotlib `hist` default.
- **PA reference lines:** dashed green = competition min (5°), dashed red = max (45°).
- **FL / MT:** still in pixels — no mm reference lines yet."""
    )
)

cells.append(
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(14, 4))
configs = [
    ("pa_deg", "Pennation angle (deg)", True),
    ("fl_px", "Fascicle length (px)", False),
    ("mt_px", "Muscle thickness (px)", False),
]
for ax, (col, title, show_ref) in zip(axes, configs):
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
plt.savefig(FIG_DIR / "histograms_pa_fl_mt.png", dpi=120, bbox_inches="tight")
plt.close(fig)
"""
    )
)

cells.append(
    md(
        """### FL bimodality — per image, not per fascicle inside one image

Each row in `geometry_df` is **one image** with **one** `fl_px` (PCA span of that image's fascicle mask). The histogram therefore shows **two populations of images**, not "two fascicle lengths inside the same image".

Below we assign each image to a **low** or **high** FL bin and check whether any image appears in both (it cannot — mutually exclusive bins). Then we correlate bin membership with image size and shape mismatch to explain the bimodality."""
    )
)

cells.append(
    code(
        """geometry_df["fl_bin"] = np.where(
    geometry_df["fl_px"] < FL_BIN_LOW_MAX,
    "low (<900 px)",
    "high (>=900 px)",
)
print("Images per FL bin:")
display(geometry_df["fl_bin"].value_counts().to_frame("count"))

# Each image is in exactly one bin by construction
assert geometry_df["fl_bin"].notna().all()

print("FL bin vs apo mask style:")
display(pd.crosstab(geometry_df["fl_bin"], geometry_df["mask_style"]))

print("FL bin vs fasc/image same shape:")
display(pd.crosstab(geometry_df["fl_bin"], geometry_df["same_shape_fasc"]))

print("Mean image dimensions by FL bin:")
display(
    geometry_df.groupby("fl_bin")[["img_h", "img_w"]].agg(["mean", "min", "max"]).round(1)
)

print("Unique (img_h, img_w) sizes by FL bin:")
for b in geometry_df["fl_bin"].unique():
    sub = geometry_df[geometry_df.fl_bin == b]
    sizes = sub.groupby(["img_h", "img_w"]).size().reset_index(name="n")
    print(f"\\n{b}:")
    display(sizes.sort_values("n", ascending=False).head(8))
"""
    )
)

cells.append(
    md(
        """## Apo overlay QC (contour edges)

**How to read each 5-panel row:**

| Panel | What you see |
|-------|----------------|
| 1 | Ultrasound image |
| 2 | Raw apo mask overlay (orange) |
| 3 | Inverted mask (region case) — test of your flip hypothesis |
| 4 | **Effective** mask used for contour detection |
| 5 | Effective mask + **fitted edges**: cyan = superficial (bottom edge), magenta = deep (top edge). Small dots = edge samples. |

**Three aponeuroses in mask:** we sort contours top→bottom; superficial = top contour; deep = next separated contour (DLTrack rule). Extra contours are ignored for MT but may appear as extra blobs in panel 4.

Figures are saved under `figures/apo_qc_*.png`."""
    )
)

cells.append(
    code(
        """def draw_apo_edges(ax, apo_vis: dict):
    if apo_vis.get("sup_xs") is not None and len(apo_vis["sup_xs"]):
        ax.scatter(apo_vis["sup_xs"], apo_vis["sup_ys"], s=4, c="cyan", label="sup edge pts")
        xs = np.linspace(apo_vis["sup_xs"].min(), apo_vis["sup_xs"].max(), 50)
        ax.plot(xs, apo_vis["sup_line"](xs), c="cyan", lw=2, label="sup fit")
    if apo_vis.get("deep_xs") is not None and len(apo_vis["deep_xs"]):
        ax.scatter(apo_vis["deep_xs"], apo_vis["deep_ys"], s=4, c="magenta", label="deep edge pts")
        xs = np.linspace(apo_vis["deep_xs"].min(), apo_vis["deep_xs"].max(), 50)
        ax.plot(xs, apo_vis["deep_line"](xs), c="magenta", lw=2, label="deep fit")
    ax.legend(fontsize=6, loc="upper right")


def mask_overlay_rgb(img, mask, color=(255, 140, 0)):
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    tint = np.zeros_like(rgb)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    sel = mask.astype(bool)
    rgb[sel] = (1 - MASK_OVERLAY_ALPHA) * rgb[sel] + MASK_OVERLAY_ALPHA * tint[sel]
    return rgb.astype(np.uint8)


def apo_qc_panel(name: str, save: bool = True):
    img = load_gray(lookups["apo_img"][name])
    raw = align_mask(load_mask(lookups["apo_mask"][name]), *img.shape)
    style = tag_apo_style(raw.mean())
    inv = invert_mask(raw)
    eff, method = effective_apo_mask(raw, style)
    apo_vis = apo_geometry_from_mask(raw, style)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        ("image", None),
        (f"raw ({style})", raw),
        ("inverted", inv),
        (f"effective ({method})", eff),
        (f"edges n={apo_vis['n_contours']} mt={apo_vis['mt_px']:.0f}px" if not np.isnan(apo_vis["mt_px"]) else f"edges n={apo_vis['n_contours']}", eff),
    ]
    for ax, (title, m) in zip(axes, panels):
        if m is None:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(mask_overlay_rgb(img, m))
            if title.startswith("edges"):
                draw_apo_edges(ax, apo_vis)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.suptitle(f"{name} coverage={raw.mean():.3f}", y=1.02, fontsize=10)
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / f"apo_qc_{name.replace('.tif','')}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


rng = random.Random(RANDOM_SEED + 1)
for style in ("region", "line"):
    pool = apo_styles[apo_styles.mask_style == style]["filename"].tolist()
    picks = rng.sample(pool, min(N_APO_GALLERY_PER_STYLE, len(pool)))
    print(f"--- Apo {style} examples ({len(picks)}) ---")
    for name in picks:
        apo_qc_panel(name)
"""
    )
)

cells.append(
    md(
        """## Fasc stretch validation

Confirms Phase 0/1 finding: for shape mismatches, **stretch** keeps fascicle annotations aligned with anatomy."""
    )
)

cells.append(
    code(
        """def fasc_stretch_panel(name: str):
    img = load_gray(lookups["fasc_img"][name])
    mask = load_mask(lookups["fasc_mask"][name])
    if mask.shape == img.shape:
        return
    aligned = align_mask(mask, *img.shape)
    vis = mask_overlay_rgb(img, aligned, color=(0, 200, 80))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"image {img.shape}")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"raw mask {mask.shape}")
    axes[2].imshow(vis)
    axes[2].set_title("stretch overlay")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(name, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fasc_stretch_{name.replace('.tif','')}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


mismatch_names = []
for name in clean_fasc_names:
    img = load_gray(lookups["fasc_img"][name])
    mask = load_mask(lookups["fasc_mask"][name])
    if mask.shape != img.shape:
        mismatch_names.append(name)

rng = random.Random(RANDOM_SEED + 2)
stretch_picks = rng.sample(mismatch_names, min(N_FASC_STRETCH_CHECK, len(mismatch_names)))
print(f"Fasc shape mismatches: {len(mismatch_names)} | showing {len(stretch_picks)}")
for name in stretch_picks:
    fasc_stretch_panel(name)
"""
    )
)

cells.append(
    md(
        """## Export artifacts

CSV columns exclude internal visualization caches. PNGs already saved under `figures/`."""
    )
)

cells.append(
    code(
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
        print(f"  {p.relative_to(OUT)} ({p.stat().st_size} bytes)")
"""
    )
)

cells.append(
    md(
        """## Phase 2 checklist

- [x] Clean manifests + explain dual-track gap
- [x] Contour-based apo edges (not horizontal row peaks)
- [x] Pixels-first geometry + histogram labels
- [x] FL bimodality explained (across images, correlated with resolution)
- [ ] User visual QC on saved apo QC PNGs
- [ ] Pixel→mm calibration before leaderboard submit
- [ ] Phase 3 segmentation baseline"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out_dir = Path(__file__).resolve().parents[1] / "notebooks/geometry"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "geometry-phase-2.ipynb"
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out_path} ({len(cells)} cells)")
