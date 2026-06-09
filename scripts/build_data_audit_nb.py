"""Generate notebooks/data-audit/data-audit.ipynb."""
import json
from pathlib import Path


def md(source: str) -> dict:
    # Jupyter renders markdown correctly only when each line ends with \n.
    lines = source.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines],
    }


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
        """# UMUD Challenge — Data Audit (Phase 0 + 1)

This notebook inventories the competition data and runs **exploratory data analysis (EDA)** focused on **data quality** before any modeling.

**Phases covered here**

- **Phase 0 — Inventory:** file counts, pairing, corrupt files, image/mask shapes, overlap between apo and fasc sets.
- **Phase 1 — Visual QC:** overlay masks on images, mask coverage statistics, alignment experiments, flag suspicious cases.

**Run environment:** Kaggle (competition data via `kagglehub`). No GPU required.

> **Tip:** Change the parameters in the *Configuration* cell below, then re-run from there downward."""
    )
)

cells.append(
    md(
        """## What is a mask? (quick primer)

In this competition, a **mask** is a second image aligned with an ultrasound image. Each pixel in the mask tells you whether that pixel belongs to an annotated structure.

- **Aponeurosis masks (`apo_masks`):** highlight regions related to the superficial and deep aponeuroses (the bright horizontal structures in the muscle).
- **Fascicle masks (`fasc_masks`):** highlight the muscle fascicle(s) — often thin oblique lines.

Masks are typically **binary**: background = 0 (black), annotated structure = nonzero (often 255 = white). They are **not** the final competition targets (`pa_deg`, `fl_mm`, `mt_mm`). They are **intermediate supervision**: you can train a model to predict masks, then use geometry rules to convert masks into the three numeric targets — or ignore masks entirely and use classical computer vision.

The organizers created masks from expert manual annotation (see the competition **Data** tab). That means mask quality and alignment directly affect any mask-based pipeline."""
    )
)

cells.append(
    md(
        """## Configuration

Edit these values and re-run the notebook to explore different samples."""
    )
)

cells.append(
    code(
        """# --- Parameters you can change ---
RANDOM_SEED = 42
N_SAMPLE_STATS = 400       # pairs to scan for shape/coverage stats (sample)
N_OVERLAY_SHOW = 6         # random overlay gallery size
N_WORST_SHOW = 4
N_BEST_SHOW = 2
N_ALIGN_SHOW = 4           # alignment comparison rows
N_SMALL_MASK_SHOW = 6      # small-mask scale-up gallery size

MASK_OVERLAY_ALPHA = 0.55  # overlay strength (higher helps thin fascicle masks)
GALLERY_TRACK = "both"     # "apo", "fasc", or "both"

APO_REGION_THRESHOLD = 0.50   # coverage above this => "region" apo mask vs "line" mask
SMALL_MASK_AREA_RATIO = 0.55  # flag pairs where mask area / image area is below this
FASC_EMPTY_THRESHOLD = 0.0    # fascicle masks at or below this coverage are "empty"
FASC_NEAR_EMPTY_THRESHOLD = 0.0005  # 0.05% — conservative; 0.1% catches ~248 extra pairs
DEFAULT_ALIGN_MODE = "stretch"  # best visual match in alignment lab so far
ALIGN_MODES = ("stretch", "center", "scale")
"""
    )
)

cells.append(
    md(
        """## Paths and file discovery

Competition data is loaded via **Kaggle Hub** (`kagglehub`). No local upload required.

We build filename → path lookups with `rglob` so nested folders still work."""
    )
)

cells.append(
    code(
        """from pathlib import Path
import random

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

COMPETITION_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
DATA_ROOT = Path(kagglehub.competition_download(COMPETITION_SLUG))
print("Competition dir:", DATA_ROOT)

DIRS = {
    "apo_img": DATA_ROOT / "apo_imgs_v1/apo_images_new_model_v1",
    "apo_mask": DATA_ROOT / "apo_masks_v1/apo_masks_new_model_v1",
    "fasc_img": DATA_ROOT / "fasc_imgs_v1/fasc_images_new_model_v1",
    "fasc_mask": DATA_ROOT / "fasc_masks_v1/fasc_masks_new_model_v1",
    "test": DATA_ROOT / "test_images_v2/test_set_v2",
}

for name, d in DIRS.items():
    print(f"{name:10s} exists={d.exists()}  path={d}")

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_lookup(directory: Path) -> dict[str, Path]:
    \"\"\"Map filename -> full path (handles nested extraction folders).\"\"\"
    return {
        p.name: p
        for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.name != "Thumbs.db"
    }


lookups = {k: build_lookup(v) for k, v in DIRS.items()}

# Compact summary — avoid dumping thousands of paths into the notebook output.
lookup_summary = pd.DataFrame(
    [{"key": k, "n_files": len(v)} for k, v in lookups.items()]
)
display(lookup_summary)
print("Sample filenames per lookup (first 5):")
for k, v in lookups.items():
    sample = sorted(v)[:5]
    print(f"  {k}: {sample}")
"""
    )
)

cells.append(
    md(
        """## Phase 0 — Inventory

Mechanical checks: how many files, do images and masks pair 1:1, and how do the apo vs fasc sets relate?"""
    )
)

cells.append(
    code(
        """def paired_ids(img_lookup, mask_lookup):
    img_ids = set(img_lookup)
    mask_ids = set(mask_lookup)
    return img_ids & mask_ids, img_ids - mask_ids, mask_ids - img_ids

rows = []
for track, img_key, mask_key in [
    ("apo", "apo_img", "apo_mask"),
    ("fasc", "fasc_img", "fasc_mask"),
]:
    common, img_only, mask_only = paired_ids(lookups[img_key], lookups[mask_key])
    rows.append({
        "track": track,
        "images": len(lookups[img_key]),
        "masks": len(lookups[mask_key]),
        "paired": len(common),
        "image_only": len(img_only),
        "mask_only": len(mask_only),
    })

inventory = pd.DataFrame(rows)
display(inventory)

apo_names = set(lookups["apo_img"])
fasc_names = set(lookups["fasc_img"])
print(f"Apo images that also appear in fasc set: {len(apo_names & fasc_names)} / {len(apo_names)}")
print(f"Fasc-only images (not in apo set): {len(fasc_names - apo_names)}")
print(f"Test .tif images: {sum(1 for p in lookups['test'] if p.lower().endswith('.tif'))}")
print(f"Test .png images: {sum(1 for p in lookups['test'] if p.lower().endswith('.png'))}")
"""
    )
)

cells.append(
    md(
        """### Fasc-only images — is that a problem?

**Short answer: no, this is expected.**

The competition provides two training tracks from Ritsche et al. (2024):

- **Aponeurosis set:** 1,048 pairs — images where aponeurosis annotation was the focus.
- **Fascicle set:** 2,761 pairs — a **larger** pool of fascicle annotations.

Every apo image also appears in the fasc set (same filename). The extra **1,713 fasc-only** images are additional fascicle training examples without a matching aponeurosis mask in the apo track.

For modeling you can:

- Train an **aponeurosis** model on 1,048 apo pairs only.
- Train a **fascicle** model on all 2,761 fasc pairs.
- They are complementary subsets, not a data error."""
    )
)

cells.append(
    code(
        """# Submission template sanity check
sample_path = DATA_ROOT / "sample_submission.csv"
if sample_path.exists():
    sample_sub = pd.read_csv(sample_path, sep=";")
    display(sample_sub.head())
    print("Columns:", list(sample_sub.columns))
else:
    print("sample_submission.csv not found at", sample_path)
"""
    )
)

cells.append(
    md(
        """### Corrupt file scan

Try opening every TIFF/PNG. Any read error is a data-quality problem."""
    )
)

cells.append(
    code(
        """def scan_readable(lookup: dict[str, Path], label: str, max_show: int = 10):
    bad = []
    for name, path in lookup.items():
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
        except Exception as e:
            bad.append((name, str(e)))
    print(f"{label}: {len(lookup)} files, {len(bad)} unreadable")
    for name, err in bad[:max_show]:
        print(f"  {name}: {err}")
    return bad

bad_files = {}
for label, key in [
    ("apo_img", "apo_img"),
    ("apo_mask", "apo_mask"),
    ("fasc_img", "fasc_img"),
    ("fasc_mask", "fasc_mask"),
    ("test", "test"),
]:
    bad_files[label] = scan_readable(lookups[key], label)
"""
    )
)

cells.append(
    md(
        """### Image / mask shape alignment

**Important:** If image and mask shapes differ, pixel-wise training loss is misaligned unless you register pairs consistently.

Many images look like **screenshots of an ultrasound machine** — the scan sits in the middle with padding/borders. Masks are sometimes stored at the **native scan resolution** (smaller canvas) while the image includes the full frame.

The alignment section below compares **stretch**, **center** (pad/center without scaling), and **scale + center** (uniform scale then center)."""
    )
)

cells.append(
    code(
        """def image_shape(path: Path):
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 2:
        return arr.shape  # H, W
    return arr.shape[:2]


def mask_coverage_from_path(path: Path) -> float:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return float((arr > 0).mean())


def analyze_pairs(track: str, img_lookup, mask_lookup, n_sample: int, seed: int):
    common = sorted(set(img_lookup) & set(mask_lookup))
    rng = random.Random(seed)
    sample = common if len(common) <= n_sample else rng.sample(common, n_sample)

    records = []
    for name in sample:
        ish = image_shape(img_lookup[name])
        msh = image_shape(mask_lookup[name])
        coverage = mask_coverage_from_path(mask_lookup[name])
        img_area = ish[0] * ish[1]
        mask_area = msh[0] * msh[1]
        records.append({
            "track": track,
            "filename": name,
            "img_h": ish[0], "img_w": ish[1],
            "mask_h": msh[0], "mask_w": msh[1],
            "same_shape": ish == msh,
            "mask_coverage": coverage,
            "mask_area_ratio": mask_area / img_area if img_area else np.nan,
        })
    return pd.DataFrame(records)

shape_df = pd.concat([
    analyze_pairs("apo", lookups["apo_img"], lookups["apo_mask"], N_SAMPLE_STATS, RANDOM_SEED),
    analyze_pairs("fasc", lookups["fasc_img"], lookups["fasc_mask"], N_SAMPLE_STATS, RANDOM_SEED + 1),
], ignore_index=True)

summary = shape_df.groupby("track").agg(
    n=("filename", "count"),
    same_shape_pct=("same_shape", lambda s: 100 * s.mean()),
    coverage_median=("mask_coverage", "median"),
    coverage_min=("mask_coverage", "min"),
    coverage_max=("mask_coverage", "max"),
    small_mask_pct=("mask_area_ratio", lambda s: 100 * (s < SMALL_MASK_AREA_RATIO).mean()),
)
display(summary.round(6))

print("Mismatched shape examples:")
display(shape_df[~shape_df["same_shape"]].head(10))
"""
    )
)

cells.append(
    md(
        """### Aponeurosis: region masks vs line masks

Apo coverage is **bimodal**: some masks cover ~90–99% of pixels (region / ROI style), others ~2–6% (thin line style). The threshold below classifies them for exploration — not a ground-truth label from the organizers."""
    )
)

cells.append(
    code(
        """apo_df = shape_df[shape_df.track == "apo"].copy()
apo_df["mask_style"] = np.where(
    apo_df["mask_coverage"] >= APO_REGION_THRESHOLD,
    "region",
    "line",
)
print("Apo mask styles in sample:")
display(apo_df["mask_style"].value_counts().to_frame("count"))

print("Apo rows sorted by coverage (lowest first):")
display(
    apo_df.sort_values("mask_coverage")[
        ["filename", "mask_coverage", "mask_style", "same_shape", "mask_area_ratio"]
    ].head(12)
)
print("Apo rows sorted by coverage (highest first):")
display(
    apo_df.sort_values("mask_coverage", ascending=False)[
        ["filename", "mask_coverage", "mask_style", "same_shape", "mask_area_ratio"]
    ].head(12)
)
"""
    )
)

cells.append(
    md(
        """### Empty or near-empty fascicle masks

Fascicle annotations can be extremely sparse (often well under 1% coverage). A mask with **zero** annotated pixels cannot supervise segmentation.

**Near-empty threshold (0.05%):** `0.0005` is a conservative QC cutoff, not an organizer-defined constant. At **0.1%** the count jumps from ~7 to ~248 pairs — too aggressive for automatic exclusion. We export empty + near-empty filenames for a **clean fasc training list** in Phase 2.

The next cell scans **all** fascicle pairs (not just the sample)."""
    )
)

cells.append(
    code(
        """print("Scanning all fascicle pairs for empty masks (may take ~1–2 min)...")
empty_fasc = []
for name in sorted(set(lookups["fasc_img"]) & set(lookups["fasc_mask"])):
    cov = mask_coverage_from_path(lookups["fasc_mask"][name])
    if cov <= FASC_EMPTY_THRESHOLD:
        empty_fasc.append({"filename": name, "mask_coverage": cov})

empty_fasc_df = pd.DataFrame(empty_fasc)
print(f"Empty fascicle masks (coverage <= {FASC_EMPTY_THRESHOLD}): {len(empty_fasc_df)}")
if len(empty_fasc_df):
    display(empty_fasc_df)

near_empty = []
for name in sorted(set(lookups["fasc_img"]) & set(lookups["fasc_mask"])):
    cov = mask_coverage_from_path(lookups["fasc_mask"][name])
    if FASC_EMPTY_THRESHOLD < cov < FASC_NEAR_EMPTY_THRESHOLD:
        near_empty.append({"filename": name, "mask_coverage": cov})
near_empty_df = pd.DataFrame(near_empty)
print(
    f"Near-empty fascicle masks ({100*FASC_NEAR_EMPTY_THRESHOLD:.2f}% > coverage > 0): "
    f"{len(near_empty_df)}"
)
if len(near_empty_df):
    display(near_empty_df)

exclude_fasc = pd.concat([
    empty_fasc_df.assign(reason="empty"),
    near_empty_df.assign(reason="near_empty"),
], ignore_index=True)
print(f"Total fasc pairs to exclude from mask training: {len(exclude_fasc)}")
print(f"Clean fasc pairs remaining: {2761 - len(exclude_fasc)}")
"""
    )
)

cells.append(
    md(
        """### Shape and coverage distributions

Aponeurosis masks are often **bimodal** (region vs line). Fascicle masks are usually **sparse** thin structures."""
    )
)

cells.append(
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, track, color in zip(axes, ["apo", "fasc"], ["tab:orange", "tab:green"]):
    sub = shape_df[shape_df.track == track]
    ax.hist(sub["mask_coverage"] * 100, bins=40, color=color, alpha=0.85)
    ax.set_title(f"{track} mask coverage (% nonzero pixels)")
    ax.set_xlabel("coverage %")
    ax.set_ylabel("count")
plt.tight_layout()
plt.show()

shape_counts = shape_df.groupby(["track", "same_shape"]).size().unstack(fill_value=0)
print("Pair counts by shape match:")
display(shape_counts)
"""
    )
)

cells.append(
    md(
        """## Phase 1 — Visual QC

We overlay masks on images. **Green** = fascicle track, **orange** = aponeurosis track.

**Bug fix from v1:** colored overlays must use `imshow(rgb)` **without** `cmap="gray"` — grayscale colormap was wiping out the green/orange tint entirely.

Ask yourself when viewing:

1. Does the mask follow visible anatomy?
2. Is the mask shifted or scaled relative to the image (shape mismatch)?
3. Is coverage suspicious (empty mask, or almost the entire frame)?"""
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


def place_mask_center(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    \"\"\"Pad/center mask on target canvas. Crops from mask center when mask is larger.\"\"\"
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
    \"\"\"Scale mask preserving aspect ratio, then center on image-sized canvas.\"\"\"
    mh, mw = mask.shape
    scale = min(target_h / mh, target_w / mw)
    new_h = max(1, int(round(mh * scale)))
    new_w = max(1, int(round(mw * scale)))
    resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
    ) > 0
    return place_mask_center(resized.astype(np.uint8), target_h, target_w)


def align_mask(mask: np.ndarray, target_h: int, target_w: int, mode: str) -> np.ndarray:
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


def overlay(
    img_gray: np.ndarray,
    mask: np.ndarray,
    color=(0, 255, 0),
    alpha=0.55,
    align_mode="center",
    draw_contours=False,
):
    h, w = img_gray.shape
    aligned = align_mask(mask, h, w, align_mode)
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    color_arr = np.zeros_like(rgb)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    m = aligned.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color_arr[m]
    vis = rgb.astype(np.uint8)
    if draw_contours and m.any():
        try:
            import cv2

            contours, _ = cv2.findContours(
                aligned.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 1)
        except Exception:
            pass
    n_annot = int(m.sum())
    return vis, aligned, n_annot


def show_pair(track, filename, img_lookup, mask_lookup, ax, color, align_mode="center"):
    img = load_gray(img_lookup[filename])
    mask = load_mask(mask_lookup[filename])
    draw_contours = track == "fasc"
    vis, _, n_annot = overlay(
        img,
        mask,
        color=color,
        alpha=MASK_OVERLAY_ALPHA,
        align_mode=align_mode,
        draw_contours=draw_contours,
    )
    cov = mask.mean() * 100
    same = mask.shape == img.shape
    ax.imshow(vis)  # no cmap — preserves green/orange overlay
    ax.set_title(
        f"{track} {filename}\\ncov={cov:.3f}% px={n_annot} same={same} mode={align_mode}",
        fontsize=8,
    )
    ax.axis("off")


def zoom_crop(vis: np.ndarray, aligned: np.ndarray, pad: int = 40):
    ys, xs = np.where(aligned > 0)
    if len(xs) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(vis.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(vis.shape[1], xs.max() + pad)
    return vis[y0:y1, x0:x1]


def focus_row(img, mask, track, filename, align_mode, axes_row, color):
    \"\"\"One row: image | raw mask | overlay | zoom for a single align mode.\"\"\"
    aligned = align_mask(mask, *img.shape, align_mode)
    vis, _, n_px = overlay(
        img,
        mask,
        color=color,
        alpha=MASK_OVERLAY_ALPHA,
        align_mode=align_mode,
        draw_contours=(track == "fasc"),
    )
    axes_row[0].imshow(img, cmap="gray")
    axes_row[0].set_title(f"image")
    axes_row[1].imshow(mask, cmap="gray")
    axes_row[1].set_title(f"raw {mask.shape}")
    axes_row[2].imshow(vis)
    axes_row[2].set_title(f"overlay ({n_px} px)")
    crop = zoom_crop(vis, aligned)
    if crop is not None:
        axes_row[3].imshow(crop)
        axes_row[3].set_title("zoom")
    else:
        axes_row[3].set_title("no px")
    for ax in axes_row:
        ax.axis("off")
    axes_row[0].set_ylabel(align_mode, fontsize=10, rotation=0, labelpad=40)
"""
    )
)

cells.append(
    code(
        """# Random sample gallery (default: stretch)
rng = random.Random(RANDOM_SEED)

def gallery_for_track(track, n_show, align_mode=DEFAULT_ALIGN_MODE):
    if track == "apo":
        img_l, mask_l, color = lookups["apo_img"], lookups["apo_mask"], (255, 140, 0)
    else:
        img_l, mask_l, color = lookups["fasc_img"], lookups["fasc_mask"], (0, 200, 80)
    ids = sorted(set(img_l) & set(mask_l))
    picks = rng.sample(ids, min(n_show, len(ids)))
    cols = 3
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, picks):
        show_pair(track, name, img_l, mask_l, ax, color, align_mode=align_mode)
    for ax in axes[len(picks):]:
        ax.axis("off")
    plt.suptitle(f"Random {track} overlays — align={align_mode} (seed={RANDOM_SEED})", y=1.02)
    plt.tight_layout()
    plt.show()

if GALLERY_TRACK in ("apo", "both"):
    gallery_for_track("apo", N_OVERLAY_SHOW)
if GALLERY_TRACK in ("fasc", "both"):
    gallery_for_track("fasc", N_OVERLAY_SHOW)
"""
    )
)

cells.append(
    md(
        """### Alignment lab: stretch vs center vs scale

For mismatched shapes, compare three registration strategies on the same files.

- **stretch:** force mask to image size (distorts aspect ratio)
- **center:** place mask in the middle without scaling (good when image has screenshot padding)
- **scale:** uniform scale then center (good when mask is a smaller version of the same scan)"""
    )
)

cells.append(
    code(
        """mismatch = shape_df[~shape_df["same_shape"]].copy()
rng = random.Random(RANDOM_SEED + 99)
if len(mismatch):
    picks = mismatch.sample(min(N_ALIGN_SHOW, len(mismatch)), random_state=RANDOM_SEED + 99)
else:
    picks = shape_df.head(0)

for _, row in picks.iterrows():
    track = row["track"]
    name = row["filename"]
    img_l = lookups["apo_img" if track == "apo" else "fasc_img"]
    mask_l = lookups["apo_mask" if track == "apo" else "fasc_mask"]
    color = (255, 140, 0) if track == "apo" else (0, 200, 80)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, mode in zip(axes, ["stretch", "center", "scale"]):
        show_pair(track, name, img_l, mask_l, ax, color, align_mode=mode)
    plt.suptitle(
        f"{track} {name} — img {row.img_h}x{row.img_w} mask {row.mask_h}x{row.mask_w}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    md(
        """### Small masks: stretch vs scale

Pairs where the mask canvas is **much smaller** than the image (low `mask_area_ratio`) — common for 800×1200 images with 556×660 masks.

Each example shows **stretch** (left) and **scale** (right) side by side."""
    )
)

cells.append(
    code(
        """small = shape_df[shape_df["mask_area_ratio"] < SMALL_MASK_AREA_RATIO].copy()
print(f"Pairs with mask_area_ratio < {SMALL_MASK_AREA_RATIO}: {len(small)} in sample")
display(
    small.sort_values("mask_area_ratio")[
        ["track", "filename", "img_h", "img_w", "mask_h", "mask_w", "mask_area_ratio"]
    ].head(12)
)

if len(small):
    picks = small.sample(min(N_SMALL_MASK_SHOW, len(small)), random_state=RANDOM_SEED + 7)
    for _, row in picks.iterrows():
        track = row["track"]
        name = row["filename"]
        img_l = lookups["apo_img" if track == "apo" else "fasc_img"]
        mask_l = lookups["apo_mask" if track == "apo" else "fasc_mask"]
        color = (255, 140, 0) if track == "apo" else (0, 200, 80)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, mode in zip(axes, ["stretch", "scale"]):
            show_pair(track, name, img_l, mask_l, ax, color, align_mode=mode)
        plt.suptitle(
            f"Small mask {track} {name} — img {row.img_h}x{row.img_w} mask {row.mask_h}x{row.mask_w}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.show()
"""
    )
)

cells.append(
    md(
        """### Focus view: image | mask | overlay | zoom × three align modes

For one file, compare **stretch**, **center**, and **scale**. Each row is one align mode with four panels.

Change `FOCUS_FILE` in the cell above to inspect other examples."""
    )
)

cells.append(
    code(
        """fasc_stats = shape_df[shape_df.track == "fasc"].sort_values("mask_coverage")
display(fasc_stats.head(N_WORST_SHOW)[["filename", "mask_coverage", "same_shape"]])
display(fasc_stats.tail(N_BEST_SHOW)[["filename", "mask_coverage", "same_shape"]])

FOCUS_FILE = fasc_stats.iloc[len(fasc_stats) // 2]["filename"]  # mid-coverage fasc example
print("FOCUS_FILE =", FOCUS_FILE, "(change this variable to inspect others)")
"""
    )
)

cells.append(
    code(
        """for track in ["apo", "fasc"]:
    img_l = lookups["apo_img" if track == "apo" else "fasc_img"]
    mask_l = lookups["apo_mask" if track == "apo" else "fasc_mask"]
    if FOCUS_FILE not in img_l or FOCUS_FILE not in mask_l:
        print(f"{track}: {FOCUS_FILE} not in this track")
        continue
    img = load_gray(img_l[FOCUS_FILE])
    mask = load_mask(mask_l[FOCUS_FILE])
    color = (255, 140, 0) if track == "apo" else (0, 200, 80)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for row_axes, mode in zip(axes, ALIGN_MODES):
        focus_row(img, mask, track, FOCUS_FILE, mode, row_axes, color)
    plt.suptitle(f"{track} — {FOCUS_FILE} (rows: stretch / center / scale)", fontsize=11)
    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    md(
        """### Test set preview

Test images use `IMG_*.tif` naming (different from training `image_*.tif`). Some test cases are **5-frame video sequences** — temporal grouping is worth exploring later."""
    )
)

cells.append(
    code(
        """test_names = sorted(lookups["test"])
print("First 10 test files:", test_names[:10])

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, name in zip(axes.ravel(), test_names[:8]):
    img = load_gray(lookups["test"][name])
    ax.imshow(img, cmap="gray")
    ax.set_title(name, fontsize=8)
    ax.axis("off")
plt.suptitle("Test image sample")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        """## Phase 0/1 checklist (fill in after you run)

| Check | What to look for | Your notes |
|-------|------------------|------------|
| Pairing | `image_only` and `mask_only` should be 0 | |
| Corrupt files | unreadable count should be 0 | |
| Shape match | ~30–40% same-shape in sample — alignment needed | |
| Apo coverage | bimodal: region (high, well above 50% coverage) vs line (low, about 1–5%) | |
| Fasc coverage | thin masks; check empty count | |
| Fasc-only 1713 | expected — extra fascicle training data | |
| Overlay fix | green/orange visible without gray cmap | |

## What comes next (Phase 2+)

- Default alignment: **stretch** (pending confirmation on more examples).
- Derive `pa_deg`, `fl_mm`, `mt_mm` from stretched masks.
- Build clean fasc subset: exclude 5 empty + 7 near-empty (0.05% threshold).
- Resolve pixel → mm calibration for FL and MT."""
    )
)

cells.append(
    code(
        """OUT = Path("/kaggle/working")
if OUT.exists():
    shape_df.to_csv(OUT / "shape_coverage_sample.csv", index=False)
    inventory.to_csv(OUT / "inventory.csv", index=False)
    if "empty_fasc_df" in globals() and len(empty_fasc_df):
        empty_fasc_df.to_csv(OUT / "empty_fasc_masks.csv", index=False)
    if "exclude_fasc" in globals() and len(exclude_fasc):
        exclude_fasc.to_csv(OUT / "exclude_fasc_masks.csv", index=False)
    # Export a few alignment figures as PNG for offline review
    fig_dir = OUT / "alignment_exports"
    fig_dir.mkdir(exist_ok=True)
    mismatch = shape_df[~shape_df["same_shape"]]
    if len(mismatch):
        row = mismatch.iloc[0]
        track, name = row["track"], row["filename"]
        img_l = lookups["apo_img" if track == "apo" else "fasc_img"]
        mask_l = lookups["apo_mask" if track == "apo" else "fasc_mask"]
        color = (255, 140, 0) if track == "apo" else (0, 200, 80)
        img = load_gray(img_l[name])
        mask = load_mask(mask_l[name])
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for row_axes, mode in zip(axes, ALIGN_MODES):
            focus_row(img, mask, track, name, mode, row_axes, color)
        plt.suptitle(f"export {track} {name}")
        plt.tight_layout()
        fig.savefig(fig_dir / f"focus_{track}_{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print("Wrote CSV summaries and alignment PNG to", OUT)
else:
    print("Not on Kaggle — skipping /kaggle/working export")
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parents[1] / "notebooks/data-audit/data-audit.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({len(cells)} cells)")
