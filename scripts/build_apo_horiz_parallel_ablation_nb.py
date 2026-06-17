"""Generate notebooks/apo-horiz-parallel-ablation — xspan_pair vs horizontality+parallelism picker."""
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
        """# UMUD — Horizontality + Parallelism Picker Ablation (Phase 3)

**GPU notebook** — on the **62 legacy `no_x_overlap` cohort** (same images as `rescued/` gallery), compare:

| Picker | Rule |
|--------|------|
| **xspan_pair** | Top-K by x-span; max x-overlap pair |
| **horiz_parallel** | Top-K by `x_span × √area × horizontality`; pair score = `overlap × horizontality(sup) × horizontality(deep) × parallelism` |

User QC tags (from manual review):
- **user_flagged** (12): `IMG_00001`, `IMG_00039`, `IMG_00111`–`IMG_00115`, `IMG_00131`–`IMG_00135`
- **user_good** (49): remaining 49 in this 62-image cohort

Panels per case (2×4):
1. raw+bbox, gray55, pred mask, overlay
2. xspan_pair edges, xspan x-ranges, horiz_parallel edges, horiz x-ranges

Outputs:
- `/kaggle/working/horiz_parallel_ablation.csv`
- `/kaggle/working/horiz_parallel_ablation_summary.json`
- Figures under `/kaggle/working/figures/horiz_parallel_ablation/`"""
    ),
    md("## Configuration"),
    code(
        """import json
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

MASK_OVERLAY_ALPHA = 0.55
APO_OVERLAY_COLOR = (255, 140, 0)

FIG_ROOT = Path("/kaggle/working/figures/horiz_parallel_ablation")
FIG_FLAGGED = FIG_ROOT / "user_flagged"
FIG_GOOD = FIG_ROOT / "user_good"
FIG_CHANGED = FIG_ROOT / "user_good_changed_pick"
for d in (FIG_ROOT, FIG_FLAGGED, FIG_GOOD, FIG_CHANGED):
    d.mkdir(parents=True, exist_ok=True)

USER_FLAGGED = {
    "IMG_00001.tif",
    "IMG_00039.tif",
    "IMG_00111.tif",
    "IMG_00112.tif",
    "IMG_00113.tif",
    "IMG_00114.tif",
    "IMG_00115.tif",
    "IMG_00131.tif",
    "IMG_00132.tif",
    "IMG_00133.tif",
    "IMG_00134.tif",
    "IMG_00135.tif",
}

# rescued folder index for cross-reference
RESCUED_INDEX = {
    "IMG_00001.tif": 1,
    "IMG_00039.tif": 13,
    "IMG_00111.tif": 29,
    "IMG_00112.tif": 30,
    "IMG_00113.tif": 31,
    "IMG_00114.tif": 32,
    "IMG_00115.tif": 33,
    "IMG_00131.tif": 42,
    "IMG_00132.tif": 43,
    "IMG_00133.tif": 44,
    "IMG_00134.tif": 45,
    "IMG_00135.tif": 46,
}


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
    return {"area": area, "x_span": x_span, "y_top": y, "y_bot": y + h, "w": w, "h": h}


def x_overlap_from_contours(sup_c: np.ndarray, deep_c: np.ndarray) -> float:
    sup_x, _ = edge_polyline(sup_c, which="bottom")
    deep_x, _ = edge_polyline(deep_c, which="top")
    if len(sup_x) == 0 or len(deep_x) == 0:
        return 0.0
    return max(0.0, min(float(sup_x.max()), float(deep_x.max())) - max(float(sup_x.min()), float(deep_x.min())))


def edge_angle_from_horizontal(contour: np.ndarray, which: str) -> float | None:
    xs, ys = edge_polyline(contour, which=which)
    if len(xs) < 2:
        return None
    line = fit_line(xs, ys)
    if line is None:
        return None
    ang = abs(float(np.degrees(np.arctan(line[1])))) % 180.0
    return float(min(ang, 180.0 - ang))


def horizontality_factor(angle_deg: float | None) -> float:
    if angle_deg is None:
        return 0.0
    return float(np.cos(np.radians(angle_deg)) ** 2)


def parallelism_factor(sup_ang: float | None, deep_ang: float | None) -> float:
    if sup_ang is None or deep_ang is None:
        return 0.0
    d = abs(sup_ang - deep_ang) % 180.0
    d = min(d, 180.0 - d)
    return float(np.cos(np.radians(d)) ** 2)


def contour_role_score(c: np.ndarray, which: str) -> float:
    f = contour_feats(c)
    ang = edge_angle_from_horizontal(c, which)
    horiz = horizontality_factor(ang)
    return f["x_span"] * float(np.sqrt(max(f["area"], 1.0))) * (0.25 + 0.75 * horiz)


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


def pick_horiz_parallel(contours: list[np.ndarray], min_sep_px: int = MIN_SEP_PX, top_k: int = TOP_K_CANDIDATES):
    if len(contours) < 2:
        return None, None
    ranked = sorted(contours, key=lambda c: max(contour_role_score(c, "bottom"), contour_role_score(c, "top")), reverse=True)
    candidates = ranked[: min(top_k, len(ranked))]
    best_pair = None
    best_score = -1.0
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
            if overlap <= 0:
                continue
            sup_ang = edge_angle_from_horizontal(sup_c, "bottom")
            deep_ang = edge_angle_from_horizontal(deep_c, "top")
            score = (
                overlap
                * horizontality_factor(sup_ang)
                * horizontality_factor(deep_ang)
                * parallelism_factor(sup_ang, deep_ang)
            )
            if score > best_score:
                best_score = score
                best_pair = (sup_c, deep_c)
    if best_pair is not None:
        return best_pair
    # fallback to xspan if no overlapping horiz-parallel pair
    return pick_best_pair_xspan(contours, min_sep_px=min_sep_px, top_k=top_k)


def apo_geometry_with_picker(apo_mask: np.ndarray, style: str, picker: str) -> dict:
    eff, method = effective_apo_mask(apo_mask, style)
    contours = find_apo_contours(eff)
    if picker == "xspan_pair":
        sup_c, deep_c = pick_best_pair_xspan(contours)
    elif picker == "horiz_parallel":
        sup_c, deep_c = pick_horiz_parallel(contours)
    else:
        sup_c, deep_c, _ = pick_superficial_deep(contours)
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
        "sup_edge_ang": None,
        "deep_edge_ang": None,
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
    out.update(
        sup_line=sup_line,
        deep_line=deep_line,
        sup_xs=sup_x,
        sup_ys=sup_y,
        deep_xs=deep_x,
        deep_ys=deep_y,
        sup_edge_ang=edge_angle_from_horizontal(sup_c, "bottom"),
        deep_edge_ang=edge_angle_from_horizontal(deep_c, "top"),
    )
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
    sup_x, deep_x = geo.get("sup_xs"), geo.get("deep_xs")
    if sup_x is None or deep_x is None or len(sup_x) == 0 or len(deep_x) == 0:
        return {"overlap_px": 0.0, "gap_px": np.nan, "sup_xmin": np.nan, "sup_xmax": np.nan, "deep_xmin": np.nan, "deep_xmax": np.nan}
    sup_lo, sup_hi = float(sup_x.min()), float(sup_x.max())
    deep_lo, deep_hi = float(deep_x.min()), float(deep_x.max())
    overlap = max(0.0, min(sup_hi, deep_hi) - max(sup_lo, deep_lo))
    gap = 0.0 if overlap > 0 else max(sup_lo, deep_lo) - min(sup_hi, deep_hi)
    return {"sup_xmin": sup_lo, "sup_xmax": sup_hi, "deep_xmin": deep_lo, "deep_xmax": deep_hi, "overlap_px": overlap, "gap_px": gap}


def pick_changed(x_geo: dict, h_geo: dict, x_stats: dict, h_stats: dict) -> bool:
    if not (np.isfinite(x_geo["mt_px"]) and np.isfinite(h_geo["mt_px"])):
        return True
    sup_shift = abs(x_stats["sup_xmin"] - h_stats["sup_xmin"]) + abs(x_stats["sup_xmax"] - h_stats["sup_xmax"])
    deep_shift = abs(x_stats["deep_xmin"] - h_stats["deep_xmin"]) + abs(x_stats["deep_xmax"] - h_stats["deep_xmax"])
    ang_shift = abs((x_geo.get("sup_edge_ang") or 0) - (h_geo.get("sup_edge_ang") or 0)) + abs(
        (x_geo.get("deep_edge_ang") or 0) - (h_geo.get("deep_edge_ang") or 0)
    )
    return bool(sup_shift > 80 or deep_shift > 80 or ang_shift > 8)


def compare_panel(case: dict, idx: int, group: str, fig_dir: Path):
    img, img_g, mask, bbox = case["img_raw"], case["img_gray55"], case["pred_mask"], case["bbox"]
    x_geo, h_geo = case["xspan_geo"], case["horiz_geo"]
    x_stats, h_stats = case["xspan_stats"], case["horiz_stats"]
    y0, y1, x0, x1 = bbox
    rescued_n = RESCUED_INDEX.get(case["image_id"], "?")

    fig, axes = plt.subplots(2, 4, figsize=(22, 8))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2))
    axes[0, 0].set_title("raw + bbox", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_g, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("gray55", fontsize=9)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"pred mask\\nn_ct={x_geo['n_contours']}", fontsize=9)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(overlay(img_g, mask))
    axes[0, 3].set_title("overlay", fontsize=9)
    axes[0, 3].axis("off")

    for ax, geo, stats, title in [
        (axes[1, 0], x_geo, x_stats, "xspan_pair"),
        (axes[1, 2], h_geo, h_stats, "horiz_parallel"),
    ]:
        ax.imshow(img_g, cmap="gray")
        if geo.get("sup_xs") is not None and len(geo["sup_xs"]):
            ax.scatter(geo["sup_xs"], geo["sup_ys"], s=4, c="cyan")
            xs = np.linspace(geo["sup_xs"].min(), geo["sup_xs"].max(), 50)
            if geo.get("sup_line") is not None:
                ax.plot(xs, geo["sup_line"](xs), c="cyan", lw=2)
        if geo.get("deep_xs") is not None and len(geo["deep_xs"]):
            ax.scatter(geo["deep_xs"], geo["deep_ys"], s=4, c="magenta")
            xs = np.linspace(geo["deep_xs"].min(), geo["deep_xs"].max(), 50)
            if geo.get("deep_line") is not None:
                ax.plot(xs, geo["deep_line"](xs), c="magenta", lw=2)
        mt = f"{geo['mt_px']:.0f}px" if np.isfinite(geo["mt_px"]) else "NaN"
        ax.set_title(f"{title}\\n{geo['mt_fail_reason']} mt={mt}", fontsize=9)
        ax.axis("off")

    def x_panel(ax, stats, title):
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.75, 0.25])
        ax.set_yticklabels(["sup", "deep"], fontsize=8)
        if np.isfinite(stats["sup_xmin"]):
            ax.barh(0.75, stats["sup_xmax"] - stats["sup_xmin"], left=stats["sup_xmin"], height=0.2, color="cyan", alpha=0.8)
        if np.isfinite(stats["deep_xmin"]):
            ax.barh(0.25, stats["deep_xmax"] - stats["deep_xmin"], left=stats["deep_xmin"], height=0.2, color="magenta", alpha=0.8)
        ax.set_title(title, fontsize=9)

    x_panel(axes[1, 1], x_stats, "xspan x-ranges")
    x_panel(axes[1, 3], h_stats, "horiz x-ranges")

    plt.suptitle(
        f"[{group} {idx:03d} rescued#{rescued_n}] {case['image_id']}  "
        f"xspan mt={x_geo['mt_px'] if np.isfinite(x_geo['mt_px']) else 'NaN'}  "
        f"horiz mt={h_geo['mt_px'] if np.isfinite(h_geo['mt_px']) else 'NaN'}  "
        f"changed_pick={case['pick_changed']}",
        y=1.02,
        fontsize=10,
    )
    plt.tight_layout()
    out = fig_dir / f"{group}_{idx:03d}_{Path(case['image_id']).stem}.png"
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
            """# First pass all test images with legacy picker to find the 62-image cohort
legacy_rows = []
for path in tqdm(test_paths, desc="find legacy no_x_overlap cohort"):
    img = load_gray(path)
    h, w = img.shape
    img_g, bbox = preprocess_gray55(img)
    _, apo_t, _ = line_learn.predict(open_rgb_256(img_g))
    mask = clip_mask_to_bbox(resize_mask_to(tensor_to_mask(apo_t), h, w), bbox)
    style = tag_apo_style(float(mask.mean()))
    legacy = apo_geometry_with_picker(mask, style, picker="legacy")
    if legacy["mt_fail_reason"] == "no_x_overlap":
        legacy_rows.append(path.name)

cohort_ids = sorted(legacy_rows)
print(f"Legacy no_x_overlap cohort: {len(cohort_ids)}")
assert len(cohort_ids) == 62, f"Expected 62, got {len(cohort_ids)}"
"""
        ),
        code(
            """rows = []
cases = []

for image_id in tqdm(cohort_ids, desc="xspan vs horiz_parallel"):
    path = TEST_DIR / image_id
    img = load_gray(path)
    h, w = img.shape
    img_g, bbox = preprocess_gray55(img)
    _, apo_t, _ = line_learn.predict(open_rgb_256(img_g))
    mask = clip_mask_to_bbox(resize_mask_to(tensor_to_mask(apo_t), h, w), bbox)
    style = tag_apo_style(float(mask.mean()))

    x_geo = apo_geometry_with_picker(mask, style, picker="xspan_pair")
    h_geo = apo_geometry_with_picker(mask, style, picker="horiz_parallel")
    x_stats = x_overlap_stats(x_geo)
    h_stats = x_overlap_stats(h_geo)

    x_ok = bool(np.isfinite(x_geo["mt_px"]))
    h_ok = bool(np.isfinite(h_geo["mt_px"]))
    user_tag = "user_flagged" if image_id in USER_FLAGGED else "user_good"
    changed = pick_changed(x_geo, h_geo, x_stats, h_stats)

    rows.append(
        {
            "image_id": image_id,
            "rescued_index": RESCUED_INDEX.get(image_id),
            "user_tag": user_tag,
            "xspan_mt_ok": x_ok,
            "horiz_mt_ok": h_ok,
            "xspan_mt_px": float(x_geo["mt_px"]) if x_ok else np.nan,
            "horiz_mt_px": float(h_geo["mt_px"]) if h_ok else np.nan,
            "xspan_fail": x_geo["mt_fail_reason"],
            "horiz_fail": h_geo["mt_fail_reason"],
            "xspan_sup_ang": x_geo.get("sup_edge_ang"),
            "xspan_deep_ang": x_geo.get("deep_edge_ang"),
            "horiz_sup_ang": h_geo.get("sup_edge_ang"),
            "horiz_deep_ang": h_geo.get("deep_edge_ang"),
            "pick_changed": changed,
            "mt_px_delta": float(h_geo["mt_px"] - x_geo["mt_px"]) if x_ok and h_ok else np.nan,
        }
    )

    cases.append(
        {
            "image_id": image_id,
            "user_tag": user_tag,
            "img_raw": img,
            "img_gray55": img_g,
            "pred_mask": mask,
            "bbox": bbox,
            "xspan_geo": x_geo,
            "horiz_geo": h_geo,
            "xspan_stats": x_stats,
            "horiz_stats": h_stats,
            "pick_changed": changed,
        }
    )

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/horiz_parallel_ablation.csv", index=False)

flagged = df[df.user_tag == "user_flagged"]
good = df[df.user_tag == "user_good"]

summary = {
    "n_cohort": int(len(df)),
    "xspan_mt_ok": int(df.xspan_mt_ok.sum()),
    "horiz_mt_ok": int(df.horiz_mt_ok.sum()),
    "user_flagged_n": int(len(flagged)),
    "user_good_n": int(len(good)),
    "flagged_pick_changed": int(flagged.pick_changed.sum()),
    "good_pick_changed": int(good.pick_changed.sum()),
    "good_broken_by_horiz": int(((good.xspan_mt_ok) & (~good.horiz_mt_ok)).sum()),
    "good_both_ok_pick_changed": int(((good.xspan_mt_ok) & (good.horiz_mt_ok) & (good.pick_changed)).sum()),
    "good_both_ok_unchanged": int(((good.xspan_mt_ok) & (good.horiz_mt_ok) & (~good.pick_changed)).sum()),
    "horiz_fail_counts": df.loc[~df.horiz_mt_ok, "horiz_fail"].value_counts().to_dict(),
}
with open("/kaggle/working/horiz_parallel_ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\\nUser-flagged rows:")
print(flagged[["rescued_index", "image_id", "xspan_mt_ok", "horiz_mt_ok", "pick_changed", "xspan_sup_ang", "horiz_sup_ang", "xspan_deep_ang", "horiz_deep_ang"]].to_string(index=False))
"""
        ),
        md("## Gallery A — user-flagged (manual QC concern)"),
        code(
            """flagged_cases = [c for c in cases if c["user_tag"] == "user_flagged"]
saved = []
for i, case in enumerate(flagged_cases, start=1):
    out = compare_panel(case, i, "flagged", FIG_FLAGGED)
    saved.append(str(out))
print(f"Saved {len(saved)} flagged figures")
"""
        ),
        md("## Gallery B — user-good (should stay stable)"),
        code(
            """good_cases = [c for c in cases if c["user_tag"] == "user_good"]
saved = []
for i, case in enumerate(good_cases, start=1):
    out = compare_panel(case, i, "good", FIG_GOOD)
    saved.append(str(out))
print(f"Saved {len(saved)} user-good figures")
"""
        ),
        md("## Gallery C — user-good where horiz_parallel changed the pick"),
        code(
            """changed_good = [c for c in good_cases if c["pick_changed"]]
print(f"user-good with changed pick: {len(changed_good)}")
saved = []
for i, case in enumerate(changed_good, start=1):
    out = compare_panel(case, i, "changed", FIG_CHANGED)
    saved.append(str(out))
print(f"Saved {len(saved)} changed-good figures under {FIG_CHANGED}")
"""
        ),
    ]
)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks/apo-horiz-parallel-ablation"
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
    (out / "apo-horiz-parallel-ablation-phase-3.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "ucheozoemena/umud-apo-horiz-parallel-ablation-phase-3",
                "title": "UMUD Apo Horiz Parallel Ablation Phase 3",
                "code_file": "apo-horiz-parallel-ablation-phase-3.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": ["ucheozoemena/umud-train-apo-gray55-phase-3"],
                "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
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
