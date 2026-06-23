"""Heuristic UMUD geometry from ultrasound pixels, adapted from AmbrosM's
public "UMUD Quick and Dirty" Kaggle notebook.

This path does not use segmentation masks. It extracts the ultrasound viewport,
recovers per-image pixel scale from tick marks, estimates aponeurosis depths from
horizontal brightness profiles, estimates fascicle slope by patch correlation,
then derives FL = MT / sin(PA).
"""
from __future__ import annotations

import math
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


def _mean_col(arr: np.ndarray, col: int) -> np.ndarray:
    v = arr[:, col]
    if v.ndim == 2:
        return v.mean(axis=-1)
    return v.astype(float)


def collect_metadata(competition_dir: str | Path, pattern: str) -> pd.DataFrame:
    """Determine px_per_cm and ultrasound viewport bbox for matching images."""
    competition_dir = str(competition_dir)
    rows = []
    for path in sorted(glob(f"{competition_dir}/{pattern}")):
        px_per_cm = -1.0
        l, t, r, b = -1, -1, -1, -1
        image_num = int(path[-8:-4])
        filetype = path[-3:]
        with Image.open(path) as img:
            a = np.asarray(img)
        if len(a.shape) == 3:
            assert a.shape[2] == 3
        height, width = a.shape[:2]

        if filetype == "png":
            first_tick = int(np.argmax(_mean_col(a, 6) > 50))
            second_minor_tick = 150 + int(np.argmax(_mean_col(a[150:], 6) > 50))
            second_major_tick = 150 + int(np.argmax(_mean_col(a[150:], 9) > 50))
            last_tick = len(a) - 1 - int(np.argmax((_mean_col(a, 6)[::-1] > 50)))
            if (second_major_tick - first_tick) < 3 * (second_minor_tick - first_tick):
                px_per_cm = float(second_major_tick - first_tick)
            else:
                px_per_cm = float(second_minor_tick - first_tick)
            hw = a.shape[1] // 2
            right = a[:, hw:]
            col_sum = right.sum(axis=(0, 2)) if right.ndim == 3 else right.sum(axis=0)
            w2 = int(np.argmin(col_sum))
            l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
        elif a.shape == (800, 1200, 3):
            if (a[87, 1147:1157] == 175).all():
                first_tick = int(np.argmax(_mean_col(a, 1150) > 50))
                second_major_tick = first_tick + 20 + int(np.argmax(_mean_col(a[first_tick + 20:], 1150) > 50))
                last_tick = len(a) - 1 - int(np.argmax((_mean_col(a, 1150)[::-1] > 50)))
                n_ticks = round((last_tick - first_tick) / (second_major_tick - first_tick))
                if n_ticks <= 14:
                    px_per_cm = float((last_tick - first_tick) / n_ticks * 2)
                    bbox_by_ticks = {
                        7: (142, 91, 1058),
                        8: (163, 91, 1037),
                        9: (211, 91, 989),
                        10: (249, 91, 951),
                        11: (282, 91, 918),
                        12: (308, 91, 892),
                        13: (331, 91, 869),
                        14: (349, 91, 851),
                    }
                    if n_ticks not in bbox_by_ticks:
                        raise AssertionError(f"Unexpected 800x1200 right-tick count: {n_ticks}")
                    l, t, r = bbox_by_ticks[n_ticks]
                    b = last_tick
                elif n_ticks == 15:
                    px_per_cm = float((last_tick - first_tick) / 3)
                    l, t, r, b = 142, 91, 1058, last_tick
                else:
                    raise AssertionError(f"Unexpected 800x1200 n_ticks: {n_ticks}")
            elif (a[42, 67:74, 0] > 115).all():
                px_per_cm = float((783 - 42) / 5)
                l, t, r, b = 171, 42, 1029, 798
            else:
                raise AssertionError("Unknown 800x1200 tick layout")
        elif a.shape == (644, 1088, 3):
            px_per_cm = float(630.5 / 5)
            l, t, r, b = 140, 0, 947, 643
        elif a.shape[0] in [512, 513]:
            if (
                (a[-5, 49, 0] == 168 and a[-5, 438, 0] == 168)
                or (a[-5, 52, 0] == 168 and a[-5, 441, 0] == 168)
                or (a[-5, 53, 0] == 168 and a[-5, 442, 0] == 168)
            ):
                px_per_cm = float((442 - 53) / 5)
            else:
                raise AssertionError("Unknown 512/513 tick layout")
            l, t, r, b = 0, 0, width, height - 10
        elif a.shape[0] == 853:
            assert len(a.shape) == 2
            if a[-5, 100] == 170 and a[-5, 934] == 170:
                px_per_cm = float((934 - 100) / 5)
            elif a[-5, 44] == 170 and a[-5, 879] == 170:
                px_per_cm = float((879 - 44) / 5)
            else:
                raise AssertionError("Unknown 853 tick layout")
            l, t, r, b = 0, 0, width, height
        else:
            raise AssertionError(f"Unknown format: {a.shape}")

        rows.append((image_num, Path(path).name, filetype, height, width, len(a.shape), px_per_cm, l, t, r, b))

    return pd.DataFrame(
        rows,
        columns=["id", "image_id", "filetype", "height", "width", "dim", "px_per_cm", "l", "t", "r", "b"],
    )


def estimate_measurements_for_row(competition_dir: str | Path, row: pd.Series) -> dict[str, float]:
    """Estimate PA/FL/MT for one metadata row."""
    path = Path(competition_dir) / "test_images_v2/test_set_v2" / str(row.image_id)
    with Image.open(path) as img:
        a = np.asarray(img)

    b = a[int(row.t) : int(row.b), int(row.l) : int(row.r)]
    if len(b.shape) == 3:
        b = b.mean(axis=-1)
    px_per_cm = float(row.px_per_cm)
    mm_per_px = 10.0 / px_per_cm

    b0 = b.mean(axis=1)
    window = min(31, len(b0) - (1 - len(b0) % 2))
    window = max(5, window if window % 2 else window - 1)
    b1 = savgol_filter(b0, window, 3)
    sup_stop = min(int(2.0 * px_per_cm), len(b1) - int(px_per_cm))
    superficial_depth_px = int(np.argmax(b1[:sup_stop]))
    min_deep_depth_px = superficial_depth_px + int(px_per_cm)
    max_deep_depth_px = min(superficial_depth_px + 5 * int(px_per_cm), len(b1))
    deep_depth_px = min_deep_depth_px + int(np.argmax(b1[min_deep_depth_px:max_deep_depth_px]))
    mt_px = deep_depth_px - superficial_depth_px

    middle_depth_px = (superficial_depth_px + deep_depth_px) // 2
    dx = 25
    ly = int(min(50, mt_px // 2 - dx))
    best_dys = []
    if ly >= 2:
        for x in range(0, b.shape[1] - dx, 25):
            y0 = max(0, middle_depth_px - ly)
            y1 = min(b.shape[0], middle_depth_px + ly)
            s_left = b[y0:y1, x]
            if len(s_left) < 2 or s_left.var() == 0:
                continue
            best_corr, best_dy = -np.inf, None
            for dy in range(-dx, dx):
                yy0 = y0 + dy
                yy1 = y1 + dy
                if yy0 < 0 or yy1 > b.shape[0]:
                    continue
                s_right = b[yy0:yy1, x + dx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    corr = np.corrcoef(s_left, s_right)[1, 0]
                if np.isfinite(corr) and corr > best_corr:
                    best_corr, best_dy = corr, dy
            if best_dy is not None:
                best_dys.append(best_dy)

    if len(best_dys) >= 5:
        best_dy = float(np.median(best_dys))
        pa_rad = abs(math.atan(best_dy / dx))
        pa_deg = min(pa_rad / math.pi * 180.0, 45.0)
    else:
        best_dy = np.nan
        pa_rad = np.nan
        pa_deg = np.nan

    if np.isfinite(pa_deg) and pa_deg >= 5:
        fl_px = mt_px / math.sin(pa_rad)
    else:
        pa_deg = 15.0
        pa_rad = math.asin(0.34)
        fl_px = mt_px / 0.34

    return {
        "superficial_depth_mm": superficial_depth_px * mm_per_px,
        "deep_depth_mm": deep_depth_px * mm_per_px,
        "mt_mm": mt_px * mm_per_px,
        "fl_mm": float(np.clip(fl_px * mm_per_px, 30.0, 200.0)),
        "pa_deg": float(pa_deg),
        "mm_per_px": mm_per_px,
        "superficial_depth_px": float(superficial_depth_px),
        "deep_depth_px": float(deep_depth_px),
        "mt_px_qd": float(mt_px),
        "pa_best_dy": float(best_dy) if np.isfinite(best_dy) else np.nan,
    }


def predict_quickdirty(competition_dir: str | Path, pattern: str = "test_images_v2/test_set_v2/IMG_*.*") -> pd.DataFrame:
    """Return quick-dirty predictions and diagnostics for all matching images."""
    meta = collect_metadata(competition_dir, pattern)
    preds = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="quickdirty geometry"):
        pred = estimate_measurements_for_row(competition_dir, row)
        preds.append(pred)
    pred_df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(preds)], axis=1)
    assert pred_df[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all()
    return pred_df


def write_submission(pred_df: pd.DataFrame, out_csv: str | Path) -> None:
    submit = pred_df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
    submit.to_csv(out_csv, index=False)
