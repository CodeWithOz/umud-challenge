"""Local calibration sprint analysis — score sensitivity, scale estimates, cohort stats."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "calibration-sprint"
OUT.mkdir(parents=True, exist_ok=True)

TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
REF_RANGES_MM = {"fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def load_debug() -> pd.DataFrame:
    path = ROOT / "tmp/kaggle-output/submission-v7/submission_debug.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_train_geometry_sample() -> pd.DataFrame:
    path = ROOT / "tmp/geometry-local-output/geometry_sample.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def implied_scales(fl_px: float, mt_px: float, fl_mm: float, mt_mm: float) -> dict[str, float]:
    return {"mm_per_px_fl": fl_mm / fl_px, "mm_per_px_mt": mt_mm / mt_px}


def score_if_true_is(
    pred: pd.DataFrame,
    true_fl: float,
    true_mt: float,
    true_pa: float | None = None,
) -> float:
    """Rough UMUD primary score if all images shared the same GT (sanity only)."""
    pa_err = (
        float(np.mean(np.abs(pred["pa_deg"] - true_pa)))
        if true_pa is not None
        else float(np.mean(np.abs(pred["pa_deg"] - pred["pa_deg"].median())))
    )
    fl_err = float(np.mean(np.abs(pred["fl_mm"] - true_fl)))
    mt_err = float(np.mean(np.abs(pred["mt_mm"] - true_mt)))
    return (pa_err / TOLERANCES["pa_deg"] + fl_err / TOLERANCES["fl_mm"] + mt_err / TOLERANCES["mt_mm"]) / 3


def main() -> None:
    debug = load_debug()
    geo = load_train_geometry_sample()

    # --- why score ~48 with MM_PER_PIXEL=1 ---
    fl_med_px = float(debug["fl_px"].median())
    mt_med_px = float(debug["mt_px"].median())
    score_at_1 = score_if_true_is(debug, true_fl=75.0, true_mt=20.0, true_pa=10.0)

    scales = np.linspace(0.04, 0.14, 41)
    sweep = []
    for s in scales:
        fl_mm = debug["fl_px"] * s
        mt_mm = debug["mt_px"] * s
        tmp = debug.assign(fl_mm=fl_mm, mt_mm=mt_mm)
        sweep.append(
            {
                "mm_per_pixel": round(float(s), 4),
                "fl_mm_median": round(float(fl_mm.median()), 2),
                "mt_mm_median": round(float(mt_mm.median()), 2),
                "score_vs_fl75_mt20": round(score_if_true_is(tmp, 75.0, 20.0, 10.0), 3),
            }
        )
    sweep_df = pd.DataFrame(sweep)
    best = sweep_df.loc[sweep_df["score_vs_fl75_mt20"].idxmin()]

    # --- train GT pixel stats by resolution cohort ---
    geo["cohort"] = geo.apply(
        lambda r: f"{int(r.img_h)}x{int(r.img_w)}" if "img_h" in geo.columns else "unknown",
        axis=1,
    )
    cohort = (
        geo.groupby("cohort")[["fl_px", "mt_px", "pa_deg"]]
        .agg(["median", "mean", "count"])
        .round(2)
    )

    # --- heuristic scale from ref-range midpoints on TRAIN GT geometry ---
    fl_mid = sum(REF_RANGES_MM["fl_mm"]) / 2
    mt_mid = sum(REF_RANGES_MM["mt_mm"]) / 2
    heuristic = {
        "mm_per_px_fl_from_train_gt": fl_mid / float(geo["fl_px"].median()),
        "mm_per_px_mt_from_train_gt": mt_mid / float(geo["mt_px"].median()),
        "mm_per_px_fl_from_test_pred": fl_mid / fl_med_px,
        "mm_per_px_mt_from_test_pred": mt_mid / mt_med_px,
    }

    summary = {
        "leaderboard_baseline_v7": 48.18203,
        "mm_per_pixel_current": 1.0,
        "test_pred_medians_px": {"fl_px": fl_med_px, "mt_px": mt_med_px, "pa_deg": float(debug["pa_deg"].median())},
        "train_gt_medians_px": {
            "fl_px": float(geo["fl_px"].median()),
            "mt_px": float(geo["mt_px"].median()),
            "pa_deg": float(geo["pa_deg"].median()),
        },
        "score_tolerances": TOLERANCES,
        "rough_score_at_mm_per_px_1_vs_ref_gt": round(score_at_1, 2),
        "best_uniform_scale_vs_fl75_mt20_pa10": best.to_dict(),
        "heuristic_scales": {k: round(v, 5) for k, v in heuristic.items()},
        "notes": [
            "No train CSV with expert fl_mm/mt_mm in competition bundle — calibration must be inferred.",
            "TIFF tags on sample test image: only width/height (no spacing).",
            "DLTrack uses manual per-image scaling — check ultrasound depth ruler in margins.",
            "FL and MT may need separate scales if geometry protocol differs from manual 3-point MT.",
        ],
    }

    sweep_df.to_csv(OUT / "scale_sweep.csv", index=False)
    cohort.to_csv(OUT / "train_gt_by_resolution.csv")
    (OUT / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
