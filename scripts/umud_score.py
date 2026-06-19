"""Official UMUD leaderboard metric (from paulritsche/umud-score on Kaggle).

Use with a ground-truth DataFrame (train geometry or hidden test labels) and a
submission-shaped prediction DataFrame. Requires no NaNs in predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLS = ("pa_deg", "fl_mm", "mt_mm")

TOLERANCES = {
    "pa_deg": 6.0,
    "fl_mm": 12.0,
    "mt_mm": 3.0,
}

WEIGHTS = {
    "pa_deg": 1.0,
    "fl_mm": 1.0,
    "mt_mm": 1.0,
}

EPS_SECONDARY = 1e-6
EPS_TERTIARY = 1e-9


class MetricError(ValueError):
    """Malformed submission for UMUD scoring."""


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> float:
    """Return official composite score (lower is better)."""
    if row_id_column_name not in solution.columns:
        raise MetricError(f"Solution missing id column '{row_id_column_name}'.")
    if row_id_column_name not in submission.columns:
        raise MetricError(f"Submission missing id column '{row_id_column_name}'.")

    for col in TARGET_COLS:
        if col not in solution.columns:
            raise MetricError(f"Solution missing ground-truth column '{col}'.")
        if col not in submission.columns:
            raise MetricError(f"Submission missing prediction column '{col}'.")

    if solution[row_id_column_name].duplicated().any():
        raise MetricError(f"Duplicate ids in solution '{row_id_column_name}'.")
    if submission[row_id_column_name].duplicated().any():
        raise MetricError(f"Duplicate ids in submission '{row_id_column_name}'.")

    merged = solution.merge(
        submission,
        on=row_id_column_name,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if len(merged) != len(solution):
        missing = len(solution) - len(merged)
        raise MetricError(f"Submission missing {missing} required row ids.")

    for col in TARGET_COLS:
        pred_col = f"{col}_pred"
        merged[pred_col] = pd.to_numeric(merged[pred_col], errors="coerce")
        if merged[pred_col].isna().any():
            raise MetricError(f"Column '{col}' must be numeric with no NaN values.")
        if np.isinf(merged[pred_col].to_numpy()).any():
            raise MetricError(f"Column '{col}' contains infinite values.")

    weight_sum = float(sum(WEIGHTS[c] for c in TARGET_COLS))
    primary = secondary = tertiary = 0.0

    for col in TARGET_COLS:
        y_true = merged[f"{col}_true"].to_numpy(dtype=float)
        y_pred = merged[f"{col}_pred"].to_numpy(dtype=float)
        tau = float(TOLERANCES[col])
        w = float(WEIGHTS[col])
        if not np.isfinite(tau) or tau <= 0:
            raise MetricError(f"Tolerance for '{col}' must be positive.")

        primary += w * float(np.mean(np.abs(y_pred - y_true))) / tau
        secondary += w * float(np.median(np.abs(y_pred - y_true))) / tau
        tertiary += w * float(np.sqrt(np.mean((y_pred - y_true) ** 2))) / tau

    primary /= weight_sum
    secondary /= weight_sum
    tertiary /= weight_sum
    return float(primary + EPS_SECONDARY * secondary + EPS_TERTIARY * tertiary)


def local_metric_report(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> pd.DataFrame:
    """Per-target MAE / MedAE / RMSE / bias for local inspection."""
    merged = solution.merge(
        submission,
        on=row_id_column_name,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    rows = []
    for col in TARGET_COLS:
        y_true = merged[f"{col}_true"].to_numpy(dtype=float)
        y_pred = merged[f"{col}_pred"].to_numpy(dtype=float)
        err = y_pred - y_true
        rows.append(
            {
                "target": col,
                "mae": float(np.mean(np.abs(err))),
                "medae": float(np.median(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "nmae": float(np.mean(np.abs(err)) / TOLERANCES[col]),
            }
        )
    return pd.DataFrame(rows)


def score_summary(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> dict:
    """Score val predictions; report partial + strict leaderboard-style metrics."""
    merged = solution.merge(
        submission,
        on=row_id_column_name,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if len(merged) == 0:
        return {
            "n_total": 0,
            "n_pred_finite": 0,
            "n_gt_finite": 0,
            "n_scorable": 0,
            "val_mt_ok_pct": float("nan"),
            "val_umud_score": float("nan"),
            "val_umud_score_strict": float("nan"),
        }

    pred_finite = merged[[f"{c}_pred" for c in TARGET_COLS]].notna().all(axis=1)
    gt_finite = merged[[f"{c}_true" for c in TARGET_COLS]].notna().all(axis=1)
    scorable = pred_finite & gt_finite

    out = {
        "n_total": int(len(merged)),
        "n_pred_finite": int(pred_finite.sum()),
        "n_gt_finite": int(gt_finite.sum()),
        "n_scorable": int(scorable.sum()),
        "val_mt_ok_pct": round(100.0 * float(pred_finite.mean()), 2),
        "val_umud_score": float("nan"),
        "val_umud_score_strict": float("nan"),
    }

    if scorable.any():
        sol = merged.loc[scorable, [row_id_column_name]].copy()
        sub = merged.loc[scorable, [row_id_column_name]].copy()
        for col in TARGET_COLS:
            sol[col] = merged.loc[scorable, f"{col}_true"].astype(float)
            sub[col] = merged.loc[scorable, f"{col}_pred"].astype(float)
        out["val_umud_score"] = round(float(score(sol, sub, row_id_column_name)), 6)

    if pred_finite.all():
        sub_all = submission[[row_id_column_name, *TARGET_COLS]].copy()
        sol_all = solution[[row_id_column_name, *TARGET_COLS]].copy()
        out["val_umud_score_strict"] = round(
            float(score(sol_all, sub_all, row_id_column_name)), 6
        )

    return out
