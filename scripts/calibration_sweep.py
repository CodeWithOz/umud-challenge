"""Phase 4 Block 1 — regenerate submission CSVs from fixed pixel geometry at multiple mm scales.

Reads v7 submission_debug.csv (MM_PER_PIXEL=1 → fl_mm == fl_px). Writes one submission
per calibration policy under tmp/kaggle-output/calibration-sweep/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = ROOT / "tmp/kaggle-output/submission-v7/submission_debug.csv"
OUT_DIR = ROOT / "tmp/kaggle-output/calibration-sweep"

TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
REF_RANGES = {"fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
REF_MID = {k: sum(v) / 2 for k, v in REF_RANGES.items()}


@dataclass(frozen=True)
class Policy:
    slug: str
    description: str
    fl_scale: float | str  # float or "cohort_fl"
    mt_scale: float | str  # float or "cohort_mt"


def load_debug() -> pd.DataFrame:
    if not DEBUG_PATH.exists():
        raise FileNotFoundError(f"Missing {DEBUG_PATH} — run submission kernel v7 first.")
    df = pd.read_csv(DEBUG_PATH)
    for col in ("pa_deg", "fl_px", "mt_px"):
        if col not in df.columns:
            raise ValueError(f"debug CSV missing {col}")
    return df


def cohort_scales_from_train_sample() -> dict[tuple[int, int], dict[str, float]]:
    """Ref-range midpoint / train GT median px per resolution cohort (200-pair sample)."""
    path = ROOT / "tmp/geometry-local-output/geometry_sample.csv"
    if not path.exists():
        return {}
    geo = pd.read_csv(path)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for (h, w), sub in geo.groupby(["img_h", "img_w"]):
        out[(int(h), int(w))] = {
            "fl": REF_MID["fl_mm"] / float(sub["fl_px"].median()),
            "mt": REF_MID["mt_mm"] / float(sub["mt_px"].median()),
        }
    return out


def resolve_scale(
    spec: float | str,
    row: pd.Series,
    cohort_lut: dict[tuple[int, int], dict[str, float]],
    kind: str,
    fallback: float,
) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if spec == "cohort_fl":
        kind = "fl"
    elif spec == "cohort_mt":
        kind = "mt"
    key = (int(row["img_h"]), int(row["img_w"]))
    if key in cohort_lut:
        return cohort_lut[key][kind]
    # Nearest train cohort by height (letterbox vs native)
    if cohort_lut:
        nearest = min(cohort_lut, key=lambda k: abs(k[0] - key[0]) + abs(k[1] - key[1]))
        return cohort_lut[nearest][kind]
    return fallback


def apply_policy(debug: pd.DataFrame, policy: Policy, cohort_lut: dict) -> pd.DataFrame:
    rows = []
    for _, r in debug.iterrows():
        fl_s = resolve_scale(policy.fl_scale, r, cohort_lut, "fl", 0.098)
        mt_s = resolve_scale(policy.mt_scale, r, cohort_lut, "mt", 0.098)
        rows.append(
            {
                "image_id": r["image_id"],
                "pa_deg": r["pa_deg"],
                "fl_mm": r["fl_px"] * fl_s,
                "mt_mm": r["mt_px"] * mt_s,
            }
        )
    return pd.DataFrame(rows)


def in_ref_range(sub: pd.DataFrame, col: str) -> float:
    lo, hi = REF_RANGES[col]
    series = sub[col]
    return float(((series >= lo) & (series <= hi)).mean())


def ref_midpoint_penalty(sub: pd.DataFrame) -> float:
    """Lower is better — normalized distance from ref midpoints (proxy for plausibility)."""
    fl_pen = float(np.mean(np.abs(sub["fl_mm"] - REF_MID["fl_mm"]) / TOLERANCES["fl_mm"]))
    mt_pen = float(np.mean(np.abs(sub["mt_mm"] - REF_MID["mt_mm"]) / TOLERANCES["mt_mm"]))
    pa_pen = float(np.mean(np.abs(sub["pa_deg"] - 25.0) / TOLERANCES["pa_deg"]))  # PA ref mid ~25°
    return (fl_pen + mt_pen + pa_pen) / 3


def build_policies() -> list[Policy]:
    uniforms = [0.08, 0.09, 0.098, 0.10, 0.11, 0.12, 0.135]
    policies = [
        Policy(f"uniform-{s:.3f}".replace(".", "p"), f"Uniform MM={s}", s, s) for s in uniforms
    ]
    policies.extend(
        [
            Policy(
                "split-gt-midpoint",
                "FL=0.135 MT=0.104 (ref midpoint / train GT median px)",
                0.135,
                0.104,
            ),
            Policy(
                "split-test-midpoint",
                "FL=0.136 MT=0.111 (ref midpoint / test pred median px)",
                0.136,
                0.111,
            ),
            Policy(
                "cohort-train-gt",
                "Per-resolution FL/MT from train geometry sample; unknown → 0.098",
                "cohort_fl",
                "cohort_mt",
            ),
            Policy(
                "cohort-depth-800",
                "800×1200 → 0.20 uniform; other cohorts → 0.098 (depth-strip heuristic from log)",
                "cohort_depth",
                "cohort_depth",
            ),
        ]
    )
    return policies


def cohort_depth_scale(row: pd.Series) -> float:
    h, w = int(row["img_h"]), int(row["img_w"])
    if h == 800 and w == 1200:
        return 0.20
    return 0.098


def apply_policy_with_depth(debug: pd.DataFrame, policy: Policy, cohort_lut: dict) -> pd.DataFrame:
    if policy.fl_scale == "cohort_depth":
        rows = []
        for _, r in debug.iterrows():
            s = cohort_depth_scale(r)
            rows.append(
                {
                    "image_id": r["image_id"],
                    "pa_deg": r["pa_deg"],
                    "fl_mm": r["fl_px"] * s,
                    "mt_mm": r["mt_px"] * s,
                }
            )
        return pd.DataFrame(rows)
    return apply_policy(debug, policy, cohort_lut)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    debug = load_debug()
    cohort_lut = cohort_scales_from_train_sample()

    results = []
    for policy in build_policies():
        sub = apply_policy_with_depth(debug, policy, cohort_lut)
        slug_dir = OUT_DIR / policy.slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(slug_dir / "submission.csv", index=False)

        row = {
            "slug": policy.slug,
            "description": policy.description,
            "n_rows": len(sub),
            "fl_mm_median": round(float(sub["fl_mm"].median()), 2),
            "mt_mm_median": round(float(sub["mt_mm"].median()), 2),
            "pa_deg_median": round(float(sub["pa_deg"].median()), 2),
            "fl_in_ref_pct": round(100 * in_ref_range(sub, "fl_mm"), 1),
            "mt_in_ref_pct": round(100 * in_ref_range(sub, "mt_mm"), 1),
            "ref_midpoint_penalty": round(ref_midpoint_penalty(sub), 4),
            "nan_rows": int(sub.isna().any(axis=1).sum()),
        }
        results.append(row)
        print(f"{policy.slug}: FL med={row['fl_mm_median']} MT med={row['mt_mm_median']} "
              f"penalty={row['ref_midpoint_penalty']}")

    res_df = pd.DataFrame(results).sort_values("ref_midpoint_penalty")
    res_df.to_csv(OUT_DIR / "sweep_results.csv", index=False)

    # Rank uniform policies separately for submit shortlist
    uniform = res_df[res_df["slug"].str.startswith("uniform-")].copy()
    best_uniform = uniform.iloc[0] if len(uniform) else None

    shortlist = []
    # Best uniform not already on leaderboard (v9 = 0.098)
    for slug in ("uniform-0p110", "uniform-0p120", "uniform-0p100", "uniform-0p090"):
        if slug in res_df["slug"].values:
            shortlist.append(slug)
            break

    for slug in ("split-gt-midpoint", "split-test-midpoint"):
        if slug in res_df["slug"].values and slug not in shortlist:
            shortlist.append(slug)

    shortlist = shortlist[:3]

    summary = {
        "source_debug": str(DEBUG_PATH),
        "n_test_images": len(debug),
        "production_baseline": {"slug": "uniform-0p098", "leaderboard_score": 2.35170},
        "cohort_lut_train_sample": {f"{h}x{w}": v for (h, w), v in cohort_lut.items()},
        "ranked_by_ref_midpoint_penalty": res_df.to_dict(orient="records"),
        "submit_shortlist": shortlist,
        "notes": [
            "ref_midpoint_penalty is offline plausibility only — not UMUD score without hidden labels.",
            "Leaderboard submits required to validate; see submit_calibration.py or kaggle competitions submit.",
        ],
    }
    (OUT_DIR / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/")
    print(f"Submit shortlist: {shortlist}")


if __name__ == "__main__":
    main()
