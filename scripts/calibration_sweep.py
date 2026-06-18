"""Regenerate submission CSVs from fixed pixel geometry at multiple mm scales.

Default: production 200-tier debug CSV. Use --uniforms for a custom grid (calibration search).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG = ROOT / "tmp/kaggle-output/block3-200tier/submission_debug.csv"
DEFAULT_OUT = ROOT / "tmp/kaggle-output/calibration-sweep-200tier"

TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
REF_RANGES = {"fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
REF_MID = {k: sum(v) / 2 for k, v in REF_RANGES.items()}


@dataclass(frozen=True)
class Policy:
    slug: str
    description: str
    fl_scale: float
    mt_scale: float


def load_debug(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for col in ("pa_deg", "fl_px", "mt_px"):
        if col not in df.columns:
            raise ValueError(f"debug CSV missing {col}")
    return df


def apply_uniform(debug: pd.DataFrame, scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": debug["image_id"],
            "pa_deg": debug["pa_deg"],
            "fl_mm": debug["fl_px"] * scale,
            "mt_mm": debug["mt_px"] * scale,
        }
    )


def in_ref_range(sub: pd.DataFrame, col: str) -> float:
    lo, hi = REF_RANGES[col]
    series = sub[col]
    return float(((series >= lo) & (series <= hi)).mean())


def ref_midpoint_penalty(sub: pd.DataFrame) -> float:
    fl_pen = float(np.mean(np.abs(sub["fl_mm"] - REF_MID["fl_mm"]) / TOLERANCES["fl_mm"]))
    mt_pen = float(np.mean(np.abs(sub["mt_mm"] - REF_MID["mt_mm"]) / TOLERANCES["mt_mm"]))
    pa_pen = float(np.mean(np.abs(sub["pa_deg"] - 25.0) / TOLERANCES["pa_deg"]))
    return (fl_pen + mt_pen + pa_pen) / 3


def slug_for(scale: float) -> str:
    return f"uniform-{scale:.3f}".replace(".", "p")


def parse_uniforms(text: str) -> list[float]:
    return sorted({float(x.strip()) for x in text.split(",") if x.strip()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        type=Path,
        default=DEFAULT_DEBUG,
        help="submission_debug.csv with fl_px/mt_px (fixed geometry)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--uniforms",
        default="0.065,0.07,0.075,0.08,0.085,0.09,0.098",
        help="Comma-separated MM_PER_PIXEL values to generate",
    )
    parser.add_argument(
        "--submit-shortlist",
        default="0.065,0.07,0.075,0.08",
        help="Subset to recommend for leaderboard (excludes already-scored if noted in summary)",
    )
    args = parser.parse_args()

    uniforms = parse_uniforms(args.uniforms)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    debug = load_debug(args.debug)
    policies = [
        Policy(slug_for(s), f"Uniform MM={s}", s, s) for s in uniforms
    ]

    results = []
    for policy in policies:
        sub = apply_uniform(debug, policy.fl_scale)
        slug_dir = out_dir / policy.slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(slug_dir / "submission.csv", index=False)

        row = {
            "slug": policy.slug,
            "mm_per_pixel": policy.fl_scale,
            "n_rows": len(sub),
            "fl_mm_median": round(float(sub["fl_mm"].median()), 2),
            "mt_mm_median": round(float(sub["mt_mm"].median()), 2),
            "fl_in_ref_pct": round(100 * in_ref_range(sub, "fl_mm"), 1),
            "mt_in_ref_pct": round(100 * in_ref_range(sub, "mt_mm"), 1),
            "ref_midpoint_penalty": round(ref_midpoint_penalty(sub), 4),
        }
        results.append(row)
        print(
            f"{policy.slug}: FL med={row['fl_mm_median']} MT med={row['mt_mm_median']} "
            f"penalty={row['ref_midpoint_penalty']}"
        )

    res_df = pd.DataFrame(results).sort_values("mm_per_pixel")
    res_df.to_csv(out_dir / "sweep_results.csv", index=False)

    shortlist_scales = parse_uniforms(args.submit_shortlist)
    shortlist = [slug_for(s) for s in shortlist_scales if slug_for(s) in res_df["slug"].values]

    summary = {
        "source_debug": str(args.debug),
        "geometry": "200-tier apo (Block 3 production)",
        "n_test_images": len(debug),
        "known_leaderboard": {
            "uniform-0p090": 2.06330,
            "uniform-0p098": 2.20146,
        },
        "uniform_grid": uniforms,
        "ranked_uniforms": res_df.to_dict(orient="records"),
        "submit_shortlist": shortlist,
        "notes": [
            "Search direction: below 0.09 (0.09 is best scored so far on 200-tier).",
            "Stop when public score rises (worse). Offline penalty is not UMUD score.",
        ],
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}/")
    print(f"Submit shortlist: {shortlist}")


if __name__ == "__main__":
    main()
