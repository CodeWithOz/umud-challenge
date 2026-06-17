# Calibration sprint (2026-06-17)

## Leaderboard context

- **v7 score:** 48.18203 (lower is better)
- **Metric:** weighted normalized MAE across PA, FL, MT
  - Tolerances: PA **6°**, FL **12 mm**, MT **3 mm** (from `paulritsche/umud-score`)
- With `MM_PER_PIXEL=1.0`, FL/MT are reported in **pixels as if they were mm** (~846 px → ~846 "mm"), so FL/MT errors dominate the score. PA is in degrees already and is a smaller contributor.

## What we do not have

| Source | Result |
|--------|--------|
| Competition bundle CSV with train `fl_mm`/`mt_mm` | **Not present** (images + masks only) |
| TIFF spacing tags (40-image Phase 2 sample + IMG_00001 test) | **Only width/height** |
| `sample_submission.csv` example rows | **2 placeholders** — not usable as hidden labels |

## What we do have

| Source | Use |
|--------|-----|
| Train GT masks (1040 dual-track) | Run same geometry as submission → `fl_px`, `mt_px` in native space |
| Competition Data tab ref ranges | FL 30–200 mm, MT 10–50 mm, PA 5–45° (sanity histograms) |
| v7 `submission_debug.csv` | Test pixel geometry (309 rows, 0% NaN) |
| DLTrack / UMUD protocol | Manual per-image scale or depth ruler — images may include left depth strip |

## Local estimates (uniform scale)

From `scripts/analyze_calibration.py` on v7 debug + Phase 2 geometry sample:

| Method | FL mm/px | MT mm/px | Notes |
|--------|----------|----------|-------|
| Ref-range midpoint / train GT median px | **0.135** | **0.104** | FL mid=115 mm, MT mid=30 mm |
| Ref-range midpoint / test pred median px | **0.136** | **0.111** | Similar |
| Single scale tuning vs synthetic GT (75/20/10) | **~0.083** | same | Not real labels — illustrative only |

**Working hypothesis:** start with a **single** `MM_PER_PIXEL ≈ 0.10–0.12` for both FL and MT, then refine after Kaggle calibration kernel (full train GT + depth-strip heuristic).

At `MM_PER_PIXEL=0.10` on v7 medians: FL ≈ **85 mm**, MT ≈ **27 mm** — inside ref ranges.

## Active Kaggle job

- **Kernel:** `umud-calibration-phase-3` (CPU)
- **Outputs:** `calibration_train_geometry.csv`, `calibration_tiff_tags.csv`, `calibration_summary.json`
- **Builder:** `scripts/build_calibration_nb.py`

## Recommended sequence (Phase 3 wrap)

1. Finish calibration kernel → pick `MM_PER_PIXEL` (possibly per-resolution cohort)
2. Wire constant(s) into `build_submission_nb.py`
3. Re-run submission with full gray55+line model (v8)
4. Leaderboard submit with calibrated mm
5. Optional: separate `MM_PER_PIXEL_FL` / `MM_PER_PIXEL_MT` if train-depth-scale vs GT-geometry disagree

## Pipeline status

See `tmp/kaggle-output/gray55-line-full-monitor.log` — prep complete; train + submission pending.
