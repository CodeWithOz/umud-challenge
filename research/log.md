# Research Log

## Current focus

_Last updated: 2026-06-17 (pre-compaction snapshot — full pipeline + calibration v3 complete)._

### Leaderboard

| Submit | Model | `MM_PER_PIXEL` | Public score | Notes |
|--------|-------|----------------|--------------|-------|
| **v7** (user submit) | micro gray55+line 50×5ep | **1.0** | **48.18203** | Lower is better; FL/MT unit error dominates |
| v7 calibrated | same | **~0.098** (planned) | — | **Next after compaction** — compare to 48.18 |
| v8 | full gray55+line 1044×10ep | 1.0 | — | **Do not submit** — 19.4% MT NaN regression |

**UMUD score** (`paulritsche/umud-score`): weighted normalized MAE; tolerances PA **6°**, FL **12 mm**, MT **3 mm**. Score ≈48 with `MM_PER_PIXEL=1` because FL/MT are pixels reported as mm (~846 / ~271).

### Best submission file today: v7 (micro)

[`umud-submission-phase-3` v7/v8](https://www.kaggle.com/code/ucheozoemena/umud-submission-phase-3) — outputs: `tmp/kaggle-output/submission-v7/`.

| Metric | v4 | v7 | **v8 (full model)** |
|--------|-----|-----|---------------------|
| Rows in CSV | 251 | **309** | 309 |
| PA/FL NaN | 0/0/43% MT | **0/0/0%** | 0/0/**19.4% MT** |
| `mt_fail_reason` | — | all **ok** | ok 249, `no_contours` **37**, `single_contour` **13**, `no_x_overlap` **10** |

**Stack (v7/v8):** fasc v14 + gray55 bbox apo infer + mask clip + **horiz_parallel** contour picker (`build_submission_nb.py`). Apo checkpoint: `apo_gray55_line_baseline.pkl` (micro in v7, full in v8).

**v6→v7 fix:** export filtered to `.tif` only (251 rows); v7 writes all 309 (251 `.tif` + 58 `.png`).

### Full gray55+line pipeline (2026-06-17) — complete

| Step | Kernel | Config | Result |
|------|--------|--------|--------|
| Prep | `umud-prep-apo-gray55-line` v2 | `PREP_RUN=4` | **1044 pairs**, 473 region→line, **111s**; dataset `umud-aligned-apo-gray55-line-full` (~2089 files) |
| Train | `umud-train-apo-gray55-phase-3` v4+ | `TRAIN_RUN=6` | **1044×10ep**, ~22 min → `apo_gray55_line_baseline.pkl` |
| Submission | `umud-submission-phase-3` v8 | full model | **Regression** — see table above |

**TRAIN_RUN=6** added in `build_train_apo_gray55_nb.py` (profile 4 was old region-GT gray55-full — do not use).

**Ops note:** `monitor_gray55_line_full.sh` must not download full prep output (2000+ PNGs). Fixed to grep-filter timing/log/pkl only. Original monitor crashed on `ush`/`rg` after train completed; backup shell completed train→v8.

### Calibration v3 — complete (`umud-calibration-phase-3`)

Outputs: `tmp/kaggle-output/calibration-sprint/` (`calibration_summary.json`, `calibration_train_geometry.csv`, `calibration_tiff_tags.csv`). Builder: `scripts/build_calibration_nb.py`; notes: `research/calibration_sprint.md`.

| Finding | Detail |
|---------|--------|
| Train mm labels in bundle | **None** — masks only |
| TIFF spacing tags | **0 / 1048** |
| Train GT geometry (1048 pairs, submission geometry) | **1044 ok**, 4 `single_contour` |
| Ref-range heuristic (FL mid=115 mm, MT mid=30 mm) | **0.093 mm/px** (FL), **0.103 mm/px** (MT) |
| **Recommended first calibrated constant** | **`MM_PER_PIXEL ≈ 0.098`** (avg of FL/MT heuristics) → v7 medians: FL ~**83 mm**, MT ~**27 mm** |
| Depth-ruler heuristic | Detected on **100%** of train — but **resolution-dependent** (do not use global median) |

**Depth scale by train resolution cohort:**

| Cohort | Train n | Depth mm/px (median) | Test n (v7) |
|--------|---------|----------------------|-------------|
| 500×760 | 45 | 0.17 | rare |
| **800×1200** | 452 | **0.20** | **239** (77% — all 58 `.png` + 181 `.tif`) |
| 1080×1640 | 551 | 0.08 | rare |

**Implication:** global depth median **0.08** is wrong for test (pulled by 1080×1640 train majority). Per-cohort depth (~0.20 on 800×1200) gives FL ~177 mm / MT ~57 mm on test — MT above ref max (50 mm). **Defer per-cohort scaling** until after uniform **~0.098** calibrated submit.

### Horiz+parallel geometry (wired in submission)

Ablation on 62 legacy `no_x_overlap` cases: **62/62 mt_ok** with xspan_pair; horiz_parallel changed **2/12 user-flagged**, **0/50 user-good** regressions. Wired into `pick_superficial_deep` in `build_submission_nb.py`.

### Blocked until post-compaction (user-approved queue)

1. **v8 MT-fail visual QC** — overlay gallery for **60 images** with MT NaN (`no_contours` 37, `single_contour` 13, `no_x_overlap` 10); same style as `umud-no-x-overlap-viz-phase-3`.
2. **Calibrated v7 submit** — wire `MM_PER_PIXEL=0.098`, rerun submission (micro model), user leaderboard submit, compare to **48.18**.
3. **v8 regression diagnosis** — after QC, decide: revert to micro apo for submit vs fix full model (longer train / connectivity postprocess / etc.).
4. **Per-cohort mm scaling** — only if uniform calibrated score + QC justify complexity.
5. Phase 4 — after Phase 3 wrap (calibrated leaderboard + v8 understood).

### Standard inference preprocessing (unchanged)

1. ROI bbox (non-black threshold) → gray55 outside bbox.
2. Apo U-Net on gray55; fasc on raw image.
3. Clip apo mask to bbox.
4. **horiz_parallel** contour pairing → PA/FL/MT in px → × `MM_PER_PIXEL` for FL/MT mm.

### Priority (superseded handoff)

**Micro proved the approach** (50 pairs, 5ep, gray55 infer + bbox mask clip):

| Model | `mt_ok` (.tif n=251) | `single_contour` | `no_x_overlap` |
|-------|----------------------|------------------|----------------|
| Baseline apo + gray55 infer | **60.6%** | 80 | 19 |
| **Gray55+line apo** (micro) | **76.5%** | **0** | 59 |

- Net **+70 fixed, −30 broken** (+40 MT OK images).
- Prep: 50 pairs, **24/50** region→line converted; train val Dice **0.518** (5ep).
- **Pending user approval:** ~~full prep `PREP_RUN=4` (1044) + train `TRAIN_RUN` full-line profile (10ep)~~ → **done 2026-06-17**; v8 regressed — micro v7 remains best geometry file.

**Standard inference preprocessing (keep regardless of model):**
1. Detect ROI bbox (non-black threshold).
2. Fill outside bbox with **RGB(55,55,55)** → grayscale 55.
3. Run apo U-Net; **zero mask pixels outside bbox** before geometry.

### Apo training strategy (decided this session)

| Approach | Result | Verdict |
|----------|--------|---------|
| Gray55 infer only (old model) | 60.6% mt_ok | Helpful preprocessing |
| Gray55 train, **region GT kept** | 54.6% mt_ok; val Dice 0.963 misleading | **Rejected** — still learns compartment fill |
| **Gray55 + region→line GT** | 76.5% mt_ok; `single_contour` 80→0 | **Proceed** — unify all apo masks as line targets |
| Split line/region models | Not run | **Deferred** — conversion cheaper than dual-model routing |
| Contrast stretch @ infer | 38.5% mt_ok (309) | **Rejected** |
| ROI crop / geometry guard | Worse or neutral | **Rejected** |

**Region→line conversion (prep-time):** for masks with `coverage ≥ 0.5`, rasterize **top + bottom foreground boundaries** as 3px polylines; line masks unchanged. Local GT check: **474/474** converted masks yield finite MT (vs 473/474 raw region).

**Training masks verified:** apo and fasc are **separate files** — never both structures in one mask. Apo GT: **574 line** + **474 region**. Model predicting fascicle + apo together is a prediction bug, not GT format.

### Letterbox collapse (context)

92/92 `no_contours` test preds share letterbox layout `(T≈26, B≈0, L≈2)` on 800×1200. Train letterbox GT is **region ~97% fg**. Model overshoots to 100% fg → region path → invert → empty.

**Gray55+bbox v3** (old model, 309 test): mt_ok **64.4%**; eliminated `no_contours` but **80/92** letterbox cases became `single_contour` (pred cov ≈ bbox area, r=0.998).

### MT failure modes (geometry glossary)

| `mt_fail_reason` | Meaning |
|------------------|---------|
| `no_contours` | No usable contours (often saturated region → invert → empty) |
| `single_contour` | Only one contour — can't separate superficial vs deep apo |
| **`no_x_overlap`** | Two contours/lines found, but **x-ranges don't overlap** — no horizontal span to measure thickness between them. Common when line preds are fragmented or horizontally offset. |
| `line_fit_fail` / `empty_edge_polyline` | Too sparse to fit edge lines |

**Tradeoff observed:** line-target training eliminates `single_contour` but increases `no_x_overlap` (19→59 in micro) — fragmented/offset line preds. Net still strongly positive.

**Visual QC (2026-06-17, kernel v1, 309 test):** 62 `no_x_overlap` cases — contour **selection** wrong (top→bottom picks fascicle fragments); mask often contains both apo bands. User confirmed via gallery.

**Contour picker ablation (2026-06-17, v1):** `xspan_pair` (top-K by x-span, max x-overlap pair) vs legacy on gray55+line preds:

| Picker | `mt_ok` (309) | `no_x_overlap` | Notes |
|--------|---------------|----------------|-------|
| legacy | **79.9%** | 62 | current submission |
| **xspan_pair** | **100%** | **0** | **62/62 rescued**, 0 broken |

Outputs: `tmp/kaggle-output/contour-picker-ablation/`. **Superseded by horiz_parallel** in submission (2026-06-17).

### Key kernels & code paths

| Step | Kernel slug | Builder | Notes |
|------|-------------|---------|-------|
| Prep gray55+line | `umud-prep-apo-gray55-line` | `scripts/build_prep_apo_gray55_line_nb.py` | `PREP_RUN=1` micro, `4` full |
| Train | `umud-train-apo-gray55-phase-3` | `scripts/build_train_apo_gray55_nb.py` | `TRAIN_RUN=5` micro, **`6` full line 1044×10ep** |
| Eval | `umud-apo-gray55-line-eval-phase-3` | `scripts/build_apo_gray55_line_eval_nb.py` | Baseline vs line model, gray55 infer |
| **no_x_overlap QC** | `umud-no-x-overlap-viz-phase-3` | `scripts/build_no_x_overlap_viz_nb.py` | 6-panel gallery; run v1 complete → `tmp/kaggle-output/no-x-overlap-viz/` (62 figs) |
| **Contour picker ablation** | `umud-apo-contour-picker-ablation-phase-3` | `scripts/build_apo_contour_picker_ablation_nb.py` | xspan_pair **100% mt_ok** vs legacy 79.9% |
| **Horiz+parallel ablation** | `umud-apo-horiz-parallel-ablation-phase-3` | `scripts/build_apo_horiz_parallel_ablation_nb.py` | 62 cohort: horiz changes 2/12 flagged, **0/50 user-good** |
| **Calibration** | `umud-calibration-phase-3` | `scripts/build_calibration_nb.py` | v3 complete → `tmp/kaggle-output/calibration-sprint/` |
| Infer ablation | `umud-apo-gray55-bbox-pipeline-phase-3-v3` | `scripts/build_apo_contrast_fill_nb.py` | v3 gray55+bbox compare |

Datasets: `umud-aligned-apo-gray55-line-timing-50` (micro), `umud-aligned-apo-gray55-line-full` (**1044 pairs**, prep v2).

Outputs: `tmp/kaggle-output/submission-v7/` (best geometry), `submission-v8-full-model/` (full model), `calibration-sprint/`, `apo-gray55-line-eval/`.

### New session handoff

> **Post-compaction queue (do not start until user signals compaction done):**
> 1. Build + run **v8 MT-fail overlay QC** notebook for 60 NaN images.
> 2. Wire **`MM_PER_PIXEL=0.098`** → rerun submission with **micro** apo (v7 stack) → calibrated leaderboard submit vs 48.18.
> 3. Diagnose v8 full-model regression from QC; decide micro vs full for production.
>
> **Do not submit v8** to leaderboard until regression understood. **Do not use** global depth median 0.08 for calibration.

**Do not retry:** gray55 train without line conversion, contrast stretch, ROI crop, geometry guard, 512px resize, `TRAIN_RUN=4` (region GT gray55-full).

---

### Apo inference experiments (archive — details below)

**ROI crop (bbox by non-black threshold + paste pred back):**
- Baseline `mt_ok` mean: **0.5437** (MT NaN ~45.6%)
- With ROI crop: `mt_ok` mean **0.4822** (MT NaN ~51.8%) — overall worsened
- **MT-fixed** (baseline NaN → crop finite): **4** total
- Of baseline `no_contours` (**92**): crop fixed **2** (remaining **90** still NaN)
- New crop failures shifted mainly to `single_contour` (**143**) and `no_x_overlap` (**17**).

**Geometry guard (high-coverage boundary mask + derive MT from boundary):**
- `mt_ok` mean unchanged: **0.5437** → **no rescue**
- MT-fixed: **0**
- `guard_applied` fraction: **0.301** (guard ran, but didn’t move NaNs).

**Gray-context fill v2** (computed dark gray, no bbox clip): mt_ok **0.544 → 0.628** on 309 test.

**Gray55+bbox v3** (RGB 55, mask clip): mt_ok **0.544 → 0.644** on 309; `no_contours` eliminated; `single_contour` 80.

**Gray55 train (region GT):** mt_ok **54.6%** on .tif — regressed. Val Dice 0.963 misleading.

**Gray55+line micro:** mt_ok **76.5%** on .tif; `single_contour` 80→0; `no_x_overlap` 19→59.

**512px resize ablation:** rejected — val Dice 0 vs 256 verify 0.008. Stay @256.

### New session handoff (superseded — see Current focus top)

**Suggested opener (resize ablation — completed, rejected 512):**

> Continue UMUD Phase 3 resize ablation from `research/log.md`. Run Kaggle prep `PREP_RUN=5` → train `TRAIN_RUN=5` → eval `umud-eval-resize-ablation-phase-3`. Compare val Dice to 256px verify (0.008). Do not full-retrain until ablation result is reviewed.

**Key code paths:** `scripts/build_prep_nb.py` (`PREP_RUN=5`), `scripts/build_train_mounted_nb.py` (`TRAIN_RUN=5`), `scripts/build_eval_resize_ablation_nb.py`. Regenerate `.ipynb` from builders before Kaggle push.

**Kaggle CLI:** `.venv/bin/kaggle` with `export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)` first in agent shells.

### Phase 3 vs Phase 4 boundary

| | Phase 3 | Phase 4 |
|---|---------|---------|
| **Goal** | First learned baseline + **first leaderboard submission** | Improve score via iteration |
| **Includes** | Full fasc + apo train, val Dice, submission notebook, **mm calibration before first submit** | Augmentation, architecture tweaks, re-trains, re-submits |
| **Ends when** | Baseline score is on the leaderboard | — |

**Your reading is correct:** full-dataset training is still Phase 3 (baseline, not iteration). Phase 4 starts when we deliberately chase a better score.

### mm calibration (Option C) — when and where

| Question | Answer |
|----------|--------|
| **When?** | After both models are trained, **before the first scored Kaggle submit** — still Phase 3 (work item 5). |
| **Where?** | Submission/inference pipeline: convert pixel geometry (FL, MT) to mm using `mm_per_pixel`. PA stays in degrees. |
| **Not blocked on** | Training — models train in pixels at 256px; calibration is a post-train multiply on measured lengths. |
| **How?** | Hunt scale from OSF/DLTrack docs, tick marks, or image metadata (TIFF tags empty in sample; deferred from Phase 2). |

**Carry-forward (not blocking Phase 3):**
- **mm calibration** — Option C: deferred until **before leaderboard submit**; build baseline in pixels first.
- **PA geometry** — prototype skews low vs competition ref range (74/200 &lt;5°); refine measure step later.
- **Val split stratification by image size** — optional improvement to try after baseline (see Phase 3 agenda).

**Constraints / budget:** Phase 3 needs Kaggle **T4** GPU (`NvidiaTeslaT4` in metadata + `--accelerator` on push). P100 breaks fastai/PyTorch. Every Kaggle push → git commit + `git push` (AGENTS.md).

---

## Competition roadmap (all phases)

High-level plan for the full pipeline. A new session should read this first for context, then **Current focus** and **Phase 2 agenda**.

| Phase | Goal | Why | Key activities | Status |
|-------|------|-----|----------------|--------|
| **0 — Inventory** | Know what files exist and whether they pair correctly | Bad pairing or corrupt files invalidate everything downstream | File counts, image/mask pairing, corrupt-file scan, apo vs fasc overlap, submission template check | **Done** |
| **1 — Visual QC** | Judge mask quality and alignment before modeling | Labels are manual masks; sparse fascicles and shape mismatch are common | Overlay galleries, coverage histograms, alignment lab (stretch/center/scale), exclude empty/near-empty fasc masks | **Done** |
| **2 — Geometry & calibration** | Turn aligned masks into PA/FL/MT; validate plausibility | Competition targets are numeric geometry, not masks; mm values need pixel scale | Stretch-align masks; implement geometry rules; hunt pixel→mm; apo region vs line tagging; export clean manifests; histograms vs physiological ranges | **Done** (calibration deferred) |
| **3 — Baseline model** | First learned pipeline (segment-then-measure) | Establish a score on the leaderboard; test whether masks support learning | fastai U-Net on clean subsets (GPU); stretch-aligned pairs; dual models (fasc + apo); geometry at inference; val split | **Next** |
| **4 — Iterate & submit** | Improve score; Kaggle submission flow | Competition metric is UMUD Score (normalized MAE; lower is better) | Augmentation, architecture tweaks, apo+fasc model design, test inference, `sample_submission.csv`, kaggle-run workflow | Not started |
| **5 — Reproducibility (if aiming for prizes)** | Top-3 require open-source, FAIR, reproducible code | Competition rules mandate public repo with license, docs, `requirements.txt` | OSI license, runnable notebook/script, documented inference | Not started |

**Approach:** Data quality before modeling (Phases 0–1). Geometry and calibration before training (Phase 2). Segment-then-measure, not classification. **Stretch** alignment when image and mask shapes differ.

---

## Phase 3 agenda (start here in new session)

### Goal

First **learned** baseline: train mask segmentation with **fastai** on Kaggle **GPU**, then derive PA/FL/MT via Phase 2 geometry at inference (segment-then-measure).

### Decisions already taken for Phase 3

| Decision | Choice | Date |
|----------|--------|------|
| Framework | **fastai** (user learning it; U-Net segmentation) | 2026-06-08 |
| Hardware | **Kaggle T4 GPU** (`NvidiaTeslaT4`) — not P100; incompatible with current fastai/PyTorch | 2026-06-12 |
| Pipeline shape | **Segment-then-measure** (DLTrack-style), not direct regression | 2026-06-10 |
| Training layout | **Two models:** fascicle on `train_fasc_clean.csv` (2,749); apo on `train_apo_all.csv` (1,048) | 2026-06-10 |
| Inference layout | Run **both** models per test image; fasc → PA/FL, apo → MT | 2026-06-10 |
| Alignment at train time | **Stretch** when `img.shape != mask.shape` | 2026-06-09 |
| Val split (v1) | Random **80/20** by image filename | 2026-06-12 |
| mm calibration | **Defer** until before submission; train/eval masks in pixels first (Option C) | 2026-06-09 |
| Data for training | **Prep notebook → Kaggle dataset → train notebook** (BirdCLEF pattern; not inline transforms) | 2026-06-13 |
| Prep output resolution | **256×256 PNG** baked at prep (NEAREST masks) — see annotation below; **512px ablation** on P1 sample (`PREP_RUN=5`) before full retrain | 2026-06-13 / 2026-06-15 |
| Segmentation loss | **Class-weighted cross-entropy** (`CrossEntropyLossFlat` + `weight=[1, w_fg]`). Fasc `w_fg=150`, apo `w_fg=15`. Unweighted CE **failed** on sparse fasc masks — see debug dossier below. | 2026-06-15 |

**Why 256px at prep (not resize at train time):** The alternative is storing full-resolution aligned PNGs and calling `Resize(384)` in fastai during training. That still reads large files from disk every epoch. Baking 256 at prep cuts file size, I/O, and GPU pixels in one step. Trade-off: resolution is fixed per dataset version — if val Dice is poor, publish a `…-512px` dataset variant (ablation first on 50 pairs) as a follow-up benchmark, not the default path until validated.

**512px ablation rationale (2026-06-15):** Native fasc images ~500–1080 × 760–1640; 256px retains ~7% of native structure pixels vs ~29% @512 (local P1 sample). Coverage *fraction* is similar at 256 and 512 (~0.26%) but absolute fg pixel count scales ~4× — relevant for U-Net learning on sparse lines.

### Optional later (do not lose)

| Idea | Notes |
|------|-------|
| **Stratify val split by image size** | FL bimodality is driven by **800×1200** vs **1080×1640** cohorts. After baseline works, try val split stratified on `(img_h, img_w)` so both resolutions appear in train and val. User wants this kept as a future experiment. |
| Refine PA geometry | Prototype PA (fascicle PCA vs deep apo slope) underestimates vs competition ref 5–45°. Improve before trusting mask-derived PA for analysis. |
| DLTrack comparison | Run DLTrack on a sample for cross-check if needed. |

Full fasc prep extrapolation @ 0.088 s/pair: ~242 s (~4 min) for 2,749 pairs (CPU).

### P2 prep results (v3 — success)

| Metric | P1 (50) | P2 (200) |
|--------|---------|----------|
| Manifest scan | 35.5 s | 40.4 s |
| Transform | 4.4 s (0.088 s/pair) | 22.9 s (0.114 s/pair) |
| Total prep | 39.9 s | 63.4 s |

P2 dataset **`umud-aligned-fasc-timing-200`** ready.

### T1 mounted train results (v6 — success)

| Metric | Inline v9 (50 fasc) | T1 mounted (50 fasc) |
|--------|---------------------|----------------------|
| sec/pair/epoch | 16.91 | **0.158** |
| Train wall-clock (1 ep) | 676 s | **7.9 s** |

**Extrapolation (mounted, 256px, resnet34):** full fasc 2,749 × 10 ep ≈ **49–72 min** train + ~5 min prep. Feasible on Kaggle T4.

### T2 mounted train results (v9 — success)

| Metric | T1 (50) | T2 (200) |
|--------|---------|----------|
| sec/pair/epoch | 0.158 | **0.107** |
| Train wall-clock (1 ep) | 7.9 s | **21.4 s** |

T2 used `kagglehub.dataset_download` fallback (mount path missing after metadata swap on same kernel).

### P4 full fasc prep results

| Metric | P3 (1374) | P4 (2749) | Projected |
|--------|-----------|-----------|-----------|
| Prep total | 187.4 s | **358.0 s** | ~316 s transform |
| Prep sec/pair | 0.115 | **0.112** | 0.114 |

Dataset **`ucheozoemena/umud-aligned-fasc-full`** ready.

### Apo track baseline — AP1/AT1 (50 pairs, 1 ep)

| Axis | AP1 prep | AT1 train | Fasc T1 (compare) |
|------|----------|-----------|-------------------|
| Total | 4.7 s | 8.0 s | 39.9 s / 7.9 s |
| sec/pair (prep) | **0.094** | — | 0.088 |
| sec/pair/epoch (train) | — | **0.161** | 0.158 |

Apo prep faster (no fasc empty-mask scan). Train rate matches fasc at N=50. Datasets: `umud-aligned-apo-timing-50`.

### Apo track — AP2/AT2 (200 pairs, 1 ep)

| Axis | AP1 (50) | AP2 (200) | Fasc T2 (compare) |
|------|----------|-----------|-------------------|
| Prep total | 4.7 s | **18.6 s** | 63.4 s |
| Prep sec/pair | 0.094 | **0.093** | 0.114 |
| Train sec/pair/epoch | 0.161 | **0.111** | 0.107 |

**Full apo projection (from AT2 rate):** prep ~2 min + train 1,048 × 0.111 × 10 ≈ **19 min** on T4.

Dataset: `ucheozoemena/umud-aligned-apo-timing-200`.

### Apo track — AP3/AT3 (524 pairs, 5 ep — 50% apo)

| Axis | Projected (AT2 rate) | Actual | In line? |
|------|----------------------|--------|----------|
| Prep total | ~49 s | **45.6 s** | Yes |
| Prep sec/pair | 0.093 | **0.087** | Yes |
| Train total (5 ep) | ~290 s | **151.5 s** | **Faster** (matches fasc T3) |
| Train sec/pair/epoch | 0.111 | **0.058** | Yes — same as fasc T3 |

**Full apo projection (from AT3):** prep ~1.5 min + train 1,048 × 0.058 × 10 ≈ **10 min**.

### AP4 full apo prep + T4 full fasc train

| Run | Pairs | Result | sec/pair (prep) or sec/pair/epoch (train) |
|-----|-------|--------|-------------------------------------------|
| **AP4 prep** | 1,048 | **93.0 s** total | **0.089** |
| **T4 train** | 2,749 × 10 ep | **1474.3 s (~24.6 min)** | **0.054** |
| **AT4 train** | 1,048 × 10 ep | **583.3 s (~9.7 min)** | **0.056** |

Both in line with scaling ladder. Models: `fasc_baseline.pkl`, `apo_baseline.pkl` exported.

### Segmentation debug dossier (2026-06-15)

**Glossary (used in this log):**

| Term | Meaning |
|------|---------|
| **GT** | Ground truth — human-annotated mask (white = structure, black = background) |
| **CE** | Cross-entropy loss (`CrossEntropyLossFlat`) — per-pixel classification loss |
| **Dice** | Overlap metric between predicted and GT structure pixels (0 = none, 1 = perfect) |
| **fasc_pca_ok** | Submission debug metric: predicted fasc mask has ≥3 foreground pixels so PCA geometry (PA/FL) can run |

**Mask sparsity (prep datasets, measured):**

| Track | Mean mask coverage | Notes |
|-------|-------------------|--------|
| Fasc | **~0.29%** foreground | Thin lines; extreme class imbalance |
| Apo | mean ~47%, **median ~6.4%** | Bimodal: line-style vs region-style |

**Unweighted T4/AT4 results (eval kernel v2):**

| Track | Val loss | Val Dice | Pred fg (debug sample) | GT fg |
|-------|----------|----------|------------------------|-------|
| fasc | 0.031 | **0.000** | **0** pixels (all background) | ~0.30% |
| apo | 0.664 | **0.0006** | ~0.22% predicted | ~41% GT val avg |

Low fasc loss + zero Dice = model learned to predict **all background** (cheap under unweighted CE).

**Submission v2 (unweighted models):** 251 `.tif` rows, comma CSV. PA/FL NaN **~97%**, MT NaN **~52%**. Chain: empty fasc pred mask → `fascicle_pca` fails → PA/FL NaN.

**Root cause:** Unweighted CE treats every pixel equally; ~99.7% background pixels dominate gradient → structure class ignored.

**Proposed fix — class-weighted CE:**

| Class | Weight | Rationale |
|-------|--------|-----------|
| Background | 1.0 | baseline |
| Fasc structure | **150** | Inverse-freq ≈ (1−0.003)/0.003 ≈ **331**; 150 is conservative half-step |
| Apo structure | **15** | Inverse-freq for median line-style apo (~6.4% fg) ≈ **14.6** |

Code: `scripts/build_train_mounted_nb.py`, `scripts/build_train_apo_mounted_nb.py` — `USE_CLASS_WEIGHTS=True`, `TRAIN_RUN=4` for full retrain.

**Verification only (NOT full retrain):** fasc **50 pairs × 5 epochs**, weighted CE 150, Kaggle train kernel v12. Evaluated via `umud-debug-phase-3` v2 against **full** val set (model still tiny):

| Metric | Unweighted full T4 | Weighted 50×5ep only |
|--------|-------------------|----------------------|
| Val Dice | 0.000 | **0.008** (still poor; proves direction) |
| Pred fg pixels (240-val-image sample) | 0 | **1,848** / 15.7M (~0.012% vs GT ~0.30%) |
| Test `fasc_pca_ok` (80 images) | 0% | **50%** |

**Notebook fixes this session:**

| Issue | Fix |
|-------|-----|
| Eval `NameError: Path` | Import `Path` in config cell |
| Eval CSV columns `<fastai.metrics.Dice object at …>` | Use `type(m).__name__.lower()` → column `dice` |
| Submission `image_id` mismatch | Use `path.name` (`IMG_00001.tif`), not stem |
| Submission 2-row NaN CSV | Template has 2 placeholder rows → write all 251 `.tif` preds |
| CSV separator | Comma (not semicolon); no competition requirement for `;` |
| Kaggle CLI auth in agent shells | `export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)` before CLI calls |

**Local `.pkl` load:** fastai 2.8 / Python 3.13 locally cannot unpickle Kaggle-exported learners (`Resolver` pickle error). **Debug and inference validation must run on Kaggle** (or pin older fastai). Datasets load fine via `kagglehub`.

**Artifacts:** `notebooks/debug-phase3/`, `scripts/build_debug_phase3_nb.py`. Debug outputs (local): `tmp/kaggle-output/debug/`, `tmp/kaggle-output/debug-v2/`.

**Pending (awaiting user approval):** Full weighted T4 + AT4 → eval v3+ → submission v3 → tune weights if Dice still low.

### Val Dice + submission v2 (2026-06-13) — summary

See **Segmentation debug dossier** above for full detail. Unweighted models unusable; weighted retrain is next step.

### P3/T3 scaling check results (50% data, 50% epochs)

**Config:** 1,374 fasc pairs (50% of 2,749 clean) · **5 epochs** (50% of 10) · 256px · resnet34 · T4

| Axis | Projected (from P2/T2 rates) | Actual | In line? |
|------|------------------------------|--------|----------|
| Prep total | ~197 s | **187.4 s** | Yes |
| Prep sec/pair | 0.114 | **0.115** | Yes |
| Train total (5 ep) | ~735 s | **397.2 s** | **Faster** — better GPU util at scale |
| Train sec/pair/epoch | 0.107 | **0.058** | **Faster** — improves with N (50→200→1374) |

**Revised full-fasc projection (from T3 rate):** prep ~5 min + train 2,749 × 0.058 × 10 ≈ **26 min** (vs ~49 min from T2 rate). Baseline extrapolation was conservative; mid-scale run confirms feasibility.

Dataset: `ucheozoemena/umud-aligned-fasc-timing-1374` · Train kernel v10 complete.

### P3/T3 projections (before run — from P1–P2 rates)

| Axis | Rate source | Projected P3/T3 |
|------|-------------|-------------------|
| Prep transform | P2: 0.114 s/pair | 1374 × 0.114 ≈ **157 s** (+ ~40 s manifest) |
| Train | T2: 0.107 s/pair/epoch | 1374 × 0.107 × 5 ≈ **735 s (~12 min)** |

Full fasc @ 10 ep projection if linear: 2749 × 0.107 × 10 ≈ **49 min** train.

### P1 prep results (v1 — upload failed; v2 fixes zip staging)

| Metric | Value |
|--------|-------|
| Clean fasc pairs | 2,749 |
| Prep targets | 50 |
| Manifest scan | 35.5 s |
| Transform (align+resize) | 4.4 s (0.088 s/pair) |
| Total prep | 39.9 s |
| Upload | v1 failed (loose folders); v2 **dataset created** despite kernel ERROR — `upload_ok` too strict on kaggle 2.0 token warning; fixed in v3 |

Full fasc prep extrapolation @ 0.088 s/pair: ~242 s (~4 min) for 2,749 pairs (CPU).

### Inline timing results (v9–v10 — superseded, kept for reference)

| Run | N fasc | Train sec | sec/pair/epoch | Total sec | Full fasc@10ep proj. |
|-----|--------|-----------|----------------|-----------|----------------------|
| 1 | 50 | 676 | 16.91 | 682 | ~103h |
| 2 | 200 | 1424 | 8.90 | 1430 | ~54h (improved GPU util; still infeasible) |

Runs 3–5 on inline train **cancelled**. Bottleneck: TIFF load + stretch-align every batch.

### BirdCLEF reference workflow ([birdclef_2026](https://github.com/CodeWithOz/birdclef_2026))

**Correct pattern** (post-`b003ac9` — Kaggle-native gen notebooks; **not** early local `scripts/` upload):

| Layer | BirdCLEF | UMUD equivalent |
|-------|----------|-----------------|
| Gen prep | `multilabel-234-v2-gen-species-1/2` CPU notebooks: competition mount → process batches → `/kaggle/working/upload/` → `kaggle datasets version/create` **from notebook** | `notebooks/prep-fasc-timing/` (`PREP_RUN` 1=50, 2=200 fasc pairs) |
| Split when session limit | gen-species-1: ranks 1–50 → `species-v2-001-050`; gen-species-2: 51+ → `species-v2-051-206` | Separate dataset slugs per timing tier; full 2,749 may need multi-notebook split later |
| Train | `multilabel-234-v2` with `dataset_sources`; `get_image_files` on mounted PNGs | `notebooks/train-mounted/` (`TRAIN_RUN` 1→timing-50, 2→timing-200) |
| Auth | Pre-authenticated on Kaggle — no secrets cell (`39becdc`) | Same |

Key commits: `b003ac9` (gen-species notebooks), `924ba26` (mount auto-extracted datasets, no unzip), `39becdc` (drop kaggle_secrets).

### Prep + train timing ladder (next — do NOT jump to full 2,749 + 1,048)

Benchmark **both axes** before full dataset, same philosophy as runs 1–2:

| Step | Prep N (fasc) | Publish dataset | Train | Epochs | Goal |
|------|---------------|-----------------|-------|--------|------|
| **P1** | 50 | `umud-aligned-fasc-timing-50` | T1: mount P1 | 1 | prep wall-clock + train wall-clock at micro N |
| **P2** | 200 | `…-timing-200` | T2: mount P2 | 1 | confirm linear-ish scaling on **both** prep and train |
| **P3** | 1,374 (50% fasc) | `…-timing-1374` | T3: 5 ep (50% of 10) | Scaling check vs P1–P2 extrapolation |

After P1–P2: extrapolate prep time for 2,749 + 1,048 and train time @ target epochs; add fp16 / resnet18 / checkpoints only after mounted-dataset train baseline exists.

**Prep notebook outputs:** `prep_timing.csv` (pairs/sec, total sec, bytes written). **Train notebook outputs:** `timing_report.csv` (as now, but no manifest scan / align).

### Phase 3 speed strategy (after inline timing — do not run inline 3–5)

**Problem:** On-the-fly TIFF load + stretch-align per sample dominates (~135s/batch at bs=8). Full dataset @ resnet34/384px/10ep is **~100h+** per track.

**Target:** Full fasc + apo training within Kaggle session budget (~9–12h) via combined optimizations.

| Lever | Expected impact | Notes |
|-------|-----------------|-------|
| **Prep notebook → Kaggle dataset** | Large — alignment once, reused every train run | **Preferred over `/kaggle/working/` cache** (working dir wiped each session). User-proven pattern from prior competition. |
| **Smaller encoder** (resnet18 vs resnet34) | ~1.5–2× | After mounted-dataset baseline |
| **Multi-session checkpointing** | Fits任意 length | `SaveModelCallback(with_opt=True)` + `learn.load()` + `start_epoch` |

### Prep dataset workflow (decision — BirdCLEF pattern)

1. **`notebooks/prep-fasc-timing/`** (CPU, internet on): competition TIFFs from `/kaggle/input/competitions/...` → stretch-align → **resize 256** (NEAREST masks) → PNG pairs + CSVs → `/kaggle/working/upload/`.
2. **Publish from notebook** via `kaggle datasets version` / `create` (subprocess). Small tiers first (50, 200 pairs).
3. **`notebooks/train-mounted/`**: `dataset_sources` only; `get_image_files`; **no** `align_mask` or manifest scan.

**Dataset layout:**

```
umud-aligned-fasc-timing-50/
  images/   masks/   manifests/train_fasc_clean.csv
```

(Full dataset: separate `umud-aligned-fasc-full` / `umud-aligned-apo-full` only after ladder extrapolation says feasible.)

### Phase 3 work items (remaining)

1. ~~Timing ladders, full prep, unweighted T4/AT4 train~~ **Done** (models exported but **segmentation failed** — see debug dossier).
2. ~~Eval + submission notebook scaffold~~ **Done** (`eval-val-dice-phase-3`, `submission-phase-3`).
3. ~~Root-cause debug~~ **Done** (`debug-phase-3`); class-weighted CE coded; **50×5ep verification** only.
4. ~~**Weighted full retrain** (fasc T4 + apo AT4 @256)~~ **Done** — fasc Dice 0.108, apo 0.039.
5. ~~Re-run eval + submission on weighted @256 models~~ **Done** (eval v4, submission v3).
6. **mm calibration** before first scored Kaggle submit (still Phase 3).
7. Reduce MT NaN rate (44.6%) — apo segmentation / geometry iteration (Phase 4).

### Key inputs from Phase 2

| File | Rows | Use |
|------|------|-----|
| `train_fasc_clean.csv` | 2,749 | Fascicle segmentation training |
| `exclude_apo_mt_invalid.csv` | 4 | Apo pairs with single-contour GT — no valid MT from mask geometry |
| `train_apo_all.csv` | 1,044 | Apo segmentation training (+ `mask_style` column; was 1,048) |
| `exclude_fasc_masks.csv` | 12 | Do not train fasc on these |
| Geometry code | `scripts/build_geometry_nb.py` | Port `align_mask`, contour geometry for inference |

Manifests available from Kaggle geometry kernel output or local `tmp/geometry-local-output/`.

---

## Phase 2 agenda (complete)

Historical checklist — all items done or explicitly deferred.

### Decisions already taken

| Decision | Rationale | Date |
|----------|-----------|------|
| **Stretch alignment** for mismatched image/mask shapes | Consistently best visual fit in alignment lab and focus views (apo and user spot-checks on fasc). Center fixed but still second-best; scale under-fills anatomy. | 2026-06-09 |
| **Exclude 12 fascicle pairs** from mask training | 5 empty (0% coverage) + 7 near-empty (0 < cov < 0.05%). Threshold 0.05% is conservative; 0.1% would exclude ~248 pairs — too aggressive. | 2026-06-09 |
| **EDA on Kaggle** via `kagglehub.competition_download` | User preference; no local data upload. Notebook: `notebooks/data-audit/`. | 2026-06-08 |
| **Apo and fasc are separate training tracks** | 1,048 apo pairs ⊆ 2,761 fasc pairs; 1,713 fasc-only is expected, not an error. | 2026-06-09 |
| **fastai is acceptable for modeling** when we get there | User is learning fastai; problem is segment-then-measure, not classification. | 2026-06-08 |
| **Deps via `uv add`** | Never `uv pip install` or bare `pip`. Documented in AGENTS.md. | 2026-06-08 |
| **Pixel → mm calibration (Option C)** | Pixels during Phases 2–3; find `mm_per_pixel` before leaderboard submit. TIFF sample (40 images) had no spacing tags. | 2026-06-09 |
| **Apo geometry by mask style** | Line → raw mask contours. Region → **invert** then contours. DLTrack-style edge fit. User approved v3 QC. | 2026-06-10 |
| **Fasc stretch validation** | Confirmed on mismatch gallery; stretch retained. | 2026-06-10 |
| **Baseline pipeline** | Segment-then-measure; separate fasc + apo training; combined inference. | 2026-06-10 |
| **Mask roles** | Fasc masks → PA/FL; apo masks → MT. | 2026-06-10 |
| **Kaggle vs local geometry notebooks** | `geometry-phase-2.ipynb` (kagglehub) vs `geometry-phase-2-local.ipynb` (`data/umud-challenge/`). Shared builder. | 2026-06-10 |
| **Val split v1** | Random 80/20 by filename for Phase 3 baseline. | 2026-06-12 |

### Open for later (not blocking Phase 3)

| Item | Notes |
|------|-------|
| **mm calibration** | Required before meaningful `fl_mm`/`mt_mm` submit; hunt OSF/DLTrack/tick marks. |
| **Val split stratified by image size** | Try after baseline — user endorsed keeping this idea. |
| **PA geometry refinement** | Prototype underestimates vs ref 5–45°. |

### Specifics to define before training

1. **Alignment function** — `stretch` when `img.shape != mask.shape`; passthrough when equal. Reuse logic from `notebooks/data-audit/data-audit.ipynb` (`align_mask(..., mode="stretch")`).
2. **Clean fasc manifest** — 2,749 pairs = 2,761 − 12. Source: `exclude_fasc_masks.csv` from audit notebook output (or regenerate in Phase 2 notebook).
3. **Geometry rules** — per competition Data tab: PA = angle fascicle–deep aponeurosis; FL = length along fascicle between aponeuroses (extrapolation if clipped); MT = perpendicular distance between superficial and deep aponeuroses (3 locations averaged in manual protocol).
4. **UMUD Score** — normalized MAE across PA, FL, MT; lower is better. Notebook: [umud-score](https://www.kaggle.com/code/paulritsche/umud-score).
5. **Submission format** — `sample_submission.csv`, semicolon-separated: `image_id;pa_deg;fl_mm;mt_mm`.

### General work items (Phase 2) — status

1. Geometry notebook (Kaggle + local) — **done** (v3).
2. Calibration hunt (TIFF tags) — **partial**; no spacing in sample; full hunt deferred to pre-submit.
3. Apo region vs line tagging — **done** (50% threshold).
4. Local validation / geometry on masks — **prototype done** on 200 dual-track sample.
5. Clean subset export — **done** (CSVs in kernel output + `tmp/geometry-local-output/`).
6. Phase 3 — **next session**.

### Artifacts from Phase 0/1

| Artifact | Location |
|----------|----------|
| Audit notebook | `notebooks/data-audit/data-audit.ipynb` |
| Kernel metadata | `notebooks/data-audit/kernel-metadata.json` |
| Notebook builder | `scripts/build_data_audit_nb.py` |
| Kaggle kernel | https://www.kaggle.com/code/ucheozoemena/umud-data-audit-phase-0-1 |
| Exclude list (from v3 run) | Kaggle output: `exclude_fasc_masks.csv` |
| Geometry notebook (Kaggle) | `notebooks/geometry/geometry-phase-2.ipynb` |
| Geometry notebook (local) | `notebooks/geometry/geometry-phase-2-local.ipynb` — uses `data/umud-challenge/`, outputs to `tmp/geometry-local-output/` |
| Geometry builder | `scripts/build_geometry_nb.py` — generates **both** notebooks |
| Kaggle kernel | https://www.kaggle.com/code/ucheozoemena/umud-geometry-phase-2 |
| Phase 2 local outputs | `tmp/geometry-local-output/` — manifests, `geometry_sample.csv`, `figures/apo_qc_*.png` |
| Git remote | `origin/main` pushed through `ad839b3` (2026-06-12) |

### Phase 2 final results (geometry v3, 2026-06-10/12)

| Metric | Value |
|--------|-------|
| Clean fasc pairs | 2,749 (12 excluded) |
| Apo pairs | 1,048 — **574 line**, **474 region** |
| Dual-track (apo ∩ clean fasc) | 1,040 (8 apo-only: fasc on exclude list) |
| Geometry sample | 200 images; MT NaN 0% with contour method (was 1% v1) |
| PA (prototype) | mean 6.9°; **74/200 below ref 5°**, 0 above 45° — prototype bias, not GT |
| FL px bimodal | low bin ≈ 800×1200 images; high bin ≈ 1080×1640 |
| MT px | mean 286; 3-point perpendicular mean (competition manual protocol) |
| Local QC PNGs | `tmp/geometry-local-output/figures/` (gitignored) |
| Histogram “ref” lines | **ref** = competition Data tab **reference** plausible ranges (PA 5–45°, FL 30–200 mm, MT 10–50 mm) for expert manual measurement — sanity guides, not submission bounds |
| User QC | v3 apo contour galleries + fasc stretch approved (2026-06-12) |

### Phase 2 session notes (for continuity)

- **Dual-track 1040 vs apo 1048:** 8 apo images have fasc masks on exclude list (`image_0086`, `0231`, `0392`, `0422`, `0491`, `0818`, `0848`, `0917`). Expected; not a bug.
- **Geometry evolution:** v1 row-peak horizontal lines (bad for curved apo) → v2/v3 OpenCV contours + linear edge fit (DLTrack-inspired). Region masks inverted; line masks raw.
- **FL histogram bimodality:** one `fl_px` per image; peaks = two **image resolution** cohorts (800×1200 vs 1080×1640), not two fascicles per image.
- **MT NaN (v1):** 1% from bad row-peak method; **0%** with contour method in v2/v3.
- **PA below ref min 5°:** 74/200 in sample — reflects **prototype** fascicle PCA vs deep apo angle, not verified ground truth. Does not block mask training.
- **Competition MT protocol:** perpendicular distance superficial↔deep at three x positions (manual); we approximate with 3-point mean on fitted edges.
- **3+ aponeuroses in mask:** sort contours top→bottom; superficial = top; deep = next separated (DLTrack rule).

---

## Experiments

| Date | Model / variation | Backbone | Training data | Key notes | Score | Status |
|---|---|---|---|---|---|---|
| 2026-06-08 | data-audit v1 | — | competition (inventory + overlays) | Initial EDA; gray cmap hid fascicle color | — | superseded |
| 2026-06-08 | data-audit v2 | — | competition | Alignment lab; apo bimodal; fasc empty scan | — | superseded |
| 2026-06-09 | data-audit v3 | — | competition | Fixed `place_mask_center`; stretch default; exclude list | — | **complete** |
| 2026-06-09 | geometry-phase-2 v1 | — | competition | Manifests; px geometry; row-peak apo QC (flawed) | — | superseded |
| 2026-06-10 | geometry-phase-2 v2 | — | competition + local | Contour edges; MT 3-point mean; FL bin = image resolution; docs | — | superseded |
| 2026-06-10 | geometry-phase-2 v3 | — | competition + local | Kaggle/local split; user QC approved; manifests + geometry CSVs | — | **complete** |
| 2026-06-12 | baseline-phase-3 v1 | resnet34 | fasc 2,749 + apo 1,048 | fastai U-Net; DataBlock `open_x`/`open_y` TypeError | — | **error** |
| 2026-06-12 | baseline-phase-3 v2 | resnet34 | fasc 2,749 + apo 1,048 | notebook not regenerated; still `open_x`/`open_y`; P100 | — | **error** |
| 2026-06-12 | baseline-phase-3 v3 | resnet34 | fasc 2,749 + apo 1,048 | T4 OK; `dataloaders()` missing `source` arg | — | **error** |
| 2026-06-12 | baseline-phase-3 v4 | resnet34 | fasc 2,749 + apo 1,048 | `TensorImage` dtype error in Resize/PIL | — | **error** |
| 2026-06-12 | baseline-phase-3 v5 | resnet34 | fasc 2,749 + apo 1,048 | `TransformBlock(fn, ImageBlock)` invalid on Kaggle fastai | — | **error** |
| 2026-06-12 | baseline-phase-3 v6 | resnet34 | fasc 2,749 + apo 1,048 | dataloader OK; ResNet34 weight download blocked (no internet) | — | **error** |
| 2026-06-12 | baseline-phase-3 v7 | resnet34 | fasc 2,749 + apo 1,048 | weights OK; loss not inferred from mask batch | — | **error** |
| 2026-06-12 | baseline-phase-3 v8 | resnet34 | fasc 2,749 + apo 1,048 | full train 10 epochs × 2; 6h+ then CANCEL_ACKNOWLEDGED; no exports | — | **cancelled** |
| 2026-06-13 | baseline-phase-3 v9 | resnet34 | fasc 50 × 1ep inline | 682s; 16.9 s/pair/ep | — | **complete** |
| 2026-06-13 | baseline-phase-3 v10 | resnet34 | fasc 200 × 1ep inline | 1430s; 8.9 s/pair/ep | — | **complete** |
| 2026-06-13 | train-mounted T4 | resnet34 | fasc 2749 × 10ep mounted | unweighted CE; 1474s; `fasc_baseline.pkl` | — | **complete** (seg useless) |
| 2026-06-13 | train-apo-mounted AT4 | resnet34 | apo 1048 × 10ep mounted | unweighted CE; 583s; `apo_baseline.pkl` | — | **complete** (seg useless) |
| 2026-06-15 | eval-val-dice v2 | — | val split 80/20 | fasc Dice 0, apo 0.0006; metric CSV column bug fixed v3 | — | **complete** |
| 2026-06-15 | submission-phase-3 v2 | — | 251 test tif | comma CSV; 97% PA/FL NaN | — | **complete** (bad models) |
| 2026-06-15 | debug-phase-3 v1/v2 | — | — | root cause: CE collapse; weighted 50×5ep verify | — | **complete** |
| 2026-06-15 | train-mounted v12 | resnet34 | fasc **50** × **5ep** weighted | verification only; Dice 0.008 | — | **complete** |
| 2026-06-15 | prep P5 + train T5 + eval resize ablation | resnet34 | fasc 50 × 5ep @512 | weighted w=150; val Dice **0.000** vs 256 verify 0.008 | — | **complete** (512 rejected) |
| 2026-06-15 | train-mounted T4 weighted | resnet34 | fasc 2749 × 10ep | `USE_CLASS_WEIGHTS`, w_fg=150 @256; val Dice **0.108** | — | **complete** |
| 2026-06-15 | train-apo-mounted AT4 weighted | resnet34 | apo 1048 × 10ep | `USE_CLASS_WEIGHTS`, w_fg=15 @256; val Dice **0.039** | — | **complete** |
| 2026-06-15 | eval-val-dice v4 | — | weighted models | fasc 0.108, apo 0.039 | — | **complete** |
| 2026-06-15 | submission-phase-3 v3 | — | 251 test tif | PA/FL NaN 0%, MT NaN 44.6% | — | **complete** |
| 2026-06-16 | apo-gray55-bbox v3 | resnet34 | old apo model | gray55+bbox clip; 309 test mt_ok 64.4%; single_contour 80 | — | **complete** |
| 2026-06-16 | train-apo-gray55 (region GT) | resnet34 | gray55-full 1044×10ep | mt_ok 54.6% — regressed vs infer-only gray55 | — | **complete** (rejected) |
| 2026-06-16 | gray55+line micro | resnet34 | gray55-line-50×5ep | region→line GT; mt_ok 76.5%; single_contour 0 | — | **complete** |
| 2026-06-17 | horiz_parallel ablation | — | 62 no_x_overlap cohort | 100% mt_ok xspan; horiz 2/12 flagged changed, 0/50 good broken | — | **complete** → wired submission |
| 2026-06-17 | submission v7 | micro gray55+line | 309 test | horiz_parallel; 0% NaN; PNG export fix; leaderboard **48.18** @ MM=1 | **48.18** | **complete** |
| 2026-06-17 | prep gray55+line full | — | PREP_RUN=4 | 1044 pairs, 473 region→line, 111s | — | **complete** |
| 2026-06-17 | train gray55+line full | resnet34 | TRAIN_RUN=6 1044×10ep | ~22 min; replaces micro checkpoint path | — | **complete** |
| 2026-06-17 | submission v8 | full gray55+line | 309 test | **MT NaN 19.4%** — 37 no_contours, 13 single, 10 no_x_overlap | — | **complete** (regressed vs v7) |
| 2026-06-17 | calibration-phase-3 v3 | — | 1048 train GT geom | depth scale resolution-dependent; **MM≈0.098** recommended uniform | — | **complete** |

---

## Decisions and reversals

| Date | Decision | Reversed? |
|------|----------|-----------|
| 2026-06-09 | Default alignment: **stretch** (not center or scale) | — |
| 2026-06-09 | Fasc near-empty threshold: **0.05%** (not 0.1%) | — |
| 2026-06-09 | `place_mask_center`: crop from mask center when mask larger than image | Replaces buggy top-left crop |
| 2026-06-09 | Calibration: **Option C** (pixels first, mm before submit) | — |
| 2026-06-09 | Apo region masks: **invert** before contour extraction | User confirmed in v3 QC |
| 2026-06-10 | Apo MT/PA edges: **contour + linear fit** (DLTrack-style), not horizontal row peaks | Replaces v1 row-peak prototype |
| 2026-06-12 | Phase 3: **fastai + Kaggle GPU**; mm calibration deferred to pre-submit | — |
| 2026-06-12 | Val split v1: random 80/20; **stratify by image size** noted for later | — |
| 2026-06-12 | **Training timing baseline** before long GPU runs; stop ladder once full-train projection is infeasible | — |
| 2026-06-13 | Full fasc@10ep ~103h @ current config; pivot to **prep dataset + train notebook** + fp16 + multi-session (runs 3–5 cancelled) | — |
| 2026-06-15 | **Unweighted CE unusable** for fasc (~0.3% fg); use **class-weighted CE** (fasc w=150, apo w=15) | — |
| 2026-06-15 | **Submission CSV:** comma separator; `image_id` = full filename (`IMG_00001.tif`) | — |
| 2026-06-13 | **Prep notebook → Kaggle dataset → train notebook** (BirdCLEF pattern) | — |
| 2026-06-13 | **256px resize baked at prep** (NEAREST masks); 384px/512px = optional dataset A/B | — |
| 2026-06-15 | **512px micro-ablation rejected** — val Dice 0 vs 256 verify 0.008; stay @256 for full train | — |
| 2026-06-10 | Dual-track 1040 not 1048: 8 apo filenames on fasc exclude list | Expected, not a data bug |
| 2026-06-10 | FL bimodality in px: driven by **800×1200 vs 1080×1640** image sizes | Not multi-fascicle per image |
| 2026-06-10 | Split geometry into Kaggle + local notebooks; shared builder | — |
| 2026-06-16 | **Apo infer:** gray55 fill RGB(55,55,55) outside bbox + mask clip to bbox | — |
| 2026-06-16 | **Apo train:** region GT → dual-boundary line targets at prep (not split models) | — |
| 2026-06-16 | **Reject** contrast stretch, ROI crop, geometry guard for MT rescue | — |
| 2026-06-16 | **Reject** gray55 train without line conversion (val Dice misleading on region GT) | — |
| 2026-06-17 | **horiz_parallel** contour picker in submission (xspan fallback) | User QC: 0/50 good regressions on 62 cohort |
| 2026-06-17 | Submission export: **all 309** image_ids (not `.tif`-only) | v6 had 251-row bug |
| 2026-06-17 | **First calibrated submit:** uniform `MM_PER_PIXEL≈0.098` on **v7 micro** before per-cohort depth scaling | Depth global median 0.08 rejected for test |
| 2026-06-17 | **Do not leaderboard-submit v8** until 60 MT-fail images QC'd | Full model regressed vs micro |

---

## Lessons

### Domain facts

- Task: predict muscle architecture from B-mode ultrasound — pennation angle (`pa_deg`), fascicle length (`fl_mm`), and muscle thickness (`mt_mm`) per test `image_id`. (Confirmed: 2026-06-04)
- Test: **309** images in `test_images_v2/test_set_v2/` — IMG_00001–00251 `.tif`, IMG_00252–00309 `.png`. (Confirmed: 2026-06-17)
- Labels are **masks**, not numeric CSVs. Targets come from **geometry** on structures (or end-to-end prediction without masks).
- Manual annotation: 2 raters, averaged; disagreement thresholds >10 mm FL, >4° PA, >1 mm MT trigger re-review (competition Data tab).
- Images often look like **ultrasound machine screenshots** (padding around scan). Masks may be at native scan resolution → shape mismatch ~60–70% of sample.
- Apo mask coverage is **bimodal**: region-style (~90%+ pixels) vs line-style (~2–6%). Fascicle masks are **sparse** (median ~0.26% coverage in 400-pair sample).
- Fasc-only 1,713 images: extra fascicle training data; all 1,048 apo filenames appear in fasc set.

### Process corrections

- Before recommending a new experiment or architecture, read the experiments table and decisions in this log. (2026-06-04)
- Kaggle notebook missing sidebar input → ask user to fix UI first, don't workaround in code. (2026-06-04)
- Kaggle dataset zips auto-extract; use `rglob` + filename lookup, not `glob('*.zip')`. (2026-06)
- Add Python deps with **`uv add`**, never `uv pip install` or bare `pip`. (2026-06-09)
- Load competition data in notebooks via **`kagglehub.competition_download`**, not local upload. (2026-06-08)
- Jupyter markdown cells need **`\n` on every line** in the JSON source array — `split("\n")` without trailing newlines renders as one paragraph. (2026-06-09)
- Colored mask overlays: **`imshow(rgb)` without `cmap="gray"`** — gray cmap removes green/orange tint. (2026-06-09)
- Markdown `~>50%~` in tables can render as strikethrough; avoid `~` around comparison symbols. (2026-06-09)
- **`place_mask_center`**: when mask exceeds target size, must crop from mask center (`mask_row_start = (mh - target_h) // 2`), not `[0:target_h]`. (2026-06-09)
- Push Kaggle notebooks **proactively** after creation unless user confirmation needed. (2026-06-08)
- **Every Kaggle kernel push** must have a matching **git commit** on `main` (AGENTS.md). (2026-06)
- Phase 2 geometry: **Kaggle notebook** (`kagglehub`) vs **local notebook** (`data/umud-challenge/`); generate both via `scripts/build_geometry_nb.py`. (2026-06-10)
- Histogram **ref** lines = competition **reference** plausible ranges for manual expert measurements, not model targets. (2026-06-12)
- FL px bimodality tracks **image dimensions** (800×1200 vs 1080×1640), not multiple fascicles per image. (2026-06-10)
- **Prep dataset pattern (BirdCLEF):** expensive transforms in **prep notebook** → **Kaggle dataset** → train notebook mounts via `dataset_sources`. Reference: [birdclef_2026](https://github.com/CodeWithOz/birdclef_2026) (`generate_spectrogram_batches.py`, `species-*` datasets, `multilabel-234` train). Benchmark **prep** and **train** at N=50→200 before full data. (2026-06-13)
- **256px at prep:** resize images+masks once when building dataset (not at train). Faster than full-res PNGs + `Resize()` in fastai. New dataset version if higher res needed. (2026-06-13)
- **512px ablation (P5/T5):** 4× more GT structure pixels did not improve val Dice (0.000 vs 0.008 @256 verify); model collapsed to all-background with same `w_fg=150`. Resolution alone is not the bottleneck — proceed @256. (2026-06-15)
- Kaggle `enable_gpu: true` defaults to **P100**, which is incompatible with current **fastai/PyTorch**. Use **T4**: `"machine_shape": "NvidiaTeslaT4"` + `kaggle kernels push --accelerator NvidiaTeslaT4`. (2026-06-12)
- **Training timing baseline (mandatory before long runs):** … v8 (2,749 fasc + 1,048 apo, 10 epochs × 2 models) ran 6h+ without this step. Run 1 (50 fasc, 1ep): **682s**, **16.9 s/pair/epoch** → **~103h** full fasc@10ep. Runs 3–5 skipped after run 2 confirms linear scale. (2026-06-13)
- **Unweighted CE + sparse fasc masks (~0.3% fg):** model collapses to all-background; val Dice ≈ 0, low loss, submission PA/FL mostly NaN. Fix: **class-weighted CE** (fasc structure weight ~150, apo ~15). Verify on Kaggle before trusting local `.pkl` load. (2026-06-15)
- **Kaggle OAuth CLI in non-interactive shells:** run `export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)` before `.venv/bin/kaggle` commands; venv activation alone is insufficient. (2026-06-15)
- **fastai `load_learner` locally:** Kaggle-exported `.pkl` may fail on local fastai 2.8 / Py 3.13 (`Resolver` pickle). Run eval/debug on Kaggle matching docker image. (2026-06-15)
- **Eval CSV metrics:** never use `str(metric)` as column name — use `type(m).__name__.lower()` (e.g. `dice`). (2026-06-15)
- **Leaderboard score with `MM_PER_PIXEL=1`:** FL/MT reported as pixels-as-mm (~846 / ~271) dominate normalized MAE; PA (degrees) is minor. Calibrate before interpreting score. v7 @ MM=1 → **48.18**. (2026-06-17)
- **Calibration:** no train mm labels or TIFF spacing tags; use ref-range midpoint vs GT/pred pixel geometry. Uniform **~0.098 mm/px** plausible first try. **Depth-ruler** scale is **resolution-dependent** — global train median (0.08) wrong for test (77% 800×1200). (2026-06-17)
- **Full gray55+line train can regress MT NaN** vs micro (v8: 19.4% vs v7: 0%) — more data ≠ better geometry; QC before swapping production model. (2026-06-17)
- **Submission export:** filter on full `pred_df` / sample_submission template — `.tif`-only filter dropped 58 `.png` rows (v6 bug). (2026-06-17)
- **Pipeline monitors:** do not download full prep dataset outputs locally; filter artifacts (timing, log, pkl, csv). (2026-06-17)

### Technical notes (Phase 0/1)

- Same-shape rate in 400-pair sample: apo **30.5%**, fasc **40.2%**.
- Empty fasc masks: **5** files. Near-empty (0.05%): **7** files. Clean fasc: **2,749**.
- Alignment ranking on reviewed examples: **stretch > center (fixed) > scale**.
- Reference pipelines: [DLTrack](https://github.com/PaulRitsche/DL_Track_US), UltraTimTrack, DUSTrack. UMUD: https://universalmuscledatabase.streamlit.app/
- Competition **reference** ranges (Data tab): PA 5–45°, FL 30–200 mm, MT 10–50 mm — for manual protocol plausibility checks.
- Dual-track geometry subset: **1,040** images (apo ∩ clean fasc); 8 apo-only due to fasc exclude list.
- Apo mask styles (full set): **574 line**, **474 region** (50% coverage threshold).
- **Apo/fasc masks are separate files** — one structure per mask; dual-track 1048 filenames but not dual-label in one file. (2026-06-16)
- **High apo val Dice can mislead** when region masks dominate (~97% fg) — metric rewards compartment fill, not MT-usable lines. (2026-06-16)
- **`no_x_overlap`:** superficial and deep apo edge lines have no shared horizontal span — MT cannot be computed. Distinct from `single_contour` (only one blob). (2026-06-16)
- **Kaggle prep upload:** notebook `kaggle` 2.0.0 fails on `datasets create` for new slugs; `pip install kaggle==2.0.2` in prep cell; retry `version` after partial create. (2026-06-16)
- **Eval model paths:** use explicit `resolve_pkl()` + `rglob` — never `name.replace('gray55_', '')` on filenames (loaded wrong model). ERROR kernels don't mount as `kernel_sources`. (2026-06-16)
- **Monitor scripts:** avoid `kaggle auth print-access-token` in loops (429 rate limit); grep `KernelWorkerStatus.ERROR` not bare `ERROR` in CLI warnings. (2026-06-16)
