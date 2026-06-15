# Research Log

## Current focus

_Last updated: 2026-06-15 (segmentation debug complete; weighted retrain **pending user approval**)._

**Best results:** _(none yet — no scored leaderboard runs)_

**Active notebooks:**

| Notebook | Kaggle slug | Status |
|----------|-------------|--------|
| Train fasc | `umud-train-mounted-phase-3` | **Unweighted T4 complete** — models collapsed; **weighted retrain not run yet** (code ready, `TRAIN_RUN=4`) |
| Train apo | `umud-train-apo-mounted-phase-3` | Same — **unweighted AT4** exported; weighted retrain pending |
| Eval val Dice | `umud-eval-val-dice-phase-3` | v3 pushed (metric column fix) |
| Submission | `umud-submission-phase-3` | v2 complete — 251-row comma CSV; ~97% PA/FL NaN (bad models) |
| Debug | `umud-debug-phase-3` | v2 complete — root-cause evidence |

**Blocked on:** User approval to run **full weighted retrain** (fasc 2749×10ep + apo 1048×10ep, ~35 min T4 total). Then re-eval + re-submission. mm calibration still deferred until before first scored submit.

**Do not use for inference:** `fasc_baseline.pkl` / `apo_baseline.pkl` from unweighted T4/AT4 — predict all-background (fasc) or near-empty (apo).

### New session handoff

**Yes — you can start a fresh chat and paste your decision.** A new agent should read this file (`research/log.md` **Current focus** + **Segmentation debug dossier**) and `AGENTS.md` before acting.

**Suggested opener for the new chat** (edit as needed):

> Continue UMUD Phase 3 from `research/log.md`. Unweighted T4/AT4 models failed (val Dice ≈ 0). Class-weighted CE is coded (`USE_CLASS_WEIGHTS=True`, fasc w=150, apo w=15, `TRAIN_RUN=4`). My decision on weighted full retrain: **[approve / reject / change weights to X]**. Run on Kaggle yourself; do not ask me to run locally.

**If approved, agent should:** (1) `git pull`, (2) push + run `umud-train-mounted-phase-3` and `umud-train-apo-mounted-phase-3` on T4, (3) re-run `umud-debug-phase-3` or `umud-eval-val-dice-phase-3`, (4) re-run `umud-submission-phase-3`, (5) update this log with Dice/NaN results.

**Key code paths:** `scripts/build_train_mounted_nb.py`, `scripts/build_train_apo_mounted_nb.py`, `scripts/build_eval_val_dice_nb.py`, `scripts/build_submission_nb.py`, `scripts/build_debug_phase3_nb.py`. Regenerate `.ipynb` from builders before Kaggle push.

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
| Prep output resolution | **256×256 PNG** baked at prep (NEAREST masks) — see annotation below | 2026-06-13 |
| Segmentation loss | **Class-weighted cross-entropy** (`CrossEntropyLossFlat` + `weight=[1, w_fg]`). Fasc `w_fg=150`, apo `w_fg=15`. Unweighted CE **failed** on sparse fasc masks — see debug dossier below. | 2026-06-15 |

**Why 256px at prep (not resize at train time):** The alternative is storing full-resolution aligned PNGs and calling `Resize(384)` in fastai during training. That still reads large files from disk every epoch. Baking 256 at prep cuts file size, I/O, and GPU pixels in one step. Trade-off: resolution is fixed per dataset version — if val Dice is poor, publish a `…-384px` dataset variant as a follow-up benchmark, not the default path.

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
4. **Weighted full retrain** (fasc T4 + apo AT4) — **pending user approval**.
5. Re-run eval + submission on weighted models; confirm Dice and NaN rates improved.
6. **mm calibration** before first scored Kaggle submit (still Phase 3).

### Key inputs from Phase 2

| File | Rows | Use |
|------|------|-----|
| `train_fasc_clean.csv` | 2,749 | Fascicle segmentation training |
| `train_apo_all.csv` | 1,048 | Apo segmentation training (+ `mask_style` column) |
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
| _pending_ | train-mounted T4 weighted | resnet34 | fasc 2749 × 10ep | `USE_CLASS_WEIGHTS`, w_fg=150 | — | **not started** |
| _pending_ | train-apo-mounted AT4 weighted | resnet34 | apo 1048 × 10ep | `USE_CLASS_WEIGHTS`, w_fg=15 | — | **not started** |

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
| 2026-06-13 | **256px resize baked at prep** (NEAREST masks); 384px = optional dataset A/B | — |
| 2026-06-10 | Dual-track 1040 not 1048: 8 apo filenames on fasc exclude list | Expected, not a data bug |
| 2026-06-10 | FL bimodality in px: driven by **800×1200 vs 1080×1640** image sizes | Not multi-fascicle per image |
| 2026-06-10 | Split geometry into Kaggle + local notebooks; shared builder | — |

---

## Lessons

### Domain facts

- Task: predict muscle architecture from B-mode ultrasound — pennation angle (`pa_deg`), fascicle length (`fl_mm`), and muscle thickness (`mt_mm`) per test `image_id`. (Confirmed: 2026-06-04)
- Training data: paired TIFF images and masks — apo 1,048 pairs, fasc 2,761 pairs. Test: 251 `.tif` in `test_images_v2/test_set_v2/`. (Confirmed: 2026-06-04)
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
- Kaggle `enable_gpu: true` defaults to **P100**, which is incompatible with current **fastai/PyTorch**. Use **T4**: `"machine_shape": "NvidiaTeslaT4"` + `kaggle kernels push --accelerator NvidiaTeslaT4`. (2026-06-12)
- **Training timing baseline (mandatory before long runs):** … v8 (2,749 fasc + 1,048 apo, 10 epochs × 2 models) ran 6h+ without this step. Run 1 (50 fasc, 1ep): **682s**, **16.9 s/pair/epoch** → **~103h** full fasc@10ep. Runs 3–5 skipped after run 2 confirms linear scale. (2026-06-13)
- **Unweighted CE + sparse fasc masks (~0.3% fg):** model collapses to all-background; val Dice ≈ 0, low loss, submission PA/FL mostly NaN. Fix: **class-weighted CE** (fasc structure weight ~150, apo ~15). Verify on Kaggle before trusting local `.pkl` load. (2026-06-15)
- **Kaggle OAuth CLI in non-interactive shells:** run `export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)` before `.venv/bin/kaggle` commands; venv activation alone is insufficient. (2026-06-15)
- **fastai `load_learner` locally:** Kaggle-exported `.pkl` may fail on local fastai 2.8 / Py 3.13 (`Resolver` pickle). Run eval/debug on Kaggle matching docker image. (2026-06-15)
- **Eval CSV metrics:** never use `str(metric)` as column name — use `type(m).__name__.lower()` (e.g. `dice`). (2026-06-15)

### Technical notes (Phase 0/1)

- Same-shape rate in 400-pair sample: apo **30.5%**, fasc **40.2%**.
- Empty fasc masks: **5** files. Near-empty (0.05%): **7** files. Clean fasc: **2,749**.
- Alignment ranking on reviewed examples: **stretch > center (fixed) > scale**.
- Reference pipelines: [DLTrack](https://github.com/PaulRitsche/DL_Track_US), UltraTimTrack, DUSTrack. UMUD: https://universalmuscledatabase.streamlit.app/
- Competition **reference** ranges (Data tab): PA 5–45°, FL 30–200 mm, MT 10–50 mm — for manual protocol plausibility checks.
- Dual-track geometry subset: **1,040** images (apo ∩ clean fasc); 8 apo-only due to fasc exclude list.
- Apo mask styles (full set): **574 line**, **474 region** (50% coverage threshold).
