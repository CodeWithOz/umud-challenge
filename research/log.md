# Research Log

## Current focus

_Last updated: 2026-06-13 (v8 cancelled — no artifacts). Refresh at session start; verify against git and Kaggle before acting._

**Best results:** _(none yet — no scored runs)_

**Active notebooks:** Phase 3: `notebooks/baseline/baseline-phase-3.ipynb` — **Kaggle v9 RUNNING** timing baseline run 1 ([kernel](https://www.kaggle.com/code/ucheozoemena/umud-baseline-phase-3-fastai-u-net)). v8 cancelled (full train, no exports). **Now:** timing runs 1→5 before full retry.

**Where we are:** Timing baseline mode implemented (`TIMING_BASELINE=True`, `TIMING_RUN=1`: fasc 50 pairs × 1 epoch). Logs `timing_report.csv` + full-run projection. After runs 1–5: pick full config within Kaggle session budget.

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

### Optional later (do not lose)

| Idea | Notes |
|------|-------|
| **Stratify val split by image size** | FL bimodality is driven by **800×1200** vs **1080×1640** cohorts. After baseline works, try val split stratified on `(img_h, img_w)` so both resolutions appear in train and val. User wants this kept as a future experiment. |
| Refine PA geometry | Prototype PA (fascicle PCA vs deep apo slope) underestimates vs competition ref 5–45°. Improve before trusting mask-derived PA for analysis. |
| DLTrack comparison | Run DLTrack on a sample for cross-check if needed. |

### Phase 3 work items (suggested order)

0. **Timing baseline first** (before any long GPU run): smallest useful subset + 1 epoch → 1–2 scaling runs (more data and/or epochs) → project wall-clock for full train. See Lessons / Process corrections.
1. Create `notebooks/baseline/` (or similar) with `kernel-metadata.json` (`enable_gpu: true`, T4, internet if pretrained).
2. Load manifests from Phase 2 (`train_fasc_clean.csv`, `train_apo_all.csv`) or regenerate in notebook.
3. fastai `SegmentationItemList` / dataloaders with stretch-aligned image–mask pairs.
4. Train fascicle model → export weights; train apo model → export weights.
5. Local or notebook val: mask IoU/Dice + optional geometry on val set (pixels OK).
6. Submission notebook: load test images, predict masks, geometry → `sample_submission.csv`.
7. mm calibration hunt before first **mm** leaderboard submit (not before mask training).

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
| 2026-06-13 | baseline-phase-3 v9 | resnet34 | fasc 50 × 1ep (timing run 1) | `TIMING_BASELINE` mode; wall-clock + projection CSV | — | **running** |

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
| 2026-06-12 | **Training timing baseline** before long GPU runs: tiny data + min epochs first, then scale to project wall-clock | — |
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
- Kaggle `enable_gpu: true` defaults to **P100**, which is incompatible with current **fastai/PyTorch**. Use **T4**: `"machine_shape": "NvidiaTeslaT4"` + `kaggle kernels push --accelerator NvidiaTeslaT4`. (2026-06-12)
- **Training timing baseline (mandatory before long runs):** Do not launch full-data, multi-epoch GPU training without a wall-clock baseline for that architecture + dataset. (1) Run smallest useful subset with minimum epochs (e.g. 1 epoch, N≈50–100 pairs). (2) Do 1–2 scaling runs (more data and/or more epochs). (3) Extrapolate time for the target config before committing GPU hours. v8 (2,749 fasc + 1,048 apo, 10 epochs × 2 models) ran 6h+ without this step. (2026-06-12)

### Technical notes (Phase 0/1)

- Same-shape rate in 400-pair sample: apo **30.5%**, fasc **40.2%**.
- Empty fasc masks: **5** files. Near-empty (0.05%): **7** files. Clean fasc: **2,749**.
- Alignment ranking on reviewed examples: **stretch > center (fixed) > scale**.
- Reference pipelines: [DLTrack](https://github.com/PaulRitsche/DL_Track_US), UltraTimTrack, DUSTrack. UMUD: https://universalmuscledatabase.streamlit.app/
- Competition **reference** ranges (Data tab): PA 5–45°, FL 30–200 mm, MT 10–50 mm — for manual protocol plausibility checks.
- Dual-track geometry subset: **1,040** images (apo ∩ clean fasc); 8 apo-only due to fasc exclude list.
- Apo mask styles (full set): **574 line**, **474 region** (50% coverage threshold).
