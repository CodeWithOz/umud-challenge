# Research Log

## Current focus

_Last updated: 2026-06-10 (Phase 2 geometry v2). Refresh at session start; verify against git and Kaggle before acting._

**Best results:** _(none yet — no scored runs)_

**Active notebooks:** `notebooks/geometry/geometry-phase-2.ipynb` — Phase 2 **v2** (contour edges, expanded docs, local-run support). `notebooks/data-audit/data-audit.ipynb` — Phase 0+1 complete (v3).

**Where we are:** Phase 2 v2 replaces v1 horizontal row peaks with DLTrack-style contour edges + fitted lines. FL bimodality explained: **800×1200 vs 1080×1640 image cohorts**, not multiple fascicles per image. Dual-track gap 1048→1040 = 8 apo images with excluded fasc masks. **Next:** user QC on saved `tmp/geometry-local-output/figures/apo_qc_*.png`; mm calibration; Phase 3.

**Constraints / budget:** Phase 2 CPU (~2.5 min local, ~5.5 min Kaggle v1). No GPU.

---

## Competition roadmap (all phases)

High-level plan for the full pipeline. A new session should read this first for context, then **Current focus** and **Phase 2 agenda**.

| Phase | Goal | Why | Key activities | Status |
|-------|------|-----|----------------|--------|
| **0 — Inventory** | Know what files exist and whether they pair correctly | Bad pairing or corrupt files invalidate everything downstream | File counts, image/mask pairing, corrupt-file scan, apo vs fasc overlap, submission template check | **Done** |
| **1 — Visual QC** | Judge mask quality and alignment before modeling | Labels are manual masks; sparse fascicles and shape mismatch are common | Overlay galleries, coverage histograms, alignment lab (stretch/center/scale), exclude empty/near-empty fasc masks | **Done** |
| **2 — Geometry & calibration** | Turn aligned masks into PA/FL/MT; validate plausibility | Competition targets are numeric geometry, not masks; mm values need pixel scale | Stretch-align masks; implement geometry rules; hunt pixel→mm; apo region vs line tagging; export clean manifests; histograms vs physiological ranges | **In progress** (v2) |
| **3 — Baseline model** | First learned pipeline (segment-then-measure) | Establish a score on the leaderboard; test whether masks support learning | fastai (or PyTorch) segmentation on clean subsets; stretch-aligned training pairs; derive PA/FL/MT at inference; local val split | Not started |
| **4 — Iterate & submit** | Improve score; Kaggle submission flow | Competition metric is UMUD Score (normalized MAE; lower is better) | Augmentation, architecture tweaks, apo+fasc model design, test inference, `sample_submission.csv`, kaggle-run workflow | Not started |
| **5 — Reproducibility (if aiming for prizes)** | Top-3 require open-source, FAIR, reproducible code | Competition rules mandate public repo with license, docs, `requirements.txt` | OSI license, runnable notebook/script, documented inference | Not started |

**Approach:** Data quality before modeling (Phases 0–1). Geometry and calibration before training (Phase 2). Segment-then-measure, not classification. **Stretch** alignment when image and mask shapes differ.

---

## Phase 2 agenda

Use this as the checklist for the next session.

### Decisions already taken

| Decision | Rationale | Date |
|----------|-----------|------|
| **Stretch alignment** for mismatched image/mask shapes | Consistently best visual fit in alignment lab and focus views (apo and user spot-checks on fasc). Center fixed but still second-best; scale under-fills anatomy. | 2026-06-09 |
| **Exclude 12 fascicle pairs** from mask training | 5 empty (0% coverage) + 7 near-empty (0 < cov < 0.05%). Threshold 0.05% is conservative; 0.1% would exclude ~248 pairs — too aggressive. | 2026-06-09 |
| **EDA on Kaggle** via `kagglehub.competition_download` | User preference; no local data upload. Notebook: `notebooks/data-audit/`. | 2026-06-08 |
| **Apo and fasc are separate training tracks** | 1,048 apo pairs ⊆ 2,761 fasc pairs; 1,713 fasc-only is expected, not an error. | 2026-06-09 |
| **fastai is acceptable for modeling** when we get there | User is learning fastai; problem is segment-then-measure, not classification. | 2026-06-08 |
| **Deps via `uv add`** | Never `uv pip install` or bare `pip`. Documented in AGENTS.md. | 2026-06-08 |

### Decisions still needed

| Decision | Options / notes | Blocking? |
|----------|-----------------|-----------|
| **Pixel → mm calibration** | Ritsche (2024) / DLTrack docs; TIFF metadata; or fixed scale per device. **Decision: Option C** — derive px geometry now, convert before submit. TIFF sample (40 images) had no spacing tags. | **Yes** for numeric targets |
| **Apo geometry by mask style** | **Line:** raw mask contours. **Region:** invert then contours (user flip hypothesis). Edges via OpenCV + linear fit (DLTrack-style), not horizontal row peaks. Superficial/deep = top two separated contours. | Yes for apo-based MT |
| **Fasc stretch validation** | Confirmed on fasc mismatch gallery (Phase 2 v2). Stretch retained. | Done |
| **Baseline pipeline shape** | Segment-then-measure (DLTrack-style) vs classical CV vs direct regression. Competition allows all; stretch + geometry is the natural first baseline. | No — can prototype geometry first |
| **Train/val split strategy** | Random by image; possibly stratify by device/muscle if metadata available. | Before training |
| **Whether to use apo masks for MT only, fasc for PA/FL** | Matches how labels were created; needs geometry design. | Phase 2 design |

### Specifics to define before training

1. **Alignment function** — `stretch` when `img.shape != mask.shape`; passthrough when equal. Reuse logic from `notebooks/data-audit/data-audit.ipynb` (`align_mask(..., mode="stretch")`).
2. **Clean fasc manifest** — 2,749 pairs = 2,761 − 12. Source: `exclude_fasc_masks.csv` from audit notebook output (or regenerate in Phase 2 notebook).
3. **Geometry rules** — per competition Data tab: PA = angle fascicle–deep aponeurosis; FL = length along fascicle between aponeuroses (extrapolation if clipped); MT = perpendicular distance between superficial and deep aponeuroses (3 locations averaged in manual protocol).
4. **UMUD Score** — normalized MAE across PA, FL, MT; lower is better. Notebook: [umud-score](https://www.kaggle.com/code/paulritsche/umud-score).
5. **Submission format** — `sample_submission.csv`, semicolon-separated: `image_id;pa_deg;fl_mm;mt_mm`.

### General work items (Phase 2 order)

1. **Geometry notebook (Kaggle)** — stretch-align masks; derive PA/FL/MT on a sample; histograms vs physiological ranges (PA 5–45°, FL 30–200 mm, MT 10–50 mm per competition).
2. **Calibration hunt** — OSF dataset (Ritsche 2024), DLTrack repo, TIFF tags.
3. **Apo region vs line tagging** — use `APO_REGION_THRESHOLD = 0.50` from audit; document counts.
4. **Local validation metric** — compare derived geometry on aligned training masks (self-consistency); later compare to DLTrack if reference available.
5. **Clean subset export** — CSVs: `train_fasc_clean.csv`, `train_apo_all.csv`, optional `exclude_fasc_masks.csv`.
6. **Phase 3 (later)** — fastai segmentation baseline on clean subsets; segment-then-measure on test.

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
| Phase 2 outputs (v1) | `tmp/kaggle-output-geometry/` — manifests + `geometry_sample.csv` |

### Phase 2 v2 results (2026-06-10)

| Metric | Value |
|--------|-------|
| Clean fasc pairs | 2,749 (12 excluded) |
| Apo pairs | 1,048 — **574 line**, **474 region** |
| Dual-track (apo ∩ clean fasc) | 1,040 (8 apo-only: fasc on exclude list) |
| Geometry sample | 200 images; MT NaN 0% with contour method (was 1% v1) |
| PA (prototype) | mean 6.9°, vs deep apo slope |
| FL px bimodal | low bin ≈ 800×1200 images; high bin ≈ 1080×1640 |
| MT px | mean 286; 3-point perpendicular mean (competition manual protocol) |
| Local QC PNGs | `tmp/geometry-local-output/figures/` (gitignored) |

---

## Experiments

| Date | Model / variation | Backbone | Training data | Key notes | Score | Status |
|---|---|---|---|---|---|---|
| 2026-06-08 | data-audit v1 | — | competition (inventory + overlays) | Initial EDA; gray cmap hid fascicle color | — | superseded |
| 2026-06-08 | data-audit v2 | — | competition | Alignment lab; apo bimodal; fasc empty scan | — | superseded |
| 2026-06-09 | data-audit v3 | — | competition | Fixed `place_mask_center`; stretch default; exclude list | — | **complete** |
| 2026-06-09 | geometry-phase-2 v1 | — | competition | Manifests; px geometry; row-peak apo QC (flawed) | — | superseded |
| 2026-06-10 | geometry-phase-2 v2 | — | competition + local | Contour edges; MT 3-point mean; FL bin = image resolution; docs | — | **complete** |

---

## Decisions and reversals

| Date | Decision | Reversed? |
|------|----------|-----------|
| 2026-06-09 | Default alignment: **stretch** (not center or scale) | — |
| 2026-06-09 | Fasc near-empty threshold: **0.05%** (not 0.1%) | — |
| 2026-06-09 | `place_mask_center`: crop from mask center when mask larger than image | Replaces buggy top-left crop |
| 2026-06-09 | Calibration: **Option C** (pixels first, mm before submit) | — |
| 2026-06-09 | Apo region masks: **invert** for line extraction (prototype; pending visual QC) | — |
| 2026-06-10 | Apo MT/PA edges: **contour + linear fit** (DLTrack-style), not horizontal row peaks | Replaces v1 row-peak prototype |
| 2026-06-10 | Dual-track 1040 not 1048: 8 apo filenames on fasc exclude list | Expected, not a data bug |
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

### Technical notes (Phase 0/1)

- Same-shape rate in 400-pair sample: apo **30.5%**, fasc **40.2%**.
- Empty fasc masks: **5** files. Near-empty (0.05%): **7** files. Clean fasc: **2,749**.
- Alignment ranking on reviewed examples: **stretch > center (fixed) > scale**.
- Reference pipelines: [DLTrack](https://github.com/PaulRitsche/DL_Track_US), UltraTimTrack, DUSTrack. UMUD: https://universalmuscledatabase.streamlit.app/
