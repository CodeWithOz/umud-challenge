# Research Log

## Current focus

_Last updated: 2026-06-04. Refresh this section at the start and end of each session; verify against git and Kaggle before acting on it._

**Best results:** _(none yet — no scored runs)_

**Active notebooks:** _(none yet)_

**Where we are:** Repo scaffolded; full competition data download in progress (~2.5 GB). No training or submission notebooks yet.

**Open questions:**

- _(none yet)_

**Constraints / budget:** _(e.g. remaining Kaggle GPU hours — update when relevant)_

---

## Experiments

| Date | Model / variation | Backbone | Training data | Key notes | Score | Status |
|---|---|---|---|---|---|---|

## Decisions and reversals

_(none yet)_

## Lessons

### Domain facts

- Task: predict muscle architecture from B-mode ultrasound — pennation angle (`pa_deg`), fascicle length (`fl_mm`), and muscle thickness (`mt_mm`) per image. (Confirmed: 2026-06-04)
- Training images are TIFF files under `apo_imgs_v1/apo_images_new_model_v1/` (e.g. `image_0000.tif`). Competition download is ~2.5 GB. (Confirmed: 2026-06-04)

### Process corrections

- Before recommending a new experiment or architecture, read the experiments table and decisions in this log; do not re-suggest approaches already tried or ruled out. Why: repeated suggestions of already-tested ideas waste GPU time and user patience. (2026-06-04)
- When a Kaggle notebook fails due to a missing or wrong sidebar input (model version, dataset, etc.), ask the user to fix it in the Kaggle UI first; do not change code to work around it. Why: restructuring code to dodge a UI fix hides the real problem and breaks intended ensemble/multi-model setups. (2026-06-04)
