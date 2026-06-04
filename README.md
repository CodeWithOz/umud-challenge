# UMUD Challenge Workspace

This repository contains local experiments, preprocessing scripts, and notebook workflows for the [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data) on Kaggle.

The competition is part of the [Universal Musculoskeletal Ultrasonography Database (UMUD)](https://link.springer.com/article/10.1186/s12880-026-02170-0) initiative. Participants build models that estimate muscle architecture from B-mode ultrasound images of lower-limb muscles — specifically **pennation angle**, **fascicle length**, and **muscle thickness**. These parameters are central to musculoskeletal research but are often measured manually; the challenge targets automated, reproducible estimation at scale. Submissions are scored on the competition leaderboard metric (see the Kaggle overview for the exact formulation).

## Current workflow

_(To be added once the training and submission pipeline is established.)_

## Competition links

- Competition home: [UMUD Challenge: Muscle Architecture in Ultrasound Data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
- Data page: [Competition data](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data)
- Overview: [Problem statement and context](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/overview)
- Rules: [Submission and usage rules](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/rules)
- Leaderboard: [Public standings](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/leaderboard)
- Code tab: [Community notebooks](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/code)
- Discussion tab: [Q&A and competition updates](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/discussion)

## Repo contents

_(Populated as notebooks, scripts, and artifacts are added.)_

- `data/umud-challenge/` — Local extracted competition files (ultrasound TIFFs, labels, submission template).
- `notebooks/` — Kaggle training and submission notebooks (per model variation).
- `scripts/` — Local preprocessing and utility scripts.
- `artifacts/` — Generated datasets and other build outputs.
- `tmp/` — Scratch space for downloads and one-off runs (gitignored).
- `research/log.md` — Structured research log (current focus, experiments, decisions, lessons) for cross-session memory.

## Environment

This project uses `uv` with the repo virtual environment:

```bash
source .venv/bin/activate
uv sync
```

Run scripts with `uv run`, for example:

```bash
uv run python main.py
```

The Kaggle CLI is installed in `.venv`; use the full path in shell commands when needed:

```bash
.venv/bin/kaggle competitions download -c umud-challenge-muscle-architecture-in-ultrasound-data -p data
```

## Notes on labels and data splits

Local extracted data under `data/umud-challenge/` (full Kaggle bundle, ~2.6 GB zip):

| Path | Contents | File count |
|------|----------|------------|
| `apo_imgs_v1/apo_images_new_model_v1/` | Aponeurosis-region ultrasound images (`.tif`) | 1,049 |
| `apo_masks_v1/apo_masks_new_model_v1/` | Matching aponeurosis masks (`.tif`) | 1,048 |
| `fasc_imgs_v1/fasc_images_new_model_v1/` | Fascicle-region ultrasound images (`.tif`) | 2,762 |
| `fasc_masks_v1/fasc_masks_new_model_v1/` | Matching fascicle masks (`.tif`) | 2,761 |
| `test_images_v2/test_set_v2/` | Held-out test images (`.tif` + `.png` variants) | 309 (251 `.tif`, 58 `.png`) |

- **Submission** — `sample_submission.csv` uses semicolon separators with columns `image_id`, `pa_deg`, `fl_mm`, `mt_mm`. Predict the three architecture measures for each test `image_id` (251 test `.tif` files in `test_images_v2/`).
- **Training labels** — Ground-truth CSVs are not a separate top-level file in this zip; training signal is provided via the paired image/mask directories above (use masks or derived geometry for supervision).

Download and extract:

```bash
.venv/bin/kaggle competitions download -c umud-challenge-muscle-architecture-in-ultrasound-data -p data
cd data && unzip -q umud-challenge-muscle-architecture-in-ultrasound-data.zip -d umud-challenge
```


Checking download progress and preparing a post-download extract script to run when it finishes.

Await