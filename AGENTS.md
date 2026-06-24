## Development Environment

- Use the existing `.venv` and `uv` for all Python tasks. Never use `pip`, `uv pip install`, or system `python`.
- Add dependencies with `uv add <package>` (updates `pyproject.toml` and lockfile). Never `uv pip install`.
- The `kaggle` CLI is at `.venv/bin/kaggle` — not on the system PATH. At session start, check `which kaggle`; if missing, activate `.venv` once for interactive shells. **Background shells never inherit the venv** — always use `.venv/bin/kaggle` (full path) there.

## Workspace Facts

- Local competition data lives under `data/umud-challenge/` (gitignored). Archive at repo root: `umud-challenge.zip`. Download with `.venv/bin/kaggle competitions download -c umud-challenge-muscle-architecture-in-ultrasound-data -p data`, then `unzip -q data/umud-challenge-muscle-architecture-in-ultrasound-data.zip -d data/umud-challenge`. Wait for the download to finish before extracting — a partial zip will fail `unzip`.
- Competition bundle is ~2.5 GB (ultrasound TIFF images under `apo_imgs_v1/`).

## Research log

1. At the start of every session, read `research/log.md` in full before taking any action — especially **Current focus**.
2. Before suggesting a new experiment, architecture, or training approach, scan the experiments table and decisions in `research/log.md` to confirm it was not already tried or ruled out. If unsure, say so and check — do not confidently recommend without verifying.
3. After every scored result or committed decision is reported, update `research/log.md` before doing anything else — including **Current focus** when best results, active notebooks, open questions, or constraints change.
4. At the end of a session (or when wrapping up a run), refresh **Current focus** so the next session has an accurate snapshot; historical detail stays in Experiments / Decisions / Lessons below.

## Kaggle Workflow Rules

**Notebook paths:**
- Kernel source inputs mount at `/kaggle/input/notebooks/{owner}/{kernel-slug}/filename`
- Dataset inputs mount at `/kaggle/input/datasets/{owner}/{dataset-slug}/`
- Competition data mounts at `/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/`
- Model inputs mount at `/kaggle/input/models/{owner}/{model-slug}/pytorch/{variation}/{version}/filename`

**Authentication:**
- Kaggle notebook environments are pre-authenticated as the notebook owner. Never add credential setup cells (`UserSecretsClient`, `KAGGLE_USERNAME`, `KAGGLE_KEY`).

**Loading competition data in notebooks:**
- Use `kagglehub.competition_download("<competition-slug>")` inside the notebook (returns a path under `/kaggle/input/competitions/...`). Do not upload local data as a Kaggle dataset for EDA.

**Prepared dataset workflow (Phase 3+ — Kaggle-native gen notebooks, [birdclef_2026](https://github.com/CodeWithOz/birdclef_2026)):**

1. **Prep notebook** (`notebooks/prep-fasc-timing/`): sole purpose is transform competition data **inside a running Kaggle CPU notebook**. Read TIFFs from `/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/` (or `kagglehub` for EDA). Stretch-align, resize, write PNGs + CSVs to `/kaggle/working/upload/`.
2. **Publish from the notebook**: `subprocess.run(['kaggle', 'datasets', 'version', ...])` with fallback to `datasets create` — no local upload step. Kaggle notebooks are pre-authenticated; never add `UserSecretsClient` / API keys.
3. **Split prep across multiple notebooks** when one session cannot finish (BirdCLEF: `gen-species-1` ranks 1–50, `gen-species-2` ranks 51+). UMUD timing ladder uses separate slugs per tier (`timing-50`, `timing-200`) for now.
4. **Train notebook** (`notebooks/train-mounted/`): `dataset_sources` in `kernel-metadata.json` — mount at `/kaggle/input/datasets/{owner}/{slug}/`. Use `get_image_files` / `rglob`; **no** inline TIFF load, align, or manifest scan.
5. **Benchmark prep and train separately** before full dataset — scale N (50 → 200 → …) on both axes. See `research/log.md` Phase 3 prep/train timing ladder.

**Not the target pattern:** early BirdCLEF local `scripts/generate_spectrogram_batches.py` + manual dataset upload (workaround before Kaggle-native gen settled).

**Dataset zip uploads:**
- Kaggle **automatically extracts zip files** when you upload them to a dataset via `kaggle datasets create` or `kaggle datasets version`. The mounted dataset directory contains extracted files, not zips.
- Extraction structure is not guaranteed to be flat: files may appear at the dataset root OR inside subdirectories named after the zip (e.g. `batch_0001/file.png`).
- **Never `glob('*.zip')` in a mounted dataset directory.** Use `rglob('*.extension')` to find files regardless of directory depth. Build a `{filename: full_path}` lookup:
  ```python
  lookup = {p.name: str(p) for pl_dir in DIRS for p in pl_dir.rglob('*.tif')}
  ```

**Git discipline:**
- Always `git pull` before committing. Kaggle auto-saves can create upstream commits that cause conflicts.
- **Every Kaggle kernel push** must have a matching **git commit on `main` first** — notebook, `kernel-metadata.json`, builder script, and `research/log.md` included. Never push to Kaggle while those changes are still uncommitted. Then `git push` to `origin` so remote stays in sync. Do not ask whether to commit files that are part of a Kaggle version you already pushed.

**Polling:**
- Use the `Monitor` tool with a single persistent background shell for Kaggle kernel status polling (one user permission for the whole loop, not one per CLI call). Activate the venv at the top of the monitor script. Use `kstatus` (not `status`) as the variable name — `status` is read-only in zsh.

**GPU accelerators:**
- Training notebooks that use **fastai** or recent **PyTorch** must use **T4** (`NvidiaTeslaT4`), not P100. `enable_gpu: true` alone defaults to P100, which is incompatible with current fastai/PyTorch on Kaggle.
- Set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` **and** pass `--accelerator NvidiaTeslaT4` on `kaggle kernels push`.
- GPU training with **pretrained encoders** needs `"enable_internet": true` (ImageNet weights download). EDA/geometry kernels can stay offline with `kagglehub`.

**Submission notebooks:**
- `kernel-metadata.json` `model_sources` may be enough on a first push (no stale sidebar entry). Only ask the user to update the Kaggle UI sidebar if submission fails with a model path `FileNotFoundError`.
- If a notebook fails because a model version, dataset, or other input is missing or wrong in the sidebar, ask the user to fix it in the Kaggle UI first — do not restructure notebook code to work around a missing input.

**Leaderboard submission — this is a CODE competition (try CLI first, manual UI as fallback):**

This competition scores a *notebook's output*, not an uploaded CSV, and re-runs the
notebook on a hidden ~2× test set for the **private** leaderboard. There are two ways
to submit a notebook version; **prefer the CLI**, fall back to the manual UI only if
the CLI route fails.

1. **CLI notebook submit (preferred).** After `kaggle kernels push` finishes and
   `kaggle kernels status <kernel>` shows `COMPLETE` (the version has run and produced
   its output), submit that version's output file:
   ```bash
   export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)
   .venv/bin/kaggle competitions submit \
     -c umud-challenge-muscle-architecture-in-ultrasound-data \
     -k ucheozoemena/umud-submission-phase-3 -v <VERSION> \
     -f submission.csv -m "block9-sN ..."
   ```
   **What this command means (verified against CLI `--help` and Kaggle docs):** with
   `-k`/`-v` set, `-f` is **not a local upload** — it names the output file
   (`submission.csv`) that *version V of the kernel already produced when it ran*. The
   command does **not** re-run the notebook; it submits that already-produced output
   for scoring. (`-f`'s help: "File for upload (full path), or the name of the output
   file produced by a kernel (for code competitions).") The hidden-test re-run for the
   private LB happens at competition close, independent of this command. The Kaggle UI
   prints this exact command when you submit a notebook manually.
   - Requires the version's run to have succeeded (output exists). Counts against the
     daily submission quota (5/day). Confirm with `kaggle competitions submissions <comp>`.
2. **Manual UI submit (fallback only).** If the CLI route errors, the user opens the
   notebook version on kaggle.com and clicks **Submit to Competition** → selects its
   `submission.csv` output. Equivalent result, just manual.
- The raw `-f <local.csv>` *without* `-k`/`-v` uploads a static CSV — this scores the
  **public** set but does **not** wire up the notebook for the private re-run. For
  this notebook/code competition, treat static CSV submits as a fallback diagnostic
  only when notebook submission is blocked or broken. Do **not** use CSVs as the
  normal experiment path, because any strong CSV result must then be recreated as an
  equivalent notebook submission for private eligibility, wasting a daily slot.
  If a CSV probe is unavoidable, label it clearly as `static` / `public probe` in
  the submission message and immediately prioritize the matching hidden-safe notebook.

## Lessons Learned

_This section is updated whenever a new lesson is discovered. Any AI agent working on this repo should add entries here proactively — do not wait to be asked._

| Date | Lesson |
|------|--------|
| 2026-06-24 | **Notebook competitions:** submit notebook outputs by default. Static CSV submits are public-only diagnostics/fallbacks, not private-final candidates; any strong CSV needs an equivalent notebook, so avoid CSV probes unless notebook submission is blocked. |
| 2026-06-23 | **MaxViT next tier:** `maxvit_tiny_rw_256` has no ImageNet pretrained weights on Kaggle timm; use **`maxvit_rmlp_tiny_rw_256`** (~28.6M, pretrained @256) as the step up from nano. |
| 2026-06-23 | **`run_block11 restore_prod`:** regenerate local prod submission notebook only — do **not** auto-push to Kaggle (avoids stray prod kernel versions after block runs). |
| 2026-06-23 | **Duplicate submit guard:** always `grep` / check `kaggle competitions submissions` for the message string before `competitions submit` — duplicate CLI calls waste daily slots (block10-cxs-s2 submitted twice). |
| 2026-06-22 | **Code-competition submit:** `kaggle competitions submit -k <kernel> -v <V> -f submission.csv` submits the output file **version V already produced** (does NOT upload a local file, does NOT re-run the kernel). Prefer this CLI route over manual UI; see Kaggle Workflow Rules → Leaderboard submission. The notebook re-runs on hidden 2× data for the private LB at competition close. |
| 2026-06-22 | Notebook submit can differ slightly from the equivalent CSV: v32 notebook scored **1.06750** vs CSV **1.06757** (geometry recomputed on Kaggle → tiny float drift, here a hair better). Submit the notebook, not the static CSV, for the private-eligible entry. |
| 2026-06 | Kaggle auto-extracts dataset zips; `glob('*.zip')` finds nothing. Use `rglob` and a filename→path lookup instead. |
| 2026-06 | Add Python deps with `uv add`, never `uv pip install` or bare `pip`. |
| 2026-06 | Every Kaggle kernel push needs a matching git commit on `main` (notebook + metadata + log) **before** the push; then `git push` to origin. Never Kaggle-ahead-of-uncommitted-git. |
| 2026-06 | fastai / modern PyTorch on Kaggle: use **T4** (`NvidiaTeslaT4`), not P100 (`enable_gpu: true` alone defaults to P100). |
| 2026-06 | **Kaggle-native gen notebooks** publish datasets via `kaggle datasets version/create` from inside the prep kernel (BirdCLEF `gen-species-1/2`, commit `b003ac9`). Split prep when session limit hit. |
| 2026-06 | `kaggle datasets` skips loose folders — zip batches to staging (BirdCLEF `batch_*.zip` + `dataset-metadata.json` alongside). Check stdout for errors; CLI may print success on failure. |
| 2026-06 | Changing `dataset_sources` on an existing kernel may not remount inputs — use `kagglehub.dataset_download` fallback or update Kaggle sidebar. |
| 2026-06 | **Unweighted CE fails** on ~0.3% fasc foreground — use class-weighted `CrossEntropyLossFlat` (see `research/log.md` debug dossier). |
| 2026-06 | **Kaggle CLI auth:** `export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)` before CLI in agent/background shells. |
| 2026-06 | **Exported `.pkl` inference/debug** on Kaggle docker, not local fastai 2.8 — pickle `Resolver` mismatch. |
| 2026-06 | Eval report: metric columns via `type(m).__name__.lower()`, not `str(m)`. Submission CSV: comma sep; `image_id` = full `.tif` filename. |
