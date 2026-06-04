## Development Environment

- Use the existing `.venv` and `uv` for all Python tasks. Never use `pip` or system `python`.
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

**Dataset zip uploads:**
- Kaggle **automatically extracts zip files** when you upload them to a dataset via `kaggle datasets create` or `kaggle datasets version`. The mounted dataset directory contains extracted files, not zips.
- Extraction structure is not guaranteed to be flat: files may appear at the dataset root OR inside subdirectories named after the zip (e.g. `batch_0001/file.png`).
- **Never `glob('*.zip')` in a mounted dataset directory.** Use `rglob('*.extension')` to find files regardless of directory depth. Build a `{filename: full_path}` lookup:
  ```python
  lookup = {p.name: str(p) for pl_dir in DIRS for p in pl_dir.rglob('*.tif')}
  ```

**Git discipline:**
- Always `git pull` before committing. Kaggle auto-saves can create upstream commits that cause conflicts.

**Polling:**
- Use the `Monitor` tool with a single persistent background shell for Kaggle kernel status polling (one user permission for the whole loop, not one per CLI call). Activate the venv at the top of the monitor script. Use `kstatus` (not `status`) as the variable name — `status` is read-only in zsh.

**Submission notebooks:**
- `kernel-metadata.json` `model_sources` may be enough on a first push (no stale sidebar entry). Only ask the user to update the Kaggle UI sidebar if submission fails with a model path `FileNotFoundError`.
- If a notebook fails because a model version, dataset, or other input is missing or wrong in the sidebar, ask the user to fix it in the Kaggle UI first — do not restructure notebook code to work around a missing input.

## Lessons Learned

_This section is updated whenever a new lesson is discovered. Any AI agent working on this repo should add entries here proactively — do not wait to be asked._

| Date | Lesson |
|------|--------|
| 2026-06 | Kaggle auto-extracts dataset zips; `glob('*.zip')` finds nothing. Use `rglob` and a filename→path lookup instead. |
