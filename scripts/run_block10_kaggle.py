"""Block 10 Kaggle runs: rv2 s2 leaderboard submit + debug-only evals for missing encoders.

Notebook kernel runs produce submission.csv + submission_debug.csv in kernel output
without using a competition submission slot. Only `competitions submit` counts.

Run:
  .venv/bin/python scripts/run_block10_kaggle.py --rv2-only
  .venv/bin/python scripts/run_block10_kaggle.py --debug-only
  .venv/bin/python scripts/run_block10_kaggle.py  # all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv/bin/kaggle"
SUBMIT_BUILD = ROOT / "scripts/build_submission_nb.py"
TRAIN_BUILD = ROOT / "scripts/build_train_apo_gray55_nb.py"
SUBMIT_DIR = ROOT / "notebooks/submission"
TRAIN_DIR = ROOT / "notebooks/train-apo-gray55"
OUT_ROOT = ROOT / "data/kaggle-outputs/block10"
KERNEL_SUBMIT = "ucheozoemena/umud-submission-phase-3"
KERNEL_TRAIN = "ucheozoemena/umud-train-apo-gray55-phase-3"
COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"
POLL_SEC = 30
SUBMIT_POLL_MAX = 150
TRAIN_POLL_MAX = 80

PROD_MODEL = "apo_gray55_line_200_maxvit_nano.pkl"
PROD_KERNEL = "umud-train-encoder-maxvit-nano-phase-3"
PROD_LABEL = "Phase 4 Block 9 s2 — maxvit geometry + calibration (PA18)"


@dataclass(frozen=True)
class ModelJob:
    slug: str
    arch: str
    pkl: str
    apo_kernel: str
    label: str
    train_run: int | None = None
    submit_lb: bool = False
    lb_msg: str = ""
    img_size: int = 256


JOBS: tuple[ModelJob, ...] = (
    ModelJob(
        slug="rv2-s2",
        arch="resnetv2_18",
        pkl="apo_gray55_line_200_rv2_18.pkl",
        apo_kernel="umud-train-encoder-resnetv2-18-phase-3",
        label="Block 10 — resnetv2_18 + Block 9 s2 calibration",
        submit_lb=True,
        lb_msg="block10-rv2-s2",
    ),
    ModelJob(
        slug="r50-debug",
        arch="resnet50",
        pkl="apo_gray55_line_200_r50.pkl",
        apo_kernel=KERNEL_TRAIN.split("/", 1)[1],
        label="Block 10 debug — resnet50 200×5ep + s2 cal",
        train_run=11,
    ),
    ModelJob(
        slug="r34-debug",
        arch="resnet34",
        pkl="apo_gray55_line_200.pkl",
        apo_kernel=KERNEL_TRAIN.split("/", 1)[1],
        label="Block 10 debug — resnet34 200×5ep + s2 cal",
        train_run=7,
    ),
    ModelJob(
        slug="cxs-debug",
        arch="convnext_small",
        pkl="apo_gray55_line_200_cxs.pkl",
        apo_kernel=KERNEL_TRAIN.split("/", 1)[1],
        label="Block 10 debug — convnext_small 200×5ep + s2 cal",
        train_run=15,
    ),
)


def kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("KAGGLE_API_TOKEN", None)
    return env


def run(cmd: list[str], env: dict[str, str] | None = None, retries: int = 2) -> subprocess.CompletedProcess[str]:
    last: subprocess.CalledProcessError | None = None
    for attempt in range(retries):
        print("+", " ".join(cmd), flush=True)
        try:
            return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            last = exc
            print(exc.stdout or "", exc.stderr or "", flush=True)
            if attempt + 1 < retries:
                time.sleep(15 * (attempt + 1))
    assert last is not None
    raise last


def poll_kernel(slug: str, env: dict[str, str], max_loops: int) -> str:
    for i in range(max_loops):
        proc = subprocess.run(
            [str(KAGGLE), "kernels", "status", slug],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        line = proc.stdout.strip().splitlines()[-1]
        print(f"  [{i+1}] {line}", flush=True)
        if "COMPLETE" in line:
            return "complete"
        if "ERROR" in line or "CANCEL" in line:
            return "error"
        time.sleep(POLL_SEC)
    return "timeout"


def parse_kernel_version(stdout: str) -> int | None:
    m = re.search(r"Kernel version (\d+) successfully pushed", stdout)
    return int(m.group(1)) if m else None


def patch_train_run(run_id: int) -> None:
    text = TRAIN_BUILD.read_text()
    text = re.sub(r"^BUILD_TRAIN_RUN = \d+", f"BUILD_TRAIN_RUN = {run_id}", text, count=1, flags=re.M)
    text = re.sub(
        r"^TRAIN_RUN = \d+  # Block 7",
        f"TRAIN_RUN = {run_id}  # Block 7",
        text,
        count=1,
        flags=re.M,
    )
    TRAIN_BUILD.write_text(text)


def patch_submission(job: ModelJob) -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(r'^BUILD_APO_MODEL_FILE = ".*?"', f'BUILD_APO_MODEL_FILE = "{job.pkl}"', text, count=1, flags=re.M)
    text = re.sub(
        r'^BUILD_APO_KERNEL_SLUG = ".*?"',
        f'BUILD_APO_KERNEL_SLUG = "{job.apo_kernel}"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(r'^BUILD_SUBMISSION_LABEL = ".*?"', f'BUILD_SUBMISSION_LABEL = "{job.label}"', text, count=1, flags=re.M)
    text = re.sub(r"^BUILD_IMG_SIZE = \d+", f"BUILD_IMG_SIZE = {job.img_size}", text, count=1, flags=re.M)
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)], env=kaggle_env())
    meta_path = SUBMIT_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        f"ucheozoemena/{job.apo_kernel}",
    ]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def restore_submission_prod() -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(r'^BUILD_APO_MODEL_FILE = ".*?"', f'BUILD_APO_MODEL_FILE = "{PROD_MODEL}"', text, count=1, flags=re.M)
    text = re.sub(r'^BUILD_APO_KERNEL_SLUG = ".*?"', f'BUILD_APO_KERNEL_SLUG = "{PROD_KERNEL}"', text, count=1, flags=re.M)
    text = re.sub(r'^BUILD_SUBMISSION_LABEL = ".*?"', f'BUILD_SUBMISSION_LABEL = "{PROD_LABEL}"', text, count=1, flags=re.M)
    text = re.sub(r"^BUILD_IMG_SIZE = \d+", "BUILD_IMG_SIZE = 256", text, count=1, flags=re.M)
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)], env=kaggle_env())
    meta_path = SUBMIT_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        f"ucheozoemena/{PROD_KERNEL}",
    ]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def analyze_debug(debug_csv: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(debug_csv)
    mtpx = df["mt_px"].to_numpy(float) if "mt_px" in df.columns else df["mt_mm"].to_numpy(float)
    mt_ok = int(np.isfinite(mtpx).sum()) if "mt_px" in df.columns else int(df["mt_mm"].notna().sum())
    n = len(df)
    fails = {}
    if "mt_fail_reason" in df.columns:
        bad = ~np.isfinite(mtpx) if "mt_px" in df.columns else df["mt_mm"].isna()
        if bad.any():
            fails = df.loc[bad, "mt_fail_reason"].value_counts().to_dict()
    return {"n": n, "mt_ok": mt_ok, "mt_ok_pct": round(100 * mt_ok / n, 2), "fail_reasons": fails}


def submit_notebook(version: int, message: str, env: dict[str, str]) -> None:
    token_proc = subprocess.run(
        [str(KAGGLE), "auth", "print-access-token"],
        capture_output=True,
        text=True,
        env=env,
    )
    if token_proc.returncode == 0 and token_proc.stdout.strip():
        env = env.copy()
        env["KAGGLE_API_TOKEN"] = token_proc.stdout.strip()
    run(
        [
            str(KAGGLE),
            "competitions",
            "submit",
            COMPETITION,
            "-k",
            KERNEL_SUBMIT,
            "-v",
            str(version),
            "-f",
            "submission.csv",
            "-m",
            message,
        ],
        env=env,
        retries=3,
    )


def run_job(job: ModelJob, env: dict[str, str]) -> dict:
    out_dir = OUT_ROOT / job.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"slug": job.slug, "arch": job.arch, "pkl": job.pkl}

    if job.train_run is not None:
        print(f"\n--- train TRAIN_RUN={job.train_run} ({job.arch}) ---", flush=True)
        patch_train_run(job.train_run)
        run([sys.executable, str(TRAIN_BUILD)], env=env)
        push = run(
            [str(KAGGLE), "kernels", "push", "-p", str(TRAIN_DIR), "--accelerator", "NvidiaTeslaT4"],
            env=env,
        )
        train_ver = parse_kernel_version(push.stdout or "")
        result["train_version"] = train_ver
        if poll_kernel(KERNEL_TRAIN, env, TRAIN_POLL_MAX) != "complete":
            result["status"] = "train_failed"
            return result

    print(f"\n--- submission eval {job.slug} ---", flush=True)
    patch_submission(job)
    push = run(
        [str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"],
        env=env,
    )
    submit_ver = parse_kernel_version(push.stdout or "")
    result["submit_version"] = submit_ver
    if poll_kernel(KERNEL_SUBMIT, env, SUBMIT_POLL_MAX) != "complete":
        result["status"] = "submit_failed"
        return result

    run([str(KAGGLE), "kernels", "output", KERNEL_SUBMIT, "-p", str(out_dir)], env=env, retries=3)
    debug = out_dir / "submission_debug.csv"
    sub = out_dir / "submission.csv"
    if not debug.exists():
        result["status"] = "missing_debug"
        return result

    stats = analyze_debug(debug)
    result.update(stats)
    result["status"] = "ok"
    print(f"  => mt_ok {stats['mt_ok']}/{stats['n']} ({stats['mt_ok_pct']}%)", flush=True)

    if job.submit_lb and submit_ver is not None:
        print(f"  => leaderboard notebook submit v{submit_ver}: {job.lb_msg}", flush=True)
        submit_notebook(submit_ver, job.lb_msg, env)
        result["lb_submitted"] = True
        result["lb_msg"] = job.lb_msg
    else:
        print("  => debug only (no competition submit)", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv2-only", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument("--restore-prod", action="store_true")
    args = parser.parse_args()

    if args.restore_prod:
        restore_submission_prod()
        run([str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"], env=kaggle_env())
        return

    if args.rv2_only:
        jobs = [JOBS[0]]
    elif args.debug_only:
        jobs = list(JOBS[1:])
    else:
        jobs = list(JOBS)

    env = kaggle_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for job in jobs:
        print(f"\n{'='*60}\nBlock 10 job: {job.slug} ({job.arch})\n{'='*60}", flush=True)
        try:
            results.append(run_job(job, env))
        except subprocess.CalledProcessError as exc:
            results.append({"slug": job.slug, "arch": job.arch, "status": f"cli_error: {exc}"})

    summary = OUT_ROOT / "kaggle_runs.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary}", flush=True)

    restore_submission_prod()
    run([str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"], env=env)


if __name__ == "__main__":
    main()
