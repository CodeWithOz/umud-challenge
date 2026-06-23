"""Block 11 — parallel train (maxvit-tiny + convnext_base) then s2 graded submits.

Run: .venv/bin/python scripts/run_block11.py
     .venv/bin/python scripts/run_block11.py --train-only
     .venv/bin/python scripts/run_block11.py --submit-only
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from block8_encoders import get_encoder

KAGGLE = ROOT / ".venv/bin/kaggle"
SUBMIT_BUILD = ROOT / "scripts/build_submission_nb.py"
SUBMIT_DIR = ROOT / "notebooks/submission"
TRAIN_GRAY55_DIR = ROOT / "notebooks/train-apo-gray55"
TRAIN_GRAY55_BUILD = ROOT / "scripts/build_train_apo_gray55_nb.py"
KERNEL_SUBMIT = "ucheozoemena/umud-submission-phase-3"
KERNEL_GRAY55 = "ucheozoemena/umud-train-apo-gray55-phase-3"
COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"
OUT_ROOT = ROOT / "data/kaggle-outputs/block11"
POLL_SEC = 30
TRAIN_POLL_MAX = 90
SUBMIT_POLL_MAX = 150

PROD_MODEL = "apo_gray55_line_200_cxs.pkl"
PROD_KERNEL = "umud-train-apo-gray55-phase-3"
PROD_LABEL = "Phase 4 Block 10 prod — convnext_small + Block 9 s2 (LB 1.04862)"


@dataclass(frozen=True)
class Block11Job:
    slug: str
    arch: str
    pkl: str
    apo_kernel: str
    lb_msg: str
    train_kernel: str
    train_dir: Path
    kind: str  # "encoder" | "gray55"


JOBS: tuple[Block11Job, ...] = (
    Block11Job(
        slug="maxvit-tiny",
        arch="maxvit_rmlp_tiny_rw_256",
        pkl="apo_gray55_line_200_maxvit_tiny.pkl",
        apo_kernel="umud-train-encoder-maxvit-tiny-phase-3",
        lb_msg="block11-maxvit-tiny-s2",
        train_kernel="ucheozoemena/umud-train-encoder-maxvit-tiny-phase-3",
        train_dir=ROOT / "notebooks/train-encoder-maxvit-tiny",
        kind="encoder",
    ),
    Block11Job(
        slug="convnext-base",
        arch="convnext_base",
        pkl="apo_gray55_line_200_cnxb.pkl",
        apo_kernel=KERNEL_GRAY55.split("/", 1)[1],
        lb_msg="block11-cnxb-s2",
        train_kernel=KERNEL_GRAY55,
        train_dir=TRAIN_GRAY55_DIR,
        kind="gray55",
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
            print((exc.stdout or "") + (exc.stderr or ""), flush=True)
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
        print(f"  [{slug}] [{i+1}] {line}", flush=True)
        if "COMPLETE" in line:
            return "complete"
        if "ERROR" in line or "CANCEL" in line:
            return "error"
        time.sleep(POLL_SEC)
    return "timeout"


def parse_version(stdout: str) -> int | None:
    m = re.search(r"Kernel version (\d+) successfully pushed", stdout)
    return int(m.group(1)) if m else None


def already_submitted(msg: str, env: dict[str, str]) -> bool:
    proc = subprocess.run(
        [str(KAGGLE), "competitions", "submissions", COMPETITION, "-v"],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0 and msg in proc.stdout


def patch_submission(job: Block11Job) -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(r'^BUILD_APO_MODEL_FILE = ".*?"', f'BUILD_APO_MODEL_FILE = "{job.pkl}"', text, count=1, flags=re.M)
    text = re.sub(
        r'^BUILD_APO_KERNEL_SLUG = ".*?"',
        f'BUILD_APO_KERNEL_SLUG = "{job.apo_kernel}"',
        text,
        count=1,
        flags=re.M,
    )
    label = f"Block 11 — {job.arch} + Block 9 s2 calibration"
    text = re.sub(r'^BUILD_SUBMISSION_LABEL = ".*?"', f'BUILD_SUBMISSION_LABEL = "{label}"', text, count=1, flags=re.M)
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)], env=kaggle_env())
    meta = json.loads((SUBMIT_DIR / "kernel-metadata.json").read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        f"ucheozoemena/{job.apo_kernel}",
    ]
    (SUBMIT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def api_token(env: dict[str, str]) -> str | None:
    tok = subprocess.run(
        [str(KAGGLE), "auth", "print-access-token"],
        capture_output=True,
        text=True,
        env=env,
    )
    if tok.returncode != 0:
        return None
    for line in reversed(tok.stdout.splitlines()):
        line = line.strip()
        if line and not line.startswith("Warning:"):
            return line
    return None


def restore_prod() -> None:
    """Regenerate local prod submission notebook only — no Kaggle push (avoids stray v41-style runs)."""
    text = SUBMIT_BUILD.read_text()
    text = re.sub(r'^BUILD_APO_MODEL_FILE = ".*?"', f'BUILD_APO_MODEL_FILE = "{PROD_MODEL}"', text, count=1, flags=re.M)
    text = re.sub(r'^BUILD_APO_KERNEL_SLUG = ".*?"', f'BUILD_APO_KERNEL_SLUG = "{PROD_KERNEL}"', text, count=1, flags=re.M)
    text = re.sub(r'^BUILD_SUBMISSION_LABEL = ".*?"', f'BUILD_SUBMISSION_LABEL = "{PROD_LABEL}"', text, count=1, flags=re.M)
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)], env=kaggle_env())
    meta = json.loads((SUBMIT_DIR / "kernel-metadata.json").read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        f"ucheozoemena/{PROD_KERNEL}",
    ]
    (SUBMIT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def push_train(job: Block11Job, env: dict[str, str]) -> dict:
    out_dir = OUT_ROOT / job.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    if job.kind == "encoder":
        run([sys.executable, str(ROOT / "scripts/build_train_encoder_nb.py"), "--slug", job.slug], env=env)
    else:
        run([sys.executable, str(TRAIN_GRAY55_BUILD)], env=env)
    push = run(
        [str(KAGGLE), "kernels", "push", "-p", str(job.train_dir), "--accelerator", "NvidiaTeslaT4"],
        env=env,
    )
    ver = parse_version(push.stdout or "")
    status = poll_kernel(job.train_kernel, env, TRAIN_POLL_MAX)
    result = {"slug": job.slug, "train_version": ver, "train_status": status}
    if status == "complete":
        run([str(KAGGLE), "kernels", "output", job.train_kernel, "-p", str(out_dir / "train")], env=env, retries=3)
    return result


def submit_job(job: Block11Job, env: dict[str, str]) -> dict:
    out_dir = OUT_ROOT / job.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if already_submitted(job.lb_msg, env):
        print(f"  SKIP LB: {job.lb_msg} already on leaderboard", flush=True)
        return {"slug": job.slug, "status": "lb_skipped_duplicate", "lb_msg": job.lb_msg}

    patch_submission(job)
    push = run(
        [str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"],
        env=env,
    )
    submit_ver = parse_version(push.stdout or "")
    if poll_kernel(KERNEL_SUBMIT, env, SUBMIT_POLL_MAX) != "complete":
        return {"slug": job.slug, "status": "submit_failed", "submit_version": submit_ver}

    run([str(KAGGLE), "kernels", "output", KERNEL_SUBMIT, "-p", str(out_dir / "submit")], env=env, retries=3)

    # Idempotency: re-check before spending a submission slot.
    if already_submitted(job.lb_msg, env):
        print(f"  SKIP LB (post-notebook): {job.lb_msg} already submitted", flush=True)
        return {"slug": job.slug, "status": "lb_skipped_duplicate", "submit_version": submit_ver}

    token_env = env.copy()
    token = api_token(env)
    if token:
        token_env["KAGGLE_API_TOKEN"] = token
    run(
        [
            str(KAGGLE),
            "competitions",
            "submit",
            COMPETITION,
            "-k",
            KERNEL_SUBMIT,
            "-v",
            str(submit_ver),
            "-f",
            "submission.csv",
            "-m",
            job.lb_msg,
        ],
        env=token_env,
        retries=2,
    )
    return {
        "slug": job.slug,
        "status": "ok",
        "submit_version": submit_ver,
        "lb_submitted": True,
        "lb_msg": job.lb_msg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    args = parser.parse_args()
    env = kaggle_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    if not args.submit_only:
        print("=== Block 11: parallel train pushes ===", flush=True)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(push_train, job, env): job for job in JOBS}
            for fut in concurrent.futures.as_completed(futs):
                job = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"  train done: {job.slug} -> {results[-1]}", flush=True)
                except Exception as exc:
                    results.append({"slug": job.slug, "train_status": f"error: {exc}"})

    if not args.train_only:
        print("\n=== Block 11: sequential graded submits (idempotent) ===", flush=True)
        for job in JOBS:
            tr = next((r for r in results if r.get("slug") == job.slug), None)
            if tr and tr.get("train_status") not in (None, "complete") and not args.submit_only:
                print(f"  skip submit {job.slug}: train {tr.get('train_status')}", flush=True)
                continue
            try:
                sub = submit_job(job, env)
                results.append(sub)
            except Exception as exc:
                results.append({"slug": job.slug, "status": f"submit_error: {exc}"})

    (OUT_ROOT / "block11_runs.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_ROOT / 'block11_runs.json'}", flush=True)
    restore_prod()


if __name__ == "__main__":
    main()
