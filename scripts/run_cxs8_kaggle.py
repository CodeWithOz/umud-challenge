"""Train convnext_small 200×8ep (TRAIN_RUN=21) and graded s2 submit.

Run: .venv/bin/python scripts/run_cxs8_kaggle.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv/bin/kaggle"
TRAIN_BUILD = ROOT / "scripts/build_train_apo_gray55_nb.py"
SUBMIT_BUILD = ROOT / "scripts/build_submission_nb.py"
TRAIN_DIR = ROOT / "notebooks/train-apo-gray55"
SUBMIT_DIR = ROOT / "notebooks/submission"
OUT_ROOT = ROOT / "data/kaggle-outputs/block12-cxs8"
KERNEL_TRAIN = "ucheozoemena/umud-train-apo-gray55-phase-3"
KERNEL_SUBMIT = "ucheozoemena/umud-submission-phase-3"
COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"
PKL = "apo_gray55_line_200_cxs8.pkl"
LB_MSG = "block12-cxs8-s2"
POLL_SEC = 30
TRAIN_POLL_MAX = 90
SUBMIT_POLL_MAX = 150


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


def patch_submission() -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(r'^BUILD_APO_MODEL_FILE = ".*?"', f'BUILD_APO_MODEL_FILE = "{PKL}"', text, count=1, flags=re.M)
    text = re.sub(
        r'^BUILD_APO_KERNEL_SLUG = ".*?"',
        'BUILD_APO_KERNEL_SLUG = "umud-train-apo-gray55-phase-3"',
        text,
        count=1,
        flags=re.M,
    )
    label = "Block 12 — convnext_small 200×8ep + Block 9 s2 calibration"
    text = re.sub(r'^BUILD_SUBMISSION_LABEL = ".*?"', f'BUILD_SUBMISSION_LABEL = "{label}"', text, count=1, flags=re.M)
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)], env=kaggle_env())
    meta = json.loads((SUBMIT_DIR / "kernel-metadata.json").read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        "ucheozoemena/umud-train-apo-gray55-phase-3",
    ]
    (SUBMIT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def restore_prod() -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(
        r'^BUILD_APO_MODEL_FILE = ".*?"',
        'BUILD_APO_MODEL_FILE = "apo_gray55_line_200_cxs.pkl"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r'^BUILD_APO_KERNEL_SLUG = ".*?"',
        'BUILD_APO_KERNEL_SLUG = "umud-train-apo-gray55-phase-3"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r'^BUILD_SUBMISSION_LABEL = ".*?"',
        'BUILD_SUBMISSION_LABEL = "Phase 4 Block 10 prod — convnext_small + Block 9 s2 (LB 1.04862)"',
        text,
        count=1,
        flags=re.M,
    )
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)], env=kaggle_env())


def main() -> None:
    env = kaggle_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if already_submitted(LB_MSG, env):
        print(f"SKIP: {LB_MSG} already on leaderboard", flush=True)
        return

    print("=== Train convnext_small 200×8ep ===", flush=True)
    run([sys.executable, str(TRAIN_BUILD)], env=env)
    push = run(
        [str(KAGGLE), "kernels", "push", "-p", str(TRAIN_DIR), "--accelerator", "NvidiaTeslaT4"],
        env=env,
    )
    train_ver = parse_version(push.stdout or "")
    train_status = poll_kernel(KERNEL_TRAIN, env, TRAIN_POLL_MAX)
    result["train_version"] = train_ver
    result["train_status"] = train_status
    if train_status != "complete":
        (OUT_ROOT / "cxs8_runs.json").write_text(json.dumps(result, indent=2))
        raise SystemExit(f"Train failed: {train_status}")

    run([str(KAGGLE), "kernels", "output", KERNEL_TRAIN, "-p", str(OUT_ROOT / "train")], env=env, retries=3)

    print("\n=== Graded submit (s2) ===", flush=True)
    patch_submission()
    push = run(
        [str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"],
        env=env,
    )
    submit_ver = parse_version(push.stdout or "")
    if poll_kernel(KERNEL_SUBMIT, env, SUBMIT_POLL_MAX) != "complete":
        result["submit_version"] = submit_ver
        result["status"] = "submit_failed"
        (OUT_ROOT / "cxs8_runs.json").write_text(json.dumps(result, indent=2))
        raise SystemExit("Submission notebook failed")

    run([str(KAGGLE), "kernels", "output", KERNEL_SUBMIT, "-p", str(OUT_ROOT / "submit")], env=env, retries=3)

    if already_submitted(LB_MSG, env):
        print(f"SKIP LB (post-notebook): {LB_MSG} already submitted", flush=True)
        result.update({"status": "lb_skipped_duplicate", "submit_version": submit_ver})
    else:
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
                LB_MSG,
            ],
            env=token_env,
            retries=2,
        )
        result.update({"status": "ok", "submit_version": submit_ver, "lb_msg": LB_MSG})

    (OUT_ROOT / "cxs8_runs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {OUT_ROOT / 'cxs8_runs.json'}", flush=True)
    restore_prod()


if __name__ == "__main__":
    main()
