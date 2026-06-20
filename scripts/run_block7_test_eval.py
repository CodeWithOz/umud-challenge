"""Run train + submission eval for Block 7 encoders not yet scored on test."""
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
OUT_DIR = ROOT / "data/kaggle-outputs/block7-test-eval"
KERNEL_TRAIN = "ucheozoemena/umud-train-apo-gray55-phase-3"
KERNEL_SUBMIT = "ucheozoemena/umud-submission-phase-3"

# convnext_tiny (14) already tested at 70.9% mt_ok — skip
ENCODERS = [
    (18, "mobilenetv3_small_100", "apo_gray55_line_200_mnv3.pkl"),
    (19, "regnetx_004", "apo_gray55_line_200_rgx004.pkl"),
    (13, "resnet18", "apo_gray55_line_200_r18.pkl"),
    (16, "efficientnet_b0", "apo_gray55_line_200_enb0.pkl"),
    (17, "efficientnet_b1", "apo_gray55_line_200_enb1.pkl"),
]

POLL_SEC = 25
TRAIN_POLL_MAX = 60
SUBMIT_POLL_MAX = 120


def kaggle_env() -> dict[str, str]:
    """Use existing auth; avoid print-access-token (rate-limited)."""
    return os.environ.copy()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


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


def patch_submission_model(pkl: str, label: str) -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(
        r'^BUILD_APO_MODEL_FILE = ".*?"',
        f'BUILD_APO_MODEL_FILE = "{pkl}"',
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
        f'BUILD_SUBMISSION_LABEL = "{label}"',
        text,
        count=1,
        flags=re.M,
    )
    SUBMIT_BUILD.write_text(text)


def restore_submission_prod() -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(
        r'^BUILD_APO_MODEL_FILE = ".*?"',
        'BUILD_APO_MODEL_FILE = "apo_gray55_line_200_r50.pkl"',
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
        'BUILD_SUBMISSION_LABEL = "Phase 4 production — 200-tier apo r50 5ep + MM=0.075"',
        text,
        count=1,
        flags=re.M,
    )
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)])


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


def analyze_debug(debug_csv: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(debug_csv)
    nan_row = df[["pa_deg", "fl_mm", "mt_mm"]].isna().any(axis=1)
    mt_ok = int((~nan_row).sum())
    n = len(df)
    fails = {}
    if "mt_fail_reason" in df.columns and nan_row.any():
        fails = df.loc[nan_row, "mt_fail_reason"].value_counts().to_dict()
    return {
        "n": n,
        "mt_ok": mt_ok,
        "mt_ok_pct": round(100 * mt_ok / n, 2),
        "mt_nan": int(nan_row.sum()),
        "fail_reasons": fails,
    }


def main() -> None:
    env = kaggle_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for run_id, arch, pkl in ENCODERS:
        slug_dir = OUT_DIR / f"run{run_id}-{arch}"
        slug_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== TRAIN_RUN={run_id} {arch} ===", flush=True)

        patch_train_run(run_id)
        run([sys.executable, str(TRAIN_BUILD)])
        run([str(KAGGLE), "kernels", "push", "-p", str(TRAIN_DIR), "--accelerator", "NvidiaTeslaT4"], env=env)

        train_status = poll_kernel(KERNEL_TRAIN, env, TRAIN_POLL_MAX)
        if train_status != "complete":
            results.append({"train_run": run_id, "arch": arch, "status": f"train_{train_status}"})
            continue

        label = f"Block 7 test eval — {arch} 200×5ep + MM=0.075"
        patch_submission_model(pkl, label)
        run([sys.executable, str(SUBMIT_BUILD)])
        run([str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"], env=env)

        submit_status = poll_kernel(KERNEL_SUBMIT, env, SUBMIT_POLL_MAX)
        if submit_status != "complete":
            results.append({"train_run": run_id, "arch": arch, "status": f"submit_{submit_status}"})
            continue

        run([str(KAGGLE), "kernels", "output", KERNEL_SUBMIT, "-p", str(slug_dir)], env=env)
        debug = slug_dir / "submission_debug.csv"
        if not debug.exists():
            results.append({"train_run": run_id, "arch": arch, "status": "missing_debug_csv"})
            continue

        stats = analyze_debug(debug)
        row = {"train_run": run_id, "arch": arch, "pkl": pkl, "status": "ok", **stats}
        results.append(row)
        print(f"  => mt_ok {row['mt_ok']}/{row['n']} ({row['mt_ok_pct']}%)", flush=True)

        if row["mt_ok_pct"] == 100.0:
            sub_csv = slug_dir / "submission.csv"
            msg = f"block7-{arch}-200x5ep"
            run(
                [str(KAGGLE), "competitions", "submit", "umud-challenge-muscle-architecture-in-ultrasound-data", "-f", str(sub_csv), "-m", msg],
                env=env,
            )
            print(f"  => leaderboard submit sent: {msg}", flush=True)

    summary_path = OUT_DIR / "test_eval_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}", flush=True)
    for r in results:
        if r.get("status") == "ok":
            print(f"  {r['arch']}: {r['mt_ok_pct']}% mt_ok ({r['mt_ok']}/{r['n']})")

    restore_submission_prod()
    run([str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"], env=env)
    patch_train_run(11)
    run([sys.executable, str(TRAIN_BUILD)])


if __name__ == "__main__":
    main()
