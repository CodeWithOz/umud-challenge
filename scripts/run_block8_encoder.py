"""Train one Block 8 encoder, run test mt_ok eval, optional leaderboard submit."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from block8_encoders import BLOCK8_ENCODERS, EncoderSpec, get_encoder

KAGGLE = ROOT / ".venv/bin/kaggle"
SUBMIT_BUILD = ROOT / "scripts/build_submission_nb.py"
SUBMIT_DIR = ROOT / "notebooks/submission"
KERNEL_SUBMIT = "ucheozoemena/umud-submission-phase-3"
OUT_ROOT = ROOT / "data/kaggle-outputs/block8"
POLL_SEC = 25


def kaggle_env() -> dict[str, str]:
    return os.environ.copy()


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


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


def patch_submission(enc: EncoderSpec) -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(
        r'^BUILD_APO_MODEL_FILE = ".*?"',
        f'BUILD_APO_MODEL_FILE = "{enc.export_name}"',
        text,
        count=1,
        flags=re.M,
    )
    kernel_slug = enc.kernel_id.split("/", 1)[1]
    text = re.sub(
        r'^BUILD_APO_KERNEL_SLUG = ".*?"',
        f'BUILD_APO_KERNEL_SLUG = "{kernel_slug}"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r'^BUILD_SUBMISSION_LABEL = ".*?"',
        f'BUILD_SUBMISSION_LABEL = "Block 8 test eval — {enc.arch} 200×5ep + MM=0.075"',
        text,
        count=1,
        flags=re.M,
    )
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)])
    meta_path = SUBMIT_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        enc.kernel_id,
    ]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def restore_submission_prod() -> None:
    text = SUBMIT_BUILD.read_text()
    text = re.sub(
        r'^BUILD_APO_MODEL_FILE = ".*?"',
        'BUILD_APO_MODEL_FILE = "apo_gray55_line_200_r18.pkl"',
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
        'BUILD_SUBMISSION_LABEL = "Phase 4 production — 200-tier apo resnet18 5ep + MM=0.075"',
        text,
        count=1,
        flags=re.M,
    )
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)])
    meta_path = SUBMIT_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        "ucheozoemena/umud-train-apo-gray55-phase-3",
    ]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def analyze_test(debug_csv: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(debug_csv)
    bad = df[["pa_deg", "fl_mm", "mt_mm"]].isna().any(axis=1)
    mt_ok = int((~bad).sum())
    n = len(df)
    fails = {}
    if "mt_fail_reason" in df.columns and bad.any():
        fails = df.loc[bad, "mt_fail_reason"].value_counts().to_dict()
    return {"n": n, "mt_ok": mt_ok, "mt_ok_pct": round(100 * mt_ok / n, 2), "fail_reasons": fails}


def run_encoder(enc: EncoderSpec, env: dict[str, str], submit_lb: bool) -> dict:
    out_dir = OUT_ROOT / enc.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = ROOT / enc.notebook_dir

    run([sys.executable, str(ROOT / "scripts/build_train_encoder_nb.py"), "--slug", enc.slug])
    run([str(KAGGLE), "kernels", "push", "-p", str(train_dir), "--accelerator", "NvidiaTeslaT4"], env=env)
    if poll_kernel(enc.kernel_id, env, 80) != "complete":
        return {"slug": enc.slug, "status": "train_failed"}

    run([str(KAGGLE), "kernels", "output", enc.kernel_id, "-p", str(out_dir / "train")], env=env)
    timing = out_dir / "train" / "timing_report.csv"
    val_row = {}
    if timing.exists():
        import pandas as pd

        val_row = pd.read_csv(timing).iloc[0].to_dict()

    patch_submission(enc)
    run([str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"], env=env)
    if poll_kernel(KERNEL_SUBMIT, env, 120) != "complete":
        return {"slug": enc.slug, "status": "submit_failed", **val_row}

    run([str(KAGGLE), "kernels", "output", KERNEL_SUBMIT, "-p", str(out_dir / "submit")], env=env)
    debug = out_dir / "submit" / "submission_debug.csv"
    if not debug.exists():
        return {"slug": enc.slug, "status": "missing_debug", **val_row}

    stats = analyze_test(debug)
    result = {"slug": enc.slug, "arch": enc.arch, "family": enc.family, "status": "ok", **val_row, **stats}
    print(f"  => test mt_ok {stats['mt_ok']}/{stats['n']} ({stats['mt_ok_pct']}%)", flush=True)

    if submit_lb and stats["mt_ok_pct"] == 100.0:
        sub_csv = out_dir / "submit" / "submission.csv"
        run(
            [
                str(KAGGLE),
                "competitions",
                "submit",
                "umud-challenge-muscle-architecture-in-ultrasound-data",
                "-f",
                str(sub_csv),
                "-m",
                f"block8-{enc.slug}-200x5ep",
            ],
            env=env,
        )
        result["lb_submitted"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="Run one encoder")
    parser.add_argument("--all", action="store_true", help="Run all Block 8 encoders in order")
    parser.add_argument("--submit-lb", action="store_true", help="Leaderboard submit if 100% test mt_ok")
    parser.add_argument("--restore-prod", action="store_true", help="Restore submission notebook to r50 prod")
    args = parser.parse_args()

    if args.restore_prod:
        restore_submission_prod()
        return

    if args.all:
        encoders = list(BLOCK8_ENCODERS)
    elif args.slug:
        encoders = [get_encoder(args.slug)]
    else:
        parser.error("Pass --slug or --all")

    env = kaggle_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for enc in encoders:
        print(f"\n=== Block 8 {enc.family} ({enc.slug}) ===", flush=True)
        results.append(run_encoder(enc, env, args.submit_lb))

    summary_path = OUT_ROOT / "results.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}", flush=True)
    restore_submission_prod()


if __name__ == "__main__":
    main()
