"""Train one Block 8 encoder, run test mt_ok eval, optional leaderboard submit."""
from __future__ import annotations

import argparse
import csv
import io
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
COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"
OUT_ROOT = ROOT / "data/kaggle-outputs/block8"
LOG_PATH = ROOT / "research/log.md"
POLL_SEC = 25
LB_POLL_SEC = 30

PROD_MODEL = "apo_gray55_line_200_maxvit_nano.pkl"
PROD_KERNEL = "umud-train-encoder-maxvit-nano-phase-3"
PROD_LABEL = "Phase 4 production — 200-tier apo maxvit-nano 5ep + MM=0.075"

# Pre-scored encoders (before LB polling was added)
PRESCORED_LB: dict[str, float] = {
    "levit128s": 1.91255,
    "resnetv2-18": 1.84197,
}


def kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    # OAuth via ~/.kaggle/credentials.json — do not inject stale KAGGLE_API_TOKEN.
    env.pop("KAGGLE_API_TOKEN", None)
    return env


def run(cmd: list[str], env: dict[str, str] | None = None, retries: int = 1) -> None:
    last_err: subprocess.CalledProcessError | None = None
    for attempt in range(retries):
        print("+", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
            if attempt + 1 < retries:
                wait = 15 * (attempt + 1)
                print(f"  retry {attempt + 2}/{retries} in {wait}s ({exc})", flush=True)
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def train_kernel_complete(slug: str, env: dict[str, str]) -> bool:
    proc = subprocess.run(
        [str(KAGGLE), "kernels", "status", slug],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0 and "COMPLETE" in proc.stdout


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


def fetch_submissions(env: dict[str, str], retries: int = 5) -> list[dict[str, str]]:
    last_err: subprocess.CalledProcessError | None = None
    for attempt in range(retries):
        proc = subprocess.run(
            [str(KAGGLE), "competitions", "submissions", COMPETITION, "-v"],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode == 0:
            reader = csv.DictReader(io.StringIO(proc.stdout))
            return list(reader)
        last_err = subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)
        wait = 20 * (attempt + 1)
        print(f"  submissions API failed ({proc.stderr.strip() or proc.stdout.strip()}); retry in {wait}s", flush=True)
        time.sleep(wait)
    assert last_err is not None
    raise last_err


def poll_lb_score(description: str, env: dict[str, str], max_loops: int = 40) -> float | None:
    auth_failures = 0
    for i in range(max_loops):
        try:
            rows = fetch_submissions(env, retries=2)
            auth_failures = 0
        except subprocess.CalledProcessError:
            auth_failures += 1
            if auth_failures >= 5:
                print(f"  => LB poll aborted after {auth_failures} auth failures", flush=True)
                return None
            time.sleep(LB_POLL_SEC)
            continue
        for row in rows:
            if row.get("description", "").strip() != description:
                continue
            status = row.get("status", "")
            score = (row.get("publicScore") or "").strip()
            print(f"  [LB {i+1}] {description}: {status} score={score or 'pending'}", flush=True)
            if "ERROR" in status:
                return None
            if "COMPLETE" in status and score:
                return float(score)
            break
        time.sleep(LB_POLL_SEC)
    return None


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
    text = re.sub(
        r"^BUILD_IMG_SIZE = \d+",
        f"BUILD_IMG_SIZE = {enc.img_size}",
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
        f'BUILD_APO_MODEL_FILE = "{PROD_MODEL}"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r'^BUILD_APO_KERNEL_SLUG = ".*?"',
        f'BUILD_APO_KERNEL_SLUG = "{PROD_KERNEL}"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r'^BUILD_SUBMISSION_LABEL = ".*?"',
        f'BUILD_SUBMISSION_LABEL = "{PROD_LABEL}"',
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(r"^BUILD_IMG_SIZE = \d+", "BUILD_IMG_SIZE = 256", text, count=1, flags=re.M)
    SUBMIT_BUILD.write_text(text)
    run([sys.executable, str(SUBMIT_BUILD)])
    meta_path = SUBMIT_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kernel_sources"] = [
        "ucheozoemena/umud-train-mounted-phase-3",
        f"ucheozoemena/{PROD_KERNEL}",
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


def load_results() -> dict[str, dict]:
    path = OUT_ROOT / "results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {row["slug"]: row for row in data}
    return data


def save_results(results: dict[str, dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ordered = [results[e.slug] for e in BLOCK8_ENCODERS if e.slug in results]
    (OUT_ROOT / "results.json").write_text(json.dumps(ordered, indent=2))


def update_log_row(result: dict) -> None:
    if not LOG_PATH.exists():
        return
    text = LOG_PATH.read_text()
    slug = result["slug"]
    lb = result.get("lb_score")
    lb_cell = f"**{lb:.5f}**" if lb is not None else ("rejected" if result.get("mt_ok_pct", 0) < 100 else "pending")
    val_dice = result.get("val_dice")
    val_umud = result.get("val_umud_score")
    val_mt = result.get("val_mt_ok_pct")
    test_pct = result.get("mt_ok_pct")
    note = result.get("log_note", "")
    row = (
        f"| `{slug}` | {val_dice:.3f} | {val_umud:.3f} | {val_mt:.1f}% | "
        f"{'**100%**' if test_pct == 100 else f'{test_pct}%'} | {lb_cell} | {note} |"
        if val_dice is not None and val_umud is not None and val_mt is not None
        else None
    )
    if row and f"| `{slug}` |" in text:
        text = re.sub(rf"\| `{re.escape(slug)}` \|[^\n]+\n", row + "\n", text, count=1)
    status_word = "**complete**" if result.get("status") == "ok" else result.get("status", "pending")
    text = re.sub(
        rf"(\| \d+ \| `{re.escape(slug)}` \|[^\|]+\|[^\|]+\|[^\|]+\| )pending( \|)",
        rf"\1{status_word}\2",
        text,
        count=1,
    )
    LOG_PATH.write_text(text)


def load_cached_eval(enc: EncoderSpec, out_dir: Path) -> dict | None:
    timing = out_dir / "train" / "timing_report.csv"
    debug = out_dir / "submit" / "submission_debug.csv"
    if not timing.exists() or not debug.exists():
        return None
    import pandas as pd

    val_row = pd.read_csv(timing).iloc[0].to_dict()
    stats = analyze_test(debug)
    result: dict = {
        "slug": enc.slug,
        "arch": enc.arch,
        "family": enc.family,
        "status": "ok",
        **val_row,
        **stats,
    }
    print(
        f"  cached eval: test mt_ok {stats['mt_ok']}/{stats['n']} ({stats['mt_ok_pct']}%)",
        flush=True,
    )
    if stats["mt_ok_pct"] < 100.0:
        result["status"] = "rejected_geometry"
        result["log_note"] = f"test mt_ok {stats['mt_ok_pct']}% — skipped LB"
    return result


def finish_lb(enc: EncoderSpec, result: dict, out_dir: Path, env: dict[str, str], submit_lb: bool, poll_lb: bool) -> dict:
    lb_msg = f"block8-{enc.slug}-200x5ep"
    if result.get("status") != "ok":
        return result
    if poll_lb:
        lb_score = poll_lb_score(lb_msg, env)
        if lb_score is not None:
            result["lb_submitted"] = True
            result["lb_score"] = lb_score
            print(f"  => LB score {lb_score:.5f}", flush=True)
            return result
    if not submit_lb:
        return result
    sub_csv = out_dir / "submit" / "submission.csv"
    if not sub_csv.exists():
        return result
    run(
        [
            str(KAGGLE),
            "competitions",
            "submit",
            COMPETITION,
            "-f",
            str(sub_csv),
            "-m",
            lb_msg,
        ],
        env=env,
        retries=3,
    )
    result["lb_submitted"] = True
    if poll_lb:
        lb_score = poll_lb_score(lb_msg, env)
        if lb_score is not None:
            result["lb_score"] = lb_score
            print(f"  => LB score {lb_score:.5f}", flush=True)
        else:
            print("  => LB score not available (timeout or error)", flush=True)
    return result


def run_encoder(
    enc: EncoderSpec,
    env: dict[str, str],
    submit_lb: bool,
    poll_lb: bool,
    skip_train_push: bool = False,
) -> dict:
    out_dir = OUT_ROOT / enc.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = ROOT / enc.notebook_dir

    cached = load_cached_eval(enc, out_dir)
    if cached is not None:
        return finish_lb(enc, cached, out_dir, env, submit_lb, poll_lb)

    run([sys.executable, str(ROOT / "scripts/build_train_encoder_nb.py"), "--slug", enc.slug])
    skip_push = skip_train_push and train_kernel_complete(enc.kernel_id, env)
    if skip_push:
        print("  train kernel already COMPLETE — skipping push", flush=True)
    if not skip_push:
        run([str(KAGGLE), "kernels", "push", "-p", str(train_dir), "--accelerator", "NvidiaTeslaT4"], env=env)
    if poll_kernel(enc.kernel_id, env, 80) != "complete":
        return {"slug": enc.slug, "arch": enc.arch, "family": enc.family, "status": "train_failed"}

    run([str(KAGGLE), "kernels", "output", enc.kernel_id, "-p", str(out_dir / "train")], env=env, retries=4)
    timing = out_dir / "train" / "timing_report.csv"
    val_row: dict = {}
    if timing.exists():
        import pandas as pd

        val_row = pd.read_csv(timing).iloc[0].to_dict()

    patch_submission(enc)
    run([str(KAGGLE), "kernels", "push", "-p", str(SUBMIT_DIR), "--accelerator", "NvidiaTeslaT4"], env=env)
    if poll_kernel(KERNEL_SUBMIT, env, 120) != "complete":
        return {"slug": enc.slug, "arch": enc.arch, "family": enc.family, "status": "submit_failed", **val_row}

    run([str(KAGGLE), "kernels", "output", KERNEL_SUBMIT, "-p", str(out_dir / "submit")], env=env, retries=4)
    debug = out_dir / "submit" / "submission_debug.csv"
    if not debug.exists():
        return {"slug": enc.slug, "arch": enc.arch, "family": enc.family, "status": "missing_debug", **val_row}

    stats = analyze_test(debug)
    result = {"slug": enc.slug, "arch": enc.arch, "family": enc.family, "status": "ok", **val_row, **stats}
    print(f"  => test mt_ok {stats['mt_ok']}/{stats['n']} ({stats['mt_ok_pct']}%)", flush=True)

    if stats["mt_ok_pct"] < 100.0:
        result["status"] = "rejected_geometry"
        result["log_note"] = f"test mt_ok {stats['mt_ok_pct']}% — skipped LB"
        return result

    return finish_lb(enc, result, out_dir, env, submit_lb, poll_lb)


def encoders_for_args(args: argparse.Namespace) -> list[EncoderSpec]:
    encs = list(BLOCK8_ENCODERS)
    if args.from_slug:
        start = next(i for i, e in enumerate(encs) if e.slug == args.from_slug)
        encs = encs[start:]
    elif args.all:
        pass
    elif args.slug:
        encs = [get_encoder(args.slug)]
    else:
        raise SystemExit("Pass --slug, --all, or --from-slug")
    if args.skip_completed:
        done = load_results()
        encs = [
            e
            for e in encs
            if e.slug not in done
            or done[e.slug].get("status") not in {"ok", "rejected_geometry", "train_failed"}
            or (done[e.slug].get("lb_submitted") and done[e.slug].get("lb_score") is None)
        ]
    return encs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="Run one encoder")
    parser.add_argument("--all", action="store_true", help="Run all Block 8 encoders in order")
    parser.add_argument("--from-slug", help="Run this encoder and all later ones in registry order")
    parser.add_argument("--skip-completed", action="store_true", help="Skip slugs already in results.json")
    parser.add_argument("--submit-lb", action="store_true", help="Leaderboard submit if 100% test mt_ok")
    parser.add_argument("--poll-lb", action="store_true", help="Poll Kaggle until LB score is ready (implies --submit-lb)")
    parser.add_argument("--skip-train-push", action="store_true", help="Poll existing train kernel only")
    parser.add_argument("--restore-prod", action="store_true", help="Restore submission notebook to production")
    args = parser.parse_args()

    if args.restore_prod:
        restore_submission_prod()
        return

    submit_lb = args.submit_lb or args.poll_lb
    poll_lb = args.poll_lb

    encoders = encoders_for_args(args)
    env = kaggle_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()

    for enc in encoders:
        if enc.slug in PRESCORED_LB and enc.slug not in results:
            results[enc.slug] = {"slug": enc.slug, "lb_score": PRESCORED_LB[enc.slug], "status": "ok"}

        print(f"\n=== Block 8 {enc.family} ({enc.slug}) ===", flush=True)
        result = run_encoder(enc, env, submit_lb, poll_lb, skip_train_push=args.skip_train_push)
        if enc.slug in PRESCORED_LB and result.get("lb_score") is None and result.get("status") == "ok":
            result["lb_score"] = PRESCORED_LB[enc.slug]
        results[enc.slug] = result
        save_results(results)
        update_log_row(result)

    print(f"\nWrote {OUT_ROOT / 'results.json'}", flush=True)
    restore_submission_prod()


if __name__ == "__main__":
    main()
