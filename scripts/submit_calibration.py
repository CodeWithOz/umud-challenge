"""Submit calibration-sweep CSVs to the UMUD competition leaderboard."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "tmp/kaggle-output/calibration-sweep-200tier"
DEFAULT_SWEEP_DIR = ROOT / "tmp/kaggle-output/calibration-sweep"
COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"
KAGGLE = ROOT / ".venv/bin/kaggle"


def kaggle_token() -> str:
    proc = subprocess.run(
        [str(KAGGLE), "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    # CLI may print upgrade warning on stdout; token is the last line.
    return proc.stdout.strip().splitlines()[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slugs",
        nargs="*",
        help="Policy slugs to submit (default: read submit_shortlist from sweep_summary.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help="Directory with per-slug submission.csv (default: sweep_summary parent)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir or SWEEP_DIR
    if not sweep_dir.exists():
        sweep_dir = DEFAULT_SWEEP_DIR
    summary_path = sweep_dir / "sweep_summary.json"
    if not summary_path.exists():
        print(f"Run scripts/calibration_sweep.py first — missing {summary_path}", file=sys.stderr)
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    slugs = args.slugs or summary.get("submit_shortlist", [])
    if not slugs:
        print("No slugs to submit", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        import os

        os.environ["KAGGLE_API_TOKEN"] = kaggle_token()

    for slug in slugs:
        csv_path = sweep_dir / slug / "submission.csv"
        if not csv_path.exists():
            print(f"Missing {csv_path}", file=sys.stderr)
            sys.exit(1)
        msg = f"phase4-cal-200tier-{slug}"
        cmd = [
            str(KAGGLE),
            "competitions",
            "submit",
            COMPETITION,
            "-f",
            str(csv_path),
            "-m",
            msg,
        ]
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
