#!/bin/zsh
# One-shot scheduled submit of Block 9 submission 2 (PA18 / FL recenter / MT21.5 shrink.45)
# just after the Kaggle daily-quota reset (00:05 UTC). Idempotent: skips if already
# submitted; self-unloads the launchd job after running. Logs are .log (gitignored).
set -u
REPO="/Users/ucheozoemena/01-projects/umud-challenge"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
CSV="$REPO/data/kaggle-outputs/block9-s1/submission_s2.csv"
KG="$REPO/.venv/bin/kaggle"
PLIST="$HOME/Library/LaunchAgents/com.uche.umud-s2-submit.plist"
LOG="$REPO/data/kaggle-outputs/block9-s2-scheduled.log"

cd "$REPO" || exit 1
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC  scheduled s2 submit ===" >> "$LOG"

export KAGGLE_API_TOKEN="$($KG auth print-access-token 2>>"$LOG")"
if [ -z "$KAGGLE_API_TOKEN" ]; then
  echo "ERROR: could not mint Kaggle access token; leaving job loaded for retry" >> "$LOG"
  exit 1
fi

# Idempotency guard: don't double-submit.
if "$KG" competitions submissions "$COMP" 2>/dev/null | grep -q 'block9-s2'; then
  echo "block9-s2 already present on leaderboard; nothing to do." >> "$LOG"
else
  "$KG" competitions submit -c "$COMP" -f "$CSV" \
    -m "block9-s2 PA18 FLrecenter MT21.5 shrink0.45 (scheduled)" >> "$LOG" 2>&1
  echo "submit exit code: $?" >> "$LOG"
fi

# One-shot: remove the launchd job so it does not run again.
launchctl unload "$PLIST" 2>>"$LOG" || true
rm -f "$PLIST"
echo "job unloaded + plist removed." >> "$LOG"
