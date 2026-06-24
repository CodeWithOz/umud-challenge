#!/bin/zsh
# One-shot scheduled submit of Block 19 cxs5-refresh blend notebook output after
# Kaggle daily-quota reset. Idempotent: skips if already submitted or if any
# completed public score is already below target.
set -u

REPO="/Users/ucheozoemena/01-projects/umud-challenge"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
KG="$REPO/.venv/bin/kaggle"
KERNEL="ucheozoemena/umud-submission-blend-qdc-cxs5"
VERSION="1"
FILE="submission.csv"
MESSAGE="block19-qdc-cxs5-refresh"
TARGET="0.6"
PLIST="$HOME/Library/LaunchAgents/com.uche.umud-block19-submit.plist"
LOG="$REPO/data/kaggle-outputs/block19-cxs5-refresh-scheduled.log"

cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")"
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC scheduled block19 notebook submit ===" >> "$LOG"

submissions="$("$KG" competitions submissions -c "$COMP" 2>>"$LOG")"
submissions_rc=$?

if [ "$submissions_rc" -ne 0 ]; then
  echo "Initial submissions check failed with rc=$submissions_rc; trying token fallback." >> "$LOG"
  token="$("$KG" auth print-access-token 2>>"$LOG" | tail -n 1)"
  if [ -n "$token" ]; then
    export KAGGLE_API_TOKEN="$token"
    submissions="$("$KG" competitions submissions -c "$COMP" 2>>"$LOG")"
    submissions_rc=$?
  fi
fi

if [ "$submissions_rc" -ne 0 ]; then
  echo "ERROR: could not read submissions; leaving job loaded for retry." >> "$LOG"
  exit 1
fi

best_score="$(printf "%s\n" "$submissions" | awk '
  /SubmissionStatus.COMPLETE/ {
    score = $NF
    if (score ~ /^[0-9]+(\.[0-9]+)?$/ && (best == "" || score + 0 < best + 0)) best = score
  }
  END { print best }
')"

if [ -n "$best_score" ]; then
  echo "Best completed public score before block19: $best_score" >> "$LOG"
  if awk "BEGIN { exit !($best_score < $TARGET) }"; then
    echo "Target already met (< $TARGET); skipping block19 submit." >> "$LOG"
    launchctl unload "$PLIST" 2>>"$LOG" || true
    rm -f "$PLIST"
    echo "job unloaded + plist removed." >> "$LOG"
    exit 0
  fi
fi

if printf "%s\n" "$submissions" | grep -q "$MESSAGE"; then
  echo "$MESSAGE already present on leaderboard; skipping." >> "$LOG"
else
  "$KG" competitions submit \
    -c "$COMP" \
    -k "$KERNEL" \
    -v "$VERSION" \
    -f "$FILE" \
    -m "$MESSAGE" >> "$LOG" 2>&1
  submit_rc=$?
  echo "submit exit code: $submit_rc" >> "$LOG"
  if [ "$submit_rc" -ne 0 ]; then
    echo "ERROR: submit failed; leaving job loaded for retry." >> "$LOG"
    exit "$submit_rc"
  fi
fi

launchctl unload "$PLIST" 2>>"$LOG" || true
rm -f "$PLIST"
echo "job unloaded + plist removed." >> "$LOG"
