#!/bin/zsh
# One-shot scheduled submit of Block 18 sequence-smoothing notebook output after
# Kaggle daily-quota reset. Idempotent: skips if already submitted or if any
# completed public score is already below target.
set -u

REPO="/Users/ucheozoemena/01-projects/umud-challenge"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
KG="$REPO/.venv/bin/kaggle"
KERNEL="ucheozoemena/umud-submission-sequence-smooth"
VERSION="1"
FILE="submission.csv"
MESSAGE="block18-seq-smooth5-mean"
TARGET="0.6"
PLIST="$HOME/Library/LaunchAgents/com.uche.umud-block18-submit.plist"
LOG="$REPO/data/kaggle-outputs/block18-sequence-smooth-scheduled.log"

cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")"
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC scheduled block18 notebook submit ===" >> "$LOG"

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
  echo "Best completed public score before block18: $best_score" >> "$LOG"
  if awk "BEGIN { exit !($best_score < $TARGET) }"; then
    echo "Target already met (< $TARGET); skipping block18 submit." >> "$LOG"
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
