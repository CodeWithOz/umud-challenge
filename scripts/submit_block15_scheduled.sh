#!/bin/zsh
# One-shot scheduled submit of Block 15 blend notebook output after the Block 13
# and Block 14 reset-submit windows. Idempotent: skips if already submitted or
# if any completed public score is already below target.
set -u

REPO="/Users/ucheozoemena/01-projects/umud-challenge"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
KG="$REPO/.venv/bin/kaggle"
KERNEL="ucheozoemena/umud-submission-blend-qdc-cxs8"
VERSION="2"
FILE="submission.csv"
MESSAGE="block15-qdc-cxs8-blend"
TARGET="0.6"
PLIST="$HOME/Library/LaunchAgents/com.uche.umud-block15-submit.plist"
LOG="$REPO/data/kaggle-outputs/block15-blend-qdc-cxs8-scheduled.log"

cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")"
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC scheduled block15 submit ===" >> "$LOG"

export KAGGLE_API_TOKEN="$($KG auth print-access-token 2>>"$LOG" | tail -n 1)"
if [ -z "$KAGGLE_API_TOKEN" ]; then
  echo "ERROR: could not mint Kaggle access token; leaving job loaded for retry" >> "$LOG"
  exit 1
fi

submissions="$("$KG" competitions submissions "$COMP" 2>>"$LOG")"
best_score="$(printf "%s\n" "$submissions" | awk '
  /SubmissionStatus.COMPLETE/ {
    score = $NF
    if (score ~ /^[0-9]+(\.[0-9]+)?$/ && (best == "" || score + 0 < best + 0)) best = score
  }
  END { print best }
')"

if [ -n "$best_score" ]; then
  echo "Best completed public score before block15: $best_score" >> "$LOG"
  if awk "BEGIN { exit !($best_score < $TARGET) }"; then
    echo "Target already met (< $TARGET); skipping block15 submit." >> "$LOG"
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
  echo "submit exit code: $?" >> "$LOG"
fi

launchctl unload "$PLIST" 2>>"$LOG" || true
rm -f "$PLIST"
echo "job unloaded + plist removed." >> "$LOG"
