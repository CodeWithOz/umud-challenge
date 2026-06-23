#!/bin/zsh
# One-shot scheduled submit of Block 13 quick-dirty notebook output just after
# Kaggle daily-quota reset. Idempotent: skips if already submitted; self-unloads.
set -u

REPO="/Users/ucheozoemena/01-projects/umud-challenge"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
KG="$REPO/.venv/bin/kaggle"
KERNEL="ucheozoemena/umud-submission-quickdirty"
VERSION="1"
FILE="submission.csv"
MESSAGE="block13-quickdirty"
PLIST="$HOME/Library/LaunchAgents/com.uche.umud-block13-submit.plist"
LOG="$REPO/data/kaggle-outputs/block13-quickdirty-scheduled.log"

cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")"
echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC scheduled block13 submit ===" >> "$LOG"

export KAGGLE_API_TOKEN="$($KG auth print-access-token 2>>"$LOG" | tail -n 1)"
if [ -z "$KAGGLE_API_TOKEN" ]; then
  echo "ERROR: could not mint Kaggle access token; leaving job loaded for retry" >> "$LOG"
  exit 1
fi

if "$KG" competitions submissions "$COMP" 2>>"$LOG" | grep -q "$MESSAGE"; then
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
