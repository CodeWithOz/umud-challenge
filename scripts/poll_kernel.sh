#!/usr/bin/env bash
# Poll Kaggle timing-baseline kernel until complete, error, or max wait.
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=".venv/bin/kaggle"
KERNEL="ucheozoemena/umud-baseline-phase-3-fastai-u-net"
POLL_SEC="${POLL_SEC:-60}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-3600}"
RUN_LABEL="${RUN_LABEL:-v9 timing run 1}"

elapsed=0
echo "=== Polling $RUN_LABEL every ${POLL_SEC}s (max ${MAX_WAIT_SEC}s) ==="
while true; do
  kstatus=$($KAGGLE kernels status "$KERNEL" 2>&1 | awk -F'"' '{print $2}')
  echo "$(date '+%H:%M:%S') $kstatus"
  case "$kstatus" in
    *COMPLETE*|*complete*)
      echo "DONE"
      exit 0
      ;;
    *ERROR*|*error*|*Cancel*|*CANCEL*)
      echo "FAILED"
      exit 1
      ;;
  esac
  if (( elapsed >= MAX_WAIT_SEC )); then
    echo "POLL_TIMEOUT"
    exit 2
  fi
  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
done
