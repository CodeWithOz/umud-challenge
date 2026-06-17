#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=.venv/bin/kaggle
LOG=tmp/kaggle-output/gray55-line-full-monitor.log
mkdir -p tmp/kaggle-output

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

wait_kernel() {
  local slug="$1"
  local label="$2"
  while true; do
    kstatus=$($KAGGLE kernels status "$slug" 2>&1 || true)
    log "$label: $kstatus"
    if echo "$kstatus" | grep -q 'KernelWorkerStatus.COMPLETE'; then return 0; fi
    if echo "$kstatus" | grep -q 'KernelWorkerStatus.ERROR'; then return 1; fi
    sleep 60
  done
}

log "=== gray55+line FULL pipeline (prep 1044 → train 10ep → submission) ==="

$KAGGLE kernels push -p notebooks/prep-apo-gray55-line
wait_kernel "ucheozoemena/umud-prep-apo-gray55-line" "prep" || exit 1
# Only fetch timing + log — full dataset is on Kaggle (2000+ PNGs; do not download locally).
$KAGGLE kernels output ucheozoemena/umud-prep-apo-gray55-line -p tmp/kaggle-output/prep-apo-gray55-line-full --force 2>&1 \
  | grep -E 'prep_timing|\.log' || true

$KAGGLE kernels push -p notebooks/train-apo-gray55 --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-train-apo-gray55-phase-3" "train" || exit 1
$KAGGLE kernels output ucheozoemena/umud-train-apo-gray55-phase-3 -p tmp/kaggle-output/train-apo-gray55-line-full --force 2>&1 \
  | grep -E 'timing_report|\.pkl|\.log' || true

$KAGGLE kernels push -p notebooks/submission --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-submission-phase-3" "submission" || exit 1
$KAGGLE kernels output ucheozoemena/umud-submission-phase-3 -p tmp/kaggle-output/submission-v8-full-model --force 2>&1 \
  | grep -E 'submission.*\.csv|\.log' || true
log "=== COMPLETE ==="
