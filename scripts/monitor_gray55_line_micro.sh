#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=.venv/bin/kaggle
LOG=tmp/kaggle-output/gray55-line-micro-monitor.log
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

log "=== gray55+line micro pipeline ==="
wait_kernel "ucheozoemena/umud-prep-apo-gray55-line" "prep" || exit 1
$KAGGLE kernels output ucheozoemena/umud-prep-apo-gray55-line -p tmp/kaggle-output/prep-apo-gray55-line --force

$KAGGLE kernels push -p notebooks/train-apo-gray55 --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-train-apo-gray55-phase-3" "train" || exit 1
$KAGGLE kernels output ucheozoemena/umud-train-apo-gray55-phase-3 -p tmp/kaggle-output/train-apo-gray55-line --force

$KAGGLE kernels push -p notebooks/apo-gray55-line-eval --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-apo-gray55-line-eval-phase-3" "eval" || exit 1
$KAGGLE kernels output ucheozoemena/umud-apo-gray55-line-eval-phase-3 -p tmp/kaggle-output/apo-gray55-line-eval --force
log "=== COMPLETE ==="
