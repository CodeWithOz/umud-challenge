#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=.venv/bin/kaggle
LOG=tmp/kaggle-output/gray55-train-eval-monitor.log
mkdir -p tmp/kaggle-output

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

wait_kernel() {
  local slug="$1"
  local label="$2"
  while true; do
    kstatus=$($KAGGLE kernels status "$slug" 2>&1 || true)
    log "$label status: $kstatus"
    if echo "$kstatus" | grep -q 'KernelWorkerStatus.COMPLETE'; then
      log "$label COMPLETE"
      return 0
    fi
    if echo "$kstatus" | grep -q 'KernelWorkerStatus.ERROR'; then
      log "$label FAILED: $kstatus"
      return 1
    fi
    sleep 60
  done
}

log "=== Gray55 train+eval monitor ==="
$KAGGLE kernels push -p notebooks/train-apo-gray55 --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-train-apo-gray55-phase-3" "train" || exit 1
$KAGGLE kernels output ucheozoemena/umud-train-apo-gray55-phase-3 -p tmp/kaggle-output/train-apo-gray55 --force

$KAGGLE kernels push -p notebooks/apo-gray55-eval --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-apo-gray55-eval-phase-3" "eval" || exit 1
$KAGGLE kernels output ucheozoemena/umud-train-apo-gray55-phase-3 -p tmp/kaggle-output/train-apo-gray55 --force
$KAGGLE kernels output ucheozoemena/umud-apo-gray55-eval-phase-3 -p tmp/kaggle-output/apo-gray55-eval --force
log "=== Gray55 train+eval COMPLETE ==="
