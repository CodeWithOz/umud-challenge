#!/usr/bin/env bash
# Post-compaction queue: v8 MT-fail viz (full model) → micro retrain → calibrated submission.
set -euo pipefail
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN=$(.venv/bin/kaggle auth print-access-token)
KAGGLE=.venv/bin/kaggle
LOG=tmp/kaggle-output/post-compaction-monitor.log
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

log "=== 1/3 v8 MT-fail viz (full model on train kernel) ==="
$KAGGLE kernels push -p notebooks/v8-mt-fail-viz --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-v8-mt-fail-viz-phase-3" "v8-viz" || exit 1
$KAGGLE kernels output ucheozoemena/umud-v8-mt-fail-viz-phase-3 -p tmp/kaggle-output/v8-mt-fail-viz --force

log "=== 2/3 micro apo retrain (TRAIN_RUN=5) for calibrated v7 ==="
$KAGGLE kernels push -p notebooks/train-apo-gray55 --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-train-apo-gray55-phase-3" "train-micro" || exit 1
$KAGGLE kernels output ucheozoemena/umud-train-apo-gray55-phase-3 -p tmp/kaggle-output/train-apo-gray55-line-micro --force

log "=== 3/3 calibrated submission (MM_PER_PIXEL=0.098) ==="
$KAGGLE kernels push -p notebooks/submission --accelerator NvidiaTeslaT4
wait_kernel "ucheozoemena/umud-submission-phase-3" "submission-calibrated" || exit 1
$KAGGLE kernels output ucheozoemena/umud-submission-phase-3 -p tmp/kaggle-output/submission-v9-calibrated --force

log "=== COMPLETE ==="
