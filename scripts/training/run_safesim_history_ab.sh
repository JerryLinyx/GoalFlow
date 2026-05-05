#!/bin/bash
# Run a controlled A/B experiment for Safe-Sim history resolution.
#
# A (baseline): history_len=4, history_stride=5  -> 2Hz history
# B (variant):  history_len=8, history_stride=2  -> 5Hz history
#
# All other settings are kept fixed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

export MAX_EPOCHS="${MAX_EPOCHS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LR="${LR:-1e-4}"
export GPUS="${GPUS:-1}"
export PRECISION="${PRECISION:-16-mixed}"
export TF_D_MODEL="${TF_D_MODEL:-256}"
export TF_NUM_LAYERS="${TF_NUM_LAYERS:-3}"
export MAX_OTHER_AGENTS="${MAX_OTHER_AGENTS:-10}"
export ANCHOR_SIZE="${ANCHOR_SIZE:-16}"
export INFER_STEPS="${INFER_STEPS:-25}"
export SEED="${SEED:-0}"
export CASE_BALANCE_EXPONENT="${CASE_BALANCE_EXPONENT:-0.5}"

echo "[History A/B] Running baseline: history_len=4, history_stride=5"
LOG_DIR="${LOG_DIR_BASELINE:-./safesim_logs_ab_history_baseline}" \
HISTORY_LEN=4 \
HISTORY_STRIDE=5 \
sh scripts/training/run_safesim_training.sh

echo "[History A/B] Running variant: history_len=8, history_stride=2"
LOG_DIR="${LOG_DIR_VARIANT:-./safesim_logs_ab_history_5hz}" \
HISTORY_LEN=8 \
HISTORY_STRIDE=2 \
sh scripts/training/run_safesim_training.sh
