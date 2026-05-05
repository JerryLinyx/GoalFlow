#!/bin/bash

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?MODEL_CHECKPOINT must point to Stage-1 best checkpoint}"

FILTERED_PATHS="${FILTERED_PATHS:-safesim/case1_filtered/data.hdf5 safesim/case2_filtered/data.hdf5 safesim/case3_filtered/data.hdf5 safesim/case4_filtered/data.hdf5 safesim/case5_filtered/data.hdf5}"
ORIGINAL_PATHS="${ORIGINAL_PATHS:-safesim/case1/data.hdf5 safesim/case2/data.hdf5 safesim/case3/data.hdf5 safesim/case4/data.hdf5 safesim/case5/data.hdf5}"

PURE_LOG_DIR="${PURE_LOG_DIR:-safesim_logs_stage2_action_goal_imitation}"
TERMINAL_LOG_ROOT="${TERMINAL_LOG_ROOT:-safesim_logs_stage2_terminal_sweep_goal_action}"
SOFTMIN_LOG_ROOT="${SOFTMIN_LOG_ROOT:-safesim_logs_stage2_softmin_sweep_goal_action}"
EVAL_ROOT="${EVAL_ROOT:-outputs/current_goal_action}"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-2e-5}"
TF_D_MODEL="${TF_D_MODEL:-1024}"
TARGET_POLICY="${TARGET_POLICY:-action}"
USE_GOAL_CONDITION="${USE_GOAL_CONDITION:-1}"
REPLAY_RATIO="${REPLAY_RATIO:-0.3}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}"
CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA:-4.0}"
SOFTMIN_WEIGHTS="${SOFTMIN_WEIGHTS:-0.0 0.001 0.0025 0.005 0.01}"

mkdir -p "${EVAL_ROOT}"

run_variant() {
  local log_dir="$1"
  local stage_name="$2"
  local term_xy="$3"
  local term_heading="$4"
  local softmin_weight="$5"
  local checkpoint_arg=""

  if [ -f "${log_dir}/checkpoints/last.ckpt" ]; then
    checkpoint_arg="${log_dir}/checkpoints/last.ckpt"
    echo "[Resume] ${stage_name} from ${checkpoint_arg}"
  else
    echo "[Start] ${stage_name}"
  fi

  env \
    HDF5_PATHS="${FILTERED_PATHS}" \
    REPLAY_HDF5_PATHS="${REPLAY_HDF5_PATHS:-${ORIGINAL_PATHS}}" \
    LOG_DIR="${log_dir}" \
    CHECKPOINT="${checkpoint_arg}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
    LR="${LR}" \
    TF_D_MODEL="${TF_D_MODEL}" \
    USE_GOAL_CONDITION="${USE_GOAL_CONDITION}" \
    TARGET_POLICY="${TARGET_POLICY}" \
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
    CFG_SCALE="${CFG_SCALE}" \
    REPLAY_RATIO="${REPLAY_RATIO}" \
    TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT}" \
    TERMINAL_XY_WEIGHT="${term_xy}" \
    TERMINAL_HEADING_WEIGHT="${term_heading}" \
    CTRL_SOFTMIN_WEIGHT="${softmin_weight}" \
    CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA}" \
    STAGE_NAME="${stage_name}" \
    SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS}" \
    sh scripts/training/run_safesim_training.sh
}

run_pure_eval() {
  local ckpt
  ckpt=$(find "${PURE_LOG_DIR}/checkpoints" -maxdepth 1 -type f -name 'best-val-*.ckpt' | sort | tail -n 1)
  if [ -z "${ckpt}" ]; then
    echo "Missing best pure imitation checkpoint in ${PURE_LOG_DIR}" >&2
    exit 1
  fi
  python scripts/analysis/evaluate_safesim_dangerous.py \
    --model_checkpoint "${ckpt}" \
    --output_dir "${EVAL_ROOT}/pure_imitation_protocol64" \
    --cfg_scales 1.0 \
    --anchor_size 16 \
    --infer_steps 25 \
    --batch_size 8 \
    --val_split 0.1 \
    --seed 0 \
    --tf_d_model 1024 \
    --max_val_samples 64 \
    --target_policy action \
    --model_label pure_imitation_goal_action
}

select_best_terminal() {
  python - <<'PY'
import csv
from pathlib import Path

root = Path("outputs/current_goal_action/terminal_eval")
rows = []
for csv_path in sorted(root.glob("termxy_*_protocol64/metrics/global.csv")):
    with csv_path.open() as f:
        row = next(csv.DictReader(f))
    row["tag"] = csv_path.parent.parent.name.replace("_protocol64", "")
    rows.append(row)

if not rows:
    raise SystemExit("No terminal evaluation rows found.")

def key(row):
    return (
        float(row["dangerous_hit_rate"]),
        float(row["hit@2m"]),
        -float(row["pred_min_dist"]),
    )

best = max(rows, key=key)
print(best["tag"])
PY
}

terminal_weights_from_tag() {
  local tag="$1"
  case "${tag}" in
    termxy_0p25_termyaw_0p05) echo "0.25 0.05" ;;
    termxy_0p25_termyaw_0p10) echo "0.25 0.10" ;;
    termxy_0p50_termyaw_0p05) echo "0.50 0.05" ;;
    termxy_0p50_termyaw_0p10) echo "0.50 0.10" ;;
    *) echo "Unknown terminal tag: ${tag}" >&2; exit 1 ;;
  esac
}

echo "[Phase 1] Finish terminal sweep"
run_variant "${TERMINAL_LOG_ROOT}/termxy_0p25_termyaw_0p10" "stage2_action_termxy_0p25_termyaw_0p10" "0.25" "0.10" "0.0"
run_variant "${TERMINAL_LOG_ROOT}/termxy_0p50_termyaw_0p05" "stage2_action_termxy_0p50_termyaw_0p05" "0.50" "0.05" "0.0"
run_variant "${TERMINAL_LOG_ROOT}/termxy_0p50_termyaw_0p10" "stage2_action_termxy_0p50_termyaw_0p10" "0.50" "0.10" "0.0"

echo "[Phase 2] Evaluate corrected pure imitation"
run_pure_eval

echo "[Phase 3] Evaluate terminal sweep"
LOG_ROOT="${TERMINAL_LOG_ROOT}" OUT_ROOT="${EVAL_ROOT}/terminal_eval" \
  sh scripts/analysis/run_safesim_terminal_sweep_eval.sh

BEST_TERMINAL_TAG=$(select_best_terminal)
read -r BEST_TERMINAL_XY BEST_TERMINAL_HEADING <<< "$(terminal_weights_from_tag "${BEST_TERMINAL_TAG}")"
echo "[Phase 4] Selected terminal base ${BEST_TERMINAL_TAG} (${BEST_TERMINAL_XY}, ${BEST_TERMINAL_HEADING})"

echo "[Phase 5] Run softmin sweep on selected terminal base"
for weight in ${SOFTMIN_WEIGHTS}; do
  weight_tag=$(printf "%s" "${weight}" | tr '.' 'p')
  run_variant \
    "${SOFTMIN_LOG_ROOT}/softmin_${weight_tag}" \
    "stage2_goal_action_${BEST_TERMINAL_TAG}_softmin_${weight_tag}" \
    "${BEST_TERMINAL_XY}" \
    "${BEST_TERMINAL_HEADING}" \
    "${weight}"
done

echo "[Phase 6] Evaluate softmin sweep"
LOG_ROOT="${SOFTMIN_LOG_ROOT}" OUT_ROOT="${EVAL_ROOT}/softmin_eval" \
  sh scripts/analysis/run_safesim_softmin_sweep_eval.sh

echo "[Done] Goal-conditioned mainline finished."
