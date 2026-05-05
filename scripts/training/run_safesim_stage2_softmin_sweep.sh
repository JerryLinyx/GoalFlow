#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?MODEL_CHECKPOINT must point to Stage-1 best checkpoint}"

FILTERED_PATHS="${FILTERED_PATHS:-safesim/case1_filtered/data.hdf5 safesim/case2_filtered/data.hdf5 safesim/case3_filtered/data.hdf5 safesim/case4_filtered/data.hdf5 safesim/case5_filtered/data.hdf5}"
ORIGINAL_PATHS="${ORIGINAL_PATHS:-safesim/case1/data.hdf5 safesim/case2/data.hdf5 safesim/case3/data.hdf5 safesim/case4/data.hdf5 safesim/case5/data.hdf5}"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-2e-5}"
TF_D_MODEL="${TF_D_MODEL:-1024}"
TARGET_POLICY="${TARGET_POLICY:-action}"
USE_GOAL_CONDITION="${USE_GOAL_CONDITION:-1}"
REPLAY_RATIO="${REPLAY_RATIO:-0.3}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"

TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}"
TERMINAL_XY_WEIGHT="${TERMINAL_XY_WEIGHT:-0.5}"
TERMINAL_HEADING_WEIGHT="${TERMINAL_HEADING_WEIGHT:-0.1}"
CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA:-4.0}"

LOG_ROOT="${LOG_ROOT:-./safesim_logs_stage2_softmin_sweep_action}"
SOFTMIN_WEIGHTS="${SOFTMIN_WEIGHTS:-0.0 0.0025 0.005 0.01 0.02}"

mkdir -p "${LOG_ROOT}"

for weight in ${SOFTMIN_WEIGHTS}; do
    weight_tag=$(printf "%s" "${weight}" | tr '.' 'p')
    run_log_dir="${LOG_ROOT}/softmin_${weight_tag}"
    stage_name="stage2_action_softmin_${weight_tag}"

    echo "[Softmin Sweep] Running weight=${weight} -> ${run_log_dir}"

    env \
      HDF5_PATHS="${FILTERED_PATHS}" \
      REPLAY_HDF5_PATHS="${REPLAY_HDF5_PATHS:-${ORIGINAL_PATHS}}" \
      LOG_DIR="${run_log_dir}" \
      MAX_EPOCHS="${MAX_EPOCHS}" \
      LR="${LR}" \
      TF_D_MODEL="${TF_D_MODEL}" \
      USE_GOAL_CONDITION="${USE_GOAL_CONDITION}" \
      TARGET_POLICY="${TARGET_POLICY}" \
      MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
      CFG_SCALE="${CFG_SCALE}" \
      REPLAY_RATIO="${REPLAY_RATIO}" \
      TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT}" \
      TERMINAL_XY_WEIGHT="${TERMINAL_XY_WEIGHT}" \
      TERMINAL_HEADING_WEIGHT="${TERMINAL_HEADING_WEIGHT}" \
      CTRL_SOFTMIN_WEIGHT="${weight}" \
      CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA}" \
      STAGE_NAME="${stage_name}" \
      SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS}" \
      sh scripts/training/run_safesim_stage2.sh || exit 1
done
