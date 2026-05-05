#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

FILTERED_PATHS="${FILTERED_PATHS:-safesim/case1_filtered/data.hdf5 safesim/case2_filtered/data.hdf5 safesim/case3_filtered/data.hdf5 safesim/case4_filtered/data.hdf5 safesim/case5_filtered/data.hdf5}"
ORIGINAL_PATHS="${ORIGINAL_PATHS:-safesim/case1/data.hdf5 safesim/case2/data.hdf5 safesim/case3/data.hdf5 safesim/case4/data.hdf5 safesim/case5/data.hdf5}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?MODEL_CHECKPOINT must point to Stage-1 best checkpoint}"

env \
  HDF5_PATHS="${FILTERED_PATHS}" \
  REPLAY_HDF5_PATHS="${REPLAY_HDF5_PATHS:-${ORIGINAL_PATHS}}" \
  LOG_DIR="${LOG_DIR:-./safesim_logs_stage2_action_imitation}" \
  MAX_EPOCHS="${MAX_EPOCHS:-20}" \
  LR="${LR:-2e-5}" \
  TF_D_MODEL="${TF_D_MODEL:-1024}" \
  USE_GOAL_CONDITION="${USE_GOAL_CONDITION:-1}" \
  TARGET_POLICY="${TARGET_POLICY:-action}" \
  MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
  CFG_SCALE="${CFG_SCALE:-1.0}" \
  REPLAY_RATIO="${REPLAY_RATIO:-0.3}" \
  TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}" \
  TERMINAL_XY_WEIGHT="0.0" \
  TERMINAL_HEADING_WEIGHT="0.0" \
  CTRL_SOFTMIN_WEIGHT="0.0" \
  CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA:-4.0}" \
  STAGE_NAME="${STAGE_NAME:-stage2_action_imitation}" \
  SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}" \
  sh scripts/training/run_safesim_training.sh
