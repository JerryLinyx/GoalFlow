#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?MODEL_CHECKPOINT must point to Stage-1 best checkpoint}"

# Shared Stage-2 settings
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

# A1: terminal-only
env \
  HDF5_PATHS="${FILTERED_PATHS}" \
  REPLAY_HDF5_PATHS="${REPLAY_HDF5_PATHS:-${ORIGINAL_PATHS}}" \
  LOG_DIR="${A1_LOG_DIR:-./safesim_logs_stage2_terminal_only}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  LR="${LR}" \
  TF_D_MODEL="${TF_D_MODEL}" \
  USE_GOAL_CONDITION="${USE_GOAL_CONDITION}" \
  TARGET_POLICY="${TARGET_POLICY}" \
  MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
  CFG_SCALE="${CFG_SCALE}" \
  REPLAY_RATIO="${REPLAY_RATIO}" \
  TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}" \
  TERMINAL_XY_WEIGHT="${A1_TERMINAL_XY_WEIGHT:-0.5}" \
  TERMINAL_HEADING_WEIGHT="${A1_TERMINAL_HEADING_WEIGHT:-0.1}" \
  CTRL_SOFTMIN_WEIGHT="0.0" \
  CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA:-4.0}" \
  STAGE_NAME="${A1_STAGE_NAME:-stage2_terminal_only}" \
  SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS}" \
  sh scripts/training/run_safesim_stage2.sh || exit 1

# A2: ctrl-softmin-only
env \
  HDF5_PATHS="${FILTERED_PATHS}" \
  REPLAY_HDF5_PATHS="${REPLAY_HDF5_PATHS:-${ORIGINAL_PATHS}}" \
  LOG_DIR="${A2_LOG_DIR:-./safesim_logs_stage2_ctrl_softmin}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  LR="${LR}" \
  TF_D_MODEL="${TF_D_MODEL}" \
  USE_GOAL_CONDITION="${USE_GOAL_CONDITION}" \
  TARGET_POLICY="${TARGET_POLICY}" \
  MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
  CFG_SCALE="${CFG_SCALE}" \
  REPLAY_RATIO="${REPLAY_RATIO}" \
  TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}" \
  TERMINAL_XY_WEIGHT="0.0" \
  TERMINAL_HEADING_WEIGHT="0.0" \
  CTRL_SOFTMIN_WEIGHT="${A2_CTRL_SOFTMIN_WEIGHT:-0.05}" \
  CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA:-4.0}" \
  STAGE_NAME="${A2_STAGE_NAME:-stage2_ctrl_softmin}" \
  SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS}" \
  sh scripts/training/run_safesim_stage2.sh
