#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

HDF5_PATHS="${HDF5_PATHS:-safesim/case1/data.hdf5 safesim/case2/data.hdf5 safesim/case3/data.hdf5 safesim/case4/data.hdf5 safesim/case5/data.hdf5}"
GOALFLOW_CKPT="${GOALFLOW_CKPT:-data/goalflow_traj_epoch_54-step_18260.ckpt}"

env \
  HDF5_PATHS="${HDF5_PATHS}" \
  LOG_DIR="${LOG_DIR:-./safesim_logs_stage1}" \
  MAX_EPOCHS="${MAX_EPOCHS:-20}" \
  LR="${LR:-5e-5}" \
  TF_D_MODEL="${TF_D_MODEL:-1024}" \
  TARGET_POLICY=raw_gt \
  INIT_MODE="${INIT_MODE:-fm_head_conservative}" \
  INIT_CHECKPOINT="${INIT_CHECKPOINT:-${GOALFLOW_CKPT}}" \
  FREEZE_LOADED_FM_EPOCHS="${FREEZE_LOADED_FM_EPOCHS:-3}" \
  CFG_SCALE="${CFG_SCALE:-1.0}" \
  STAGE_NAME="${STAGE_NAME:-stage1}" \
  SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}" \
  sh scripts/training/run_safesim_training.sh
