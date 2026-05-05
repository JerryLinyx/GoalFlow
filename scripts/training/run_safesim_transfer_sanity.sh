#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

COMMON_ORIGINAL_PATHS="safesim/case1/data.hdf5 safesim/case2/data.hdf5 safesim/case3/data.hdf5 safesim/case4/data.hdf5 safesim/case5/data.hdf5"
GOALFLOW_CKPT="${GOALFLOW_CKPT:-data/goalflow_traj_epoch_54-step_18260.ckpt}"
BASE_DIR="${BASE_DIR:-./safesim_logs_transfer_sanity}"

for MODE in none fm_head_conservative fm_head_extended; do
  LOG_DIR="${BASE_DIR}/${MODE}"
  EXTRA_ARGS=""
  if [ "${MODE}" != "none" ]; then
    EXTRA_ARGS="INIT_MODE=${MODE} INIT_CHECKPOINT=${GOALFLOW_CKPT} FREEZE_LOADED_FM_EPOCHS=1"
  fi
  env \
    HDF5_PATHS="${COMMON_ORIGINAL_PATHS}" \
    LOG_DIR="${LOG_DIR}" \
    MAX_EPOCHS=1 \
    TF_D_MODEL=1024 \
    NUM_WORKERS="${NUM_WORKERS:-0}" \
    LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-0.1}" \
    LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}" \
    CFG_SCALE=1.0 \
    TARGET_POLICY=raw_gt \
    STAGE_NAME="sanity_${MODE}" \
    SAVE_EVERY_N_EPOCHS=1 \
    ${EXTRA_ARGS} \
    sh scripts/training/run_safesim_training.sh
done
