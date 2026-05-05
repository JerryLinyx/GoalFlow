#!/bin/bash
# ============================================================
# SafeSim training script with optional GoalFlow FM-head transfer and mixed replay.
# ============================================================

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

# ======================== Data Paths ========================
HDF5_PATHS="${HDF5_PATHS:-safesim/case1/data.hdf5 safesim/case2/data.hdf5 safesim/case3/data.hdf5 safesim/case4/data.hdf5 safesim/case5/data.hdf5}"
REPLAY_HDF5_PATHS="${REPLAY_HDF5_PATHS:-}"

# Alternatives:
# HDF5_PATHS="safesim/case1_filtered/data.hdf5 safesim/case2_filtered/data.hdf5 safesim/case3_filtered/data.hdf5 safesim/case4_filtered/data.hdf5 safesim/case5_filtered/data.hdf5"
# HDF5_PATHS="safesim/case1_best/data.hdf5"

# ======================== Training Config ========================
MAX_EPOCHS="${MAX_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
GPUS="${GPUS:-1}"
PRECISION="${PRECISION:-16-mixed}"
LOG_DIR="${LOG_DIR:-./safesim_logs_cfg_base}"
CHECKPOINT="${CHECKPOINT:-}"
SEED="${SEED:-0}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-5}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
STAGE_NAME="${STAGE_NAME:-baseline}"

# ======================== Model Config ========================
TF_D_MODEL="${TF_D_MODEL:-256}"
TF_NUM_LAYERS="${TF_NUM_LAYERS:-3}"
MAX_OTHER_AGENTS="${MAX_OTHER_AGENTS:-10}"
ANCHOR_SIZE="${ANCHOR_SIZE:-10}"
INFER_STEPS="${INFER_STEPS:-100}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-5}"
HISTORY_LEN="${HISTORY_LEN:-4}"
HISTORY_STRIDE="${HISTORY_STRIDE:-5}"
CFG_SCALE="${CFG_SCALE:-1.5}"
CONDITION_DROPOUT_PROB="${CONDITION_DROPOUT_PROB:-0.15}"
USE_GOAL_CONDITION="${USE_GOAL_CONDITION:-0}"
TARGET_POLICY="${TARGET_POLICY:-raw_gt}"
REPLAY_RATIO="${REPLAY_RATIO:-0.3}"
TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}"
TERMINAL_XY_WEIGHT="${TERMINAL_XY_WEIGHT:-0.0}"
TERMINAL_HEADING_WEIGHT="${TERMINAL_HEADING_WEIGHT:-0.0}"
CTRL_SOFTMIN_WEIGHT="${CTRL_SOFTMIN_WEIGHT:-0.0}"
CTRL_SOFTMIN_BETA="${CTRL_SOFTMIN_BETA:-4.0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
INIT_MODE="${INIT_MODE:-none}"
FREEZE_LOADED_FM_EPOCHS="${FREEZE_LOADED_FM_EPOCHS:-0}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"

CHECKPOINT_ARGS=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_ARGS="--checkpoint ${CHECKPOINT}"
fi

REPLAY_ARGS=""
if [ -n "${REPLAY_HDF5_PATHS}" ]; then
    REPLAY_ARGS="--replay_hdf5_paths ${REPLAY_HDF5_PATHS} --replay_ratio ${REPLAY_RATIO}"
fi

INIT_ARGS=""
if [ "${INIT_MODE}" != "none" ]; then
    INIT_ARGS="--init_mode ${INIT_MODE} --init_checkpoint ${INIT_CHECKPOINT} --freeze_loaded_fm_epochs ${FREEZE_LOADED_FM_EPOCHS}"
fi

MODEL_INIT_ARGS=""
if [ -n "${MODEL_CHECKPOINT}" ]; then
    MODEL_INIT_ARGS="--model_checkpoint ${MODEL_CHECKPOINT}"
fi

python navsim/agents/goalflow/run_safesim_training.py \
    --hdf5_paths ${HDF5_PATHS} \
    --max_epochs ${MAX_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --gpus ${GPUS} \
    --precision ${PRECISION} \
    --log_dir ${LOG_DIR} \
    --seed ${SEED} \
    --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --limit_train_batches ${LIMIT_TRAIN_BATCHES} \
    --limit_val_batches ${LIMIT_VAL_BATCHES} \
    --tf_d_model ${TF_D_MODEL} \
    --tf_num_layers ${TF_NUM_LAYERS} \
    --max_other_agents ${MAX_OTHER_AGENTS} \
    --anchor_size ${ANCHOR_SIZE} \
    --infer_steps ${INFER_STEPS} \
    --temporal_stride ${TEMPORAL_STRIDE} \
    --history_len ${HISTORY_LEN} \
    --history_stride ${HISTORY_STRIDE} \
    --cfg_scale ${CFG_SCALE} \
    --condition_dropout_prob ${CONDITION_DROPOUT_PROB} \
    --use_goal_condition ${USE_GOAL_CONDITION} \
    --target_policy ${TARGET_POLICY} \
    --trajectory_weight ${TRAJECTORY_WEIGHT} \
    --terminal_xy_weight ${TERMINAL_XY_WEIGHT} \
    --terminal_heading_weight ${TERMINAL_HEADING_WEIGHT} \
    --ctrl_softmin_weight ${CTRL_SOFTMIN_WEIGHT} \
    --ctrl_softmin_beta ${CTRL_SOFTMIN_BETA} \
    --stage_name ${STAGE_NAME} \
    ${REPLAY_ARGS} \
    ${INIT_ARGS} \
    ${MODEL_INIT_ARGS} \
    ${CHECKPOINT_ARGS}
