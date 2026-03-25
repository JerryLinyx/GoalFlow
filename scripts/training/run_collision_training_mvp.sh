#!/bin/bash
# ==========================================================
# GoalFlow Collision Training — MVP on test cache
#
# Uses existing test feature cache (10-step trajectories).
# GoalFlowCollisionAgent.forward() automatically:
#   1. Pads 10→11 steps to match model noise dim (B,12,30)
#   2. Syncs targets['trajectory'] → features['gt_trajs']
#
# This is for quick validation only.
# Full training should use trainval cache (11-step).
# ==========================================================

# ===== PATHS (edit these) =====
FEATURE_CACHE=${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}/feature_cache_test
V99_PRETRAINED_PATH=$NAVSIM_DEVKIT_ROOT/data/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth
CHECKPOINT_PATH=$NAVSIM_DEVKIT_ROOT/data/goalflow_traj_epoch_54-step_18260.ckpt
VOC_PATH=$NAVSIM_DEVKIT_ROOT/data/cluster_points_8192_.npy

# ===== MVP SETTINGS =====
MAX_EPOCHS=5          # quick validation: 5 epochs
BATCH_SIZE=2          # small batch for CPU/single GPU
PRECISION=32          # use 32 for CPU/MPS compatibility

echo "=== GoalFlow Collision Training MVP ==="
echo "Cache:      $FEATURE_CACHE"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Epochs:     $MAX_EPOCHS"
echo "Batch:      $BATCH_SIZE"
echo ""

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=goalflow_agent_collision \
  experiment_name=collision_mvp \
  scene_filter=navtest \
  split=test \
  cache_path=$FEATURE_CACHE \
  use_cache_without_dataset=True \
  trainer.params.max_epochs=$MAX_EPOCHS \
  trainer.params.precision=$PRECISION \
  agent.config.training=True \
  agent.config.has_navi=True \
  agent.config.start=True \
  agent.config.freeze_perception=True \
  agent.config.only_perception=False \
  agent.config.train_scale=0.1 \
  agent.config.tf_d_model=1024 \
  agent.config.trajectory_weight=50.0 \
  agent.config.agent_class_weight=0.2 \
  agent.config.agent_box_weight=0.05 \
  agent.config.bev_semantic_weight=0.2 \
  agent.config.agent_loss=True \
  agent.config.adv_mode=True \
  dataloader.params.batch_size=$BATCH_SIZE \
  agent.config.v99_pretrained_path=$V99_PRETRAINED_PATH \
  agent.checkpoint_path=$CHECKPOINT_PATH \
  agent.config.voc_path=$VOC_PATH
