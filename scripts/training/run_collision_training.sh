#!/bin/bash
# ==========================================================
# GoalFlow Collision Training
# Trains FM to generate collision trajectories toward agents
#
# Differences from normal training:
#   - Uses CollisionTargetBuilder (collision trajectories as GT)
#   - Perception is frozen (reuses pretrained V99 backbone)
#   - Goal point (navi) automatically points to target agent
# ==========================================================

FEATURE_CACHE=''  # set your feature_cache path (or use run_collision_cache.sh)
V99_PRETRAINED_PATH=$NAVSIM_DEVKIT_ROOT/data/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth
# Start from the pretrained normal GoalFlow checkpoint for faster convergence
CHECKPOINT_PATH=$NAVSIM_DEVKIT_ROOT/data/goalflow_traj_epoch_54-step_18260.ckpt
VOC_PATH=$NAVSIM_DEVKIT_ROOT/data/cluster_points_8192_.npy

FREEZE_PERCEPTION=True   # perception already trained, only fine-tune FM decoder

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=goalflow_agent_collision \
experiment_name=collision_train \
scene_filter=navtrain \
split=trainval \
cache_path=$FEATURE_CACHE \
trainer.params.max_epochs=50 \
agent.config.training=True \
agent.config.has_navi=True \
agent.config.start=True \
agent.config.freeze_perception=$FREEZE_PERCEPTION \
agent.config.only_perception=False \
agent.config.train_scale=0.1 \
agent.config.tf_d_model=1024 \
agent.config.trajectory_weight=50.0 \
agent.config.agent_class_weight=0.2 \
agent.config.agent_box_weight=0.05 \
agent.config.bev_semantic_weight=0.2 \
agent.config.agent_loss=True \
dataloader.params.batch_size=2 \
use_cache_without_dataset=True \
agent.config.v99_pretrained_path=$V99_PRETRAINED_PATH \
agent.checkpoint_path=$CHECKPOINT_PATH \
agent.config.voc_path=$VOC_PATH
