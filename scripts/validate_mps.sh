#!/bin/bash
# MPS MVP validation script
# Runs 1 batch through the full pipeline to verify adv_mode selection logic works
# Usage: bash scripts/validate_mps.sh

set -e

export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-$(pwd)}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/Users/linyuxuan/navsim_data}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-$(pwd)/exp}
export HYDRA_FULL_ERROR=1

FEATURE_CACHE=$NAVSIM_EXP_ROOT/feature_cache_test
CHECKPOINT_PATH=$NAVSIM_DEVKIT_ROOT/data/goalflow_traj_epoch_54-step_18260.ckpt
VOC_PATH=$NAVSIM_DEVKIT_ROOT/data/cluster_points_8192_.npy

echo "=============================="
echo " GoalFlow MPS MVP Validation  "
echo "=============================="
echo "NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "Feature cache:      $FEATURE_CACHE"
echo ""

# Detect MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

echo ""
echo "[Step] Running fast_dev_run=true (1 batch, adv_mode=False → normal mode)"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=goalflow_agent_traj \
    experiment_name=validate_mps_normal \
    scene_filter=navtest \
    split=test \
    cache_path=$FEATURE_CACHE \
    trainer.params.fast_dev_run=true \
    trainer.params.accelerator=mps \
    trainer.params.precision=32 \
    trainer.params.max_epochs=1 \
    agent.config.training=False \
    agent.config.has_navi=True \
    agent.config.start=True \
    agent.config.use_nearest=True \
    agent.config.adv_mode=False \
    agent.config.anchor_size=4 \
    agent.config.infer_steps=5 \
    agent.config.freeze_perception=True \
    agent.config.tf_d_model=1024 \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=0 \
    use_cache_without_dataset=True \
    agent.config.voc_path=$VOC_PATH \
    agent.checkpoint_path=$CHECKPOINT_PATH
echo "✅ Normal mode passed"

echo ""
echo "[Step] Running fast_dev_run=true (1 batch, adv_mode=True → adversarial mode)"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=goalflow_agent_traj \
    experiment_name=validate_mps_adv \
    scene_filter=navtest \
    split=test \
    cache_path=$FEATURE_CACHE \
    trainer.params.fast_dev_run=true \
    trainer.params.accelerator=mps \
    trainer.params.precision=32 \
    trainer.params.max_epochs=1 \
    agent.config.training=False \
    agent.config.has_navi=True \
    agent.config.start=True \
    agent.config.use_nearest=True \
    agent.config.adv_mode=True \
    agent.config.adv_agent_idx=0 \
    agent.config.adv_traj_step=8 \
    agent.config.anchor_size=4 \
    agent.config.infer_steps=5 \
    agent.config.freeze_perception=True \
    agent.config.tf_d_model=1024 \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=0 \
    use_cache_without_dataset=True \
    agent.config.voc_path=$VOC_PATH \
    agent.checkpoint_path=$CHECKPOINT_PATH
echo "✅ Adversarial mode passed"

echo ""
echo "=============================="
echo " MVP Validation Complete ✅   "
echo "=============================="
