# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

GoalFlow is a CVPR 2025 research project for end-to-end autonomous driving trajectory planning. It uses goal-driven Flow Matching to generate multimodal trajectories on the [NAVSIM](https://github.com/autonomousvision/navsim) benchmark. The codebase extends NAVSIM with custom agents, models, and training infrastructure.

## Environment Setup

```bash
conda create -n goalflow python=3.10
conda activate goalflow
pip install -r requirements.txt
pip install -e nuplan-devkit
pip install -e .
```

Key dependencies: PyTorch 2.0 (CUDA 11.8), PyTorch Lightning 2.2, Hydra 1.3, nuplan-devkit.

Set `NAVSIM_DEVKIT_ROOT` to the repo root before running any scripts. Dataset paths are configured in `navsim/planning/script/config/common/default_common.yaml` and `navsim/planning/script/config/metric_caching/default_metric_caching.yaml`.

## Key Commands

### Data Preparation (Cache)
```bash
sh scripts/cahce/run_dataset_cache_test.sh       # cache test features (required for eval)
sh scripts/cahce/run_dataset_cache_trainval.sh   # cache train features
sh scripts/cahce/run_metric_caching_test.sh      # cache test metrics (required for eval)
sh scripts/cahce/run_metric_caching_trainval.sh  # cache train metrics
sh scripts/generate/run_generate_dac_label.sh    # generate DAC score labels (optional)
```

### Training
```bash
sh scripts/training/run_goalflow_training_perception.sh  # Step 1: train perception module
sh scripts/training/run_goalflow_training_traj.sh        # Step 2: train trajectory planner
sh scripts/training/run_goalflow_training_navi.sh        # Step 3: train goal point module (optional)
```

Training is driven by `navsim/planning/script/run_training.py` via Hydra. Config overrides are passed as CLI args.

### Evaluation
```bash
sh scripts/generate/run_generate_trajs.sh        # generate trajectories → stored in log dir
sh scripts/evaluation/run_goalflow_trajs.sh      # compute PDMS score on generated trajectories
sh scripts/generate/run_generate_navi.sh         # evaluate goal point construction module (optional)
```

Evaluation is a two-step process: generate trajectories first, then score them separately.

## Architecture

### Module Structure

```
navsim/
├── agents/
│   └── goalflow/               # GoalFlow agent implementation
│       ├── goalflow_agent_traj.py    # Trajectory planning agent (PyTorch Lightning module)
│       ├── goalflow_agent_navi.py    # Goal point construction agent
│       ├── goalflow_model_traj.py    # Trajectory decoder model (Flow Matching)
│       ├── goalflow_model_navi.py    # Goal point scorer/constructor
│       ├── goalflow_config.py        # GoalFlowConfig dataclass (all hyperparams)
│       ├── goalflow_features.py      # Feature/target builders for dataset
│       ├── goalflow_loss.py          # Loss functions
│       ├── v99_backbone.py           # VoVNet-V99 image backbone
│       └── diffusion_es.py           # Flow Matching / diffusion utilities
├── planning/
│   ├── script/                 # Entry points (run_training.py, run_generate_trajs.py, etc.)
│   │   └── config/             # Hydra configs (agents, scene filters, training params)
│   └── training/               # Dataset, dataloader, agent_lightning_module.py
├── common/                     # Shared dataclasses, enums, dataloader
└── evaluate/                   # Metric computation
```

### Core Design

- **Agent abstraction**: All agents inherit from `AbstractAgent`. The Lightning training loop wraps agents in `AgentLightningModule`.
- **Feature pipeline**: `GoalFlowFeatureBuilder` and `GoalFlowTargetBuilder` preprocess sensor data into cached tensors. All training and inference operates on this feature cache.
- **Two-model design**: `GoalFlowTrajModel` handles trajectory generation; `GoalFlowModel` (navi) handles goal point scoring. They share the V99 perception backbone.
- **Goal point vocabulary**: Precomputed cluster centroids in `cluster_points_8192_.npy`. Goal points are selected from this vocabulary during inference.
- **Configuration via Hydra**: `GoalFlowConfig` dataclass is the single source of truth for all hyperparameters. Scripts pass overrides as CLI args (`agent.config.xxx=value`).
- **Evaluation flow**: PDMS (PDM Score) is computed separately from inference. Trajectories are saved as files during generation, then loaded for metric computation.

### Data Files (placed in `data/`)
- `goalflow_traj_epoch=54-step=18260.ckpt` — main model checkpoint
- `cluster_points_8192_.npy` — goal point vocabulary
- `goal_point_scores.gz` — precomputed goal point DAC + distance scores
- `goalflow_navi_epoch=99-step=132500.ckpt` — goal point construction checkpoint (optional)
- `depth_pretrained_v99-...pth` — V99 backbone pretrained weights (for training only)
