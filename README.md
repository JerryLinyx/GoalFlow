# GoalFlow / SafeSim Workspace

English | [简体中文](README.zh-CN.md)

This repository now serves two purposes:

1. the original **GoalFlow (CVPR 2025)** codebase and paper-era materials
2. an actively developed **SafeSim dangerous trajectory generation** research line built on top of GoalFlow

The root README is the entrypoint for the **current workspace state**. The original paper-style README has been preserved at:

- [docs/legacy/goalflow-original-readme.md](docs/legacy/goalflow-original-readme.md)

## Current Project Focus

The active SafeSim line studies dangerous trajectory generation with:

- explicit `goal_point` conditioning
- protocol-based evaluation
- structured comparison between imitation, terminal guidance, and softmin guidance

The current formulation is:

```text
history/start state + scene context + goal_point -> trajectory
```

## Training Regimes

The repository currently contains three distinct training regimes:

1. **Full training / from-scratch baseline**
   - SafeSim Baseline
   - trained without GoalFlow transfer
   - used as the main non-transfer reference

2. **Transfer and prior alignment**
   - Stage1 Prior Alignment
   - initializes from transferred GoalFlow-compatible weights
   - aligns the model on `original` data before dangerous fine-tuning

3. **Corrected fine-tuning mainline**
   - Goal-conditioned Pure Imitation
   - Goal-conditioned Terminal Ablation
   - Goal-conditioned Softmin Ablation
   - these are the current fine-tuning experiments and the main source of the latest results

## Experiment Snapshot Table

The table below tracks the main experiment scenarios that summarize the current
project state.

| Scenario | Role | Supervision | Goal Point | Status | dangerous_hit_rate | hit@2m | pred_min_dist | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| SafeSim Baseline | full-training reference | legacy baseline line | no explicit goal token | evaluated | 0.2812 | 0.1562 | 12.9143 | historical qualitative anchor |
| Stage1 Prior Alignment | transfer reference | `raw_gt` on `original` | no explicit goal token | evaluated | 0.2188 | 0.0781 | 13.4077 | transfer + prior-alignment stage before fine-tuning |
| Goal-conditioned Pure Imitation | fine-tuning baseline | `action` | explicit `goal_point` | evaluated | 0.4219 | 0.3125 | 9.6171 | corrected fine-tuning baseline |
| Goal-conditioned Terminal Ablation | fine-tuning ablation | `action` | explicit `goal_point` | evaluated | 0.4688 | 0.3125 | 9.2413 | best terminal-only is `xy=0.25, heading=0.05` |
| Goal-conditioned Softmin Ablation | fine-tuning ablation | `action` | explicit `goal_point` | evaluated | 0.5156 | 0.3438 | 4.6922 | best softmin is `0.0025` |

## Checkpoints and Artifacts

For reproducibility and result inspection, use the shareable checkpoint bundle:

- [goalflow_safesim_checkpoints_20260505.zip](https://drive.google.com/file/d/1FRYlWvijY_QJUC8Bcm4n8JTSROpBVJW-/view?usp=drive_link)

Artifact-to-scenario mapping inside the bundle:

| Folder in zip | Scenario in this README | What it contains |
|---|---|---|
| `baseline/` | SafeSim Baseline | from-scratch baseline checkpoint and evaluation summary |
| `stage1/` | Stage1 Prior Alignment | transfer/prior-alignment checkpoint used before fine-tuning |
| `pure_imitation_goal_action/` | Goal-conditioned Pure Imitation | corrected pure imitation checkpoint and formal evaluation summary |
| `best_terminal_only/` | Goal-conditioned Terminal Ablation | best terminal-only checkpoint (`terminal_xy=0.25`, `terminal_heading=0.05`) and formal evaluation summary |
| `best_softmin/` | Goal-conditioned Softmin Ablation | best softmin checkpoint (`softmin=0.0025` on the best terminal base) and formal evaluation summary |

The bundle also includes a top-level `MANIFEST.md` with the same mapping.

## Current Takeaway

The current corrected mainline is complete:

- goal-conditioned pure imitation
- terminal sweep
- softmin sweep
- protocol-based evaluation for all current valid runs

## Conclusions and Tradeoffs

- **Best dangerous-task result:** Goal-conditioned softmin ablation (`softmin = 0.0025` on top of the best terminal base) gives the strongest dangerous metrics:
  - `dangerous_hit_rate = 0.5156`
  - `hit@2m = 0.3438`
  - `pred_min_dist = 4.6922`
- **Best terminal-only result:** `terminal_xy = 0.25`, `terminal_heading = 0.05` is the strongest terminal-only variant:
  - `dangerous_hit_rate = 0.4688`
  - `pred_min_dist = 9.2413`
- **Best baseline for controlled comparison:** goal-conditioned pure imitation is the clean corrected baseline:
  - `dangerous_hit_rate = 0.4219`
  - `pred_min_dist = 9.6171`

Main tradeoff:

- adding `terminal` improves dangerous-task success over pure imitation, but increases motion aggressiveness
- adding small `softmin` improves dangerous-task success further, but pushes trajectories farther toward off-road and high-jerk behavior
- therefore, **softmin is best if the objective is danger maximization**, while **pure imitation / terminal-only are better if the objective is a more conservative balance**

The current practical takeaway is:

- **best overall by dangerous metrics:** goal-conditioned softmin (`0.0025`)
- **best balanced intermediate point:** terminal-only (`0.25 / 0.05`)

## Read First

1. Current evaluation protocol  
   [docs/metrics/safesim-dangerous-metrics-v1.md](docs/metrics/safesim-dangerous-metrics-v1.md)

2. Current experiment index  
   [docs/reports/2026-05-04-safesim-experiment-index.md](docs/reports/2026-05-04-safesim-experiment-index.md)

3. General docs index  
   [docs/README.md](docs/README.md)

4. Original GoalFlow install/train/test docs  
   [docs/install.md](docs/install.md)  
   [docs/train.md](docs/train.md)  
   [docs/test.md](docs/test.md)

## Main Experiment Lines

These are the main experiment lines for the current SafeSim workspace.

| Experiment | Purpose | Status | Entry |
|---|---|---|---|
| Stage 1 prior alignment | Transfer GoalFlow FM head into SafeSim structured-input training | Completed | [scripts/training/run_safesim_stage1.sh](scripts/training/run_safesim_stage1.sh) |
| Goal-conditioned pure imitation | Corrected baseline with `target_policy=action` and explicit `goal_point` | Trained and evaluated | [scripts/training/run_safesim_stage2_action_imitation.sh](scripts/training/run_safesim_stage2_action_imitation.sh) |
| Goal-conditioned terminal sweep | Ablation over `terminal_xy` / `terminal_heading` with corrected setup | Trained and evaluated | [scripts/training/run_safesim_stage2_terminal_sweep.sh](scripts/training/run_safesim_stage2_terminal_sweep.sh) |
| Goal-conditioned softmin sweep | Small-weight sweep on the selected corrected terminal base | Trained and evaluated | [scripts/training/run_safesim_stage2_softmin_sweep.sh](scripts/training/run_safesim_stage2_softmin_sweep.sh) |
| Goal-action mainline orchestrator | Finish terminal sweep, run evaluation, choose terminal base, launch softmin sweep | Completed | [scripts/training/run_safesim_goal_action_mainline.sh](scripts/training/run_safesim_goal_action_mainline.sh) |

Historical and superseded experiments are still available in the archive and
experiment index, but they are not the focus of this README.

## Scenario Sections

### Original GoalFlow context

This section is the paper-era reference point for the repository.

![GoalFlow main figure](assets/main_fig.png)

### Scenario 1: SafeSim baseline reference

This is a historical from-scratch SafeSim reference kept as a qualitative
anchor.

![SafeSim baseline reference](assets/safesim_current/baseline_proxy64_examples.png)

### Scenario 2: Stage1 prior-alignment reference

This section shows the transferred SafeSim model before corrected Stage2 and
serves as the main transfer reference.

![Stage1 prior alignment reference](assets/safesim_current/stage1_proxy64_examples.png)

### Scenario 3: Goal-conditioned pure imitation

This is the first corrected baseline under:

- `target_policy = action`
- explicit `goal_point`
- no `terminal`
- no `softmin`

This is the corrected baseline that the new ablations are compared against.

![Goal-conditioned pure imitation](assets/safesim_current/pure_imitation_goal_action_examples.png)

### Scenario 4: Goal-conditioned terminal ablation

This family is now fully evaluated. The best terminal-only variant is:

- `terminal_xy = 0.25`
- `terminal_heading = 0.05`

It improves dangerous-task metrics over corrected pure imitation.

![Best corrected terminal-only examples](assets/safesim_current/terminal_goal_action_best_examples.png)


### Scenario 5: Goal-conditioned softmin ablation

This family is now fully evaluated. The best corrected softmin variant is:

- base terminal: `terminal_xy = 0.25`, `terminal_heading = 0.05`
- `softmin = 0.0025`

This is the strongest dangerous-task result on the corrected mainline.

![Best corrected softmin examples](assets/safesim_current/softmin_goal_action_best_examples.png)

## Current Result Files

The latest corrected mainline outputs are organized under:

- pure imitation:
  [outputs/current_goal_action/pure_imitation_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/pure_imitation_protocol64)
- terminal sweep:
  [outputs/current_goal_action/terminal_eval](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/terminal_eval)
- softmin sweep:
  [outputs/current_goal_action/softmin_eval](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/softmin_eval)

These are the authoritative result directories for the corrected goal-conditioned line.

## Main Code Entry Points

### Original GoalFlow

- Trajectory model: [navsim/agents/goalflow/goalflow_model_traj.py](navsim/agents/goalflow/goalflow_model_traj.py)
- Goal-point model: [navsim/agents/goalflow/goalflow_model_navi.py](navsim/agents/goalflow/goalflow_model_navi.py)
- Original trajectory training script: [scripts/training/run_goalflow_training_traj.sh](scripts/training/run_goalflow_training_traj.sh)

### SafeSim Research Line

- Common training entry: [navsim/agents/goalflow/run_safesim_training.py](navsim/agents/goalflow/run_safesim_training.py)
- Agent: [navsim/agents/goalflow/safesim_agent.py](navsim/agents/goalflow/safesim_agent.py)
- Dataset: [navsim/agents/goalflow/safesim_dataset.py](navsim/agents/goalflow/safesim_dataset.py)
- Encoder: [navsim/agents/goalflow/safesim_encoder.py](navsim/agents/goalflow/safesim_encoder.py)
- Model: [navsim/agents/goalflow/safesim_model.py](navsim/agents/goalflow/safesim_model.py)
- Config: [navsim/agents/goalflow/safesim_config.py](navsim/agents/goalflow/safesim_config.py)

## Evaluation

Formal comparison should use the protocol evaluator, not training loss alone.

- Main evaluator: [scripts/analysis/evaluate_safesim_dangerous.py](scripts/analysis/evaluate_safesim_dangerous.py)
- Terminal sweep evaluation: [scripts/analysis/run_safesim_terminal_sweep_eval.sh](scripts/analysis/run_safesim_terminal_sweep_eval.sh)
- Softmin sweep evaluation: [scripts/analysis/run_safesim_softmin_sweep_eval.sh](scripts/analysis/run_safesim_softmin_sweep_eval.sh)

## Environment

Two environment manifests now coexist on purpose:

- [requirements.txt](/Users/linyuxuan/workSpace/GoalFlow/requirements.txt): legacy/original GoalFlow paper-oriented dependency set
- [requirements.safesim-current.txt](/Users/linyuxuan/workSpace/GoalFlow/requirements.safesim-current.txt): current validated environment snapshot for the corrected SafeSim goal-conditioned mainline

For the corrected SafeSim mainline, prefer:

```bash
conda create -n goalflow python=3.10
conda activate goalflow
pip install -r requirements.safesim-current.txt
pip install -e nuplan-devkit
pip install -e .
```

For the legacy paper workflow, keep using `requirements.txt`.

Legacy setup:

```bash
conda create -n goalflow python=3.10
conda activate goalflow
pip install -r requirements.txt
pip install -e nuplan-devkit
pip install -e .
```

For the original paper workflow, continue with:

- [docs/install.md](docs/install.md)
- [docs/train.md](docs/train.md)
- [docs/test.md](docs/test.md)

## Paper and Legacy Materials

- Paper: [GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving](https://arxiv.org/abs/2503.05689)
- Project page: [GoalFlow Project Page](https://zebinx.github.io/HomePage-of-GoalFlow/)
- Original archived README: [docs/legacy/goalflow-original-readme.md](docs/legacy/goalflow-original-readme.md)
