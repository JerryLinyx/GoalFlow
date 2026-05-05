# SafeSim V2 Filtered Baseline

> Archived document. This baseline note belongs to an older SafeSim V2 line and
> is preserved for historical reference only.

## Purpose

This document freezes the current V2 model as the official baseline for all subsequent fine-tuning and loss-guidance experiments.

The goal is to keep:

- the checkpoint,
- the data split,
- the evaluation protocol,
- and the reported metrics

stable and easy to reference in future comparisons.

## Baseline Definition

### Data

- split: `filtered`
- files:
  - `safesim/case1_filtered/data.hdf5`
  - `safesim/case2_filtered/data.hdf5`
  - `safesim/case3_filtered/data.hdf5`
  - `safesim/case4_filtered/data.hdf5`
  - `safesim/case5_filtered/data.hdf5`

### Split Protocol

- split unit: scene-level
- seed: `0`
- train / val samples: `2493 / 279`
- train / val scenes: `277 / 31`
- case balance sampler: `1 / sqrt(N_case)`

Reference file:

- [split_summary.json](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_filtered_case_v2/split_summary.json)

### Model

Architecture:

- map CNN encoder
- agent MLP + GRU encoder
- explicit `ego-ctrl pair token`
- fusion transformer over `[CLS, pair, map, agent]`
- decoder cross-attention to `all_tokens`
- `role × case` binding for `ego` and `ctrl`
- `pair × case` binding for pair token

### Training

- objective: pure imitation / Flow Matching L1
- max epochs: `60`
- early stopping patience: `10`
- selection metric: `val_primary_metric`
- inference config during validation:
  - `anchor_size=64`
  - `infer_steps=100`
  - `cfg_scale=1.0`

## Official Baseline Checkpoint

- best checkpoint:
  [best-primary-03-0.6745.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_filtered_case_v2/checkpoints/best-primary-03-0.6745.ckpt)

- last checkpoint:
  [last.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_filtered_case_v2/checkpoints/last.ckpt)

## Updated Plan A Baseline

The baseline above is the original V2 filtered baseline.

After adding a stabler evaluation protocol in Plan A:

- deterministic validation noise
- no early stopping interference
- physical plausibility diagnostics

the working baseline for future fine-tuning exploration is now:

- checkpoint:
  [best-primary-09-0.6329.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_filtered_case_v2_plan_a/checkpoints/best-primary-09-0.6329.ckpt)
- log dir:
  [safesim_logs_filtered_case_v2_plan_a](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_filtered_case_v2_plan_a)

Best observed Plan A metrics so far:

- `val_primary_metric = 0.6329`
- `val_bbox_collision_rate = 0.5914`
- `val_hit_2m = 0.3513`
- `val_hit_4m = 0.6559`
- `val_pred_better_than_gt_rate = 0.6380`
- `val_random_better_than_gt_rate = 0.5448`
- `val_pred_min_dist = 6.8096m`
- `val_random_min_dist = 8.9324m`

Important interpretation:

- this Plan A baseline is not numerically better than the earlier V2 best on task metrics;
- it is kept because the protocol is cleaner and now exposes physical plausibility diagnostics;
- it is therefore the more appropriate baseline for later fine-tuning experiments.

## Best Validation Metrics

Best epoch: `3`

- `val_primary_metric = 0.6745`
- `val_bbox_collision_rate = 0.6308`
- `val_hit_2m = 0.3728`
- `val_hit_4m = 0.6738`
- `val_pred_better_than_gt_rate = 0.6380`
- `val_random_better_than_gt_rate = 0.5484`
- `val_pred_min_dist = 6.1838m`
- `val_random_min_dist = 9.1926m`
- `val_candidate_mean_min_dist = 9.0005m`
- `val_loss = 0.1449`

## Interpretation

This baseline is the current best task-aligned model under:

- filtered data only
- no auxiliary collision-aware loss
- no fine-tuning from a normal-driving checkpoint
- no candidate prior
- no inference guidance beyond nearest-to-ctrl candidate selection

Important implication:

- this checkpoint is **not** the best imitation model
- it is the best task model under the current evaluation rule

This means all future experiments should compare primarily against:

- `val_primary_metric`
- `val_bbox_collision_rate`
- `val_hit_2m`
- `val_pred_min_dist`

and not against `val_loss` alone.

## Known Limitation

The current baseline still uses imitation-only training.

Observed behavior:

- task metrics peak early
- later epochs can continue improving imitation loss while hurting collision-oriented metrics

Therefore this baseline should be treated as:

- a strong architecture baseline
- not the final objective-aligned solution

## Next Comparison Targets

All future experiments should compare back to this baseline unless explicitly stated otherwise.

Immediate next candidates:

1. inference-only CFG sweep on the same checkpoint
2. normal-trajectory pretrain + dangerous-data fine-tune
3. imitation + small collision-aware auxiliary loss
