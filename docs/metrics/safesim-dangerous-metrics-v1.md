# SafeSim Dangerous Metrics Protocol v1.1

This protocol defines the official evaluation scheme for SafeSim dangerous trajectory generation.

## Decision Flow

Checkpoint selection is:

1. Evaluate all Tier A/B/C/D/E metrics.
2. Filter checkpoints by acceptance gates.
3. Rank the remaining checkpoints by:
   1. `dangerous_hit_rate` descending
   2. `hit@2m` descending
   3. `pred_min_dist` ascending
   4. `pred_better_than_gt_rate` descending when enabled

If no checkpoint passes gates, report `selection_status=no_valid_checkpoint` and still emit the top diagnostic checkpoint.

## Metric Tiers

### Tier A: Dangerous task metrics

- `dangerous_hit_rate ↑`
- `hit@2m ↑`
- `hit@4m ↑`
- `pred_min_dist ↓`
- `random_dangerous_hit_rate ↑`
- `random_min_dist ↓`
- `pred_better_than_gt_rate ↑`

`pred_better_than_gt_rate` is reported only when `target_policy != raw_gt`.

Definitions:

- `dangerous_hit_rate`: any-time oriented bbox overlap between predicted ego and ctrl
- `hit@τ`: `min_t ||pred_xy[t] - ctrl_xy[t]|| <= τ`
- `pred_min_dist`: `min_t ||pred_xy[t] - ctrl_xy[t]||`

### Tier B: Trajectory quality metrics

- `ADE_vs_target ↓`
- `FDE_vs_target ↓`
- `deviation_ADE_vs_raw_gt`
- `deviation_FDE_vs_raw_gt`
- `minADE@6_vs_target ↓`
- `minFDE@6_vs_target ↓`
- `meanADE@6_vs_target ↓`
- `meanFDE@6_vs_target ↓`
- `candidate_ADE_std@6 ↑`
- `MissRate@2m_vs_target ↓`
- `MissRate@4m_vs_target ↓`

`deviation_*_vs_raw_gt` is a deviation signal, not a dangerous-task success metric.

`minADE/minFDE` are always computed with `K=6`, never with `K=64`.

### Tier C: Physical plausibility metrics

- `mean_speed`
- `mean_accel ↓`
- `mean_jerk ↓`
- `max_jerk ↓`
- `first_step_speed_error ↓`
- `first_step_heading_error ↓`
- `low_motion_rate ↓`

Definitions:

- `low_motion_rate`: predicted path length `< 1.0m`
- `first_step_speed_error`:
  - `v_hist = ||h[-1] - h[-2]|| / dt`
  - `v_pred = ||pred[0] - origin|| / dt`
  - error = `|v_pred - v_hist|`
- `first_step_heading_error`:
  - `θ_hist = atan2(h[-1]_y - h[-2]_y, h[-1]_x - h[-2]_x)`
  - `θ_pred = atan2(pred[0]_y, pred[0]_x)`
  - error = `abs(wrap_angle(θ_pred - θ_hist))`

### Tier D: Scene consistency metrics

- `offroad_rate ↓`
- `non_drivable_occupancy_rate ↓`

v1 uses center-point occupancy only. A sample is off-road if any future center point lies on a non-drivable pixel or outside map bounds.

Map assumption for v1:

- local `x ∈ [-60, 60]` spans the full map width
- local `y ∈ [-15, 15]` spans the full map height

### Tier E: Candidate diversity metrics

- `candidate_min_dist_spread ↑`
- `candidate_xy_std ↑`

Definitions:

- `candidate_min_dist_spread = p90(min_dist) - p10(min_dist)` over candidates
- `candidate_xy_std`: mean XY standard deviation across the first 6 candidates

## Case-wise Reporting

All formal evaluations must output:

- global metrics
- per-case Tier A metrics for cases 1–5
- per-case sample counts
- case summary statistics:
  - `case_mean`
  - `case_std`
  - `case_gap = max - min`
  - `case_min`

## Confidence Intervals

Binary metrics use Wilson 95% confidence intervals:

- `dangerous_hit_rate`
- `hit@2m`
- `hit@4m`
- `pred_better_than_gt_rate`
- `low_motion_rate`
- `offroad_rate`

Continuous metrics use bootstrap 95% confidence intervals with 1000 resamples:

- `pred_min_dist`
- `ADE/FDE`
- `mean_jerk`
- `mean_accel`
- `first_step_speed_error`
- `first_step_heading_error`

## Acceptance Gates

A checkpoint is valid only if:

- `low_motion_rate <= 0.05`
- `mean_accel <= 6.0`
- `mean_jerk <= 15.0`
- `max_jerk <= 40.0`
- `first_step_speed_error <= 5.0`
- `offroad_rate <= 0.10`

## Output Layout

```text
outputs/{run_id}/
  metrics/
    global.csv
    per_case.csv
    confidence.json
    best_summary.json
  qualitative/
    success/
      examples.png
    failure_safe/
      examples.png
    failure_unphysical/
      examples.png
  summary.md
```

## Qualitative Categories

- `success`: `dangerous_hit=1` and sample passes plausibility checks
- `failure_safe`: `dangerous_hit=0` and sample passes plausibility checks
- `failure_unphysical`: sample is dangerous or near-dangerous, but fails plausibility checks
