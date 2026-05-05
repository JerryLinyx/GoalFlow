# SafeSim History Resolution A/B Test

> Archived document. This A/B note belongs to an older SafeSim line and is kept
> only for experiment history.

## Purpose

This document defines the first controlled A/B experiment for Safe-Sim history resolution.

Goal:

- test whether increasing history frequency improves dangerous ego-trajectory generation quality
- do this without changing loss, decoder structure, selector, or data split rules

## Fixed Protocol

- dataset: five-case `filtered`
- conditioning: `case_id` enabled
- loss: pure imitation
- selector: nearest-to-ctrl with `anchor_size=64`
- split: scene-level, seed `0`
- case balancing: `1 / sqrt(N_case)`
- checkpoint selection: `val_primary_metric`

## A/B Definition

### Screening Budget Override

This first A/B pass is a screening experiment, not a final leaderboard run.

- `anchor_size = 16`
- `infer_steps = 25`
- `max_epochs = 8`

Reason:

- full R1 validation with `anchor_size=64` and `infer_steps=100` is too expensive for iterative A/B
- both A and B still use the same inference budget, so the comparison remains fair
- any winning variant must later be rerun under the full evaluation budget

### A. Baseline

- `history_len = 4`
- `history_stride = 5`
- effective history frequency: `2Hz`

### B. Variant

- `history_len = 8`
- `history_stride = 2`
- effective history frequency: `5Hz`

## Why This A/B Matters

The current model only sees a sparse history. That limits how well it can infer:

- closing speed
- short-horizon steering trend
- timing of collision opportunity
- ego-ctrl relative motion consistency

This experiment isolates whether denser history alone provides a measurable improvement before changing model structure.

## Success Criteria

The variant is considered better only if it improves task-aligned validation metrics, not just imitation loss.

Primary indicators:

- `val_primary_metric`
- `val_bbox_collision_rate`
- `val_hit_2m`
- `val_pred_better_than_gt_rate`

Bias control:

- always compare selected metrics with random-candidate metrics
- do not treat `selected pred_min_dist` alone as evidence of better generation quality

## Output Artifacts

- baseline log dir: `safesim_logs_ab_history_baseline`
- variant log dir: `safesim_logs_ab_history_5hz`
- comparison summary: `outputs/history_ab/summary.md`

## Result Notes

### Outcome

This screening A/B did **not** show evidence that increasing history frequency alone helps under the current architecture and single-seed protocol.

Best checkpoints under the shared screening budget:

- baseline (`history_len=4, history_stride=5`): `epoch 1`
- variant (`history_len=8, history_stride=2`): `epoch 1`

Best-metric comparison:

| Metric | Baseline | Variant |
| --- | ---: | ---: |
| `val_primary_metric` | `0.6203` | `0.5972` |
| `val_bbox_collision_rate` | `0.5806` | `0.5591` |
| `val_hit_2m` | `0.3333` | `0.3190` |
| `val_hit_4m` | `0.6344` | `0.5986` |
| `val_pred_better_than_gt_rate` | `0.6272` | `0.6201` |
| `val_random_better_than_gt_rate` | `0.5950` | `0.5412` |
| `val_pred_min_dist` | `6.6368m` | `7.3583m` |
| `val_random_min_dist` | `8.2431m` | `9.1436m` |
| `val_candidate_mean_min_dist` | `8.2130m` | `9.1152m` |
| `val_loss` | `0.1464` | `0.1438` |

### Interpretation

- Higher-frequency history slightly improved imitation loss, but that did **not** translate into better task-aligned collision metrics in this run family.
- Both the selected and random-candidate distance metrics got worse in the high-frequency variant, which argues against the gap being only a selector artifact.
- The evidence strength is still limited: this was a single-seed screening run, and both arms reached their best validation point at `epoch 1`, which indicates high early-epoch variance.
- Under the current encoder, adding more history likely increases input detail without giving the model a stronger mechanism to exploit ego-ctrl interaction, but this remains a hypothesis rather than a proven causal result.

### Decision

- keep the baseline history setting for now
- do **not** treat denser history alone as a high-priority standalone improvement
- prioritize explicit `ego-ctrl` interaction modeling and decoder access to richer scene tokens before revisiting higher-frequency history
- rerun the history ablation only after stronger interaction modeling is in place and with multi-seed evaluation

Detailed run summary is available at:

- [outputs/history_ab/summary.md](/Users/linyuxuan/workSpace/GoalFlow/outputs/history_ab/summary.md)

Follow-on architecture decision:

- [docs/archive/plans/2026-04-15-safesim-v2-design.md](/Users/linyuxuan/workSpace/GoalFlow/docs/archive/plans/2026-04-15-safesim-v2-design.md)
