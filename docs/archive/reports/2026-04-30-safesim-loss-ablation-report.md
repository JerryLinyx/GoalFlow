# SafeSim Loss 消融报告

> Archived document. This report summarizes an older ablation line that used
> superseded supervision and evaluation assumptions.
## 实验设置
- 数据：`filtered` dangerous validation split
- 评估协议：`max_val_samples=64`, `anchor_size=16`, `infer_steps=25`, `cfg_scale=1.0`
- 对比方法：`baseline`, `Train Imitation (A0)`, `A1 terminal only`, `A2 softmin only`, `A3 terminal + softmin`
- 说明：qualitative panels 选取的是每组协议评估中的代表性样本，不是严格按同一 scene 一一对齐的对照图。
## 主表
| Method | dangerous_hit_rate | hit@2m | hit@4m | pred_min_dist | pred_better_than_gt_rate | low_motion_rate | first_step_speed_error | mean_accel | mean_jerk | offroad_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.3125 | 0.2031 | 0.3906 | 12.54 | 0.5938 | 0.0000 | 2.66 | 4.72 | 15.96 | 0.4531 |
| Train Imitation (A0) | 0.2656 | 0.1406 | 0.4688 | 10.80 | 0.5625 | 0.7812 | 5.44 | 1.29 | 4.47 | 0.2500 |
| A1 Terminal Only | 0.3281 | 0.1562 | 0.4844 | 10.08 | 0.6719 | 0.0000 | 5.87 | 3.80 | 14.05 | 0.2344 |
| A2 Softmin Only | 0.4375 | 0.2656 | 0.4688 | 6.26 | 0.6250 | 0.0000 | 80.56 | 67.68 | 230.00 | 0.7969 |
| A3 Terminal + Softmin | 0.4219 | 0.1875 | 0.5312 | 5.53 | 0.6875 | 0.0000 | 48.42 | 18.30 | 59.98 | 0.5156 |

## Qualitative Board
![loss ablation board](/Users/linyuxuan/workSpace/GoalFlow/outputs/group_meeting_2026_04_30/loss_ablation_board.png)
## 做了哪些修改
- `Baseline`：from-scratch SafeSim 参考组。
- `Train Imitation (A0)`：使用 GoalFlow transfer + Stage1 + Stage2，但 Stage2 的 loss 仍然是纯 imitation。
- `A1 Terminal Only`：加入 terminal position 和 heading loss，用来抑制 low-motion collapse，并让轨迹具有更明确的终局状态。
- `A2 Softmin Only`：加入针对 ctrl-distance 的 softmin loss，直接奖励最接近危险目标的轨迹。
- `A3 Terminal + Softmin`：同时使用终局引导和危险接近引导。
## 主要结论
- `A0` 的 low-motion collapse 最严重（`low_motion_rate = 0.7813`）。这就是轨迹缩在原点附近的定量表现。
- `A1` 基本消除了这种 collapse（`low_motion_rate = 0.0`），同时动力学指标仍然相对可控，是当前最平衡的消融版本。
- `A2` 的危险指标最强，但动力学明显恶化：`first_step_speed_error`、`mean_accel`、`mean_jerk` 和 `offroad_rate` 都非常大。
- `A3` 相比纯 imitation 在危险指标上也更强，但整体仍然过于激进，物理合理性不足。
- 如果目标是在不明显破坏物理质量的前提下提升危险性，那么与原始 `baseline` 相比，`A1` 是当前最干净、最可继续推进的方向。
## 为什么会出现原点塌缩
- 在 `A0` 中，Stage2 对危险 target 使用的是纯 imitation，但没有显式的运动约束或终局约束。
- 在 ego-local 坐标系里，当前时刻的位置就是原点。对于一个保守的平均解来说，停留在原点附近可以降低 imitation error。
- 这就是为什么 `A0` 在视觉上会表现为“缩在原点附近”，也是为什么 `low_motion_rate` 是跟踪这一失败模式的关键指标。
- `A1`、`A2` 和 `A3` 都通过显式鼓励轨迹离开原点，修复了这一类 failure mode。
## 下一步建议
- 以 `A1` 作为下一阶段的 base variant。
- 重新引入 dangerous-approach 项，但权重应明显小于当前 `A2/A3`。
- 下一步优先加入轻量 continuity regularizer，而不是继续增大 softmin 的权重。
- 后续所有实验继续沿用同一套 protocol，并把 `low_motion_rate`、`first_step_speed_error` 和 `mean_jerk` 作为 guardrail metrics。
