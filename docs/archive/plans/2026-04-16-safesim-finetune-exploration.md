# SafeSim Fine-Tune Exploration Plan

> Archived document. This fine-tune plan predates the current corrected
> goal-conditioned action-target training line.

## Goal

在当前 SafeSim V2 架构上，探索一条比“直接在 `filtered` 上从头训练”更稳的路线：

- 先学正常、顺滑、物理合理的轨迹先验；
- 再在危险数据上做定向微调；
- 最终判断是否能同时提升：
  - 物理合理性；
  - case-conditioned 危险生成质量；
  - 训练曲线稳定性。

## Why This Plan

当前已经得到两个比较稳定的结论：

1. 当前 V2 结构已经足够表达任务，不再是第一瓶颈。
2. 纯 imitation 在小规模 `filtered` 数据上会较快走向：
   - 更像 GT 均值；
   - 但不一定更平滑；
   - 也不一定继续提升 collision-oriented metric。

因此，下一阶段的重点不是继续堆结构，而是：

- 用更大的正常轨迹分布提供运动先验；
- 再用危险数据做任务对齐。

## Fixed Baseline For Comparison

后续所有微调实验默认对照：

- checkpoint:
  [best-primary-09-0.6329.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_filtered_case_v2_plan_a/checkpoints/best-primary-09-0.6329.ckpt)
- model family: SafeSim V2
- data family: `filtered`
- training objective: imitation-only Flow Matching L1
- evaluation family:
  - `val_primary_metric`
  - `val_bbox_collision_rate`
  - `val_hit_2m`
  - `val_pred_min_dist`
  - plausibility diagnostics (`mean_accel`, `mean_jerk`, `accel_gap`, `jerk_gap`)

## Candidate Fine-Tune Strategy

### Stage 1: Normal / Broad Trajectory Pretrain

目的：

- 学到更稳定、更平滑的轨迹分布；
- 建立更强的运动先验；
- 降低危险数据小样本训练时的 mode collapse 和轨迹抖动。

首选数据候选：

- 五个 case 的 `original`

原因：

- 数据量更大；
- 覆盖更广；
- 虽然不全是碰撞样本，但更适合做基础轨迹预训练。

建议设置：

- 架构：保持当前 SafeSim V2 不变
- loss：仍然是 imitation-only Flow Matching L1
- selection：训练时不改
- 目标：只学轨迹先验，不追求 collision metric 最优

### Stage 2: Dangerous Fine-Tune on `filtered`

目的：

- 在保留正常轨迹先验的基础上；
- 用 `filtered` 数据把分布往危险模式上拉；
- 尤其观察：
  - bbox collision
  - hit@2m
  - pred_min_dist
  - 物理平滑性是否优于从头训练 baseline

建议设置：

- 从 Stage 1 checkpoint 初始化
- 学习率低于 pretrain
- epoch 数明显少于 from-scratch 训练
- 默认不改 loss 形式，先只看 pretrain + finetune 是否带来收益

## Experimental Order

### FT-1

`original pretrain -> filtered finetune`

这是第一优先实验。

目的：

- 验证更大数据量 + 更宽轨迹分布是否能显著改善物理平滑性。

### FT-2

`original pretrain -> filtered finetune + CFG sweep`

如果 FT-1 成立，再测 inference guidance 是否还能抬高危险指标。

### FT-3

`original pretrain -> filtered finetune + small auxiliary collision loss`

只有在 FT-1 明显优于当前 baseline，但仍然不够危险时，才进入这一步。

## Acceptance Criteria

一个 fine-tune 方案被认为优于当前 baseline，需要满足：

1. `val_primary_metric` 不低于 baseline；
2. `bbox_collision_rate` 或 `hit@2m` 至少一项改善；
3. `pred_mean_accel / pred_mean_jerk` 明显下降；
4. `accel_gap / jerk_gap` 相对 baseline 缩小；
5. 可视化轨迹在主观上不再出现明显折返和锯齿。

## Non-Goals

当前阶段先不做：

- RL fine-tuning
- DPO / preference optimization
- 直接移除 imitation loss
- 继续扩大 V2 结构复杂度

## Immediate Next Step

先不直接开跑大训练。

下一步先做：

1. 整理 pretrain / finetune 的固定协议；
2. 确认是否直接使用 `original` 作为 pretrain 数据；
3. 锁定：
   - pretrain epoch
   - finetune epoch
   - 学习率
   - checkpoint 选择规则

然后再启动第一轮 FT-1。
