# SafeSim / GoalFlow 实验总览（2026-05-04）

## 这份文档的作用

这是一份**当前有效**的实验入口和结果总览，用来回答三件事：

1. 我们做过哪些实验  
2. 每个实验的入口脚本、日志目录、结果目录在哪里  
3. 哪些结论仍然有效，哪些已经因为 bug 修复而失效或需要重做

这份文档应该作为当前 SafeSim/GoalFlow 微调线的**总入口**。

---

## 当前建议阅读顺序

1. 仓库总览：
   [README.md](/Users/linyuxuan/workSpace/GoalFlow/README.md)
2. 评估协议：
   [safesim-dangerous-metrics-v1.md](/Users/linyuxuan/workSpace/GoalFlow/docs/metrics/safesim-dangerous-metrics-v1.md)
3. 旧的 loss 消融报告（注意：包含已过时实验）：
   [2026-04-30-safesim-loss-ablation-report.md](/Users/linyuxuan/workSpace/GoalFlow/docs/archive/reports/2026-04-30-safesim-loss-ablation-report.md)
4. 当前这份实验总览：
   [2026-05-04-safesim-experiment-index.md](/Users/linyuxuan/workSpace/GoalFlow/docs/reports/2026-05-04-safesim-experiment-index.md)

---

## 一条贯穿始终的项目路线

为了避免把历史实验、修 bug 过程和当前有效结果混在一起，可以按下面这条线理解整个项目：

1. **原版 GoalFlow**  
   提供 flow-matching 轨迹生成思路，以及显式 `goal / navigation point` 条件化的建模背景。

2. **Stage1 transfer**  
   把 GoalFlow 的 FM head 和轨迹先验迁移到 SafeSim 结构化输入线上，得到一个可用的迁移起点。

3. **旧 Stage2 线（现已归档）**  
   这条线曾经尝试 pure imitation / terminal / softmin，但后来确认 `nearest_action_sample` 监督路径有 bug，因此这些实验只保留调试价值。

4. **corrected mainline**  
   修复 supervision / evaluation 问题，恢复显式 `goal_point`，重新建立：
   - goal-conditioned pure imitation
   - terminal sweep
   - softmin sweep

5. **当前最新结论**  
   corrected mainline 已经完成训练与正式测评；危险指标已经显著提升，但物理合理性仍然是主要瓶颈。

---

## 关键结论

### 1. 旧的 Stage2 主线不能再直接当正式结论

我们已经确认：

- `target_policy=nearest_action_sample` 在当前 `filtered` 数据里是坏的
- `action_sample_positions / action_sample_yaws` 为全零

因此，以下实验的**绝对结果**不能再作为正式结论：

- `A0` pure imitation（旧）
- `A1` terminal only（旧）
- `A2` softmin only（旧）
- `A3` terminal + softmin（旧）
- `temporal_stride=2`（旧）

这些实验仍然保留**诊断价值**，但不能作为最终方法对比。

### 2. 当前正确主线

当前应该使用：

- `target_policy = action`
- 显式 `goal_point`
- Goal-conditioned imitation / finetune
- 统一的 corrected mainline orchestration

也就是说，当前新的实验主线是：

```text
history/start state + scene context + goal_point -> trajectory
```

而不是旧的：

```text
scene context -> imitate full trajectory
```

### 3. 评估口径已经固定

正式比较必须使用：

- [evaluate_safesim_dangerous.py](/Users/linyuxuan/workSpace/GoalFlow/scripts/analysis/evaluate_safesim_dangerous.py)
- [safesim-dangerous-metrics-v1.md](/Users/linyuxuan/workSpace/GoalFlow/docs/metrics/safesim-dangerous-metrics-v1.md)

不要再用只看 `val_loss` 的方式决定哪个实验更好。

---

## 当前训练与评估入口

### 通用训练入口

- [run_safesim_training.py](/Users/linyuxuan/workSpace/GoalFlow/navsim/agents/goalflow/run_safesim_training.py)
- [run_safesim_training.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_training.sh)

### 通用评估入口

- [evaluate_safesim_dangerous.py](/Users/linyuxuan/workSpace/GoalFlow/scripts/analysis/evaluate_safesim_dangerous.py)

### 关键实验脚本

- Stage 1：
  [run_safesim_stage1.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_stage1.sh)
- 旧 Stage 2 主线：
  [run_safesim_stage2.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_stage2.sh)
- 修复后的 goal-conditioned pure imitation：
  [run_safesim_stage2_action_imitation.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_stage2_action_imitation.sh)
- terminal sweep：
  [run_safesim_stage2_terminal_sweep.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_stage2_terminal_sweep.sh)
- softmin sweep：
  [run_safesim_stage2_softmin_sweep.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_stage2_softmin_sweep.sh)
- corrected mainline orchestration：
  [run_safesim_goal_action_mainline.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_goal_action_mainline.sh)

### sweep 评估脚本

- terminal sweep eval：
  [run_safesim_terminal_sweep_eval.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/analysis/run_safesim_terminal_sweep_eval.sh)
- softmin sweep eval：
  [run_safesim_softmin_sweep_eval.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/analysis/run_safesim_softmin_sweep_eval.sh)

---

## 实验注册表

### A. 可作为历史参考，但不能当最终正式结论

| 名称 | 说明 | target_policy | 结果位置 | 结论状态 |
|---|---|---|---|---|
| Baseline | from-scratch SafeSim 基线 | 旧线 | [safesim_logs_cfg_base](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_cfg_base) | 可参考 |
| Stage1 | GoalFlow FM head transfer + original prior alignment | `raw_gt` | [safesim_logs_stage1](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage1) | 仍有效 |
| A0 / A1 / A2 / A3（旧） | 旧的 pure imitation / terminal / softmin / mixed 消融 | `nearest_action_sample` | [2026-04-30-safesim-loss-ablation-report.md](/Users/linyuxuan/workSpace/GoalFlow/docs/archive/reports/2026-04-30-safesim-loss-ablation-report.md) | 过时 |
| stride=2（旧） | 只改 temporal stride 的历史尝试 | `nearest_action_sample` | 本地旧日志已清理；结论仅保留在历史讨论中 | 过时 |

### B. 修复后的 action-target 线

| 名称 | 说明 | target_policy | 结果位置 | 状态 |
|---|---|---|---|---|
| action-target sweep（中间线） | 无显式 goal 的 action-target terminal / softmin 尝试 | `action` | [outputs/ablation_compare](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare) | 已完成，但已被 corrected mainline 取代 |
| goal-conditioned imitation | 显式 `goal_point` + pure imitation | `action` | [safesim_logs_stage2_action_goal_imitation](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_action_goal_imitation) | 已训练并测评 |
| goal-conditioned terminal sweep | 显式 `goal_point` + terminal sweep | `action` | [safesim_logs_stage2_terminal_sweep_goal_action](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_terminal_sweep_goal_action) | 已训练并测评 |
| goal-conditioned softmin sweep | 在 best terminal base 上做 small softmin sweep | `action` | [safesim_logs_stage2_softmin_sweep_goal_action](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_softmin_sweep_goal_action) | 已训练并测评 |
| goal-action mainline | 统一跑测评、选 best terminal、启动 softmin sweep | `action` | [run_safesim_goal_action_mainline.sh](/Users/linyuxuan/workSpace/GoalFlow/scripts/training/run_safesim_goal_action_mainline.sh) | **已完成** |

---

## 评估结果目录

统一评估结果主要在：

- [outputs/ablation_compare](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare)

其中包括：

- 历史 ablation：
  - [stage2_imitation_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/stage2_imitation_protocol64)
  - [stage2_terminal_only_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/stage2_terminal_only_protocol64)
  - [stage2_ctrl_softmin_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/stage2_ctrl_softmin_protocol64)
  - [stage2_terminal_softmin_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/stage2_terminal_softmin_protocol64)
- softmin sweep：
  - [softmin_0p0_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/softmin_0p0_protocol64)
  - [softmin_0p0025_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/softmin_0p0025_protocol64)
  - [softmin_0p005_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/softmin_0p005_protocol64)
  - [softmin_0p01_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/softmin_0p01_protocol64)
  - [softmin_0p02_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/softmin_0p02_protocol64)
- terminal sweep：
  - [termxy_0p25_termyaw_0p05_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/termxy_0p25_termyaw_0p05_protocol64)
  - [termxy_0p25_termyaw_0p10_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/termxy_0p25_termyaw_0p10_protocol64)
  - [termxy_0p50_termyaw_0p05_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/termxy_0p50_termyaw_0p05_protocol64)
  - [termxy_0p50_termyaw_0p10_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/ablation_compare/termxy_0p50_termyaw_0p10_protocol64)

---

## 当前推荐的结论口径

### 可以继续引用的

- Stage1 transfer 本身有效
- `nearest_action_sample` 在当前数据里无效
- 评估必须用协议脚本，不看单独 `val_loss`
- `terminal` 和 `softmin` 方向本身有研究价值

### 不能直接继续引用的

- “旧 A0/A1/A2/A3 谁最好”的绝对结论
- “pure imitation 一定失败”的最终结论
- “temporal_stride=2 一定没用”的最终结论

这些都已经在 corrected mainline 下重新验证完毕，因此现在讨论主结果时应优先引用 corrected mainline，而不是旧线。

---

## 当前 corrected mainline 结果

Goal-action mainline orchestration 已经完成。它完成了：

1. goal-conditioned pure imitation 的正式协议评估
2. corrected terminal sweep 的正式协议评估
3. best terminal base 选择
4. small softmin sweep 训练
5. softmin sweep 的正式协议评估

其中 corrected pure imitation baseline 仍然是关键基线：

- 结果目录：
  [pure_imitation_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/pure_imitation_protocol64)
- 配置：
  - `target_policy=action`
  - `use_goal_condition=1`
  - `terminal=0`
  - `softmin=0`

它回答的是：

> 在不缺 `goal_point` 条件的情况下，pure imitation 本身能不能正常工作。

---

## 当前最佳 corrected 结果

### Best terminal-only

- run:
  [termxy_0p25_termyaw_0p05_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/terminal_eval/termxy_0p25_termyaw_0p05_protocol64)
- key metrics:
  - `dangerous_hit_rate = 0.4688`
  - `hit@2m = 0.3125`
  - `pred_min_dist = 9.2413`
  - `mean_jerk = 21.1557`
  - `offroad_rate = 0.4844`

### Best softmin

- run:
  [softmin_0p0025_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/softmin_eval/softmin_0p0025_protocol64)
- key metrics:
  - `dangerous_hit_rate = 0.5156`
  - `hit@2m = 0.3438`
  - `hit@4m = 0.6719`
  - `pred_min_dist = 4.6922`
  - `mean_jerk = 29.1909`
  - `offroad_rate = 0.5625`

### Main takeaway

- corrected pure imitation 已经成立，不再出现旧主线那种监督错误导致的结论污染
- terminal-only 能进一步提高 dangerous-task 指标
- small softmin 能把 dangerous-task 指标继续推高
- 当前下一步的主要工作重点是**物理合理性**，而不是继续补主线实验

## 后续建议

1. 不再继续补主线实验缺口，因为 corrected mainline 已完成
2. 下一轮工作聚焦在：
   - 更强的物理合理性约束
   - 或重新定义 goal / control interaction 以降低 off-road 和 jerk
3. 不要再基于旧的无 goal / 坏 target 实验扩张

---

## 建议外部分享、不进 Git 的目录

这些目录体积较大，已经被 `.gitignore` 排除，建议走网盘或共享文件服务器：

- [safesim_logs_stage1](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage1)
- [safesim_logs_cfg_base](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_cfg_base)
- [safesim_logs_stage2_action_goal_imitation](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_action_goal_imitation)
- [safesim_logs_stage2_terminal_sweep_goal_action](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_terminal_sweep_goal_action)

建议提交到 Git 的则是：

- 训练/评估代码
- 脚本
- 文档
- 测试
- 小体积 README 图片

### 最小分享清单

如果不想共享整目录，建议至少共享下面这些文件：

- Stage1
  - [best-val-16-0.0389.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage1/checkpoints/best-val-16-0.0389.ckpt)
  - [last.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage1/checkpoints/last.ckpt)
  - [split_summary.json](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage1/split_summary.json)
- Goal-conditioned pure imitation
  - [best-val-13-0.0193.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_action_goal_imitation/checkpoints/best-val-13-0.0193.ckpt)
  - [last.ckpt](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_action_goal_imitation/checkpoints/last.ckpt)
  - [metrics.csv](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_action_goal_imitation/csv_logs/version_0/metrics.csv)
  - [split_summary.json](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_action_goal_imitation/split_summary.json)
- Goal-conditioned terminal sweep
  - [termxy_0p25_termyaw_0p05 best](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_terminal_sweep_goal_action/termxy_0p25_termyaw_0p05/checkpoints/best-val-11-0.2204.ckpt)
  - [termxy_0p25_termyaw_0p10 best](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_terminal_sweep_goal_action/termxy_0p25_termyaw_0p10/checkpoints/best-val-15-0.2216.ckpt)
  - [termxy_0p50_termyaw_0p05 best](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_terminal_sweep_goal_action/termxy_0p50_termyaw_0p05/checkpoints/best-val-18-0.3756.ckpt)
  - [termxy_0p50_termyaw_0p10 best](/Users/linyuxuan/workSpace/GoalFlow/safesim_logs_stage2_terminal_sweep_goal_action/termxy_0p50_termyaw_0p10/checkpoints/best-val-09-0.3676.ckpt)
  - 每组对应的 `last.ckpt`
  - 每组对应的 `csv_logs/version_*/metrics.csv`
  - 每组对应的 `split_summary.json`

### 已整理好的共享包

如果当前是在本地工作区内做交接，已经整理好的最小共享包目录是：

- [goalflow_safesim_checkpoints_20260505](/Users/linyuxuan/workSpace/GoalFlow/release_artifacts/goalflow_safesim_checkpoints_20260505)
- [goalflow_safesim_checkpoints_20260505.zip](/Users/linyuxuan/workSpace/GoalFlow/release_artifacts/goalflow_safesim_checkpoints_20260505.zip)

外部共享下载链接：

- [goalflow_safesim_checkpoints_20260505.zip](https://drive.google.com/file/d/1FRYlWvijY_QJUC8Bcm4n8JTSROpBVJW-/view?usp=drive_link)

其中包含：
- baseline
- Stage1
- corrected pure imitation
- best terminal-only
- best softmin

对应关系如下：

| zip 内文件夹 | 对应实验线 | 说明 |
|---|---|---|
| `baseline/` | SafeSim Baseline | from-scratch baseline checkpoint 与测评摘要 |
| `stage1/` | Stage1 Prior Alignment | transfer / prior-alignment checkpoint |
| `pure_imitation_goal_action/` | corrected pure imitation | goal-conditioned pure imitation 的 checkpoint 与测评摘要 |
| `best_terminal_only/` | corrected terminal-only | 最优 terminal-only 配置的 checkpoint 与测评摘要 |
| `best_softmin/` | corrected softmin | 当前最优 softmin 配置的 checkpoint 与测评摘要 |

并附带每组的正式测评摘要与清单说明。
