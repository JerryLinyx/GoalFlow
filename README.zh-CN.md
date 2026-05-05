# GoalFlow / SafeSim 工作区

[English](README.md) | 简体中文

这个仓库当前同时承担两类内容：

1. 原版 **GoalFlow（CVPR 2025）** 代码与论文时期资料
2. 基于 GoalFlow 扩展出来的 **SafeSim 危险轨迹生成** 研究主线

根目录 README 现在作为**当前工作区状态**的入口。原始论文风格 README 已归档到：

- [docs/legacy/goalflow-original-readme.md](docs/legacy/goalflow-original-readme.md)

## 当前项目重点

当前 SafeSim 主线聚焦于危险轨迹生成，核心要素包括：

- 显式 `goal_point` 条件
- 正式协议测评
- imitation、terminal、softmin 三类方法的结构化比较

当前使用的训练形式是：

```text
history/start state + scene context + goal_point -> trajectory
```

## 训练形态

当前仓库里实际上有三种训练形态：

1. **全量训练 / from-scratch baseline**
   - SafeSim Baseline
   - 不使用 GoalFlow transfer
   - 作为主要的非迁移参考组

2. **迁移 + 先验对齐**
   - Stage1 Prior Alignment
   - 从可迁移的 GoalFlow 权重出发
   - 先在 `original` 数据上恢复轨迹先验，再进入后续微调

3. **修正后的微调主线**
   - Goal-conditioned Pure Imitation
   - Goal-conditioned Terminal Ablation
   - Goal-conditioned Softmin Ablation
   - 这些实验共同构成当前的 fine-tuning 主线，也是最新结果的主要来源

## 实验总表

下面这张总表只展示当前项目状态里最重要的几条实验线。

| Scenario | 角色 | 监督 | Goal Point | 状态 | dangerous_hit_rate | hit@2m | pred_min_dist | 说明 |
|---|---|---|---|---|---:|---:|---:|---|
| SafeSim Baseline | 全量训练参考组 | 旧 baseline 训练线 | 无显式 goal token | 已评估 | 0.2812 | 0.1562 | 12.9143 | 历史上的定性参照 |
| Stage1 Prior Alignment | 迁移参考组 | `raw_gt` on `original` | 无显式 goal token | 已评估 | 0.2188 | 0.0781 | 13.4077 | corrected Stage2 之前的迁移 + 先验对齐阶段 |
| Goal-conditioned Pure Imitation | 微调 baseline | `action` | 显式 `goal_point` | 已完成正式测评 | 0.4219 | 0.3125 | 9.6171 | 当前 corrected fine-tuning baseline |
| Goal-conditioned Terminal Ablation | 微调 ablation | `action` | 显式 `goal_point` | 已完成正式测评 | 0.4688 | 0.3125 | 9.2413 | 最优 terminal-only 为 `xy=0.25, heading=0.05` |
| Goal-conditioned Softmin Ablation | 微调 ablation | `action` | 显式 `goal_point` | 已完成正式测评 | 0.5156 | 0.3438 | 4.6922 | 最优 softmin 为 `0.0025` |

## Checkpoint 与实验产物

要复现实验或检查结果，直接使用下面这个共享 checkpoint 包：

- [goalflow_safesim_checkpoints_20260505.zip](https://drive.google.com/file/d/1FRYlWvijY_QJUC8Bcm4n8JTSROpBVJW-/view?usp=drive_link)

zip 包内文件夹与 scenario 的对应关系：

| zip 内文件夹 | 对应的 scenario | 内容 |
|---|---|---|
| `baseline/` | SafeSim Baseline | from-scratch baseline checkpoint 与正式测评摘要 |
| `stage1/` | Stage1 Prior Alignment | transfer / prior-alignment checkpoint，用于后续微调初始化 |
| `pure_imitation_goal_action/` | Goal-conditioned Pure Imitation | corrected pure imitation checkpoint 与正式测评摘要 |
| `best_terminal_only/` | Goal-conditioned Terminal Ablation | 最优 terminal-only checkpoint（`terminal_xy=0.25`, `terminal_heading=0.05`）与正式测评摘要 |
| `best_softmin/` | Goal-conditioned Softmin Ablation | 最优 softmin checkpoint（基于最佳 terminal base，`softmin=0.0025`）与正式测评摘要 |

zip 顶层还包含一个 `MANIFEST.md`，对这些对应关系做了同样说明。

## 当前结论

当前 corrected mainline 已经完整跑通，包括：

- goal-conditioned pure imitation
- terminal sweep
- softmin sweep
- 所有当前有效实验的正式协议测评

## 结论与 Tradeoff

- **危险指标最强的方案：** Goal-conditioned softmin ablation  
  在最佳 terminal base 上取 `softmin = 0.0025` 时，当前主线取得了最强的危险指标：
  - `dangerous_hit_rate = 0.5156`
  - `hit@2m = 0.3438`
  - `pred_min_dist = 4.6922`
- **最佳 terminal-only：** `terminal_xy = 0.25`, `terminal_heading = 0.05`
  - `dangerous_hit_rate = 0.4688`
  - `pred_min_dist = 9.2413`
- **最干净的 corrected baseline：** goal-conditioned pure imitation
  - `dangerous_hit_rate = 0.4219`
  - `pred_min_dist = 9.6171`

当前最核心的 tradeoff 是：

- 加 `terminal` 后，危险指标相对 pure imitation 会提升，但轨迹也会更激进
- 再加小权重 `softmin` 后，危险指标还能继续提升，但 off-road 和 jerk 问题会进一步加重
- 因此，**如果只看危险任务指标，softmin 是当前最优；如果看更保守、更平衡的中间方案，terminal-only 更合适**

目前最直接的项目结论是：

- **按危险指标排名的最优方案：** goal-conditioned softmin (`0.0025`)
- **当前最平衡的中间点：** terminal-only (`0.25 / 0.05`)

## 建议阅读顺序

1. 当前测评协议  
   [docs/metrics/safesim-dangerous-metrics-v1.md](docs/metrics/safesim-dangerous-metrics-v1.md)

2. 当前实验总索引  
   [docs/reports/2026-05-04-safesim-experiment-index.md](docs/reports/2026-05-04-safesim-experiment-index.md)

3. 文档总索引  
   [docs/README.md](docs/README.md)

4. 原版 GoalFlow 的安装 / 训练 / 测试文档  
   [docs/install.md](docs/install.md)  
   [docs/train.md](docs/train.md)  
   [docs/test.md](docs/test.md)

## 当前主要实验线

下面这些实验线构成了当前 SafeSim 主线。

| 实验 | 作用 | 状态 | 入口 |
|---|---|---|---|
| Stage 1 prior alignment | 将 GoalFlow FM head 迁移到 SafeSim 结构化输入训练 | 已完成 | [scripts/training/run_safesim_stage1.sh](scripts/training/run_safesim_stage1.sh) |
| Goal-conditioned pure imitation | 使用 `target_policy=action` 和显式 `goal_point` 的修正 baseline | 已训练并测评 | [scripts/training/run_safesim_stage2_action_imitation.sh](scripts/training/run_safesim_stage2_action_imitation.sh) |
| Goal-conditioned terminal sweep | 在修正后的设置下扫描 `terminal_xy` / `terminal_heading` | 已训练并测评 | [scripts/training/run_safesim_stage2_terminal_sweep.sh](scripts/training/run_safesim_stage2_terminal_sweep.sh) |
| Goal-conditioned softmin sweep | 在选定 terminal base 后进行 small softmin 扫描 | 已训练并测评 | [scripts/training/run_safesim_stage2_softmin_sweep.sh](scripts/training/run_safesim_stage2_softmin_sweep.sh) |
| Goal-action mainline orchestrator | 跑完 terminal、完成评估、选 terminal base、启动 softmin sweep | 已完成 | [scripts/training/run_safesim_goal_action_mainline.sh](scripts/training/run_safesim_goal_action_mainline.sh) |

更详细的发展过程和历史实验放在实验总索引与归档中，不在这份 README 里展开。

## 各 Scenario 分节

### Original GoalFlow context

这一节是仓库的论文时期背景参考。

![GoalFlow main figure](assets/main_fig.png)

### Scenario 1: SafeSim baseline reference

这是历史上的 from-scratch SafeSim 参考组，保留在这里作为定性的对照锚点。

![SafeSim baseline reference](assets/safesim_current/baseline_proxy64_examples.png)

### Scenario 2: Stage1 prior-alignment reference

这一节展示 corrected Stage2 之前、迁移完成后的先验状态。

![Stage1 prior alignment reference](assets/safesim_current/stage1_proxy64_examples.png)

### Scenario 3: Goal-conditioned pure imitation

这是修复后第一条真正干净的 baseline，满足：

- `target_policy = action`
- 显式 `goal_point`
- 不加 `terminal`
- 不加 `softmin`

它是当前 corrected mainline 的基线。

![Goal-conditioned pure imitation](assets/safesim_current/pure_imitation_goal_action_examples.png)

### Scenario 4: Goal-conditioned terminal ablation

这一族已经完成正式测评。当前最优的 terminal-only 配置是：

- `terminal_xy = 0.25`
- `terminal_heading = 0.05`

它比 corrected pure imitation 有更高的危险指标。

![Best corrected terminal-only examples](assets/safesim_current/terminal_goal_action_best_examples.png)

### Scenario 5: Goal-conditioned softmin ablation

这一族也已经完成正式测评。当前最优的 corrected softmin 配置是：

- terminal base: `terminal_xy = 0.25`, `terminal_heading = 0.05`
- `softmin = 0.0025`

这是 corrected mainline 里危险性最强的一组。

![Best corrected softmin examples](assets/safesim_current/softmin_goal_action_best_examples.png)

## 当前结果目录

最新 corrected mainline 的正式结果统一放在：

- pure imitation：
  [outputs/current_goal_action/pure_imitation_protocol64](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/pure_imitation_protocol64)
- terminal sweep：
  [outputs/current_goal_action/terminal_eval](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/terminal_eval)
- softmin sweep：
  [outputs/current_goal_action/softmin_eval](/Users/linyuxuan/workSpace/GoalFlow/outputs/current_goal_action/softmin_eval)

这些目录是当前 goal-conditioned corrected 主线的权威结果位置。

## 主要代码入口

### 原版 GoalFlow

- 轨迹模型：[navsim/agents/goalflow/goalflow_model_traj.py](navsim/agents/goalflow/goalflow_model_traj.py)
- Goal-point 模型：[navsim/agents/goalflow/goalflow_model_navi.py](navsim/agents/goalflow/goalflow_model_navi.py)
- 原版轨迹训练脚本：[scripts/training/run_goalflow_training_traj.sh](scripts/training/run_goalflow_training_traj.sh)

### SafeSim 研究主线

- 通用训练入口：[navsim/agents/goalflow/run_safesim_training.py](navsim/agents/goalflow/run_safesim_training.py)
- Agent：[navsim/agents/goalflow/safesim_agent.py](navsim/agents/goalflow/safesim_agent.py)
- Dataset：[navsim/agents/goalflow/safesim_dataset.py](navsim/agents/goalflow/safesim_dataset.py)
- Encoder：[navsim/agents/goalflow/safesim_encoder.py](navsim/agents/goalflow/safesim_encoder.py)
- Model：[navsim/agents/goalflow/safesim_model.py](navsim/agents/goalflow/safesim_model.py)
- Config：[navsim/agents/goalflow/safesim_config.py](navsim/agents/goalflow/safesim_config.py)

## 测评

正式比较必须使用协议评估，不能只依赖训练 `loss`。

- 主评估脚本：[scripts/analysis/evaluate_safesim_dangerous.py](scripts/analysis/evaluate_safesim_dangerous.py)
- terminal sweep 评估：[scripts/analysis/run_safesim_terminal_sweep_eval.sh](scripts/analysis/run_safesim_terminal_sweep_eval.sh)
- softmin sweep 评估：[scripts/analysis/run_safesim_softmin_sweep_eval.sh](scripts/analysis/run_safesim_softmin_sweep_eval.sh)

## 环境

现在仓库里刻意保留了两套依赖清单：

- [requirements.txt](/Users/linyuxuan/workSpace/GoalFlow/requirements.txt)：原版 GoalFlow / 论文时期环境
- [requirements.safesim-current.txt](/Users/linyuxuan/workSpace/GoalFlow/requirements.safesim-current.txt)：当前 corrected SafeSim goal-conditioned mainline 的已验证环境快照

如果你要复现当前 corrected SafeSim 主线，优先使用：

```bash
conda create -n goalflow python=3.10
conda activate goalflow
pip install -r requirements.safesim-current.txt
pip install -e nuplan-devkit
pip install -e .
```

如果你要走原版论文工作流，则继续使用 `requirements.txt`。

原版环境安装方式：

```bash
conda create -n goalflow python=3.10
conda activate goalflow
pip install -r requirements.txt
pip install -e nuplan-devkit
pip install -e .
```

如果要走原版论文代码流程，继续参考：

- [docs/install.md](docs/install.md)
- [docs/train.md](docs/train.md)
- [docs/test.md](docs/test.md)

## 论文与原版资料

- Paper: [GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving](https://arxiv.org/abs/2503.05689)
- Project page: [GoalFlow Project Page](https://zebinx.github.io/HomePage-of-GoalFlow/)
- 原版归档 README：[docs/legacy/goalflow-original-readme.md](docs/legacy/goalflow-original-readme.md)
