# SafeSim Plan A Stable Baseline Implementation Plan

> Archived document. This implementation plan belongs to an older baseline
> protocol and is no longer the active source of truth.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the SafeSim V2 filtered baseline under a stabler training protocol so the resulting checkpoint and curves are trustworthy enough to serve as the foundation for later pretrain/fine-tune experiments.

**Architecture:** Keep the current V2 model bundle unchanged (`pair token + role×case/pair×case + all-token cross-attention`) and keep imitation-only Flow Matching training unchanged. Only modify the training/validation protocol: deterministic validation noise, sparse heavy task metrics, optional early stopping, and physical plausibility diagnostics.

**Tech Stack:** PyTorch Lightning, PyTorch 2.x, SafeSim custom dataset/model/agent pipeline.

---

### Task 1: Freeze the Plan A protocol in code-facing docs

**Files:**
- Create: `docs/plans/2026-04-16-safesim-plan-a-stable-baseline.md`
- Modify: `PRD.md`
- Modify: `docs/baselines/2026-04-16-safesim-v2-filtered-baseline.md`

**Steps:**
1. Record the motivation for Plan A:
   - current task metrics are too stochastic to compare epochs cleanly
   - current baseline is useful but too noisy to act as the only formal reference
   - physical plausibility must stay ahead of aggressive collision alignment
2. Freeze the protocol assumptions:
   - V2 architecture unchanged
   - imitation loss unchanged
   - scene-level split unchanged
   - deterministic validation inference
   - sparse heavy metrics
   - no early stopping for the formal rerun
3. State the new formal baseline criteria:
   - best checkpoint chosen from sparse heavy-metric epochs
   - physical plausibility diagnostics reported alongside collision metrics

### Task 2: Add protocol knobs to SafeSimConfig and training CLI

**Files:**
- Modify: `navsim/agents/goalflow/safesim_config.py`
- Modify: `navsim/agents/goalflow/run_safesim_training.py`
- Modify: `scripts/training/run_safesim_training.sh`

**Steps:**
1. Add deterministic evaluation settings:
   - `deterministic_eval_noise: bool`
   - `heavy_metrics_every_n_epochs: int`
2. Add training-loop controls:
   - `enable_early_stopping: bool`
3. Add matching CLI and shell flags/defaults for the formal baseline run.
4. Set the shell defaults to the Plan A baseline protocol:
   - new log dir
   - heavy metrics every 2 epochs
   - early stopping disabled

### Task 3: Make validation inference deterministic

**Files:**
- Modify: `navsim/agents/goalflow/safesim_agent.py`
- Modify: `navsim/agents/goalflow/safesim_model.py`

**Steps:**
1. Build a deterministic per-sample seed from:
   - `file_idx`
   - `scene_key`
   - `timestep`
2. In validation/test only, generate:
   - fixed `inference_noise`
   - fixed `random_candidate_index`
3. Pass these tensors into model inference so each epoch evaluates the same Monte Carlo candidates.
4. Keep training noise untouched.

### Task 4: Make heavy task metrics sparse and add plausibility diagnostics

**Files:**
- Modify: `navsim/agents/goalflow/safesim_agent.py`
- Modify: `navsim/agents/goalflow/safesim_metrics.py`

**Steps:**
1. Run `val_loss` every validation epoch.
2. Run inference-heavy collision metrics only every `heavy_metrics_every_n_epochs`.
3. Add trajectory plausibility diagnostics for prediction and GT:
   - mean acceleration magnitude
   - mean jerk magnitude
   - max jerk magnitude
4. Log these as diagnostics only, not as optimization targets.

### Task 5: Rerun the formal baseline and capture the result

**Files:**
- Modify: `docs/baselines/2026-04-16-safesim-v2-filtered-baseline.md`
- Create/Update: training log dir produced by `scripts/training/run_safesim_training.sh`

**Steps:**
1. Launch the formal baseline run with the Plan A protocol.
2. Let it complete the planned epochs unless there is a hard failure.
3. Record:
   - best checkpoint
   - heavy-metric epochs
   - final loss curve
   - plausibility diagnostics
4. Update the baseline document to distinguish:
   - old noisy baseline
   - new Plan A stable baseline

### Task 6: Verification

**Files:**
- Test: `py_compile` over modified Python files
- Test: a short Lightning smoke run

**Steps:**
1. Run `python -m py_compile` on modified SafeSim files.
2. Run a fast smoke training pass to confirm:
   - deterministic validation noise path works
   - sparse heavy metrics path works
   - no early stopping path works
   - plausibility metrics log without crashing
3. Start the full formal baseline run only after the smoke pass succeeds.
