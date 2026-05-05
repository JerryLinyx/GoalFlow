# GoalFlow FM-Transfer Fine-Tune Workflow

> Archived document. This workflow contains assumptions that were later
> invalidated by supervision bug fixes and should be treated as historical.

## Summary

This workflow keeps the SafeSim structured-input stack and adds a sanity-gated
GoalFlow FM-head transfer path.

The current gate decisions are already fixed from the completed sanity runs:

- target supervision on `filtered`: `nearest_action_sample`
- transfer mode: `fm_head_conservative`
- FM latent width: `tf_d_model=1024`

The implementation has four stages:

1. `filtered` target audit
2. 1-epoch transfer sanity check
3. Stage 1 prior alignment on `original`
4. Stage 2 dangerous fine-tune on `filtered` with mixed replay

There is also a top-level orchestrator:

```bash
sh scripts/training/run_safesim_finetune_workflow.sh
```

It runs:
- audit
- sanity
- Stage 1
- Stage 2
- offline dangerous evaluation

and automatically:
- picks `recommended_policy` from `audit_summary.json`
- resolves the Stage 1 best checkpoint
- forwards that checkpoint into Stage 2

## Commands

### 1. Audit filtered supervision target

```bash
python scripts/analysis/audit_safesim_filtered_targets.py \
  --output_dir outputs/safesim_target_audit
```

Outputs:
- `outputs/safesim_target_audit/audit_summary.json`
- `outputs/safesim_target_audit/audit_summary.md`

Observed result:
- `recommended_policy == nearest_action_sample`
- current audit numbers:
  - `raw_gt`: mean min dist `8.8826`, dangerous hit rate `0.3000`
  - `action`: mean min dist `7.0413`, dangerous hit rate `0.3500`
  - `nearest_action_sample`: mean min dist `6.4094`, dangerous hit rate `0.3500`

Decision:
- Stage 2 should use `TARGET_POLICY=nearest_action_sample`

### 2. 1-epoch transfer sanity runs

```bash
sh scripts/training/run_safesim_transfer_sanity.sh
```

This runs:
- random init
- conservative FM-head transfer
- extended FM-head transfer

All sanity transfer runs use `tf_d_model=1024`, because the released GoalFlow
trajectory checkpoint uses 1024-dim FM latents.

Observed result from the completed sanity gate:
- `random init (none_clean)`: `val/loss = 0.139241`
- `fm_head_conservative`: `val/loss = 0.115738`
- `fm_head_extended`: `val/loss = 0.133899`

Decision:
- `fm_head_conservative` beats random init by about `16.65%`
- `fm_head_extended` is worse than conservative
- Stage 1 and Stage 2 should use `fm_head_conservative`

### 3. Stage 1 prior alignment

```bash
sh scripts/training/run_safesim_stage1.sh
```

Defaults:
- `original` data
- `fm_head_conservative`
- `20` epochs
- first `3` epochs freeze loaded FM modules

### 4. Stage 2 dangerous fine-tune

```bash
MODEL_CHECKPOINT=/abs/path/to/stage1_best.ckpt \
sh scripts/training/run_safesim_stage2.sh
```

Defaults:
- `filtered` primary data
- `original` replay data
- mixed replay ratio `30% original / 70% filtered`
- `nearest_action_sample` supervision target
- every epoch checkpoint saved

### 5. Offline dangerous evaluation

```bash
python scripts/analysis/evaluate_safesim_dangerous.py \
  --checkpoint_dir /abs/path/to/stage2/checkpoints \
  --output_dir outputs/stage2_eval \
  --tf_d_model 1024
```

Outputs:
- `dangerous_eval.csv`
- `summary.md`
- one best-scene panel per `cfg_scale`

## Notes

- `dangerous_hit_rate` is the renamed "bbox collision rate" metric and is
  treated as "higher is better".
- The first implementation keeps the current split policy unchanged for
  comparability with existing baselines.
- `transfer_report.json` and `split_summary.json` are written into each run
  directory automatically.
- The current go-forward defaults are:
  - `TF_D_MODEL=1024`
  - `INIT_MODE=fm_head_conservative`
  - `TARGET_POLICY=nearest_action_sample` for Stage 2
