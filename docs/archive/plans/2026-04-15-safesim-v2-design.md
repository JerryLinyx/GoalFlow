# Safe-Sim V2 Design

> Archived document. This design note describes an older SafeSim V2 iteration.

## 1. Purpose

This document defines the next architecture iteration for Safe-Sim dangerous ego-trajectory generation.

The goal of V2 is not to redesign the whole stack. The goal is to remove the most important task-alignment bottlenecks in the current model while keeping the Flow Matching planner and the current training protocol intact.

The design is driven by three observations from the current system:

- the current encoder already produces useful scene tokens, but the decoder only consumes a single compressed `CLS` summary;
- the task is fundamentally conditional on the `ego`-`ctrl` relationship, but that relationship is not represented explicitly;
- stronger case conditioning is needed, but it should be achieved with a minimal structural change rather than by increasing model size aggressively.

## 2. Current Grounded Constraints

The following constraints are taken directly from the current implementation:

- the map branch is a CNN and already produces `196` map tokens;
- the agent branch is `MLP + GRU` and produces one token per agent;
- the fusion module already uses `nn.TransformerEncoder`;
- the fusion output already contains `all_tokens`, but `SafeSimSceneEncoder.forward()` currently discards them and returns only `scene_context`;
- the Flow Matching head already uses `ParallelAttentionLayer`, which supports cross-attention, but the current instantiation disables cross-attention and uses self-attention only.

This means V2 can be implemented as a targeted extension of the existing code path. We do not need a new backbone, a new dataset format, or a new trajectory head.

## 3. Recommendation

The recommended V2 scope is:

1. add an explicit `ego-ctrl pair token`;
2. let the FM decoder cross-attend to `all_tokens` while keeping `CLS`;
3. strengthen case conditioning with explicit `role × case` and `pair × case` bindings;
4. keep history frequency unchanged for now;
5. defer Safe-Sim candidate-prior work to a later iteration.

This is the smallest change set that directly addresses the current task mismatch:

- who the target vehicle is,
- how ego relates to that target,
- and how fine-grained scene structure reaches the decoder.

## 4. Alternatives Considered

### A. Keep Current Encoder and Only Increase History Frequency

This was already tested in a first screening A/B (`4@2Hz` vs `8@5Hz`).

Current conclusion:

- denser history did not show task-aligned improvement under the current architecture;
- the evidence is not strong enough to prove higher-frequency history is useless;
- however, it is strong enough to say it should not be the next standalone priority.

Reason:

- the current encoder compresses each agent into one token;
- there is no explicit `ego-ctrl` interaction token;
- the decoder still reads only global summary information.

### B. Inflate the Encoder With More Transformer Layers

This is not the recommended next step.

Reason:

- the fusion layer already uses a transformer;
- the current bottleneck is not the absence of attention, but the lack of task-specific structure and decoder access to rich tokens;
- the current filtered dataset size remains modest, so blind model inflation raises overfitting risk without solving the key conditioning problem.

### C. Use Safe-Sim Candidate Priors Immediately

This remains promising but is not the first V2 step.

Reason:

- there are multiple qualitatively different ways to use `action_sample_positions`;
- the simplest route is candidate-based reranking, but that is still a separate research track;
- we should first improve the base model's conditional representation so later candidate-prior experiments are easier to interpret.

## 5. Chosen V2 Architecture

### 5.1 Keep SceneFusion Transformer

The current `SceneFusionEncoder` should stay.

Reason:

- it already performs the right type of global fusion for `[CLS, map, agent]` tokens;
- it is not the main bottleneck;
- replacing it would create unnecessary experimental drift.

V2 should therefore treat the fusion transformer as a stable component and build around it.

### 5.2 Add an Explicit Ego-Ctrl Pair Token

V2 should create one additional token representing the relationship between `ego` and `ctrl`.

Reference features for the pair token:

- `delta_x`
- `delta_y`
- `delta_yaw`
- relative speed
- Euclidean distance
- TTC
- approaching flag

Reference TTC definition:

- compute line-of-sight direction from ego to ctrl;
- project relative velocity onto that line to get closing speed;
- if `closing_speed <= 0`, set `ttc = 10`;
- otherwise compute `ttc_raw = distance / max(closing_speed, eps)`;
- clip `ttc_raw` to `[0, 10]`;
- set `approaching_flag = 1` only when `closing_speed > 0`, otherwise `0`.

Reason for `approaching_flag`:

- `ttc = 10` can mean either "far but approaching slowly" or "not approaching";
- the flag disambiguates those two geometries with one cheap feature.

Recommended placement:

- append the pair token to the fusion input sequence;
- do not replace `CLS`;
- keep the sequence structure as `[CLS, pair, map tokens, agent tokens]`.

Reason:

- this makes the most important task relationship explicit;
- it introduces very few parameters;
- it provides a clean place to inject case information later.

### 5.3 Expose All Encoder Tokens to the Decoder

The current decoder path compresses all scene information into `scene_context = CLS`.

V2 should preserve `CLS`, but additionally expose `all_tokens` to the FM decoder through cross-attention.

Recommended implementation:

- keep the current self-attention path over `[scene token(s), trajectory tokens]`;
- additionally enable cross-attention from trajectory tokens to `all_tokens`;
- do not replace the current path with pooling;
- do not remove `CLS`.

Reason:

- `CLS` remains useful as a global summary;
- cross-attending to `all_tokens` removes the current information bottleneck;
- the existing `ParallelAttentionLayer` already supports cross-attention, so this can be implemented with limited code churn.

### 5.4 Strengthen Case Conditioning With Explicit Bindings

Current `case_id` conditioning is CLS-only.

V2 should keep that path, but add stronger bindings to the task-critical tokens.

Recommended rule:

- use a joint `role_case_embedding(role, case_id)` for `ego` and `ctrl` tokens;
- use a dedicated `pair_case_embedding(case_id)` for the `ego-ctrl pair token`;
- keep the existing global `case_embedding` on `CLS`;
- do not inject case into every map token in the first V2 iteration.

The intended token construction is:

- `ego_token = ego_token + role_embedding(ego) + role_case_embedding(ego, case_id)`
- `ctrl_token = ctrl_token + role_embedding(ctrl) + role_case_embedding(ctrl, case_id)`
- `pair_token = pair_mlp(pair_features) + pair_type_embedding + pair_case_embedding(case_id)`

Other-agent tokens should keep the normal role embedding in the first implementation. If later evidence shows case-specific behavior among reactive agents matters, `other × case` can be enabled as a separate ablation.

Initialization rule:

- `role_case_embedding` must be zero-initialized;
- `pair_case_embedding` must be zero-initialized.

Reason:

- zero initialization makes the first forward pass equivalent to the current shared-role baseline for those terms;
- the case-specific offsets are then learned only when training evidence supports them;
- without zero initialization, the V2 model starts with random case-specific offsets, which makes the A/B comparison against V1 less controlled.

Reason:

- the collision type changes the meaning of the ego-ctrl relationship, not just the global scene prior;
- `ego` in Case 1 and `ego` in Case 3 should not necessarily share the same conditional role representation;
- `ctrl` in a T-Bone scenario has a different interaction role than `ctrl` in a Rear-End scenario;
- the parameter cost is small: `num_roles × (num_cases + 1) × D` is negligible compared with the current model size.

This is a stronger choice than simple additive case injection. It is justified because the task is explicitly case-conditioned and the case labels encode different collision geometries.

Intentional redundancy:

- case is injected into `CLS`, `ego`, `ctrl`, and `pair`;
- this is intentional, not an oversight;
- `CLS` carries global case context while role-case and pair-case embeddings provide local interaction-specific conditioning.

If later ablations show that `CLS` case injection is redundant after role-case and pair-case binding, it can be removed in a separate `V2 - CLS_case` ablation.

## 6. What V2 Explicitly Does Not Change

V2 should not change:

- the Safe-Sim dataset format;
- the main imitation objective;
- the future trajectory output rate;
- the current `history_len=4, history_stride=5` baseline;
- the candidate selection policy;
- the core FM trajectory normalization scheme.

These are held constant so that V2 measures the effect of better structural conditioning rather than a bundle of unrelated changes.

## 7. Implementation Sequence

The recommended implementation order is:

1. modify the scene encoder to return both `scene_context` and `all_tokens`;
2. add pair-token construction in the encoder input path;
3. inject case embedding into `ego`, `ctrl`, and `pair`;
4. modify the FM decoder layers to consume `all_tokens` through cross-attention;
5. keep the existing validation protocol and rerun the filtered baseline as the new V2 base.

This order minimizes regression risk:

- pair-token construction is isolated to the encoder;
- decoder cross-attention uses existing attention infrastructure;
- metrics and sampling remain fixed.

## 8. Evaluation Rule For V2

V2 must be evaluated under the current R1/R2 protocol:

- same scene-level split;
- same `WeightedRandomSampler`;
- same selected vs random candidate metrics;
- same checkpoint selection rule using `val_primary_metric`.

History frequency is not to be revisited until V2 has been tested. If V2 shows a clear improvement, then a second history-resolution ablation should be rerun on top of the stronger interaction-aware architecture.

### 8.1 Bundle Result Versus Component Attribution

The first full V2 run answers:

- is the combined V2 architecture worth adopting over V1?

It does not answer:

- which component caused the gain or regression?

Therefore V2 must be followed by leave-one-out ablations under the same screening budget:

- `V2 - cross_attention`: keep pair token and role-case binding, but use the old CLS-only decoder path;
- `V2 - pair_token`: keep cross-attention and role-case binding, but remove the explicit pair token;
- `V2 - role_case`: keep pair token and cross-attention, but revert role-case / pair-case binding to the current CLS-only case path;
- optional `V2 - CLS_case`: keep role-case and pair-case binding, but remove global case injection from `CLS`.

The bundle result is the adoption test. The leave-one-out results are the attribution test.

## 9. Expected Effect

If V2 works as intended, the expected changes are:

- higher `bbox_collision_rate`;
- more stable `pred_better_than_gt_rate`;
- stronger case-wise consistency;
- reduced gap between selected and random candidate quality;
- better evidence that improvements come from the model rather than only from selector behavior.

The expected effect is not "perfect collision generation". The expected effect is a cleaner, more target-specific base model that is better aligned with the downstream accident-generation objective.

Case-wise consistency should be measured, not judged visually.

Recommended reporting:

- `case_mean_primary = mean(val_case_i_primary_metric for i in 1..5)`;
- `case_min_primary = min(val_case_i_primary_metric for i in 1..5)`;
- `case_std_primary = std(val_case_i_primary_metric for i in 1..5)`;
- `case_gap_primary = max(val_case_i_primary_metric) - min(val_case_i_primary_metric)`.

Expected direction:

- the global mean should improve or stay stable;
- `case_min_primary` should improve if weaker cases benefit;
- `case_std_primary` and `case_gap_primary` must be interpreted together with the mean, because larger spread can mean either stronger case specialization or worse imbalance.

## 10. Decision

The chosen next iteration is:

> Keep the current fusion transformer, add an explicit `ego-ctrl pair token`, let the decoder see `all_tokens`, and strengthen case conditioning on task-relevant tokens before revisiting history-frequency or candidate-prior changes.

The case-conditioning choice for V2 is intentionally not the weakest additive version. We will bind case to role for `ego` and `ctrl`, and bind case to the pair token, because the model should learn case-specific interpretations of the same actor roles.

This is the most defensible next step under the current evidence.

## 11. Implementation Status

Implemented files:

- `navsim/agents/goalflow/safesim_config.py`
- `navsim/agents/goalflow/safesim_encoder.py`
- `navsim/agents/goalflow/safesim_model.py`
- `navsim/agents/goalflow/run_safesim_training.py`
- `scripts/training/run_safesim_training.sh`

Implemented behavior:

- `PairEncoder` constructs the explicit ego-ctrl pair token;
- `role_case_embedding` is added to ego and ctrl tokens;
- `pair_case_embedding` is added to the pair token;
- both new case-binding embedding tables are zero-initialized;
- `SafeSimSceneEncoder` returns `scene_context`, `all_tokens`, and `token_mask`;
- the FM decoder can cross-attend from trajectory tokens to `all_tokens`;
- default training script logs V2 runs to `safesim_logs_filtered_case_v2`;
- config flags support leave-one-out ablations:
  - `--no-use_pair_token`
  - `--no-use_role_case_embedding` (disables both role-case and pair-case binding)
  - `--no-use_all_token_cross_attention`
  - `--no-use_cls_case_embedding`

Smoke-test result:

- training forward/backward runs on a small Safe-Sim batch;
- inference returns `trajectory [B, 11, 3]` and `trajectory_candidates [B, anchor_size, 11, 3]`;
- zero-initialization was verified for the new case-binding embeddings.

### 11.1 Post-V2 Inference Guidance Track

The next optimization step after the V2 bundle is not another checkpoint rule change.

The next low-cost experiment is classifier-free guidance (CFG) at inference time.

Implementation status:

- `cfg_scale` is now exposed in `SafeSimConfig`;
- the denoiser now supports conditional and unconditional passes through `guided_denoise(...)`;
- unconditional CFG uses condition dropout only, not trajectory-token dropout;
- the training entry points expose `--cfg_scale` and `CFG_SCALE`.

Interpretation rule:

- `cfg_scale = 1.0` is the current baseline behavior
- `cfg_scale > 1.0` amplifies case-conditioned denoising directions

This experiment is intentionally inference-only:

- no retraining
- no loss change
- no selector change

The purpose is diagnostic:

- if CFG improves collision-oriented metrics, the current V2 model already contains useful case-conditioned structure that was under-amplified at inference time
- if CFG fails, the next priority should shift to auxiliary collision-aware losses rather than stronger selector logic

Current screening evidence:

- a small fixed-val-subset CPU screening run was executed on the V2 best checkpoint
- evaluation budget for the screening run was reduced to `anchor_size=16`, `infer_steps=25`
- results on 8 validation samples were:
  - `w=1.0`: `primary=0.9212`, `bbox=0.8750`, `hit@2m=0.3750`, `pred_min=2.3462`
  - `w=1.5`: `primary=1.0838`, `bbox=1.0000`, `hit@2m=0.7500`, `pred_min=1.6738`
  - `w=2.0`: `primary=0.8100`, `bbox=0.7500`, `hit@2m=0.5000`, `pred_min=2.2030`
  - `w=2.5`: `primary=0.9075`, `bbox=0.8750`, `hit@2m=0.2500`, `pred_min=2.2627`
  - `w=3.0`: `primary=0.6700`, `bbox=0.6250`, `hit@2m=0.3750`, `pred_min=3.4973`

Preliminary conclusion:

- the patch behaves sensibly;
- moderate guidance (`w≈1.5`) is promising;
- this is only a screening result and must not be treated as a full-val conclusion.

Next required step:

- rerun the sweep on the full validation split under the standard V2 evaluation protocol before making adoption decisions.
