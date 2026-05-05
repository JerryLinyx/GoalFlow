from dataclasses import dataclass, field
from typing import List


@dataclass
class SafeSimConfig:
    """Configuration for the original SafeSim trajectory imitation setup."""

    # ======================== Data ========================
    hdf5_paths: List[str] = field(default_factory=list)
    temporal_stride: int = 5
    future_len: int = 11
    future_stride: int = 5
    action_horizon: int = 32
    history_len: int = 4
    history_stride: int = 5
    max_other_agents: int = 10
    scene_timesteps: int = 100
    target_policy: str = "raw_gt"  # raw_gt | action | nearest_action_sample

    # ======================== Map Encoder ========================
    map_input_size: int = 224
    map_channels: int = 1
    map_embed_dim: int = 128

    # ======================== Agent Encoder ========================
    agent_feat_dim: int = 6
    agent_embed_dim: int = 128
    num_roles: int = 3
    num_cases: int = 5  # kept for dataset compatibility; not used by the original model
    goal_feat_dim: int = 3

    # ======================== Scene Fusion ========================
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.1
    use_case_condition: bool = True
    use_goal_condition: bool = False

    # ======================== Flow Matching ========================
    train_scale: float = 0.1
    test_scale: float = 0.1
    infer_steps: int = 100
    alpha: float = 3.0
    anchor_size: int = 10
    start: bool = True
    cur_sampling: bool = True

    # ======================== Trajectory Normalization ========================
    x_scale: float = 60.0
    y_scale: float = 15.0
    heading_scale: float = 3.14159265
    rotation_times: int = 10

    # ======================== Training ========================
    training: bool = True
    lr: float = 1e-4
    step_size: int = 20
    gamma: float = 0.8
    trajectory_weight: float = 1.0
    terminal_xy_weight: float = 0.0
    terminal_heading_weight: float = 0.0
    ctrl_softmin_weight: float = 0.0
    ctrl_softmin_beta: float = 4.0
    freeze_loaded_fm_epochs: int = 0
    init_mode: str = "none"  # none | fm_head_conservative | fm_head_extended
    init_checkpoint: str = ""
    model_checkpoint: str = ""
    stage_name: str = "baseline"

    # ======================== Inference ========================
    use_nearest: bool = True
    cfg_scale: float = 1.0
    condition_dropout_prob: float = 0.15

    # ======================== Checkpoint ========================
    checkpoint_path: str = ''
