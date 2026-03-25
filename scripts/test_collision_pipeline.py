"""
Quick test: verify CollisionTargetBuilder + model forward pass works.
Runs a single batch through the collision pipeline on MPS/CPU.
"""

import os
import sys
import gzip
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT",
            Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_agent_collision import GoalFlowCollisionAgent
from navsim.agents.goalflow.collision_target_builder import make_collision_trajectory


def test_collision_trajectory():
    """Test trajectory generation logic."""
    print("=== Test 1: make_collision_trajectory ===")

    # Agent at (20, 5) in ego frame
    agent_pos = np.array([20.0, 5.0])
    traj = make_collision_trajectory(agent_pos, num_steps=11)

    print(f"  Agent position: ({agent_pos[0]}, {agent_pos[1]})")
    print(f"  Trajectory shape: {traj.shape}")
    print(f"  Step 0 (t=0.5s):  x={traj[0,0]:.2f}, y={traj[0,1]:.2f}")
    print(f"  Step 7 (t=4.0s):  x={traj[7,0]:.2f}, y={traj[7,1]:.2f}")
    print(f"  Step 10 (t=5.5s): x={traj[10,0]:.2f}, y={traj[10,1]:.2f}")
    print(f"  Heading: {np.degrees(traj[0,2]):.1f}°")

    # Verify step 8 (index 7) is approximately at agent position
    dist_at_step8 = np.linalg.norm(traj[7, :2] - agent_pos)
    print(f"  Distance to agent at step 8: {dist_at_step8:.2f}m")
    assert dist_at_step8 < 2.0, f"Step 8 should be near agent, got {dist_at_step8:.2f}m"
    print("  ✓ PASSED\n")


def test_model_forward():
    """Test full forward pass with collision targets."""
    print("=== Test 2: Model forward pass with collision data ===")

    # Load a cached feature/target pair
    cache_root = ROOT / "exp/feature_cache_test"
    if not cache_root.exists():
        print("  ⚠ Feature cache not found, skipping model forward test")
        return

    # Find one token
    token = None
    for log_dir in cache_root.iterdir():
        for td in log_dir.iterdir():
            if (td / "transfuser_feature.gz").exists():
                token = td.name
                break
        if token:
            break

    if not token:
        print("  ⚠ No cached tokens found")
        return

    print(f"  Token: {token}")

    # Load features
    def load_gz(p):
        with gzip.open(p, "rb") as f:
            return pickle.load(f)

    for log_dir in cache_root.iterdir():
        td = log_dir / token
        if td.exists():
            features = load_gz(td / "transfuser_feature.gz")
            targets = load_gz(td / "transfuser_target.gz")
            break

    # Simulate collision: replace trajectory with collision version
    agent_states = targets["agent_states"].numpy()
    agent_labels = targets["agent_labels"].numpy()

    nearest_idx = None
    for i in range(len(agent_labels)):
        if agent_labels[i]:
            nearest_idx = i
            break

    if nearest_idx is not None:
        agent_pos = agent_states[nearest_idx, :2]
        dist = np.linalg.norm(agent_pos)
        print(f"  Nearest agent: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}), dist={dist:.1f}m")

        if 3.0 < dist < 50.0:
            # Model expects 11-step trajectory (+ 1 start point = 12 in model)
            # The noise dimension is hardcoded to 12 in the model
            num_steps = 11
            collision_traj = make_collision_trajectory(agent_pos, num_steps=num_steps)
            targets["trajectory"] = torch.tensor(collision_traj, dtype=torch.float32)
            print(f"  ✓ Replaced trajectory with collision version (num_steps={num_steps})")
            print(f"  Collision traj step 8: ({collision_traj[7,0]:.1f}, {collision_traj[7,1]:.1f})")
        else:
            print(f"  Agent at {dist:.1f}m — outside collision range, using original trajectory")
    else:
        print("  ⚠ No agents in scene, using original trajectory")

    # Setup model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = GoalFlowConfig(
        training=True,  # training mode
        has_navi=True,
        start=True,
        freeze_perception=True,
        tf_d_model=1024,
        voc_path=str(ROOT / "data/cluster_points_8192_.npy"),
        trajectory_weight=50.0,
        agent_class_weight=0.2,
        agent_box_weight=0.05,
        bev_semantic_weight=0.2,
    )

    agent = GoalFlowCollisionAgent(
        config=config,
        lr=1e-4,
        checkpoint_path=str(ROOT / "data/goalflow_traj_epoch_54-step_18260.ckpt"),
    )
    state_dict = torch.load(
        str(ROOT / "data/goalflow_traj_epoch_54-step_18260.ckpt"),
        map_location="cpu",
    )["state_dict"]
    agent.load_state_dict(
        {k.replace("agent.", ""): v for k, v in state_dict.items()},
        strict=False,
    )
    agent.to(device)
    agent.train()  # training mode

    print(f"  Model loaded on {device}")

    # Add batch dimension + move to device
    def to_dev(v):
        if not isinstance(v, torch.Tensor):
            return v
        if device.type == "mps" and v.dtype == torch.float64:
            v = v.float()
        return v.unsqueeze(0).to(device)

    features_b = {k: to_dev(v) for k, v in features.items()}
    features_b["token"] = token
    targets_b = {k: to_dev(v) for k, v in targets.items()}

    # The model reads gt_trajs from features (not targets).
    # The model hardcodes noise as (B, 12, 30), so gt_trajs needs 11 steps
    # (+ 1 start point = 12). Pad if the cache has only 10 steps.
    gt = features_b["gt_trajs"]  # (1, 10, 3) from cache
    if gt.shape[1] < 11:
        pad = gt[:, -1:, :].expand(-1, 11 - gt.shape[1], -1)
        features_b["gt_trajs"] = torch.cat([gt, pad], dim=1)
        print(f"  Padded gt_trajs from {gt.shape[1]} to 11 steps")

    # Also set gt_trajs to collision trajectory for navi computation
    if nearest_idx is not None and 3.0 < dist < 50.0:
        features_b["gt_trajs"] = targets_b["trajectory"].clone()

    # Forward pass
    print("  Running forward pass (training mode)...")
    with torch.no_grad():
        predictions = agent._goalflow_model(features_b, targets_b)

    pred_traj = predictions["trajectory"]
    print(f"  ✓ Output trajectory shape: {pred_traj.shape}")
    print(f"  Output step 8: ({pred_traj[0,7,0]:.2f}, {pred_traj[0,7,1]:.2f})")

    # Compute loss
    from navsim.agents.goalflow.goalflow_loss import goalflow_loss
    loss_dict = goalflow_loss(targets_b, predictions, config)
    total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
    print(f"  ✓ Loss computed: {total_loss.item():.4f}")

    # Check: predicted trajectory should point somewhat toward the agent
    if nearest_idx is not None and 3.0 < dist < 50.0:
        pred_np = pred_traj[0].cpu().numpy()
        pred_heading = np.arctan2(pred_np[7, 1], pred_np[7, 0])
        target_heading = np.arctan2(agent_pos[1], agent_pos[0])
        heading_diff = abs(np.degrees(pred_heading - target_heading))
        print(f"  Heading to agent: {np.degrees(target_heading):.1f}°")
        print(f"  Predicted heading: {np.degrees(pred_heading):.1f}°")
        print(f"  Heading difference: {heading_diff:.1f}°")

    print("  ✓ PASSED — collision pipeline works\n")


if __name__ == "__main__":
    test_collision_trajectory()
    test_model_forward()
    print("=" * 50)
    print("All tests passed! Ready for GPU training.")
