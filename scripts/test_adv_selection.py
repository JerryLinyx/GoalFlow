"""
Unit test for adversarial trajectory selection logic.
No dataset or checkpoint needed — pure tensor validation.

Run: python scripts/test_adv_selection.py
"""
import torch
import sys

def minmax(t):
    batch_min = torch.min(t, dim=-1).values.unsqueeze(-1)
    batch_max = torch.max(t, dim=-1).values.unsqueeze(-1)
    return (t - batch_min) / (batch_max - batch_min + 1e-8)


def normal_selection(pred_trajs, navi, ep_score_weight=0.0):
    """Original: select trajectory closest to navigation goal."""
    distances = torch.norm(pred_trajs[:, :, 8, :2] - navi, dim=-1)
    if ep_score_weight > 0.0:
        distances_norm = minmax(distances)
        progress = torch.norm(pred_trajs[:, :, 8, :2], dim=-1)
        progress_norm = minmax(progress)
        scores = (1. - ep_score_weight) * distances_norm - ep_score_weight * progress_norm
    else:
        scores = distances
    return torch.argmin(scores, dim=1)


def adv_selection(pred_trajs, agent_states, adv_agent_idx=0, adv_traj_step=8):
    """New: select trajectory closest to adversarial agent."""
    adv_pos = agent_states[:, adv_agent_idx, :2].unsqueeze(1)   # (B, 1, 2)
    distances = torch.norm(pred_trajs[:, :, adv_traj_step, :2] - adv_pos, dim=-1)
    return torch.argmin(distances, dim=1)


def test_normal_selects_nearest_to_goal():
    """Normal mode should pick trajectory nearest to navi goal."""
    B, A, T = 2, 8, 12  # batch, anchor_size, timesteps
    pred_trajs = torch.randn(B, A, T, 3)

    # Force trajectory index 3 to be closest to navi for all batches
    navi = torch.zeros(B, 1, 2)
    pred_trajs[:, 3, 8, :2] = torch.tensor([0.1, 0.1])   # very close to (0,0)
    pred_trajs[:, :3, 8, :2] += 10.0                      # far away
    pred_trajs[:, 4:, 8, :2] += 10.0

    selected = normal_selection(pred_trajs, navi)
    assert (selected == 3).all(), f"Expected index 3, got {selected}"
    print("✅ test_normal_selects_nearest_to_goal passed")


def test_adv_selects_nearest_to_agent():
    """Adv mode should pick trajectory nearest to target agent position."""
    B, A, T = 2, 8, 12
    pred_trajs = torch.randn(B, A, T, 3) * 20  # spread out far

    # Place adv agent at (5, 5)
    agent_states = torch.zeros(B, 30, 5)
    agent_states[:, 0, :2] = torch.tensor([5.0, 5.0])  # nearest agent at (5,5)

    # Force trajectory index 5 to be closest to (5,5)
    pred_trajs[:, 5, 8, :2] = torch.tensor([5.1, 5.1])  # very close
    pred_trajs[:, :5, 8, :2] += 50.0
    pred_trajs[:, 6:, 8, :2] += 50.0

    selected = adv_selection(pred_trajs, agent_states, adv_agent_idx=0, adv_traj_step=8)
    assert (selected == 5).all(), f"Expected index 5, got {selected}"
    print("✅ test_adv_selects_nearest_to_agent passed")


def test_adv_different_from_normal():
    """Adv mode and normal mode should typically select different trajectories."""
    B, A, T = 1, 16, 12
    pred_trajs = torch.randn(B, A, T, 3)

    navi = torch.tensor([[[10.0, 10.0]]])         # goal far top-right
    agent_states = torch.zeros(B, 30, 5)
    agent_states[:, 0, :2] = torch.tensor([-10.0, -10.0])  # agent far bottom-left

    # Best for navi: close to (10,10)
    pred_trajs[:, 2, 8, :2] = torch.tensor([10.1, 10.1])
    # Best for adv: close to (-10,-10)
    pred_trajs[:, 7, 8, :2] = torch.tensor([-10.1, -10.1])

    normal_idx = normal_selection(pred_trajs, navi)
    adv_idx = adv_selection(pred_trajs, agent_states)

    assert normal_idx.item() == 2, f"Normal expected 2, got {normal_idx.item()}"
    assert adv_idx.item() == 7,    f"Adv expected 7, got {adv_idx.item()}"
    assert normal_idx.item() != adv_idx.item(), "Normal and adv should select different trajectories"
    print(f"✅ test_adv_different_from_normal passed  (normal={normal_idx.item()}, adv={adv_idx.item()})")


def test_adv_agent_idx_selection():
    """adv_agent_idx should control which agent is targeted."""
    B, A, T = 1, 8, 12
    pred_trajs = torch.randn(B, A, T, 3) * 20

    agent_states = torch.zeros(B, 30, 5)
    agent_states[:, 0, :2] = torch.tensor([5.0, 0.0])   # agent 0 at (5, 0)
    agent_states[:, 1, :2] = torch.tensor([-5.0, 0.0])  # agent 1 at (-5, 0)

    # traj 1 close to agent 0, traj 6 close to agent 1
    pred_trajs[:, 1, 8, :2] = torch.tensor([5.1, 0.0])
    pred_trajs[:, 6, 8, :2] = torch.tensor([-5.1, 0.0])

    idx_agent0 = adv_selection(pred_trajs, agent_states, adv_agent_idx=0)
    idx_agent1 = adv_selection(pred_trajs, agent_states, adv_agent_idx=1)

    assert idx_agent0.item() == 1, f"Expected 1 for agent_idx=0, got {idx_agent0.item()}"
    assert idx_agent1.item() == 6, f"Expected 6 for agent_idx=1, got {idx_agent1.item()}"
    print(f"✅ test_adv_agent_idx_selection passed  (agent0→traj{idx_agent0.item()}, agent1→traj{idx_agent1.item()})")


def test_batch_independence():
    """Each batch item should independently select based on its own agent position."""
    B, A, T = 3, 8, 12
    pred_trajs = torch.randn(B, A, T, 3) * 20

    agent_states = torch.zeros(B, 30, 5)
    # Each batch has agent at different positions
    agent_states[0, 0, :2] = torch.tensor([10.0, 0.0])
    agent_states[1, 0, :2] = torch.tensor([0.0, 10.0])
    agent_states[2, 0, :2] = torch.tensor([-10.0, -10.0])

    # Closest trajectory differs per batch
    pred_trajs[0, 2, 8, :2] = torch.tensor([10.1, 0.0])
    pred_trajs[1, 5, 8, :2] = torch.tensor([0.0, 10.1])
    pred_trajs[2, 7, 8, :2] = torch.tensor([-10.1, -10.1])

    selected = adv_selection(pred_trajs, agent_states)
    expected = torch.tensor([2, 5, 7])
    assert (selected == expected).all(), f"Expected {expected}, got {selected}"
    print(f"✅ test_batch_independence passed  (selected={selected.tolist()})")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on: {device}\n")

    test_normal_selects_nearest_to_goal()
    test_adv_selects_nearest_to_agent()
    test_adv_different_from_normal()
    test_adv_agent_idx_selection()
    test_batch_independence()

    print(f"\n🎉 All tests passed on {device}")
    print("\nNext step: bash scripts/validate_mps.sh  (requires feature cache)")
