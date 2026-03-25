"""
Visualize collision trajectories as BEV animation (GIF/MP4).

Shows:
  - Green trajectory: original safe driving GT
  - Red trajectory: fabricated collision trajectory (toward target agent)
  - Blue box: target agent
  - Gray boxes: other agents
  - Ego vehicle: orange box at origin

Usage:
    # From existing feature cache (no model needed):
    python scripts/visualize_collision_bev.py --cache_path exp/feature_cache_test --num_scenes 5

    # With trained collision model predictions:
    python scripts/visualize_collision_bev.py --cache_path exp/feature_cache_test \
        --checkpoint data/collision_model.ckpt --num_scenes 5

Output:
    exp/viz_collision/<token>.gif — animated BEV showing collision trajectory
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import torch
import gzip
import pickle

ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT",
            Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from navsim.agents.goalflow.collision_target_builder import make_collision_trajectory


def load_gz(p: Path) -> Dict:
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def draw_vehicle(ax, x, y, heading, length, width, color="gray", alpha=0.7, label=None):
    """Draw a vehicle as a rotated rectangle."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    corners = np.array([
        [-length/2, -width/2],
        [ length/2, -width/2],
        [ length/2,  width/2],
        [-length/2,  width/2],
    ])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rot.T + np.array([x, y])
    polygon = plt.Polygon(corners, closed=True, facecolor=color, edgecolor="black",
                          linewidth=0.8, alpha=alpha, label=label, zorder=5)
    ax.add_patch(polygon)
    # Direction arrow
    dx, dy = 0.6 * length * cos_h, 0.6 * length * sin_h
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0), zorder=6)


def render_frame(
    agent_states: np.ndarray,
    agent_labels: np.ndarray,
    safe_traj: np.ndarray,
    collision_traj: np.ndarray,
    step_idx: int,
    token: str,
    pred_traj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render one BEV frame for a specific time step."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    # Grid
    for g in range(-60, 61, 10):
        ax.axhline(g, color="#2a2a4a", linewidth=0.3, zorder=0)
        ax.axvline(g, color="#2a2a4a", linewidth=0.3, zorder=0)

    # Draw agents
    target_drawn = False
    for i in range(len(agent_labels)):
        if not agent_labels[i]:
            continue
        x, y, h, l, w = agent_states[i]
        if i == 0 and not target_drawn:
            # Target agent = blue
            draw_vehicle(ax, x, y, h, l, w, color="#4fc3f7", alpha=0.9, label="Target Agent")
            ax.plot(x, y, "o", color="#4fc3f7", markersize=8, zorder=10)
            target_drawn = True
        else:
            draw_vehicle(ax, x, y, h, l, w, color="#666666", alpha=0.5)

    # Draw ego at origin
    draw_vehicle(ax, 0, 0, 0, 4.5, 2.0, color="#ff8c00", alpha=0.9, label="Ego")

    # Draw full trajectories (faded)
    ax.plot(safe_traj[:, 0], safe_traj[:, 1], "o-", color="#4caf50", markersize=3,
            linewidth=1.0, alpha=0.3, zorder=3)
    ax.plot(collision_traj[:, 0], collision_traj[:, 1], "o-", color="#f44336", markersize=3,
            linewidth=1.0, alpha=0.3, zorder=3)

    # Draw trajectories up to current step (bright)
    s = min(step_idx + 1, len(safe_traj))
    ax.plot(safe_traj[:s, 0], safe_traj[:s, 1], "o-", color="#4caf50", markersize=5,
            linewidth=2.5, alpha=0.9, label="Safe GT", zorder=7)
    s = min(step_idx + 1, len(collision_traj))
    ax.plot(collision_traj[:s, 0], collision_traj[:s, 1], "s-", color="#f44336", markersize=5,
            linewidth=2.5, alpha=0.9, label="Collision Traj", zorder=7)

    # Draw model prediction if available
    if pred_traj is not None:
        s = min(step_idx + 1, len(pred_traj))
        ax.plot(pred_traj[:s, 0], pred_traj[:s, 1], "^-", color="#ffeb3b", markersize=5,
                linewidth=2.5, alpha=0.9, label="Model Pred", zorder=8)

    # Draw ego moving along collision trajectory
    if step_idx < len(collision_traj):
        ex, ey = collision_traj[step_idx, 0], collision_traj[step_idx, 1]
        eh = collision_traj[step_idx, 2]
        draw_vehicle(ax, ex, ey, eh, 4.5, 2.0, color="#ff6b35", alpha=0.7)

    # Time annotation
    t = (step_idx + 1) * 0.5
    ax.text(0.02, 0.98, f"t = {t:.1f}s  (step {step_idx+1}/{len(collision_traj)})",
            transform=ax.transAxes, fontsize=14, color="white", fontweight="bold",
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#4fc3f7", alpha=0.8))

    # Collision warning at step 8
    if step_idx >= 7:
        ax.text(0.5, 0.95, "⚠ COLLISION", transform=ax.transAxes, fontsize=18,
                color="#f44336", fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#f44336", alpha=0.9))

    ax.set_xlim(-10, 60)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, facecolor="#1a1a2e", edgecolor="#4fc3f7",
              labelcolor="white", framealpha=0.8)
    ax.set_title(f"Collision BEV — {token[:8]}...", color="white", fontsize=12)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_color("#2a2a4a")

    fig.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def create_collision_gif(
    token: str,
    features: Dict,
    targets: Dict,
    out_path: Path,
    pred_traj: Optional[np.ndarray] = None,
):
    """Create a GIF showing collision trajectory animation."""
    agent_states = targets["agent_states"].numpy()
    agent_labels = targets["agent_labels"].numpy()
    safe_traj = targets["trajectory"].numpy()  # original safe trajectory

    # Find target agent
    target_pos = None
    for i in range(len(agent_labels)):
        if agent_labels[i]:
            target_pos = agent_states[i, :2]
            break

    if target_pos is None:
        print(f"  {token}: no agents, skipping")
        return False

    dist = np.linalg.norm(target_pos)
    if dist < 3.0 or dist > 50.0:
        print(f"  {token}: agent at {dist:.1f}m (out of range), skipping")
        return False

    # Generate collision trajectory
    collision_traj = make_collision_trajectory(target_pos, num_steps=len(safe_traj))

    print(f"  {token}: agent at ({target_pos[0]:.1f}, {target_pos[1]:.1f}), "
          f"dist={dist:.1f}m → generating {len(collision_traj)}-step collision")

    # Render frames
    frames = []
    num_steps = len(collision_traj)
    for step in range(num_steps):
        img = render_frame(
            agent_states, agent_labels,
            safe_traj, collision_traj,
            step, token, pred_traj,
        )
        frames.append(img)
    # Hold last frame
    for _ in range(4):
        frames.append(frames[-1])

    # Save as GIF
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            out_path, save_all=True, append_images=pil_frames[1:],
            duration=400, loop=0,  # 400ms per frame
        )
        print(f"  → Saved: {out_path}")
        return True
    except ImportError:
        # Fallback: save individual frames
        for i, f in enumerate(frames):
            plt.imsave(out_path.parent / f"{token}_frame{i:02d}.png", f)
        print(f"  → Saved {len(frames)} PNG frames (install Pillow for GIF)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Visualize collision BEV trajectories")
    parser.add_argument("--cache_path", type=str,
                        default=str(ROOT / "exp/feature_cache_test"),
                        help="Path to feature cache")
    parser.add_argument("--num_scenes", type=int, default=5,
                        help="Number of scenes to visualize")
    parser.add_argument("--out_dir", type=str,
                        default=str(ROOT / "exp/viz_collision"),
                        help="Output directory for GIFs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: collision model checkpoint for predictions")
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache: {cache_path}")
    print(f"Output: {out_dir}")
    print()

    # Collect all tokens
    all_tokens = []
    for log_dir in cache_path.iterdir():
        if not log_dir.is_dir():
            continue
        for td in log_dir.iterdir():
            if (td / "transfuser_feature.gz").exists() and (td / "transfuser_target.gz").exists():
                all_tokens.append(td)

    print(f"Found {len(all_tokens)} cached scenes")

    # Optionally load model for predictions
    pred_model = None
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading collision model: {args.checkpoint}")
        from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
        from navsim.agents.goalflow.goalflow_agent_collision import GoalFlowCollisionAgent

        config = GoalFlowConfig(
            training=False, has_navi=True, start=True,
            freeze_perception=True, tf_d_model=1024, infer_steps=5,
            anchor_size=64, adv_mode=True, adv_agent_idx=0, adv_traj_step=8,
            voc_path=str(ROOT / "data/cluster_points_8192_.npy"),
        )
        pred_model = GoalFlowCollisionAgent(config=config, lr=1e-4, checkpoint_path=args.checkpoint)
        sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        pred_model.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()}, strict=False)
        pred_model.eval()
        print("  Model loaded for inference")

    # Generate GIFs
    count = 0
    for td in all_tokens:
        if count >= args.num_scenes:
            break

        token = td.name
        features = load_gz(td / "transfuser_feature.gz")
        targets = load_gz(td / "transfuser_target.gz")

        # Optional: get model prediction
        pred_traj_np = None
        if pred_model is not None:
            with torch.no_grad():
                features_b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                              for k, v in features.items()}
                features_b["token"] = token
                targets_b = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                             for k, v in targets.items()}
                preds = pred_model.forward(features_b, targets_b)
                pred_traj_np = preds["trajectory"][0].cpu().numpy()
                # Denormalize: take first rotation only (first 3 of 30)
                if pred_traj_np.shape[-1] == 30:
                    pred_traj_np = pred_traj_np[:, :3]

        out_path = out_dir / f"{token}.gif"
        ok = create_collision_gif(token, features, targets, out_path, pred_traj_np)
        if ok:
            count += 1

    print(f"\n{'='*50}")
    print(f"Generated {count} collision BEV GIFs in {out_dir}")


if __name__ == "__main__":
    main()
