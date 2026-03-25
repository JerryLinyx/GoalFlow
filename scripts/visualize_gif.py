"""
Generate BEV driving GIFs from GoalFlow — reproduces the README driving videos.

Since sensor_blobs (camera images) may not be available locally, this script
renders BEV-only GIFs (map + agents + predicted trajectory animation).
If sensor blobs ARE available, pass --with_cameras for the full 3×3 grid.

Usage:
    # BEV-only (no sensor blobs needed):
    python scripts/visualize_gif.py --num_scenes 4 --out_dir exp/viz_gif

    # Full camera+BEV (requires sensor_blobs):
    python scripts/visualize_gif.py --num_scenes 4 --with_cameras

Output:
    exp/viz_gif/<token>.gif   — animated BEV driving scene
"""

import os
import sys
import argparse
import random
import io
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon as MPoly
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT",
            Path(__file__).resolve().parents[1]))
os.environ.setdefault("NUPLAN_MAPS_ROOT",
                      str(Path(os.environ.get("OPENSCENE_DATA_ROOT",
                               "/Users/linyuxuan/navsim_data")) / "maps"))
sys.path.insert(0, str(ROOT))

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Scene
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.plots import (
    configure_bev_ax, configure_ax,
    plot_cameras_frame,
    frame_plot_to_pil,
)
from navsim.visualization.config import TRAJECTORY_CONFIG


# ── load GoalFlow agent for trajectory prediction ─────────────────────────────

def load_agent(checkpoint: str, voc_path: str, device_str: str = "cpu"):
    import torch
    from navsim.agents.goalflow.goalflow_agent_traj import GoalFlowTrajAgent
    from navsim.agents.goalflow.goalflow_config import GoalFlowConfig

    device = torch.device(device_str)
    config = GoalFlowConfig(
        training=False, has_navi=True, start=True,
        use_nearest=True, adv_mode=False,
        anchor_size=64, infer_steps=5,
        freeze_perception=True, tf_d_model=1024,
        voc_path=voc_path,
    )
    agent = GoalFlowTrajAgent(config=config, lr=1e-4, checkpoint_path=checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    agent.load_state_dict(
        {k.replace("agent.", ""): v for k, v in state_dict.items()},
        strict=False,
    )
    agent.eval()
    agent.to(device)
    return agent, device


def predict_from_cache(token: str, agent, device, cache_root: Path):
    import torch, gzip, pickle
    import numpy as np

    def load_gz(p):
        with gzip.open(p, "rb") as f:
            return pickle.load(f)

    for log_dir in cache_root.iterdir():
        td = log_dir / token
        if td.exists():
            features = load_gz(td / "transfuser_feature.gz")
            targets  = load_gz(td / "transfuser_target.gz")
            break
    else:
        return None

    def to_dev(v):
        if not isinstance(v, torch.Tensor): return v
        if device.type == "mps" and v.dtype == torch.float64: v = v.float()
        return v.to(device)

    features = {k: to_dev(v) for k, v in features.items()}
    features["token"] = token
    targets  = {k: to_dev(v.unsqueeze(0)) if isinstance(v, torch.Tensor) else v
                for k, v in targets.items()}
    features = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}

    with torch.no_grad():
        out = agent._goalflow_model(features, targets)

    traj = out["trajectory"].squeeze(0).cpu().numpy()  # (T, 3)
    return traj


# ── BEV frame rendering ────────────────────────────────────────────────────────

def render_bev_frame(scene: Scene,
                     frame_idx: int,
                     pred_traj: np.ndarray,
                     history_traj: np.ndarray) -> Image.Image:
    """Render a single BEV frame with map, agents, history, prediction."""
    from navsim.common.dataclasses import Trajectory

    fig, ax = plt.subplots(1, 1, figsize=(6, 8), facecolor="#0d1117")

    # map + agents
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

    # history path (grey dashes)
    if history_traj is not None and len(history_traj) > 1:
        ax.plot(history_traj[:, 1], history_traj[:, 0],
                "--", color="#888888", lw=1.5, alpha=0.7, zorder=5)

    # predicted future (gold)
    if pred_traj is not None and len(pred_traj) > 0:
        ax.plot(pred_traj[:, 1], pred_traj[:, 0],
                "o-", color="#ffd700", lw=2.0, ms=4, zorder=6,
                label="GoalFlow prediction")

    # GT future (green dashes) — frames after current
    n_future = scene.scene_metadata.num_future_frames
    future_start = frame_idx + 1
    if future_start < len(scene.frames):
        future_poses = np.array([
            f.ego_status.ego_pose
            for f in scene.frames[future_start:]
        ])
        if len(future_poses) > 0:
            # transform to ego-relative at current frame
            ego = scene.frames[frame_idx].ego_status.ego_pose
            dx  = future_poses[:, 0] - ego[0]
            dy  = future_poses[:, 1] - ego[1]
            cos_h, sin_h = np.cos(-ego[2]), np.sin(-ego[2])
            rel_x = cos_h * dx - sin_h * dy
            rel_y = sin_h * dx + cos_h * dy
            ax.plot(rel_y, rel_x, "o--", color="#4ade80",
                    lw=1.5, ms=3, alpha=0.8, zorder=5, label="Ground truth")

    # frame counter
    n_hist = scene.scene_metadata.num_history_frames
    t_sec  = (frame_idx - n_hist + 1) * 0.5    # 0.5s per step
    ax.text(0.02, 0.97, f"t = {t_sec:+.1f}s",
            transform=ax.transAxes, va="top", ha="left",
            color="white", fontsize=11, fontfamily="monospace",
            bbox=dict(facecolor="#1a1a2e", edgecolor="none", alpha=0.8))

    configure_bev_ax(ax)
    configure_ax(ax)
    ax.legend(loc="lower right", fontsize=8,
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default=str(
        Path(os.environ.get("OPENSCENE_DATA_ROOT",
             "/Users/linyuxuan/navsim_data")) / "navsim_logs/test"))
    parser.add_argument("--sensor_dir",  default=str(
        Path(os.environ.get("OPENSCENE_DATA_ROOT",
             "/Users/linyuxuan/navsim_data")) / "sensor_blobs/test"))
    parser.add_argument("--cache_dir",   default=str(ROOT / "exp/feature_cache_test"))
    parser.add_argument("--checkpoint",  default=str(
        ROOT / "data/goalflow_traj_epoch_54-step_18260.ckpt"))
    parser.add_argument("--voc_path",    default=str(ROOT / "data/cluster_points_8192_.npy"))
    parser.add_argument("--out_dir",     default=str(ROOT / "exp/viz_gif"))
    parser.add_argument("--num_scenes",  type=int, default=4)
    parser.add_argument("--with_cameras", action="store_true",
                        help="Render full 3×3 camera+BEV grid (needs sensor_blobs)")
    parser.add_argument("--duration_ms", type=int, default=500,
                        help="Milliseconds per GIF frame")
    parser.add_argument("--seed",        type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── device
    import torch
    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    print(f"Device: {device_str}")

    # ── load GoalFlow agent
    print("Loading GoalFlow checkpoint …")
    agent, device = load_agent(args.checkpoint, args.voc_path, device_str)
    print("Checkpoint loaded ✓")

    # ── scene loader (no sensors for BEV mode)
    sensor_cfg = (SensorConfig.build_all_sensors(include=[3])
                  if args.with_cameras
                  else SensorConfig.build_no_sensors())
    loader = SceneLoader(
        data_path    = Path(args.data_dir),
        sensor_blobs_path = Path(args.sensor_dir),
        scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10),
        sensor_config = sensor_cfg,
    )

    cache_root = Path(args.cache_dir)
    # only pick tokens that exist in the cache
    cached = {p.name for log in cache_root.iterdir() for p in log.iterdir()
              if (p / "transfuser_feature.gz").exists()}
    candidates = [t for t in loader.tokens if t in cached]

    random.seed(args.seed)
    tokens = random.sample(candidates, min(args.num_scenes, len(candidates)))
    print(f"Rendering {len(tokens)} GIFs …")

    for i, token in enumerate(tokens):
        print(f"  [{i+1}/{len(tokens)}] {token}")
        scene = loader.get_scene_from_token(token)
        n_hist = scene.scene_metadata.num_history_frames   # 4

        # predicted trajectory from cache
        pred_traj = predict_from_cache(token, agent, device, cache_root)

        # history poses (in ego frame at current step) — just for trailing path
        # we'll regenerate per-frame, so set to None here
        history_traj = None

        frames: List[Image.Image] = []

        # animate from history start through all future frames
        all_frame_indices = list(range(len(scene.frames)))

        if args.with_cameras:
            # full 3×3 camera grid (needs sensor blobs)
            frames = frame_plot_to_pil(
                plot_cameras_frame, scene, all_frame_indices)
        else:
            # BEV-only
            for fi in tqdm(all_frame_indices, desc=f"  {token[:16]}", leave=False):
                img = render_bev_frame(scene, fi, pred_traj, history_traj)
                frames.append(img)

        if frames:
            gif_path = out_dir / f"{token}.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=args.duration_ms,
                loop=0,
            )
            print(f"    saved → {gif_path.name}  ({len(frames)} frames)")

    print(f"\n✅ Done. GIFs: {out_dir}/")


if __name__ == "__main__":
    main()
