"""
GoalFlow inference visualization script.
Loads cached features directly (no raw sensor data needed).
Runs the model and renders BEV + trajectory comparisons.

Usage:
    python scripts/visualize_results.py \
        --num_scenes 5 \
        --out_dir exp/viz \
        --adv_mode          # optional: enable adversarial selection

Outputs (per scene):
    exp/viz/<token>_bev.png        — BEV: map + agents + GT vs predicted traj
    exp/viz/<token>_compare.png    — side-by-side normal vs adv (if --adv_mode)
    exp/viz/summary.png            — grid of all scenes
"""

import os
import sys
import gzip
import pickle
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── project root on path ───────────────────────────────────────────────────────
ROOT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from navsim.agents.goalflow.goalflow_agent_traj import GoalFlowTrajAgent
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_features import GoalFlowFeatureBuilder, GoalFlowTargetBuilder


# ── helpers ────────────────────────────────────────────────────────────────────

def load_gz(path: Path) -> Dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_cache_token(cache_root: Path, token: str) -> Tuple[Dict, Dict]:
    """Load features + targets for a single token from the feature cache."""
    # walk cache dirs to find the token folder
    for log_dir in cache_root.iterdir():
        token_dir = log_dir / token
        if token_dir.exists():
            features = load_gz(token_dir / "transfuser_feature.gz")
            targets  = load_gz(token_dir / "transfuser_target.gz")
            features["token"] = token
            return features, targets
    raise FileNotFoundError(f"Token {token} not found in cache {cache_root}")


def collect_tokens(cache_root: Path, n: int, seed: int = 42) -> List[str]:
    """Collect up to n token names from the cache."""
    tokens = []
    for log_dir in sorted(cache_root.iterdir()):
        if not log_dir.is_dir():
            continue
        for token_dir in sorted(log_dir.iterdir()):
            if (token_dir / "transfuser_feature.gz").exists():
                tokens.append(token_dir.name)
    random.seed(seed)
    random.shuffle(tokens)
    return tokens[:n]


def add_batch_dim(d: Dict) -> Dict:
    return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in d.items()}


def run_inference(model: GoalFlowTrajAgent,
                  features: Dict,
                  targets: Dict,
                  device: torch.device) -> np.ndarray:
    """Run forward pass and return trajectory (T, 3) numpy array."""
    features_b = add_batch_dim(features)
    targets_b  = add_batch_dim(targets)

    def to_device(v, dev):
        if not isinstance(v, torch.Tensor):
            return v
        # MPS does not support float64 — cast to float32
        if dev.type == "mps" and v.dtype == torch.float64:
            v = v.float()
        return v.to(dev)

    features_b = {k: to_device(v, device) for k, v in features_b.items()}
    targets_b  = {k: to_device(v, device) for k, v in targets_b.items()}

    model.eval()
    model.to(device)
    with torch.no_grad():
        out = model._goalflow_model(features_b, targets_b)

    traj = out["trajectory"].squeeze(0).cpu().numpy()   # (T, 3) x,y,heading
    return traj


def draw_bev(ax: plt.Axes,
             agent_states: np.ndarray,
             agent_labels: np.ndarray,
             gt_traj: np.ndarray,
             pred_traj: np.ndarray,
             pred_traj_adv: np.ndarray = None,
             title: str = "") -> plt.Axes:
    """
    Draw a BEV scene on an existing axes.
    Coordinate system: x = forward (up), y = left (right)
    """
    ax.set_facecolor("#1a1a2e")

    # ── road grid ──────────────────────────────────────────
    for x in np.arange(-30, 31, 10):
        ax.axhline(x, color="#2a2a4e", lw=0.5, zorder=0)
        ax.axvline(x, color="#2a2a4e", lw=0.5, zorder=0)

    # ── surrounding agents (vehicles) ──────────────────────
    for i, (state, label) in enumerate(zip(agent_states, agent_labels)):
        if not label:
            continue
        x, y, heading, length, width = state
        # draw as oriented rectangle
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        corners = np.array([
            [ length/2,  width/2],
            [ length/2, -width/2],
            [-length/2, -width/2],
            [-length/2,  width/2],
        ])
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = (rot @ corners.T).T + np.array([x, y])
        color = "#ff6b6b" if i == 0 else "#ffa500"   # nearest agent = red
        poly = plt.Polygon(corners[:, [1, 0]], closed=True,
                           facecolor=color, edgecolor="white",
                           linewidth=0.8, alpha=0.85, zorder=3)
        ax.add_patch(poly)

    # ── ego vehicle (at origin) ─────────────────────────────
    ego = plt.Rectangle((-1, -2), 2, 4,
                        facecolor="#4ecdc4", edgecolor="white",
                        linewidth=1.2, zorder=4)
    ax.add_patch(ego)
    ax.annotate("EGO", (0, 0), ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=5)

    # ── ground truth trajectory ─────────────────────────────
    if gt_traj is not None and len(gt_traj) > 0:
        ax.plot(gt_traj[:, 1], gt_traj[:, 0],
                "o--", color="#a8e6cf", lw=1.5, ms=3,
                label="GT (human)", zorder=6)

    # ── predicted trajectory ────────────────────────────────
    if pred_traj is not None and len(pred_traj) > 0:
        ax.plot(pred_traj[:, 1], pred_traj[:, 0],
                "s-", color="#ffd93d", lw=2.0, ms=4,
                label="GoalFlow (normal)", zorder=7)

    # ── adversarial trajectory ──────────────────────────────
    if pred_traj_adv is not None and len(pred_traj_adv) > 0:
        ax.plot(pred_traj_adv[:, 1], pred_traj_adv[:, 0],
                "^-", color="#ff4757", lw=2.0, ms=4,
                label="GoalFlow (adv)", zorder=8)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 40)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)", color="white", fontsize=8)
    ax.set_ylabel("forward (m)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.legend(loc="upper right", fontsize=7,
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    return ax


def draw_traj_info(ax: plt.Axes,
                   token: str,
                   gt_traj: np.ndarray,
                   pred_traj: np.ndarray,
                   pred_adv: np.ndarray = None) -> plt.Axes:
    """Print numerical stats as text panel."""
    ax.set_facecolor("#0f0f23")
    ax.axis("off")

    def ade(a, b):
        n = min(len(a), len(b))
        return float(np.mean(np.linalg.norm(a[:n, :2] - b[:n, :2], axis=-1)))

    lines = [
        f"Token: {token[:16]}...",
        "",
        f"GT traj steps:   {len(gt_traj)}",
        f"Pred steps:      {len(pred_traj)}",
        "",
        f"ADE (normal):    {ade(gt_traj, pred_traj):.2f} m",
    ]
    if pred_adv is not None:
        lines.append(f"ADE (adv):       {ade(gt_traj, pred_adv):.2f} m")
        # nearest agent collision proxy
        lines.append("")
        lines.append(f"→ adv trades navigation accuracy")
        lines.append(f"  for proximity to target agent")

    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8.5, color="#e0e0e0",
            fontfamily="monospace",
            linespacing=1.6)
    return ax


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",    default=str(ROOT / "exp/feature_cache_test"))
    parser.add_argument("--checkpoint",   default=str(ROOT / "data/goalflow_traj_epoch_54-step_18260.ckpt"))
    parser.add_argument("--voc_path",     default=str(ROOT / "data/cluster_points_8192_.npy"))
    parser.add_argument("--out_dir",      default=str(ROOT / "exp/viz"))
    parser.add_argument("--num_scenes",   type=int, default=6)
    parser.add_argument("--adv_mode",     action="store_true",
                        help="Also run adversarial selection and show comparison")
    parser.add_argument("--adv_agent_idx", type=int, default=0)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── device ─────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── load model ─────────────────────────────────────────
    print("Loading checkpoint...")
    config = GoalFlowConfig(
        training=False,
        has_navi=True,
        start=True,
        use_nearest=True,
        adv_mode=False,
        anchor_size=64,
        infer_steps=5,
        freeze_perception=True,
        tf_d_model=1024,
        voc_path=args.voc_path,
    )
    agent = GoalFlowTrajAgent(
        config=config,
        lr=1e-4,
        checkpoint_path=args.checkpoint,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    agent.load_state_dict(
        {k.replace("agent.", ""): v for k, v in state_dict.items()},
        strict=False,
    )
    agent.eval()
    print("Checkpoint loaded ✓")

    # adversarial config (same model, different scoring)
    if args.adv_mode:
        config_adv = GoalFlowConfig(
            training=False,
            has_navi=True,
            start=True,
            use_nearest=True,
            adv_mode=True,
            adv_agent_idx=args.adv_agent_idx,
            adv_traj_step=8,
            anchor_size=64,
            infer_steps=5,
            freeze_perception=True,
            tf_d_model=1024,
            voc_path=args.voc_path,
        )
        agent_adv = GoalFlowTrajAgent(
            config=config_adv, lr=1e-4, checkpoint_path=args.checkpoint)
        agent_adv.load_state_dict(
            {k.replace("agent.", ""): v for k, v in state_dict.items()},
            strict=False,
        )
        agent_adv.eval()

    # ── collect tokens ─────────────────────────────────────
    cache_root = Path(args.cache_dir)
    tokens = collect_tokens(cache_root, args.num_scenes, seed=args.seed)
    print(f"Visualizing {len(tokens)} scenes...")

    summary_axes = []

    for i, token in enumerate(tokens):
        print(f"  [{i+1}/{len(tokens)}] {token}")
        try:
            features, targets = load_cache_token(cache_root, token)
        except FileNotFoundError as e:
            print(f"    ⚠ skipped: {e}")
            continue

        # ── inference ──────────────────────────────────────
        pred_traj  = run_inference(agent, features, targets, device)
        pred_adv   = None
        if args.adv_mode:
            pred_adv = run_inference(agent_adv, features, targets, device)

        gt_traj = targets.get("trajectory", None)
        if gt_traj is not None:
            gt_traj = gt_traj.numpy() if isinstance(gt_traj, torch.Tensor) else gt_traj
            # GT traj shape: (T, 3) or (T, 2)
            if gt_traj.ndim == 1:
                gt_traj = gt_traj.reshape(-1, 3)

        agent_states = features.get("agent_states", torch.zeros(30, 5)).numpy()
        agent_labels = features.get("agent_labels", torch.zeros(30, dtype=torch.bool)).numpy()
        if agent_states.ndim == 1:
            agent_states = agent_states.reshape(-1, 5)

        # ── plot ───────────────────────────────────────────
        fig = plt.figure(figsize=(14 if args.adv_mode else 9, 6),
                         facecolor="#0f0f23")

        if args.adv_mode:
            gs = GridSpec(1, 3, figure=fig, wspace=0.3,
                          left=0.05, right=0.97, top=0.92, bottom=0.1)
            ax_normal = fig.add_subplot(gs[0])
            ax_adv    = fig.add_subplot(gs[1])
            ax_info   = fig.add_subplot(gs[2])

            draw_bev(ax_normal, agent_states, agent_labels,
                     gt_traj, pred_traj, title="Normal Mode")
            draw_bev(ax_adv, agent_states, agent_labels,
                     gt_traj, pred_traj, pred_adv, title="Adversarial Mode")
            draw_traj_info(ax_info, token, gt_traj, pred_traj, pred_adv)
        else:
            gs = GridSpec(1, 2, figure=fig, wspace=0.3,
                          left=0.05, right=0.97, top=0.92, bottom=0.1)
            ax_bev  = fig.add_subplot(gs[0])
            ax_info = fig.add_subplot(gs[1])

            draw_bev(ax_bev, agent_states, agent_labels,
                     gt_traj, pred_traj, title="GoalFlow Prediction")
            draw_traj_info(ax_info, token, gt_traj, pred_traj)

        fig.suptitle(f"GoalFlow  |  scene {i+1}/{len(tokens)}",
                     color="white", fontsize=11, y=0.98)

        out_path = out_dir / f"{token}_bev.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    saved → {out_path.name}")

        summary_axes.append((token, pred_traj, pred_adv, gt_traj,
                              agent_states, agent_labels))

    # ── summary grid ───────────────────────────────────────
    n = len(summary_axes)
    if n == 0:
        print("No scenes rendered.")
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 5),
                             facecolor="#0f0f23")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for j, (token, pred, pred_adv, gt, states, labels) in enumerate(summary_axes):
        draw_bev(axes_flat[j], states, labels, gt, pred, pred_adv,
                 title=token[:20] + "…")

    for j in range(len(summary_axes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("GoalFlow — Scene Summary", color="white", fontsize=13)
    summary_path = out_dir / "summary.png"
    fig.savefig(summary_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n✅ Done. Summary: {summary_path}")
    print(f"   Individual PNGs: {out_dir}/")


if __name__ == "__main__":
    main()
