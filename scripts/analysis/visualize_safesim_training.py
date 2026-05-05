#!/usr/bin/env python3
"""
Visualize SafeSim training progress and trajectory predictions.

Outputs:
  - loss_curves.png
  - trajectory_examples.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep matplotlib cache writable in restricted environments.
MPL_DIR = ROOT / "tmp" / "mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp" / "xdg-cache"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from navsim.agents.goalflow.safesim_agent import SafeSimAgent
from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_dataset import SafeSimTemporalDataset, safesim_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SafeSim training artifacts")
    parser.add_argument(
        "--metrics",
        type=str,
        default="safesim_logs/csv_logs/version_1/metrics.csv",
        help="Path to Lightning CSV metrics file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path. If omitted, the nearest saved checkpoint to --target_epoch is used.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="safesim_logs/checkpoints",
        help="Directory containing saved checkpoints",
    )
    parser.add_argument(
        "--target_epoch",
        type=int,
        default=25,
        help="Human-facing target epoch to visualize. Nearest saved checkpoint is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/safesim_epoch25_viz",
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=6,
        help="Number of validation examples to visualize",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size for visualization",
    )
    parser.add_argument(
        "--anchor_size",
        type=int,
        default=64,
        help="Number of trajectory candidates to sample during inference",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=100,
        help="Number of denoising steps during inference",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio used during training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Scene split seed used during training",
    )
    parser.add_argument(
        "--hdf5_paths",
        nargs="+",
        default=[
            "safesim/case1/data.hdf5",
            "safesim/case2/data.hdf5",
            "safesim/case3/data.hdf5",
            "safesim/case4/data.hdf5",
            "safesim/case5/data.hdf5",
        ],
        help="SafeSim HDF5 files used for training",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint_dir: Path, checkpoint: str, target_epoch: int) -> Path:
    if checkpoint:
        return Path(checkpoint)

    ckpts = sorted(checkpoint_dir.glob("safesim-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def parse_epoch(path: Path) -> int:
        return int(path.stem.split("-")[1])

    # Saved checkpoints are zero-based. Match the nearest saved epoch.
    return min(ckpts, key=lambda path: abs(parse_epoch(path) - (target_epoch - 1)))


def plot_loss_curves(metrics_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    train = df[["epoch", "train/loss_epoch"]].dropna()
    val = df[["epoch", "val/loss"]].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    if not train.empty:
        ax.plot(train["epoch"], train["train/loss_epoch"], label="Train Loss", linewidth=2)
    if not val.empty:
        ax.plot(val["epoch"], val["val/loss"], label="Val Loss", linewidth=2)
        best_idx = val["val/loss"].idxmin()
        best_row = val.loc[best_idx]
        ax.scatter([best_row["epoch"]], [best_row["val/loss"]], color="red", zorder=3)
        ax.annotate(
            f"best val={best_row['val/loss']:.4f}\nepoch={int(best_row['epoch'])}",
            (best_row["epoch"], best_row["val/loss"]),
            textcoords="offset points",
            xytext=(10, -10),
        )

    ax.set_title("SafeSim Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return df


def build_val_loader(args: argparse.Namespace) -> DataLoader:
    config = SafeSimConfig(
        hdf5_paths=args.hdf5_paths,
        training=False,
        anchor_size=args.anchor_size,
        infer_steps=args.infer_steps,
    )
    dataset = SafeSimTemporalDataset(config, split="train")
    dataset_size = len(dataset)
    val_size = max(1, int(round(dataset_size * args.val_split)))
    train_size = max(1, dataset_size - val_size)
    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, dataset_size - train_size], generator=generator)
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=safesim_collate_fn,
    )


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_predictions(
    checkpoint_path: Path,
    loader: DataLoader,
    num_examples: int,
    anchor_size: int,
    infer_steps: int,
):
    config = SafeSimConfig(
        hdf5_paths=[],
        training=False,
        anchor_size=anchor_size,
        infer_steps=infer_steps,
    )
    device = _resolve_device()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    agent = SafeSimAgent(config)
    agent.load_state_dict(state_dict, strict=True)
    agent.to(device).eval()
    agent._config.training = False
    agent.model._config.training = False

    examples = []
    seen_scene_keys = set()
    pred_min_dists = []
    gt_min_dists = []

    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            outputs = agent.forward(batch)
            pred = outputs["trajectory"].cpu()
            gt = batch["future_trajectory"].cpu()
            ctrl = batch["ctrl_future"].cpu()

            pred_dist = torch.norm(pred[..., :2] - ctrl[..., :2], dim=-1)
            gt_dist = torch.norm(gt[..., :2] - ctrl[..., :2], dim=-1)

            pred_min_dists.extend(pred_dist.min(dim=-1).values.tolist())
            gt_min_dists.extend(gt_dist.min(dim=-1).values.tolist())

            for idx in range(pred.shape[0]):
                if len(examples) >= num_examples:
                    break
                scene_key = batch["scene_key"][idx]
                if scene_key in seen_scene_keys:
                    continue
                seen_scene_keys.add(scene_key)
                examples.append(
                    {
                        "scene_key": scene_key,
                        "timestep": int(batch["timestep"][idx]),
                        "pred": pred[idx],
                        "gt": gt[idx],
                        "ctrl": ctrl[idx],
                        "pred_min": float(pred_dist[idx].min()),
                        "gt_min": float(gt_dist[idx].min()),
                    }
                )
            if len(examples) >= num_examples:
                break

    return examples, pred_min_dists, gt_min_dists


def plot_trajectory_examples(examples, output_path: Path):
    if not examples:
        raise ValueError("No examples collected for trajectory visualization")

    cols = 3
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten()

    for ax in axes[len(examples):]:
        ax.axis("off")

    for ax, ex in zip(axes, examples):
        pred = ex["pred"].numpy()
        gt = ex["gt"].numpy()
        ctrl = ex["ctrl"].numpy()

        ax.plot([0], [0], "ko", label="Ego Start")
        ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=2, label="Pred Ego")
        ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", linewidth=2, linestyle="--", label="GT Ego")
        ax.plot(ctrl[:, 0], ctrl[:, 1], color="#1f77b4", linewidth=2, label="Ctrl Agent")

        ax.set_title(
            f"{ex['scene_key']}\nt={ex['timestep']}  pred_min={ex['pred_min']:.2f}m  gt_min={ex['gt_min']:.2f}m",
            fontsize=10,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(alpha=0.3)
        ax.axis("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("SafeSim Trajectory Visualization", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics)
    checkpoint_path = resolve_checkpoint(Path(args.checkpoint_dir), args.checkpoint, args.target_epoch)
    print(f"Using checkpoint: {checkpoint_path}")

    plot_loss_curves(metrics_path, output_dir / "loss_curves.png")
    loader = build_val_loader(args)
    examples, pred_min_dists, gt_min_dists = collect_predictions(
        checkpoint_path, loader, args.num_examples, args.anchor_size, args.infer_steps
    )
    plot_trajectory_examples(examples, output_dir / "trajectory_examples.png")

    print(f"Saved loss curves to: {output_dir / 'loss_curves.png'}")
    print(f"Saved trajectory examples to: {output_dir / 'trajectory_examples.png'}")
    if pred_min_dists:
        print(f"mean_pred_min_dist={sum(pred_min_dists) / len(pred_min_dists):.4f}")
        print(f"mean_gt_min_dist={sum(gt_min_dists) / len(gt_min_dists):.4f}")


if __name__ == "__main__":
    main()
