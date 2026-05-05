#!/usr/bin/env python3
"""
Render a qualitative panel of Stage-2 predictions where the predicted
trajectory is visibly present in the figure.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, random_split

from navsim.agents.goalflow.safesim_agent import SafeSimAgent
from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_dataset import SafeSimTemporalDataset, safesim_collate_fn
from navsim.agents.goalflow.safesim_metrics import min_distance_per_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tf_d_model", type=int, default=1024)
    parser.add_argument("--anchor_size", type=int, default=16)
    parser.add_argument("--infer_steps", type=int, default=25)
    parser.add_argument("--search_samples", type=int, default=128)
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--min_pred_extent", type=float, default=2.0)
    parser.add_argument("--exclude_summary", type=str, default="")
    parser.add_argument("--title_suffix", type=str, default="")
    parser.add_argument("--output_name", type=str, default="trajectory_examples_visible.png")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hdf5_paths",
        nargs="+",
        default=[
            "safesim/case1_filtered/data.hdf5",
            "safesim/case2_filtered/data.hdf5",
            "safesim/case3_filtered/data.hdf5",
            "safesim/case4_filtered/data.hdf5",
            "safesim/case5_filtered/data.hdf5",
        ],
    )
    return parser.parse_args()


def build_loader(args):
    config = SafeSimConfig(
        hdf5_paths=args.hdf5_paths,
        training=False,
        anchor_size=args.anchor_size,
        infer_steps=args.infer_steps,
        cfg_scale=1.0,
        tf_d_model=args.tf_d_model,
    )
    dataset = SafeSimTemporalDataset(config, split="eval", target_policy="raw_gt")
    dataset_size = len(dataset)
    val_size = max(1, int(round(dataset_size * args.val_split)))
    train_size = max(1, dataset_size - val_size)
    _, val_dataset = random_split(
        dataset,
        [train_size, dataset_size - train_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    subset = Subset(val_dataset, list(range(min(args.search_samples, len(val_dataset)))))
    loader = DataLoader(
        subset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=safesim_collate_fn,
    )
    return loader, config


def load_agent(args, config):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = SafeSimAgent(config)
    agent.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    agent.to(device).eval()
    agent._config.training = False
    agent.model._config.training = False
    return agent, device


def pick_examples(loader, agent, device, args):
    excluded = set()
    if args.exclude_summary:
        payload = json.loads(Path(args.exclude_summary).read_text())
        excluded = {(ex["scene_key"], ex["timestep"]) for ex in payload.get("examples", [])}

    examples = []
    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            outputs = agent.forward(batch)
            pred = outputs["trajectory"]
            gt = batch["future_trajectory"]
            ctrl = batch["ctrl_future"]

            pred_min = min_distance_per_sample(pred, ctrl)
            gt_min = min_distance_per_sample(gt, ctrl)

            for i in range(pred.shape[0]):
                p = pred[i].detach().cpu()
                g = gt[i].detach().cpu()
                c = ctrl[i].detach().cpu()
                pred_extent = torch.norm(p[:, :2], dim=-1).max().item()
                ctrl_extent = torch.norm(c[:, :2], dim=-1).max().item()
                pred_end_gap = torch.norm(p[-1, :2] - g[-1, :2]).item()
                score = pred_extent + 0.25 * pred_end_gap
                examples.append(
                    {
                        "scene_key": batch["scene_key"][i],
                        "timestep": int(batch["timestep"][i]),
                        "pred": p,
                        "gt": g,
                        "ctrl": c,
                        "pred_min": float(pred_min[i]),
                        "gt_min": float(gt_min[i]),
                        "pred_extent": pred_extent,
                        "ctrl_extent": ctrl_extent,
                        "pred_end_gap": pred_end_gap,
                        "score": score,
                    }
                )

    visible = [
        ex
        for ex in examples
        if ex["pred_extent"] >= args.min_pred_extent
        and ex["ctrl_extent"] >= 2.0
        and (ex["scene_key"], ex["timestep"]) not in excluded
    ]
    visible.sort(key=lambda ex: (-ex["score"], ex["pred_min"]))
    return visible[: args.num_examples]


def render_examples(examples, output_path: Path, title_suffix: str):
    cols = 3
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.8 * rows))
    axes = axes.flatten()
    for ax in axes[len(examples):]:
        ax.axis("off")

    for ax, ex in zip(axes, examples):
        pred = ex["pred"].numpy()
        gt = ex["gt"].numpy()
        ctrl = ex["ctrl"].numpy()
        ax.scatter([0], [0], color="black", s=24, zorder=6, label="Ego Start")
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            color="#d62728",
            linewidth=2.5,
            marker="o",
            markersize=3,
            zorder=5,
            label="Pred Ego",
        )
        ax.scatter([pred[-1, 0]], [pred[-1, 1]], color="#d62728", s=40, marker="x", zorder=7, label="Pred End")
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            color="#2ca02c",
            linewidth=2,
            linestyle="--",
            zorder=3,
            label="GT Ego",
        )
        ax.scatter([gt[-1, 0]], [gt[-1, 1]], color="#2ca02c", s=28, marker="s", zorder=4, label="GT End")
        ax.plot(
            ctrl[:, 0],
            ctrl[:, 1],
            color="#1f77b4",
            linewidth=2,
            marker="^",
            markersize=3,
            zorder=2,
            label="Ctrl Agent",
        )
        ax.scatter([ctrl[-1, 0]], [ctrl[-1, 1]], color="#1f77b4", s=28, marker="D", zorder=4, label="Ctrl End")
        ax.set_title(
            f"{ex['scene_key']}\nt={ex['timestep']} pred_min={ex['pred_min']:.2f} gt_min={ex['gt_min']:.2f} extent={ex['pred_extent']:.1f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.axis("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:7], labels[:7], loc="upper center", ncol=4)
    title = "Stage 2 Typical Visible Predictions"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, config = build_loader(args)
    agent, device = load_agent(args, config)
    examples = pick_examples(loader, agent, device, args)
    render_examples(examples, out_dir / args.output_name, args.title_suffix)

    summary = {
        "selected_count": len(examples),
        "selection_rule": {
            "search_samples": args.search_samples,
            "min_pred_extent": args.min_pred_extent,
            "ranking": "score = pred_extent + 0.25 * pred_end_gap",
            "exclude_summary": args.exclude_summary,
        },
        "examples": [
            {k: v for k, v in ex.items() if k not in {"pred", "gt", "ctrl"}} for ex in examples
        ],
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(out_dir / args.output_name)
    print(out_dir / "selection_summary.json")


if __name__ == "__main__":
    main()
