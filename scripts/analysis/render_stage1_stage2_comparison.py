#!/usr/bin/env python3
"""
Render a side-by-side Stage-1 vs Stage-2 qualitative comparison on filtered
dangerous-validation samples, plus aggregate proxy metrics on the same subset.
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
from navsim.agents.goalflow.safesim_metrics import (
    bbox_collision_rate_per_sample,
    better_than_gt_rate,
    hit_rate_from_min_dist,
    min_distance_per_sample,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tf_d_model", type=int, default=1024)
    parser.add_argument("--anchor_size", type=int, default=16)
    parser.add_argument("--infer_steps", type=int, default=25)
    parser.add_argument("--search_samples", type=int, default=128)
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--min_extent", type=float, default=2.0)
    parser.add_argument("--min_history_extent", type=float, default=1.0)
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


def load_agent(checkpoint_path: str, config: SafeSimConfig, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = SafeSimAgent(config)
    agent.load_state_dict(checkpoint["state_dict"], strict=True)
    agent.to(device).eval()
    agent._config.training = False
    agent.model._config.training = False
    return agent


def compute_metrics(pred, gt, ctrl, ego_extent, ctrl_extent):
    pred_min = min_distance_per_sample(pred, ctrl)
    gt_min = min_distance_per_sample(gt, ctrl)
    return {
        "pred_min": pred_min,
        "gt_min": gt_min,
        "dangerous_hit": bbox_collision_rate_per_sample(pred, ctrl, ego_extent, ctrl_extent),
        "hit2": hit_rate_from_min_dist(pred_min, 2.0),
        "hit4": hit_rate_from_min_dist(pred_min, 4.0),
        "better": better_than_gt_rate(pred_min, gt_min),
    }


def summarize(metric_chunks):
    def cat(name):
        return torch.cat([chunk[name].detach().cpu() for chunk in metric_chunks], dim=0)

    pred_min = cat("pred_min")
    gt_min = cat("gt_min")
    dangerous_hit = cat("dangerous_hit")
    hit2 = cat("hit2")
    hit4 = cat("hit4")
    better = cat("better")
    return {
        "dangerous_hit_rate": float(dangerous_hit.mean()),
        "hit@2m": float(hit2.mean()),
        "hit@4m": float(hit4.mean()),
        "pred_better_than_gt_rate": float(better.mean()),
        "pred_min_dist": float(pred_min.mean()),
        "gt_min_dist": float(gt_min.mean()),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, config = build_loader(args)
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    stage1 = load_agent(args.stage1_checkpoint, config, device)
    stage2 = load_agent(args.stage2_checkpoint, config, device)

    stage1_metrics = []
    stage2_metrics = []
    candidate_examples = []

    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            out1 = stage1.forward(batch)
            out2 = stage2.forward(batch)

            gt = batch["future_trajectory"]
            ctrl = batch["ctrl_future"]
            ego_extent = batch["ego_extent_future"]
            ctrl_extent = batch["ctrl_extent_future"]

            m1 = compute_metrics(out1["trajectory"], gt, ctrl, ego_extent, ctrl_extent)
            m2 = compute_metrics(out2["trajectory"], gt, ctrl, ego_extent, ctrl_extent)
            stage1_metrics.append(m1)
            stage2_metrics.append(m2)

            for i in range(gt.shape[0]):
                pred1 = out1["trajectory"][i].detach().cpu()
                pred2 = out2["trajectory"][i].detach().cpu()
                gt_i = gt[i].detach().cpu()
                ctrl_i = ctrl[i].detach().cpu()
                history_i = batch["agent_history"][i, 0, :, :2].detach().cpu()
                extent1 = torch.norm(pred1[:, :2], dim=-1).max().item()
                extent2 = torch.norm(pred2[:, :2], dim=-1).max().item()
                ctrl_extent_i = torch.norm(ctrl_i[:, :2], dim=-1).max().item()
                history_extent_i = torch.norm(history_i[:, :2], dim=-1).max().item()
                if (
                    extent1 < args.min_extent
                    or extent2 < args.min_extent
                    or ctrl_extent_i < 2.0
                    or history_extent_i < args.min_history_extent
                ):
                    continue
                score = (m1["pred_min"][i] - m2["pred_min"][i]).item() + 0.1 * extent2
                candidate_examples.append(
                    {
                        "scene_key": batch["scene_key"][i],
                        "timestep": int(batch["timestep"][i]),
                        "stage1_pred": pred1,
                        "stage2_pred": pred2,
                        "gt": gt_i,
                        "ctrl": ctrl_i,
                        "history": history_i,
                        "stage1_pred_min": float(m1["pred_min"][i]),
                        "stage2_pred_min": float(m2["pred_min"][i]),
                        "gt_min": float(m1["gt_min"][i]),
                        "stage1_hit": float(m1["dangerous_hit"][i]),
                        "stage2_hit": float(m2["dangerous_hit"][i]),
                        "extent1": extent1,
                        "extent2": extent2,
                        "history_extent": history_extent_i,
                        "score": score,
                    }
                )

    stage1_summary = summarize(stage1_metrics)
    stage2_summary = summarize(stage2_metrics)

    candidate_examples.sort(key=lambda ex: (-ex["score"], ex["stage2_pred_min"]))
    selected = []
    seen_scene_keys = set()
    for ex in candidate_examples:
        if ex["scene_key"] in seen_scene_keys:
            continue
        selected.append(ex)
        seen_scene_keys.add(ex["scene_key"])
        if len(selected) >= args.num_examples:
            break

    cols = 3
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4.8 * rows))
    axes = axes.flatten()
    for ax in axes[len(selected):]:
        ax.axis("off")

    for ax, ex in zip(axes, selected):
        pred1 = ex["stage1_pred"].numpy()
        pred2 = ex["stage2_pred"].numpy()
        gt = ex["gt"].numpy()
        ctrl = ex["ctrl"].numpy()
        history = ex["history"].numpy()
        ax.plot(
            history[:, 0],
            history[:, 1],
            color="#7f7f7f",
            linewidth=1.8,
            linestyle="--",
            marker="s",
            markersize=3.0,
            zorder=1,
            label="History",
        )
        ax.scatter([0], [0], color="black", s=22, zorder=6, label="Ego Start")
        ax.plot(pred1[:, 0], pred1[:, 1], color="#ff7f0e", linewidth=2.1, marker="o", markersize=2.8, zorder=4, label="Stage1")
        ax.scatter([pred1[-1, 0]], [pred1[-1, 1]], color="#ff7f0e", s=30, marker="x", zorder=6)
        ax.plot(pred2[:, 0], pred2[:, 1], color="#d62728", linewidth=2.5, marker="o", markersize=3.0, zorder=5, label="Stage2")
        ax.scatter([pred2[-1, 0]], [pred2[-1, 1]], color="#d62728", s=36, marker="x", zorder=7)
        ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", linewidth=1.9, linestyle="--", zorder=3, label="GT")
        ax.plot(ctrl[:, 0], ctrl[:, 1], color="#1f77b4", linewidth=1.9, marker="^", markersize=2.8, zorder=2, label="Ctrl")
        ax.set_title(
            f"{ex['scene_key']}\nt={ex['timestep']} s1={ex['stage1_pred_min']:.2f} s2={ex['stage2_pred_min']:.2f} gt={ex['gt_min']:.2f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.axis("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:6], labels[:6], loc="upper center", ncol=6)
    fig.suptitle("Stage 1 vs Stage 2 on Filtered Dangerous Scenes", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "stage1_vs_stage2_examples.png", dpi=220)
    plt.close(fig)

    payload = {
        "stage1_summary": stage1_summary,
        "stage2_summary": stage2_summary,
        "selected_examples": [
            {
                k: v
                for k, v in ex.items()
                if k not in {"stage1_pred", "stage2_pred", "gt", "ctrl", "history"}
            }
            for ex in selected
        ],
        "settings": {
            "search_samples": args.search_samples,
            "anchor_size": args.anchor_size,
            "infer_steps": args.infer_steps,
            "dataset": "filtered validation subset",
            "min_extent": args.min_extent,
            "min_history_extent": args.min_history_extent,
        },
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Stage 1 vs Stage 2 Proxy Comparison",
        "",
        "| model | dangerous_hit_rate | hit@2m | hit@4m | pred_better_than_gt_rate | pred_min_dist | gt_min_dist |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| stage1 | {stage1_summary['dangerous_hit_rate']:.4f} | {stage1_summary['hit@2m']:.4f} | {stage1_summary['hit@4m']:.4f} | {stage1_summary['pred_better_than_gt_rate']:.4f} | {stage1_summary['pred_min_dist']:.4f} | {stage1_summary['gt_min_dist']:.4f} |",
        f"| stage2 | {stage2_summary['dangerous_hit_rate']:.4f} | {stage2_summary['hit@2m']:.4f} | {stage2_summary['hit@4m']:.4f} | {stage2_summary['pred_better_than_gt_rate']:.4f} | {stage2_summary['pred_min_dist']:.4f} | {stage2_summary['gt_min_dist']:.4f} |",
        "",
        "Selected scenes are ranked by Stage-2 improvement in min distance while requiring visible history and visible Stage-1/Stage-2 future motion.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))

    print(out_dir / "stage1_vs_stage2_examples.png")
    print(out_dir / "summary.md")
    print(out_dir / "comparison_summary.json")


if __name__ == "__main__":
    main()
