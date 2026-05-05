#!/usr/bin/env python3
"""
Audit whether filtered raw GT trajectories are already dangerous enough, or whether
Stage-2 should supervise against action-derived dangerous targets instead.
"""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_dataset import SafeSimTemporalDataset
from navsim.agents.goalflow.safesim_metrics import bbox_collision_rate_per_sample, min_distance_per_sample


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs/safesim_target_audit")
    return parser.parse_args()


def evaluate_policy(dataset, indices):
    min_dists = []
    hit_rates = []
    examples = []
    for idx in indices:
        sample = dataset[idx]
        traj = sample["training_target_trajectory"].unsqueeze(0)
        ctrl = sample["ctrl_future"].unsqueeze(0)
        ego_extent = sample["ego_extent_future"].unsqueeze(0)
        ctrl_extent = sample["ctrl_extent_future"].unsqueeze(0)
        min_dist = float(min_distance_per_sample(traj, ctrl)[0])
        hit_rate = float(bbox_collision_rate_per_sample(traj, ctrl, ego_extent, ctrl_extent)[0])
        min_dists.append(min_dist)
        hit_rates.append(hit_rate)
        examples.append(
            {
                "scene_key": sample["scene_key"],
                "timestep": int(sample["timestep"]),
                "target_source": sample["target_source"],
                "min_dist": min_dist,
                "dangerous_hit": hit_rate,
            }
        )
    return {
        "mean_pred_min_dist": sum(min_dists) / len(min_dists),
        "dangerous_hit_rate": sum(hit_rates) / len(hit_rates),
        "examples": examples,
        "available": True,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SafeSimConfig(hdf5_paths=args.hdf5_paths, training=False)
    raw_dataset = SafeSimTemporalDataset(config, split="audit", target_policy="raw_gt")
    action_dataset = SafeSimTemporalDataset(config, split="audit", target_policy="action")
    sample_summary = {
        "available": False,
        "error": None,
        "mean_pred_min_dist": None,
        "dangerous_hit_rate": None,
        "examples": [],
    }
    sample_dataset = SafeSimTemporalDataset(config, split="audit", target_policy="nearest_action_sample")

    rng = random.Random(args.seed)
    total = len(raw_dataset)
    indices = sorted(rng.sample(range(total), min(args.num_samples, total)))

    raw_summary = evaluate_policy(raw_dataset, indices)
    action_summary = evaluate_policy(action_dataset, indices)
    try:
        sample_summary = evaluate_policy(sample_dataset, indices)
    except RuntimeError as exc:
        sample_summary["error"] = str(exc)

    raw_dist = raw_summary["mean_pred_min_dist"]
    raw_hit = raw_summary["dangerous_hit_rate"]
    action_dist = action_summary["mean_pred_min_dist"]
    action_hit = action_summary["dangerous_hit_rate"]

    recommend_action = action_hit > raw_hit + 0.10 or action_dist < raw_dist * 0.90
    recommended_policy = "action" if recommend_action else "raw_gt"

    payload = {
        "num_samples": len(indices),
        "indices": indices,
        "raw_gt": raw_summary,
        "action": action_summary,
        "nearest_action_sample": sample_summary,
        "recommended_policy": recommended_policy,
        "decision_rule": {
            "action_if_hit_rate_margin_gt": 0.10,
            "action_if_mean_min_dist_ratio_lt": 0.90,
        },
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# SafeSim Filtered Target Audit",
        "",
        f"- Samples audited: {len(indices)}",
        f"- Recommended policy: `{recommended_policy}`",
        "",
        "| Policy | Mean Min Dist (m) | Dangerous Hit Rate |",
        "|---|---:|---:|",
        f"| raw_gt | {raw_summary['mean_pred_min_dist']:.4f} | {raw_summary['dangerous_hit_rate']:.4f} |",
        f"| action | {action_summary['mean_pred_min_dist']:.4f} | {action_summary['dangerous_hit_rate']:.4f} |",
        (
            f"| nearest_action_sample | {sample_summary['mean_pred_min_dist']:.4f} | "
            f"{sample_summary['dangerous_hit_rate']:.4f} |"
            if sample_summary["available"]
            else f"| nearest_action_sample | N/A | N/A |"
        ),
        "",
        "Decision:",
        f"- Recommend `{recommended_policy}` for Stage 2 supervision.",
    ]
    if not sample_summary["available"] and sample_summary["error"]:
        lines.extend([
            "",
            "Nearest-action-sample status:",
            f"- Invalid in current dataset: `{sample_summary['error']}`",
        ])
    (output_dir / "audit_summary.md").write_text("\n".join(lines))
    print(f"Saved audit report to {output_dir}")


if __name__ == "__main__":
    main()
