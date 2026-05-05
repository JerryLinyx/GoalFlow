#!/usr/bin/env python3
"""
Protocolized offline dangerous evaluation for SafeSim checkpoints.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split

from navsim.agents.goalflow.safesim_agent import SafeSimAgent
from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_dataset import SafeSimTemporalDataset, safesim_collate_fn
from navsim.agents.goalflow.safesim_metrics import (
    ade_per_sample,
    bbox_collision_rate_per_sample,
    better_than_gt_rate,
    bootstrap_mean_confidence_interval,
    candidate_min_dist_spread_per_sample,
    candidate_trajectory_error_metrics,
    candidate_xy_std_per_sample,
    dangerous_hit_rate_per_sample,
    fde_per_sample,
    first_step_heading_error_per_sample,
    first_step_speed_error_per_sample,
    hit_rate_from_min_dist,
    low_motion_rate_per_sample,
    min_distance_per_sample,
    miss_rate_from_fde,
    non_drivable_occupancy_rate_per_sample,
    offroad_rate_per_sample,
    trajectory_kinematics_stats,
    wilson_confidence_interval,
)


GATE_THRESHOLDS = {
    "low_motion_rate": 0.05,
    "mean_accel": 6.0,
    "mean_jerk": 15.0,
    "max_jerk": 40.0,
    "first_step_speed_error": 5.0,
    "offroad_rate": 0.10,
}

TIER_A_CASE_METRICS = [
    "dangerous_hit_rate",
    "hit@2m",
    "hit@4m",
    "pred_better_than_gt_rate",
    "pred_min_dist",
]

BINARY_CI_METRICS = [
    "dangerous_hit_rate",
    "hit@2m",
    "hit@4m",
    "pred_better_than_gt_rate",
    "low_motion_rate",
    "offroad_rate",
]

CONTINUOUS_CI_METRICS = [
    "pred_min_dist",
    "ADE_vs_target",
    "FDE_vs_target",
    "deviation_ADE_vs_raw_gt",
    "deviation_FDE_vs_raw_gt",
    "mean_jerk",
    "max_jerk",
    "mean_accel",
    "first_step_speed_error",
    "first_step_heading_error",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cfg_scales", nargs="+", type=float, default=[1.0, 1.5])
    parser.add_argument("--anchor_size", type=int, default=64)
    parser.add_argument("--infer_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--tf_d_model", type=int, default=1024)
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--model_checkpoint", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument(
        "--init_mode",
        type=str,
        default="none",
        choices=["none", "fm_head_conservative", "fm_head_extended"],
    )
    parser.add_argument("--model_label", type=str, default="")
    parser.add_argument(
        "--target_policy",
        type=str,
        default="raw_gt",
        choices=["raw_gt", "action", "nearest_action_sample"],
    )
    parser.add_argument("--topk_metrics", type=int, default=6)
    parser.add_argument("--base_dt", type=float, default=0.1)
    parser.add_argument("--x_scale", type=float, default=60.0)
    parser.add_argument("--y_scale", type=float, default=15.0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    return parser.parse_args()


def build_val_loader(args):
    config = SafeSimConfig(
        hdf5_paths=args.hdf5_paths,
        training=False,
        anchor_size=args.anchor_size,
        infer_steps=args.infer_steps,
        tf_d_model=args.tf_d_model,
        target_policy=args.target_policy,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
    )
    dataset = SafeSimTemporalDataset(config, split="eval", target_policy=args.target_policy)
    dataset_size = len(dataset)
    val_size = max(1, int(round(dataset_size * args.val_split)))
    train_size = max(1, dataset_size - val_size)
    generator = torch.Generator().manual_seed(args.seed)
    _, val_dataset = random_split(dataset, [train_size, dataset_size - train_size], generator=generator)
    if args.max_val_samples > 0:
        val_count = min(args.max_val_samples, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(val_count)))
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=safesim_collate_fn,
    )
    return loader, config


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_tf_d_model_from_checkpoint(checkpoint_payload) -> int | None:
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    if "model.scene_encoder.fusion.cls_token" in state_dict:
        return int(state_dict["model.scene_encoder.fusion.cls_token"].shape[-1])
    if "model.trajectory_encoder.weight" in state_dict:
        return int(state_dict["model.trajectory_encoder.weight"].shape[0])
    return None


def infer_use_goal_condition_from_checkpoint(checkpoint_payload) -> bool:
    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    if "model.scene_encoder.fusion.goal_encoder.mlp.0.weight" in state_dict:
        return True
    type_key = "model.scene_encoder.fusion.type_embedding.weight"
    if type_key in state_dict:
        return int(state_dict[type_key].shape[0]) >= 4
    return False


def load_agent(checkpoint_path: Path | None, args, cfg_scale: float, device):
    inferred_tf_d_model = args.tf_d_model
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        inferred = infer_tf_d_model_from_checkpoint(checkpoint)
        if inferred is not None:
            inferred_tf_d_model = inferred
    elif args.model_checkpoint:
        checkpoint = torch.load(str(args.model_checkpoint), map_location="cpu", weights_only=False)
        inferred = infer_tf_d_model_from_checkpoint(checkpoint)
        if inferred is not None:
            inferred_tf_d_model = inferred

    inferred_use_goal_condition = infer_use_goal_condition_from_checkpoint(checkpoint) if checkpoint is not None else False

    config = SafeSimConfig(
        hdf5_paths=[],
        training=False,
        anchor_size=args.anchor_size,
        infer_steps=args.infer_steps,
        cfg_scale=cfg_scale,
        tf_d_model=inferred_tf_d_model,
        use_goal_condition=inferred_use_goal_condition,
        target_policy=args.target_policy,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
    )
    agent = SafeSimAgent(config)
    if checkpoint_path is not None:
        agent.load_state_dict(checkpoint["state_dict"], strict=True)
    elif args.model_checkpoint:
        agent.load_state_dict(checkpoint["state_dict"], strict=True)
    elif args.init_mode != "none":
        if not args.init_checkpoint:
            raise ValueError("--init_checkpoint is required when --init_mode is enabled")
        transfer_report = agent.model.load_goalflow_fm_head(args.init_checkpoint, args.init_mode)
        if not transfer_report["loaded_modules"]:
            raise RuntimeError(
                f"Transfer init_mode={args.init_mode} loaded no modules from {args.init_checkpoint}"
            )
    agent.to(device).eval()
    agent._config.training = False
    agent.model._config.training = False
    return agent


def _safe_mean(values: Iterable[float]):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return None
    if np.isnan(values).all():
        return None
    return float(np.nanmean(values))


def _binary_metric_with_ci(df: pd.DataFrame, metric_name: str):
    valid = df[metric_name].dropna()
    if valid.empty:
        return None, None
    successes = int(valid.sum())
    total = int(valid.shape[0])
    return float(valid.mean()), wilson_confidence_interval(successes, total)


def _continuous_metric_with_ci(df: pd.DataFrame, metric_name: str, bootstrap_samples: int, seed: int):
    valid = df[metric_name].dropna().to_numpy()
    if valid.size == 0:
        return None, None
    mean = float(valid.mean())
    ci = bootstrap_mean_confidence_interval(
        valid,
        num_bootstrap=bootstrap_samples,
        confidence=0.95,
        seed=seed,
    )
    return mean, ci


def _sample_plausibility_flags(sample_row: Dict[str, float]) -> Dict[str, bool]:
    return {
        "low_motion_ok": sample_row["low_motion_rate"] <= GATE_THRESHOLDS["low_motion_rate"],
        "mean_accel_ok": sample_row["mean_accel"] <= GATE_THRESHOLDS["mean_accel"],
        "mean_jerk_ok": sample_row["mean_jerk"] <= GATE_THRESHOLDS["mean_jerk"],
        "max_jerk_ok": sample_row["max_jerk"] <= GATE_THRESHOLDS["max_jerk"],
        "first_step_speed_ok": sample_row["first_step_speed_error"] <= GATE_THRESHOLDS["first_step_speed_error"],
        "offroad_ok": sample_row["offroad_rate"] <= GATE_THRESHOLDS["offroad_rate"],
    }


def _sample_passes_plausibility(sample_row: Dict[str, float]) -> bool:
    flags = _sample_plausibility_flags(sample_row)
    return all(flags.values())


def evaluate_agent(agent: SafeSimAgent, loader, args, eval_config: SafeSimConfig, checkpoint_label: str):
    device = next(agent.parameters()).device
    dt_seconds = eval_config.future_stride * args.base_dt

    sample_rows: List[Dict[str, float]] = []
    render_records: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            outputs = agent.forward(batch)
            pred = outputs["trajectory"]
            random_pred = outputs["random_trajectory"]
            candidates = outputs["trajectory_candidates"]
            raw_gt = batch["future_trajectory"]
            target = batch["training_target_trajectory"]
            ctrl = batch["ctrl_future"]
            ego_extent = batch["ego_extent_future"]
            ctrl_extent = batch["ctrl_extent_future"]
            drivable_map = batch["drivable_map"]
            history_xy = batch["agent_history"][:, 0, :, :2]

            pred_min = min_distance_per_sample(pred, ctrl)
            random_min = min_distance_per_sample(random_pred, ctrl)
            gt_min = min_distance_per_sample(raw_gt, ctrl)

            dangerous_hit = dangerous_hit_rate_per_sample(pred, ctrl, ego_extent, ctrl_extent)
            random_dangerous_hit = bbox_collision_rate_per_sample(random_pred, ctrl, ego_extent, ctrl_extent)
            hit2 = hit_rate_from_min_dist(pred_min, 2.0)
            hit4 = hit_rate_from_min_dist(pred_min, 4.0)

            raw_gt_better = better_than_gt_rate(pred_min, gt_min)
            ade_target = ade_per_sample(pred, target)
            fde_target = fde_per_sample(pred, target)
            ade_raw_gt = ade_per_sample(pred, raw_gt)
            fde_raw_gt = fde_per_sample(pred, raw_gt)

            candidate_target_metrics = candidate_trajectory_error_metrics(candidates, target, k=args.topk_metrics)
            miss2_target = miss_rate_from_fde(fde_target, 2.0)
            miss4_target = miss_rate_from_fde(fde_target, 4.0)

            kinematics = trajectory_kinematics_stats(pred, dt_seconds)
            low_motion = low_motion_rate_per_sample(pred)
            first_speed_err = first_step_speed_error_per_sample(pred, history_xy, dt_seconds)
            first_heading_err = first_step_heading_error_per_sample(pred, history_xy)
            offroad = offroad_rate_per_sample(pred, drivable_map, x_scale=args.x_scale, y_scale=args.y_scale)
            non_drivable_occupancy = non_drivable_occupancy_rate_per_sample(
                pred,
                drivable_map,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
            )
            candidate_spread = candidate_min_dist_spread_per_sample(candidates, ctrl)
            candidate_xy_std = candidate_xy_std_per_sample(candidates, k=args.topk_metrics)

            target_sources = batch["target_source"]
            for idx in range(pred.shape[0]):
                target_source = target_sources[idx]
                pred_better = float(raw_gt_better[idx]) if target_source != "raw_gt" else np.nan
                row = {
                    "checkpoint": checkpoint_label,
                    "cfg_scale": float(eval_config.cfg_scale),
                    "scene_key": batch["scene_key"][idx],
                    "timestep": int(batch["timestep"][idx]),
                    "case_id": int(batch["case_id"][idx]),
                    "target_policy": args.target_policy,
                    "target_source": target_source,
                    "dangerous_hit_rate": float(dangerous_hit[idx]),
                    "random_dangerous_hit_rate": float(random_dangerous_hit[idx]),
                    "hit@2m": float(hit2[idx]),
                    "hit@4m": float(hit4[idx]),
                    "pred_min_dist": float(pred_min[idx]),
                    "random_min_dist": float(random_min[idx]),
                    "pred_better_than_gt_rate": pred_better,
                    "ADE_vs_target": float(ade_target[idx]),
                    "FDE_vs_target": float(fde_target[idx]),
                    "deviation_ADE_vs_raw_gt": float(ade_raw_gt[idx]),
                    "deviation_FDE_vs_raw_gt": float(fde_raw_gt[idx]),
                    "minADE@6_vs_target": float(candidate_target_metrics["minADE"][idx]),
                    "minFDE@6_vs_target": float(candidate_target_metrics["minFDE"][idx]),
                    "meanADE@6_vs_target": float(candidate_target_metrics["meanADE"][idx]),
                    "meanFDE@6_vs_target": float(candidate_target_metrics["meanFDE"][idx]),
                    "candidate_ADE_std@6": float(candidate_target_metrics["ade_std"][idx]),
                    "MissRate@2m_vs_target": float(miss2_target[idx]),
                    "MissRate@4m_vs_target": float(miss4_target[idx]),
                    "mean_speed": float(kinematics["mean_speed"][idx]),
                    "mean_accel": float(kinematics["mean_accel"][idx]),
                    "mean_jerk": float(kinematics["mean_jerk"][idx]),
                    "max_jerk": float(kinematics["max_jerk"][idx]),
                    "first_step_speed_error": float(first_speed_err[idx]),
                    "first_step_heading_error": float(first_heading_err[idx]),
                    "low_motion_rate": float(low_motion[idx]),
                    "offroad_rate": float(offroad[idx]),
                    "non_drivable_occupancy_rate": float(non_drivable_occupancy[idx]),
                    "candidate_min_dist_spread": float(candidate_spread[idx]),
                    "candidate_xy_std": float(candidate_xy_std[idx]),
                }
                sample_rows.append(row)

                render_row = dict(row)
                render_row.update(
                    {
                        "history": history_xy[idx].detach().cpu(),
                        "pred": pred[idx].detach().cpu(),
                        "raw_gt": raw_gt[idx].detach().cpu(),
                        "target": target[idx].detach().cpu(),
                        "ctrl": ctrl[idx].detach().cpu(),
                    }
                )
                render_records.append(render_row)

    return sample_rows, render_records


def aggregate_case_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    case_rows = []
    grouped = sample_df.groupby(["checkpoint", "cfg_scale", "case_id"], sort=True)
    for (checkpoint, cfg_scale, case_id), group in grouped:
        row = {
            "checkpoint": checkpoint,
            "cfg_scale": cfg_scale,
            "case_id": int(case_id),
            "count": int(group.shape[0]),
        }
        for metric in TIER_A_CASE_METRICS:
            row[metric] = _safe_mean(group[metric])
        case_rows.append(row)
    return pd.DataFrame(case_rows)


def attach_case_summary(global_df: pd.DataFrame, per_case_df: pd.DataFrame) -> pd.DataFrame:
    if per_case_df.empty:
        return global_df
    output_rows = []
    grouped = per_case_df.groupby(["checkpoint", "cfg_scale"], sort=False)
    for _, global_row in global_df.iterrows():
        row = dict(global_row)
        key = (global_row["checkpoint"], global_row["cfg_scale"])
        if key in grouped.groups:
            group = grouped.get_group(key)
            for metric in TIER_A_CASE_METRICS:
                values = group[metric].dropna().to_numpy(dtype=np.float64)
                if values.size == 0:
                    row[f"case_mean_{metric}"] = None
                    row[f"case_std_{metric}"] = None
                    row[f"case_gap_{metric}"] = None
                    row[f"case_min_{metric}"] = None
                else:
                    row[f"case_mean_{metric}"] = float(values.mean())
                    row[f"case_std_{metric}"] = float(values.std())
                    row[f"case_gap_{metric}"] = float(values.max() - values.min())
                    row[f"case_min_{metric}"] = float(values.min())
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def evaluate_result_row(sample_df: pd.DataFrame, checkpoint: str, cfg_scale: float, args) -> Tuple[Dict, Dict]:
    row_df = sample_df[(sample_df["checkpoint"] == checkpoint) & (sample_df["cfg_scale"] == cfg_scale)]
    summary = {
        "checkpoint": checkpoint,
        "cfg_scale": float(cfg_scale),
        "target_policy": args.target_policy,
        "count": int(row_df.shape[0]),
    }

    # Tier A
    for metric in [
        "dangerous_hit_rate",
        "random_dangerous_hit_rate",
        "hit@2m",
        "hit@4m",
        "pred_min_dist",
        "random_min_dist",
        "pred_better_than_gt_rate",
    ]:
        summary[metric] = _safe_mean(row_df[metric])

    # Tier B
    for metric in [
        "ADE_vs_target",
        "FDE_vs_target",
        "deviation_ADE_vs_raw_gt",
        "deviation_FDE_vs_raw_gt",
        "minADE@6_vs_target",
        "minFDE@6_vs_target",
        "meanADE@6_vs_target",
        "meanFDE@6_vs_target",
        "candidate_ADE_std@6",
        "MissRate@2m_vs_target",
        "MissRate@4m_vs_target",
    ]:
        summary[metric] = _safe_mean(row_df[metric])

    # Tier C/D/E
    for metric in [
        "mean_speed",
        "mean_accel",
        "mean_jerk",
        "max_jerk",
        "first_step_speed_error",
        "first_step_heading_error",
        "low_motion_rate",
        "offroad_rate",
        "non_drivable_occupancy_rate",
        "candidate_min_dist_spread",
        "candidate_xy_std",
    ]:
        summary[metric] = _safe_mean(row_df[metric])

    gate_pass = (
        (summary["low_motion_rate"] is not None and summary["low_motion_rate"] <= GATE_THRESHOLDS["low_motion_rate"])
        and (summary["mean_accel"] is not None and summary["mean_accel"] <= GATE_THRESHOLDS["mean_accel"])
        and (summary["mean_jerk"] is not None and summary["mean_jerk"] <= GATE_THRESHOLDS["mean_jerk"])
        and (summary["max_jerk"] is not None and summary["max_jerk"] <= GATE_THRESHOLDS["max_jerk"])
        and (
            summary["first_step_speed_error"] is not None
            and summary["first_step_speed_error"] <= GATE_THRESHOLDS["first_step_speed_error"]
        )
        and (summary["offroad_rate"] is not None and summary["offroad_rate"] <= GATE_THRESHOLDS["offroad_rate"])
    )
    summary["gate_pass"] = gate_pass

    confidence = {}
    for metric in BINARY_CI_METRICS:
        mean_value, ci = _binary_metric_with_ci(row_df, metric)
        confidence[metric] = {
            "mean": mean_value,
            "ci95": list(ci) if ci is not None else None,
        }
    for metric in CONTINUOUS_CI_METRICS:
        mean_value, ci = _continuous_metric_with_ci(row_df, metric, args.bootstrap_samples, args.seed)
        confidence[metric] = {
            "mean": mean_value,
            "ci95": list(ci) if ci is not None else None,
        }

    return summary, confidence


def rank_results(global_df: pd.DataFrame) -> Tuple[pd.Series, str]:
    sortable = global_df.copy()
    sortable["_pred_better_sort"] = sortable["pred_better_than_gt_rate"].fillna(-1.0)
    valid = sortable[sortable["gate_pass"]]
    source = valid if not valid.empty else sortable
    status = "valid" if not valid.empty else "no_valid_checkpoint"
    best = source.sort_values(
        by=["dangerous_hit_rate", "hit@2m", "pred_min_dist", "_pred_better_sort"],
        ascending=[False, False, True, False],
    ).iloc[0]
    return best, status


def render_examples(examples: List[Dict], output_path: Path, title: str):
    if not examples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[len(examples):]:
        ax.axis("off")

    for ax, ex in zip(axes, examples):
        history = ex["history"].numpy()
        pred = ex["pred"].numpy()
        raw_gt = ex["raw_gt"].numpy()
        target = ex["target"].numpy()
        ctrl = ex["ctrl"].numpy()

        ax.plot(history[:, 0], history[:, 1], color="#7f7f7f", linestyle="--", linewidth=1.8, marker="s", markersize=3, zorder=1, label="History")
        ax.scatter([0], [0], color="black", s=20, zorder=6, label="Ego Start")
        ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=2.4, marker="o", markersize=3, zorder=5, label="Pred")
        ax.plot(raw_gt[:, 0], raw_gt[:, 1], color="#2ca02c", linewidth=1.8, linestyle="--", zorder=3, label="Raw GT")
        if ex["target_source"] != "raw_gt":
            ax.plot(target[:, 0], target[:, 1], color="#9467bd", linewidth=1.8, linestyle=":", zorder=4, label="Target")
        ax.plot(ctrl[:, 0], ctrl[:, 1], color="#1f77b4", linewidth=1.8, marker="^", markersize=3, zorder=2, label="Ctrl")
        ax.set_title(
            f"{ex['scene_key']}\ncase={ex['case_id']} t={ex['timestep']} hit={int(ex['dangerous_hit_rate'])} "
            f"min={ex['pred_min_dist']:.2f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.axis("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def select_qualitative_examples(render_records: List[Dict], num_examples: int):
    success = []
    failure_safe = []
    failure_unphysical = []

    for row in render_records:
        plausible = _sample_passes_plausibility(row)
        if row["dangerous_hit_rate"] >= 0.5 and plausible:
            success.append(row)
        elif row["dangerous_hit_rate"] < 0.5 and plausible:
            failure_safe.append(row)
        elif (row["dangerous_hit_rate"] >= 0.5 or row["pred_min_dist"] <= 4.0) and not plausible:
            failure_unphysical.append(row)

    success.sort(key=lambda row: (row["pred_min_dist"], row["case_id"], row["scene_key"]))
    failure_safe.sort(key=lambda row: (row["pred_min_dist"], row["case_id"], row["scene_key"]))
    failure_unphysical.sort(
        key=lambda row: (
            -(row["low_motion_rate"] > GATE_THRESHOLDS["low_motion_rate"]),
            -(row["offroad_rate"] > GATE_THRESHOLDS["offroad_rate"]),
            -row["max_jerk"],
            row["pred_min_dist"],
        )
    )
    return {
        "success": success[:num_examples],
        "failure_safe": failure_safe[:num_examples],
        "failure_unphysical": failure_unphysical[:num_examples],
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    qualitative_dir = output_dir / "qualitative"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    loader, eval_config = build_val_loader(args)
    device = resolve_device()
    sample_rows: List[Dict] = []
    render_rows_by_run: Dict[Tuple[str, float], List[Dict]] = {}

    runs = []
    if args.checkpoint_dir:
        checkpoints = sorted(Path(args.checkpoint_dir).glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {args.checkpoint_dir}")
        for checkpoint_path in checkpoints:
            runs.append((Path(checkpoint_path).name, checkpoint_path))
    else:
        if not args.model_checkpoint and args.init_mode == "none":
            raise ValueError(
                "Provide --checkpoint_dir for trained checkpoints, or use --model_checkpoint / "
                "--init_checkpoint with --init_mode for single-model evaluation."
            )
        label = args.model_label or Path(args.model_checkpoint).name if args.model_checkpoint else f"{args.init_mode}:{Path(args.init_checkpoint).name}"
        runs.append((label, None))

    for cfg_scale in args.cfg_scales:
        for label, checkpoint_path in runs:
            agent = load_agent(checkpoint_path, args, cfg_scale, device)
            rows, render_records = evaluate_agent(agent, loader, args, eval_config=SafeSimConfig(
                hdf5_paths=[],
                training=False,
                anchor_size=args.anchor_size,
                infer_steps=args.infer_steps,
                cfg_scale=cfg_scale,
                tf_d_model=args.tf_d_model,
                target_policy=args.target_policy,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
                future_stride=eval_config.future_stride,
            ), checkpoint_label=label)
            sample_rows.extend(rows)
            render_rows_by_run[(label, float(cfg_scale))] = render_records

    sample_df = pd.DataFrame(sample_rows)
    if sample_df.empty:
        raise RuntimeError("No evaluation samples were produced")

    per_case_df = aggregate_case_rows(sample_df)
    global_rows = []
    confidence_payload = {}
    for (checkpoint, cfg_scale), _group in sample_df.groupby(["checkpoint", "cfg_scale"], sort=True):
        summary, confidence = evaluate_result_row(sample_df, checkpoint, cfg_scale, args)
        global_rows.append(summary)
        confidence_payload[f"{checkpoint}|cfg={cfg_scale:.1f}"] = confidence

    global_df = pd.DataFrame(global_rows)
    global_df = attach_case_summary(global_df, per_case_df)

    best_row, selection_status = rank_results(global_df)
    best_key = (best_row["checkpoint"], float(best_row["cfg_scale"]))
    best_render_rows = render_rows_by_run[best_key]
    selected_examples = select_qualitative_examples(best_render_rows, args.num_examples)

    for category, examples in selected_examples.items():
        category_dir = qualitative_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        render_examples(
            examples,
            category_dir / "examples.png",
            title=f"{category.replace('_', ' ').title()} @ {best_row['checkpoint']} cfg={best_row['cfg_scale']:.1f}",
        )

    global_df.to_csv(metrics_dir / "global.csv", index=False)
    per_case_df.to_csv(metrics_dir / "per_case.csv", index=False)
    (metrics_dir / "confidence.json").write_text(json.dumps(confidence_payload, indent=2))

    best_summary = {
        "selection_status": selection_status,
        "selection_rule": "gate_filter_then_dangerous_hit_rate",
        "gate_thresholds": GATE_THRESHOLDS,
        "best_global_metrics": best_row.to_dict(),
        "confidence": confidence_payload[f"{best_row['checkpoint']}|cfg={best_row['cfg_scale']:.1f}"],
        "qualitative_counts": {key: len(value) for key, value in selected_examples.items()},
    }
    (metrics_dir / "best_summary.json").write_text(json.dumps(best_summary, indent=2, default=str))

    summary_lines = [
        "# SafeSim Dangerous Metrics Protocol v1.1",
        "",
        f"- target_policy: `{args.target_policy}`",
        f"- selection_status: `{selection_status}`",
        f"- best checkpoint: `{best_row['checkpoint']}`",
        f"- best cfg_scale: `{best_row['cfg_scale']:.1f}`",
        f"- gate_pass: `{bool(best_row['gate_pass'])}`",
        "",
        "| checkpoint | cfg_scale | gate_pass | dangerous_hit_rate | hit@2m | hit@4m | pred_min_dist | pred_better_than_gt_rate | low_motion_rate | offroad_rate | mean_jerk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sortable = global_df.sort_values(by=["dangerous_hit_rate", "hit@2m", "pred_min_dist"], ascending=[False, False, True])
    for _, row in sortable.iterrows():
        pred_better = "N/A" if pd.isna(row["pred_better_than_gt_rate"]) else f"{row['pred_better_than_gt_rate']:.4f}"
        summary_lines.append(
            f"| {row['checkpoint']} | {row['cfg_scale']:.1f} | {int(bool(row['gate_pass']))} | "
            f"{row['dangerous_hit_rate']:.4f} | {row['hit@2m']:.4f} | {row['hit@4m']:.4f} | "
            f"{row['pred_min_dist']:.4f} | {pred_better} | {row['low_motion_rate']:.4f} | "
            f"{row['offroad_rate']:.4f} | {row['mean_jerk']:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    print(f"Saved protocolized evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
