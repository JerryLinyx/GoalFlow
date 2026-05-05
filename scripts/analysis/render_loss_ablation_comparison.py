"""
Render scene-aligned qualitative comparisons across five SafeSim loss ablation methods.

Each selected scene is shown as one row with five columns:
  baseline, A0 imitation, A1 terminal only, A2 softmin only, A3 terminal+softmin.

The script automatically selects three representative scenes:
  1. collapse_recovery: A0 collapses near the origin while A1/A3 recover visible motion
  2. balanced_improvement: A1 provides cleaner motion with improved dangerous metrics
  3. aggressive_softmin: A2/A3 become much more dangerous but also more aggressive
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.goalflow.safesim_config import SafeSimConfig
from scripts.analysis.evaluate_safesim_dangerous import (
    build_val_loader,
    evaluate_agent,
    load_agent,
    resolve_device,
)


METHODS = [
    ("baseline", Path("safesim_logs_cfg_base/checkpoints/safesim-54-0.0318.ckpt")),
    ("A0_imitation", Path("safesim_logs_stage2/checkpoints/best-val-17-0.0033.ckpt")),
    ("A1_terminal_only", Path("safesim_logs_stage2_terminal_only/checkpoints/safesim-07-0.0928.ckpt")),
    ("A2_ctrl_softmin_only", Path("safesim_logs_stage2_ctrl_softmin/checkpoints/safesim-05-0.3295.ckpt")),
    ("A3_terminal_plus_softmin", Path("safesim_logs_stage2_terminal_softmin/checkpoints/best-val-12-0.3437.ckpt")),
]


def make_args(max_val_samples: int):
    return SimpleNamespace(
        hdf5_paths=[
            "safesim/case1_filtered/data.hdf5",
            "safesim/case2_filtered/data.hdf5",
            "safesim/case3_filtered/data.hdf5",
            "safesim/case4_filtered/data.hdf5",
            "safesim/case5_filtered/data.hdf5",
        ],
        batch_size=4,
        val_split=0.1,
        seed=0,
        anchor_size=16,
        infer_steps=25,
        tf_d_model=256,
        target_policy="nearest_action_sample",
        x_scale=60.0,
        y_scale=15.0,
        topk_metrics=6,
        base_dt=0.1,
        max_val_samples=max_val_samples,
        cfg_scales=[1.0],
        num_examples=3,
        init_mode="none",
        init_checkpoint="",
        model_checkpoint="",
        model_label="",
        checkpoint_dir="",
        output_dir="",
        bootstrap_samples=200,
    )


def pred_extent(pred: np.ndarray) -> float:
    steps = np.linalg.norm(np.diff(pred[:, :2], axis=0), axis=-1)
    return float(steps.sum())


def collect_records(max_val_samples: int):
    args = make_args(max_val_samples)
    loader, eval_config = build_val_loader(args)
    device = resolve_device()
    combined = {}

    for label, checkpoint_path in METHODS:
        agent = load_agent(checkpoint_path, args, 1.0, device)
        _, render_records = evaluate_agent(
            agent,
            loader,
            args,
            eval_config=SafeSimConfig(
                hdf5_paths=[],
                training=False,
                anchor_size=args.anchor_size,
                infer_steps=args.infer_steps,
                cfg_scale=1.0,
                tf_d_model=agent._config.tf_d_model,
                target_policy=args.target_policy,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
                future_stride=eval_config.future_stride,
            ),
            checkpoint_label=label,
        )
        for row in render_records:
            key = (row["scene_key"], int(row["timestep"]))
            entry = combined.setdefault(
                key,
                {
                    "scene_key": row["scene_key"],
                    "timestep": int(row["timestep"]),
                    "case_id": int(row["case_id"]),
                    "raw_gt": row["raw_gt"],
                    "target": row["target"],
                    "ctrl": row["ctrl"],
                    "history": row["history"],
                    "methods": {},
                },
            )
            row_copy = dict(row)
            row_copy["pred_extent"] = pred_extent(row["pred"].numpy())
            entry["methods"][label] = row_copy
    return combined


def valid_entries(combined):
    required = {name for name, _ in METHODS}
    entries = []
    for entry in combined.values():
        if required.issubset(entry["methods"].keys()):
            entries.append(entry)
    return entries


def choose_examples(entries):
    used = set()
    picked = []

    def pick_best(name, score_fn):
        candidates = []
        for entry in entries:
            if entry["scene_key"] in used:
                continue
            methods = entry["methods"]
            candidates.append((score_fn(methods), entry))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            score, entry = candidates[0]
            used.add(entry["scene_key"])
            picked.append((name, score, entry))

    def collapse_score(m):
        return (
            max(0.0, 1.0 - m["A0_imitation"]["pred_extent"]) * 50.0
            + m["A1_terminal_only"]["pred_extent"] * 2.0
            + m["A3_terminal_plus_softmin"]["dangerous_hit_rate"] * 10.0
            - m["A3_terminal_plus_softmin"]["offroad_rate"] * 5.0
        )

    def balanced_score(m):
        return (
            m["A1_terminal_only"]["dangerous_hit_rate"] * 20.0
            + (1.0 / (1.0 + m["A1_terminal_only"]["pred_min_dist"])) * 20.0
            + m["A1_terminal_only"]["pred_extent"]
            - m["A1_terminal_only"]["offroad_rate"] * 10.0
            - m["A1_terminal_only"]["mean_jerk"] * 0.2
        )

    def aggressive_score(m):
        return (
            m["A2_ctrl_softmin_only"]["dangerous_hit_rate"] * 20.0
            + m["A3_terminal_plus_softmin"]["dangerous_hit_rate"] * 15.0
            - m["A1_terminal_only"]["dangerous_hit_rate"] * 10.0
            + m["A2_ctrl_softmin_only"]["pred_extent"]
            + m["A3_terminal_plus_softmin"]["pred_extent"]
        )

    pick_best("collapse_recovery", collapse_score)
    pick_best("balanced_terminal", balanced_score)
    pick_best("aggressive_softmin", aggressive_score)
    return picked


def render_rows(selected, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_order = [name for name, _ in METHODS]
    method_titles = {
        "baseline": "Baseline",
        "A0_imitation": "A0 Imitation",
        "A1_terminal_only": "A1 Terminal",
        "A2_ctrl_softmin_only": "A2 Softmin",
        "A3_terminal_plus_softmin": "A3 Term+Soft",
    }

    rows = len(selected)
    cols = len(method_order)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.6 * rows))
    axes = np.atleast_2d(axes)

    for row_idx, (category, _score, entry) in enumerate(selected):
        for col_idx, method in enumerate(method_order):
            ax = axes[row_idx, col_idx]
            rec = entry["methods"][method]
            history = entry["history"].numpy()
            pred = rec["pred"].numpy()
            raw_gt = entry["raw_gt"].numpy()
            target = entry["target"].numpy()
            ctrl = entry["ctrl"].numpy()

            ax.plot(history[:, 0], history[:, 1], color="#7f7f7f", linestyle="--", linewidth=1.6, marker="s", markersize=2.5, zorder=1)
            ax.scatter([0], [0], color="black", s=16, zorder=5)
            ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=2.2, marker="o", markersize=2.5, zorder=4)
            ax.plot(raw_gt[:, 0], raw_gt[:, 1], color="#2ca02c", linewidth=1.6, linestyle="--", zorder=2)
            ax.plot(target[:, 0], target[:, 1], color="#9467bd", linewidth=1.4, linestyle=":", zorder=3)
            ax.plot(ctrl[:, 0], ctrl[:, 1], color="#1f77b4", linewidth=1.6, marker="^", markersize=2.5, zorder=2)

            ax.set_title(
                f"{method_titles[method]}\nmin={rec['pred_min_dist']:.2f} "
                f"hit={int(rec['dangerous_hit_rate'])} ext={rec['pred_extent']:.1f}",
                fontsize=9,
            )
            ax.grid(alpha=0.25)
            ax.axis("equal")

            if col_idx == 0:
                ax.set_ylabel(
                    f"{category}\n{entry['scene_key']}\ncase={entry['case_id']} t={entry['timestep']}",
                    fontsize=9,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color="#7f7f7f", linestyle="--", marker="s", label="History"),
            mlines.Line2D([], [], color="#d62728", marker="o", label="Pred"),
            mlines.Line2D([], [], color="#2ca02c", linestyle="--", label="Raw GT"),
            mlines.Line2D([], [], color="#9467bd", linestyle=":", label="Target"),
            mlines.Line2D([], [], color="#1f77b4", marker="^", label="Ctrl"),
        ]
        labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.suptitle("Three Additional Scene-Aligned Comparisons Across Five Methods", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    out_dir = Path("outputs/group_meeting_2026_04_30")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = collect_records(max_val_samples=128)
    entries = valid_entries(combined)
    selected = choose_examples(entries)

    render_rows(selected, out_dir / "loss_ablation_three_more_groups.png")

    summary = {
        "num_entries_with_all_methods": len(entries),
        "selected": [
            {
                "category": category,
                "score": float(score),
                "scene_key": entry["scene_key"],
                "timestep": entry["timestep"],
                "case_id": entry["case_id"],
                "methods": {
                    method: {
                        "pred_min_dist": float(entry["methods"][method]["pred_min_dist"]),
                        "dangerous_hit_rate": float(entry["methods"][method]["dangerous_hit_rate"]),
                        "pred_extent": float(entry["methods"][method]["pred_extent"]),
                        "offroad_rate": float(entry["methods"][method]["offroad_rate"]),
                        "mean_jerk": float(entry["methods"][method]["mean_jerk"]),
                    }
                    for method, _ in METHODS
                },
            }
            for category, score, entry in selected
        ],
    }
    (out_dir / "loss_ablation_three_more_groups.json").write_text(json.dumps(summary, indent=2))
    print(out_dir / "loss_ablation_three_more_groups.png")
    print(out_dir / "loss_ablation_three_more_groups.json")


if __name__ == "__main__":
    main()
