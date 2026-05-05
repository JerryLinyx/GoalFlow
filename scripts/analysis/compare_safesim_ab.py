#!/usr/bin/env python3
"""
Compare two SafeSim training runs and emit a compact markdown summary.

Usage:
  python scripts/analysis/compare_safesim_ab.py \
    --baseline safesim_logs_ab_history_baseline \
    --variant safesim_logs_ab_history_5hz \
    --output outputs/history_ab/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_metrics(metrics_path: Path) -> List[Dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _find_latest_metrics_path(run_dir: Path) -> Path:
    csv_root = run_dir / "csv_logs"
    candidates = sorted(csv_root.glob("version_*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found under {csv_root}")
    return candidates[-1]


def _best_epoch_row(rows: List[Dict[str, str]], metric: str) -> Dict[str, str]:
    best_row = None
    best_val = None
    for row in rows:
        current = _parse_float(row.get(metric, ""))
        if current is None:
            continue
        if best_val is None or current > best_val:
            best_val = current
            best_row = row
    if best_row is None:
        raise ValueError(f"No rows contain metric '{metric}'.")
    return best_row


def _load_run(run_dir: Path) -> Tuple[Dict[str, str], Dict]:
    metrics_path = _find_latest_metrics_path(run_dir)
    split_path = run_dir / "split_summary.json"
    rows = _read_metrics(metrics_path)
    best = _best_epoch_row(rows, "val_primary_metric")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    return best, split_payload


def _metric(row: Dict[str, str], key: str) -> str:
    value = _parse_float(row.get(key, ""))
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    variant_dir = Path(args.variant)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_best, baseline_split = _load_run(baseline_dir)
    variant_best, variant_split = _load_run(variant_dir)

    lines = [
        "# SafeSim History A/B Summary",
        "",
        "## Protocol",
        "",
        f"- Baseline run dir: `{baseline_dir}`",
        f"- Variant run dir: `{variant_dir}`",
        f"- Baseline history: len=4, stride=5",
        f"- Variant history: len=8, stride=2",
        f"- Split unit: `{baseline_split['split_unit']}`",
        f"- Seed: `{baseline_split['seed']}`",
        "",
        "## Best Epoch Comparison",
        "",
        "| Metric | Baseline | Variant |",
        "| --- | ---: | ---: |",
        f"| Epoch | {baseline_best['epoch']} | {variant_best['epoch']} |",
        f"| val_primary_metric | {_metric(baseline_best, 'val_primary_metric')} | {_metric(variant_best, 'val_primary_metric')} |",
        f"| val_bbox_collision_rate | {_metric(baseline_best, 'val_bbox_collision_rate')} | {_metric(variant_best, 'val_bbox_collision_rate')} |",
        f"| val_hit_2m | {_metric(baseline_best, 'val_hit_2m')} | {_metric(variant_best, 'val_hit_2m')} |",
        f"| val_hit_4m | {_metric(baseline_best, 'val_hit_4m')} | {_metric(variant_best, 'val_hit_4m')} |",
        f"| val_pred_better_than_gt_rate | {_metric(baseline_best, 'val_pred_better_than_gt_rate')} | {_metric(variant_best, 'val_pred_better_than_gt_rate')} |",
        f"| val_random_better_than_gt_rate | {_metric(baseline_best, 'val_random_better_than_gt_rate')} | {_metric(variant_best, 'val_random_better_than_gt_rate')} |",
        f"| val_pred_min_dist | {_metric(baseline_best, 'val_pred_min_dist')} | {_metric(variant_best, 'val_pred_min_dist')} |",
        f"| val_random_min_dist | {_metric(baseline_best, 'val_random_min_dist')} | {_metric(variant_best, 'val_random_min_dist')} |",
        f"| val_candidate_mean_min_dist | {_metric(baseline_best, 'val_candidate_mean_min_dist')} | {_metric(variant_best, 'val_candidate_mean_min_dist')} |",
        f"| val_loss | {_metric(baseline_best, 'val_loss')} | {_metric(variant_best, 'val_loss')} |",
        "",
        "## Notes",
        "",
        "- `val_primary_metric` is the current checkpoint-selection metric.",
        "- `selected` and `random` metrics should be read together to detect nearest-selector artifacts.",
        "- Case-stratified analysis should still be checked in the raw `metrics.csv` before drawing final conclusions.",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
