import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navsim.agents.goalflow.safesim_metrics import (
    ade_per_sample,
    bbox_collision_rate_per_sample,
    candidate_min_dist_spread_per_sample,
    candidate_trajectory_error_metrics,
    candidate_xy_std_per_sample,
    first_step_heading_error_per_sample,
    first_step_speed_error_per_sample,
    fde_per_sample,
    low_motion_rate_per_sample,
    offroad_rate_per_sample,
    wilson_confidence_interval,
)


def test_ade_fde_per_sample():
    pred = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    ade = ade_per_sample(pred, target)
    fde = fde_per_sample(pred, target)
    assert torch.allclose(ade, torch.tensor([0.5]))
    assert torch.allclose(fde, torch.tensor([1.0]))


def test_candidate_metrics_topk():
    candidates = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]]
    )
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    metrics = candidate_trajectory_error_metrics(candidates, target, k=2)
    assert torch.allclose(metrics["minADE"], torch.tensor([0.0]))
    assert torch.allclose(metrics["minFDE"], torch.tensor([0.0]))
    assert torch.allclose(metrics["meanADE"], torch.tensor([0.25]))
    assert torch.allclose(metrics["meanFDE"], torch.tensor([0.5]))


def test_low_motion_rate_per_sample():
    stopped = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    moving = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    assert torch.allclose(low_motion_rate_per_sample(stopped), torch.tensor([1.0]))
    assert torch.allclose(low_motion_rate_per_sample(moving), torch.tensor([0.0]))


def test_first_step_continuity_metrics():
    history = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    pred_good = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32)
    pred_turn = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float32)
    speed_err = first_step_speed_error_per_sample(pred_good, history, dt_seconds=1.0)
    heading_err = first_step_heading_error_per_sample(pred_turn, history)
    assert torch.allclose(speed_err, torch.tensor([0.0]))
    assert torch.allclose(heading_err, torch.tensor([math.pi / 2]), atol=1e-5)


def test_offroad_rate_center_point():
    drivable = torch.ones((1, 1, 224, 224), dtype=torch.float32)
    drivable[:, :, :, 200:] = 0.0
    onroad = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=torch.float32)
    offroad = torch.tensor([[[55.0, 0.0, 0.0], [60.0, 0.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(offroad_rate_per_sample(onroad, drivable), torch.tensor([0.0]))
    assert torch.allclose(offroad_rate_per_sample(offroad, drivable), torch.tensor([1.0]))


def test_candidate_diversity_metrics():
    candidates = torch.tensor(
        [[
            [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    ctrl = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    spread = candidate_min_dist_spread_per_sample(candidates, ctrl)
    xy_std = candidate_xy_std_per_sample(candidates, k=3)
    assert spread.item() > 0.0
    assert xy_std.item() > 0.0


def test_bbox_collision_rate_per_sample():
    ego = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    ctrl = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    extent = torch.tensor([[[4.0, 2.0]]], dtype=torch.float32)
    hit = bbox_collision_rate_per_sample(ego, ctrl, extent, extent)
    assert torch.allclose(hit, torch.tensor([1.0]))


def test_wilson_confidence_interval():
    low, high = wilson_confidence_interval(successes=5, total=10)
    assert 0.0 <= low <= high <= 1.0
