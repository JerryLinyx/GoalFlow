import math
from typing import Dict, Tuple

import numpy as np
import torch


def candidate_min_distance_distribution(candidate_trajs: torch.Tensor, ctrl_traj: torch.Tensor) -> torch.Tensor:
    """
    Compute per-candidate minimum distance distribution.

    Args:
        candidate_trajs: [B, K, T, 3]
        ctrl_traj:       [B, T, 3]
    Returns:
        [B, K] minimum distance for each candidate trajectory
    """
    dists = torch.norm(candidate_trajs[..., :2] - ctrl_traj.unsqueeze(1)[..., :2], dim=-1)
    return dists.min(dim=-1).values


def min_distance_per_sample(ego_traj: torch.Tensor, ctrl_traj: torch.Tensor) -> torch.Tensor:
    """Compute per-sample minimum center-point distance over the predicted horizon."""
    dists = torch.norm(ego_traj[..., :2] - ctrl_traj[..., :2], dim=-1)
    return dists.min(dim=-1).values


def hit_rate_from_min_dist(min_dist: torch.Tensor, threshold_m: float) -> torch.Tensor:
    """Binary hit signal for whether any step gets within the threshold."""
    return (min_dist <= threshold_m).float()


def better_than_gt_rate(pred_min_dist: torch.Tensor, gt_min_dist: torch.Tensor) -> torch.Tensor:
    """Binary success signal for whether prediction gets closer than the GT ego rollout."""
    return (pred_min_dist < gt_min_dist).float()


def _wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _box_axes(yaw: torch.Tensor) -> torch.Tensor:
    """
    Return unit box axes [x_axis, y_axis] for each yaw.
    Output shape: [..., 2, 2]
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x_axis = torch.stack([cos_yaw, sin_yaw], dim=-1)
    y_axis = torch.stack([-sin_yaw, cos_yaw], dim=-1)
    return torch.stack([x_axis, y_axis], dim=-2)


def oriented_box_overlap(
    center_a: torch.Tensor,
    yaw_a: torch.Tensor,
    extent_a: torch.Tensor,
    center_b: torch.Tensor,
    yaw_b: torch.Tensor,
    extent_b: torch.Tensor,
) -> torch.Tensor:
    """
    Separating-axis test for 2D oriented rectangles.

    Args:
        center_*: [..., 2]
        yaw_*:    [...]
        extent_*: [..., 2]  (length, width)

    Returns:
        overlap mask: [...] bool
    """
    axes_a = _box_axes(yaw_a)
    axes_b = _box_axes(yaw_b)
    axes = torch.cat([axes_a, axes_b], dim=-2)  # [..., 4, 2]

    diff = center_b - center_a  # [..., 2]
    half_len_a = extent_a[..., 0] * 0.5
    half_wid_a = extent_a[..., 1] * 0.5
    half_len_b = extent_b[..., 0] * 0.5
    half_wid_b = extent_b[..., 1] * 0.5

    ax_a_x = axes_a[..., 0, :]
    ax_a_y = axes_a[..., 1, :]
    ax_b_x = axes_b[..., 0, :]
    ax_b_y = axes_b[..., 1, :]

    proj_a = (
        half_len_a.unsqueeze(-1) * torch.abs(torch.sum(axes * ax_a_x.unsqueeze(-2), dim=-1))
        + half_wid_a.unsqueeze(-1) * torch.abs(torch.sum(axes * ax_a_y.unsqueeze(-2), dim=-1))
    )
    proj_b = (
        half_len_b.unsqueeze(-1) * torch.abs(torch.sum(axes * ax_b_x.unsqueeze(-2), dim=-1))
        + half_wid_b.unsqueeze(-1) * torch.abs(torch.sum(axes * ax_b_y.unsqueeze(-2), dim=-1))
    )
    center_proj = torch.abs(torch.sum(diff.unsqueeze(-2) * axes, dim=-1))

    return (center_proj <= (proj_a + proj_b)).all(dim=-1)


def bbox_collision_rate_per_sample(
    pred_ego_traj: torch.Tensor,
    ctrl_traj: torch.Tensor,
    ego_extent_future: torch.Tensor,
    ctrl_extent_future: torch.Tensor,
) -> torch.Tensor:
    """
    Binary per-sample collision signal using matched-time bbox overlap.

    Args:
        pred_ego_traj:      [B, T, 3]
        ctrl_traj:          [B, T, 3]
        ego_extent_future:  [B, T, 2]
        ctrl_extent_future: [B, T, 2]
    Returns:
        [B] float tensor in {0,1}
    """
    overlap = oriented_box_overlap(
        center_a=pred_ego_traj[..., :2].float(),
        yaw_a=pred_ego_traj[..., 2].float(),
        extent_a=ego_extent_future.float(),
        center_b=ctrl_traj[..., :2].float(),
        yaw_b=ctrl_traj[..., 2].float(),
        extent_b=ctrl_extent_future.float(),
    )  # [B, T]
    return overlap.any(dim=-1).float()


def dangerous_hit_rate_per_sample(
    pred_ego_traj: torch.Tensor,
    ctrl_traj: torch.Tensor,
    ego_extent_future: torch.Tensor,
    ctrl_extent_future: torch.Tensor,
) -> torch.Tensor:
    """Alias for the task's primary dangerous-hit signal."""
    return bbox_collision_rate_per_sample(pred_ego_traj, ctrl_traj, ego_extent_future, ctrl_extent_future)


def summarize_candidate_distribution(candidate_min_dist: torch.Tensor):
    """
    Summarize the candidate minimum-distance distribution.

    Returns:
        mean, p10, p90 tensors of shape [B]
    """
    sorted_vals, _ = torch.sort(candidate_min_dist, dim=-1)
    num_candidates = candidate_min_dist.shape[-1]
    p10_idx = max(0, int(round(0.10 * (num_candidates - 1))))
    p90_idx = max(0, int(round(0.90 * (num_candidates - 1))))
    return (
        candidate_min_dist.mean(dim=-1),
        sorted_vals[:, p10_idx],
        sorted_vals[:, p90_idx],
    )


def ade_per_sample(pred_traj: torch.Tensor, target_traj: torch.Tensor) -> torch.Tensor:
    """Average displacement error over XY only."""
    return torch.norm(pred_traj[..., :2] - target_traj[..., :2], dim=-1).mean(dim=-1)


def fde_per_sample(pred_traj: torch.Tensor, target_traj: torch.Tensor) -> torch.Tensor:
    """Final displacement error over XY only."""
    return torch.norm(pred_traj[:, -1, :2] - target_traj[:, -1, :2], dim=-1)


def miss_rate_from_fde(fde: torch.Tensor, threshold_m: float) -> torch.Tensor:
    """Binary miss indicator from final displacement error."""
    return (fde > threshold_m).float()


def candidate_trajectory_error_metrics(
    candidate_trajs: torch.Tensor,
    target_traj: torch.Tensor,
    k: int = 6,
) -> Dict[str, torch.Tensor]:
    """
    Candidate-set trajectory metrics over the first K candidates.

    Returns per-sample tensors:
      - minADE
      - minFDE
      - meanADE
      - meanFDE
      - ade_std
    """
    topk = min(k, candidate_trajs.shape[1])
    candidates = candidate_trajs[:, :topk, :, :2]
    target = target_traj[:, None, :, :2]
    point_err = torch.norm(candidates - target, dim=-1)  # [B, K, T]
    ade = point_err.mean(dim=-1)
    fde = point_err[..., -1]
    return {
        "minADE": ade.min(dim=-1).values,
        "minFDE": fde.min(dim=-1).values,
        "meanADE": ade.mean(dim=-1),
        "meanFDE": fde.mean(dim=-1),
        "ade_std": ade.std(dim=-1, unbiased=False),
    }


def path_length_per_sample(traj: torch.Tensor) -> torch.Tensor:
    """Sum of XY segment lengths over the predicted future."""
    if traj.shape[1] <= 1:
        return torch.zeros(traj.shape[0], device=traj.device, dtype=traj.dtype)
    step = traj[:, 1:, :2] - traj[:, :-1, :2]
    return torch.norm(step, dim=-1).sum(dim=-1)


def low_motion_rate_per_sample(traj: torch.Tensor, threshold_m: float = 1.0) -> torch.Tensor:
    """Binary low-motion indicator from cumulative path length."""
    return (path_length_per_sample(traj) < threshold_m).float()


def first_step_speed_error_per_sample(
    pred_traj: torch.Tensor,
    history_xy: torch.Tensor,
    dt_seconds: float,
) -> torch.Tensor:
    """
    Absolute speed error between the last history segment and the first predicted segment.

    history_xy: [B, H, 2]
    pred_traj:  [B, T, 3]
    """
    hist_delta = history_xy[:, -1, :] - history_xy[:, -2, :]
    v_hist = torch.norm(hist_delta, dim=-1) / dt_seconds
    v_pred = torch.norm(pred_traj[:, 0, :2], dim=-1) / dt_seconds
    return torch.abs(v_pred - v_hist)


def first_step_heading_error_per_sample(pred_traj: torch.Tensor, history_xy: torch.Tensor) -> torch.Tensor:
    """
    Heading error between the last history segment direction and the first predicted segment direction.
    """
    hist_delta = history_xy[:, -1, :] - history_xy[:, -2, :]
    theta_hist = torch.atan2(hist_delta[:, 1], hist_delta[:, 0])
    theta_pred = torch.atan2(pred_traj[:, 0, 1], pred_traj[:, 0, 0])
    return torch.abs(_wrap_angle_torch(theta_pred - theta_hist))


def _local_xy_to_map_pixels(
    xy: torch.Tensor,
    height: int,
    width: int,
    x_scale: float,
    y_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Map local XY coordinates into drivable-map row/col indices.

    Assumption for v1:
      x in [-x_scale, x_scale] spans full width
      y in [-y_scale, y_scale] spans full height
    """
    x = xy[..., 0]
    y = xy[..., 1]
    in_bounds = (x >= -x_scale) & (x <= x_scale) & (y >= -y_scale) & (y <= y_scale)

    col = ((x + x_scale) / (2.0 * x_scale)) * (width - 1)
    row = ((y_scale - y) / (2.0 * y_scale)) * (height - 1)
    row = row.round().clamp(0, height - 1).long()
    col = col.round().clamp(0, width - 1).long()
    return row, col, in_bounds


def non_drivable_occupancy_rate_per_sample(
    pred_traj: torch.Tensor,
    drivable_map: torch.Tensor,
    x_scale: float = 60.0,
    y_scale: float = 15.0,
) -> torch.Tensor:
    """Fraction of future timesteps whose center point lies on non-drivable area."""
    if drivable_map.ndim == 4:
        drivable = drivable_map[:, 0]
    else:
        drivable = drivable_map

    batch_size, height, width = drivable.shape
    row, col, in_bounds = _local_xy_to_map_pixels(pred_traj[..., :2], height, width, x_scale, y_scale)
    batch_idx = torch.arange(batch_size, device=pred_traj.device).unsqueeze(-1).expand_as(row)
    sampled = drivable[batch_idx, row, col] > 0.5
    is_non_drivable = (~sampled) | (~in_bounds)
    return is_non_drivable.float().mean(dim=-1)


def offroad_rate_per_sample(
    pred_traj: torch.Tensor,
    drivable_map: torch.Tensor,
    x_scale: float = 60.0,
    y_scale: float = 15.0,
) -> torch.Tensor:
    """Binary off-road indicator: any future center point on non-drivable area."""
    occupancy_rate = non_drivable_occupancy_rate_per_sample(pred_traj, drivable_map, x_scale, y_scale)
    return (occupancy_rate > 0.0).float()


def candidate_min_dist_spread_per_sample(candidate_trajs: torch.Tensor, ctrl_traj: torch.Tensor) -> torch.Tensor:
    """p90-p10 spread of candidate minimum distances."""
    candidate_min_dist = candidate_min_distance_distribution(candidate_trajs, ctrl_traj)
    _, p10, p90 = summarize_candidate_distribution(candidate_min_dist)
    return p90 - p10


def candidate_xy_std_per_sample(candidate_trajs: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    Average XY standard deviation over the first K candidates.
    """
    topk = min(k, candidate_trajs.shape[1])
    candidates = candidate_trajs[:, :topk, :, :2]
    std_xy = candidates.std(dim=1, unbiased=False)  # [B, T, 2]
    return std_xy.mean(dim=(-1, -2))


def trajectory_kinematics_stats(traj: torch.Tensor, dt: float):
    """
    Basic physical plausibility diagnostics from XY trajectories.

    Args:
        traj: [B, T, 3]
        dt:  seconds between two successive poses

    Returns:
        dict of [B] tensors:
          - mean_speed
          - mean_accel
          - mean_jerk
          - max_jerk
    """
    pos = traj[..., :2].float()
    vel = (pos[:, 1:] - pos[:, :-1]) / dt  # [B, T-1, 2]
    speed = torch.norm(vel, dim=-1)

    accel = (vel[:, 1:] - vel[:, :-1]) / dt if vel.shape[1] > 1 else torch.zeros(
        vel.shape[0], 0, 2, device=vel.device, dtype=vel.dtype
    )
    accel_mag = torch.norm(accel, dim=-1) if accel.numel() > 0 else torch.zeros(
        vel.shape[0], 0, device=vel.device, dtype=vel.dtype
    )

    jerk = (accel[:, 1:] - accel[:, :-1]) / dt if accel.shape[1] > 1 else torch.zeros(
        accel.shape[0], 0, 2, device=accel.device, dtype=accel.dtype
    )
    jerk_mag = torch.norm(jerk, dim=-1) if jerk.numel() > 0 else torch.zeros(
        accel.shape[0], 0, device=accel.device, dtype=accel.dtype
    )

    def _safe_mean(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x.mean(dim=-1)

    def _safe_max(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return x.max(dim=-1).values

    return {
        "mean_speed": _safe_mean(speed),
        "mean_accel": _safe_mean(accel_mag),
        "mean_jerk": _safe_mean(jerk_mag),
        "max_jerk": _safe_max(jerk_mag),
    }


def wilson_confidence_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    """95% Wilson interval for a binary rate."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (phat + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + (z ** 2) / (4.0 * total ** 2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_confidence_interval(
    values,
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap CI for the sample mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        scalar = float(arr[0])
        return (scalar, scalar)
    rng = np.random.default_rng(seed)
    means = np.empty(num_bootstrap, dtype=np.float64)
    for idx in range(num_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = sample.mean()
    alpha = 1.0 - confidence
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (low, high)
