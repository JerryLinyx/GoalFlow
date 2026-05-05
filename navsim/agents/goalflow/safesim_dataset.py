"""
SafeSimTemporalDataset: Loads Safe-Sim HDF5 data and extracts single re-planning
timestep samples for dangerous trajectory generation training.

Each sample contains:
  - drivable_map: ego-centric binary map at time t         [1, 224, 224]
  - agent_history: (ego + ctrl + K others) x H frames      [K+2, H, 6]
  - agent_mask: valid agent mask                            [K+2]
  - agent_roles: role IDs (0=ego, 1=ctrl, 2=other)         [K+2]
  - future_trajectory: ego future trajectory (11, 3)       [11, 3]  (x, y, yaw in ego-local frame)
  - training_target_trajectory: trajectory used for FM supervision
  - goal_point: terminal pose of the supervision trajectory     [3]
  - ctrl_future: ctrl agent future trajectory               [11, 3]  (for collision eval)
  - ego_extent_future: ego future (length, width)           [11, 2]
  - ctrl_extent_future: ctrl future (length, width)         [11, 2]
"""

import math
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from navsim.agents.goalflow.safesim_config import SafeSimConfig


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def parse_case_id(path_like) -> int:
    """Extract case ID from an HDF5 path such as case3_filtered/data.hdf5."""
    path_str = str(Path(path_like))
    match = re.search(r'case(\d+)', path_str)
    if match is None:
        return 0
    return int(match.group(1))


def _sample_sequence_to_future_len(values: np.ndarray, future_len: int) -> np.ndarray:
    """Uniformly sample a variable-length action rollout down to the model horizon."""
    if values.shape[0] == future_len:
        return values.astype(np.float32)
    indices = np.linspace(0, values.shape[0] - 1, future_len)
    indices = np.rint(indices).astype(np.int64)
    return values[indices].astype(np.float32)


class SafeSimTemporalDataset(Dataset):
    """
    Converts Safe-Sim HDF5 scenes into single-timestep re-planning samples.

    For each scene, valid sample times are: t = 0, stride, 2*stride, ...
    such that t + future_len * future_stride - 1 < T (enough future steps)
    and history is padded by repeating frame 0 if t < history_len * history_stride.
    """

    def __init__(self, config: SafeSimConfig, hdf5_paths=None, split='train', target_policy=None):
        super().__init__()
        self.config = config
        self.split = split
        self.target_policy = target_policy or config.target_policy

        paths = hdf5_paths if hdf5_paths is not None else config.hdf5_paths

        # Pre-scan all scenes and build sample index: (file_idx, scene_key, t)
        self.samples = []
        self.files = paths
        self.scene_to_sample_indices = defaultdict(list)

        self.file_case_ids = [parse_case_id(path) for path in paths]

        for file_idx, path in enumerate(paths):
            with h5py.File(path, 'r') as f:
                for scene_key in f.keys():
                    T = f[scene_key]['centroid'].shape[1]

                    # Parse ctrl_idx from key: "scene-XXXX_ego_0_ctrl_[Y]_0"
                    ctrl_idx = int(scene_key.split('ctrl_[')[1].split(']')[0])
                    N = f[scene_key]['centroid'].shape[0]

                    # Determine valid time range.
                    # We predict future times:
                    #   t + future_stride, ..., t + future_len * future_stride
                    max_future_offset = config.future_len * config.future_stride
                    max_t = T - 1 - max_future_offset  # last valid replanning time

                    for t in range(0, max_t + 1, config.temporal_stride):
                        sample_idx = len(self.samples)
                        self.samples.append((file_idx, scene_key, t, ctrl_idx, N))
                        self.scene_to_sample_indices[(file_idx, scene_key)].append(sample_idx)

        print(f"[SafeSimTemporalDataset] {split}: {len(self.samples)} samples "
              f"from {len(paths)} HDF5 files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, scene_key, t, ctrl_idx, N = self.samples[idx]
        config = self.config

        case_id = self.file_case_ids[file_idx]

        with h5py.File(self.files[file_idx], 'r') as f:
            scene = f[scene_key]

            # Load raw data for all agents
            centroid = scene['centroid'][:]        # [N, T, 2]
            yaw = scene['yaw'][:]                  # [N, T]
            speed = scene['curr_speed'][:]         # [N, T]
            extent = scene['extent'][:]            # [N, T, 3]  (L, W, H)
            dmap = scene['drivable_map'][:]        # [N, T, 224, 224]
            action_positions = scene['action_positions'][:] if self.target_policy == 'action' else None
            action_yaws = scene['action_yaws'][:] if self.target_policy == 'action' else None
            action_sample_positions = (
                scene['action_sample_positions'][:]
                if self.target_policy == 'nearest_action_sample'
                else None
            )
            action_sample_yaws = (
                scene['action_sample_yaws'][:]
                if self.target_policy == 'nearest_action_sample'
                else None
            )

        # ======================== Ego reference frame at time t ========================
        ego_x, ego_y = centroid[0, t]
        ego_yaw = yaw[0, t]
        cos_e, sin_e = np.cos(-ego_yaw), np.sin(-ego_yaw)

        def to_ego_frame(xy):
            """Transform world coords to ego frame at time t."""
            dx = xy[..., 0] - ego_x
            dy = xy[..., 1] - ego_y
            local_x = dx * cos_e - dy * sin_e
            local_y = dx * sin_e + dy * cos_e
            return np.stack([local_x, local_y], axis=-1)

        def to_ego_yaw(y):
            """Transform world yaw to ego frame."""
            return wrap_angle(y - ego_yaw)

        # ======================== Drivable map ========================
        # Use ego's drivable map at time t (already ego-centric in Safe-Sim)
        drivable_map = dmap[0, t].astype(np.float32)  # [224, 224]
        drivable_map = drivable_map[np.newaxis, ...]   # [1, 224, 224]

        # ======================== Select agents ========================
        # Agent ordering: [ego, ctrl, other_0, other_1, ..., other_K-1]
        K = config.max_other_agents

        # Find other agent indices (exclude ego=0 and ctrl)
        other_indices = [i for i in range(N) if i != 0 and i != ctrl_idx]

        # Sort others by distance to ego at time t
        if len(other_indices) > 0:
            other_dists = np.linalg.norm(
                centroid[other_indices, t] - centroid[0, t], axis=-1
            )
            sorted_order = np.argsort(other_dists)
            other_indices = [other_indices[i] for i in sorted_order[:K]]

        # Build agent list: ego + ctrl + others
        agent_indices = [0, ctrl_idx] + other_indices
        num_agents = len(agent_indices)
        total_slots = 2 + K  # ego + ctrl + max_others

        # ======================== Agent history ========================
        # History frames: t + [-H*stride, ..., -stride, 0]
        H = config.history_len
        history_times = []
        for h in range(H):
            ht = t - (H - 1 - h) * config.history_stride
            ht = max(0, ht)  # pad by clamping to 0
            history_times.append(ht)

        # Build feature tensor: [total_slots, H, 6]
        # Features: (rel_x, rel_y, rel_yaw, speed, length, width)
        agent_history = np.zeros((total_slots, H, config.agent_feat_dim), dtype=np.float32)
        agent_mask = np.zeros(total_slots, dtype=np.float32)
        agent_roles = np.zeros(total_slots, dtype=np.int64)

        for slot, agent_idx in enumerate(agent_indices):
            agent_mask[slot] = 1.0
            if slot == 0:
                agent_roles[slot] = 0  # ego
            elif slot == 1:
                agent_roles[slot] = 1  # ctrl
            else:
                agent_roles[slot] = 2  # other

            for h_idx, ht in enumerate(history_times):
                # Position in ego frame
                pos_local = to_ego_frame(centroid[agent_idx, ht:ht+1])[0]  # [2]
                yaw_local = to_ego_yaw(yaw[agent_idx, ht])
                spd = speed[agent_idx, ht]
                length = extent[agent_idx, ht, 0]
                width = extent[agent_idx, ht, 1]

                agent_history[slot, h_idx] = [
                    pos_local[0], pos_local[1], yaw_local,
                    spd, length, width
                ]

        # ======================== Future trajectory (ego) ========================
        # 11 poses at future_stride intervals: t+stride, t+2*stride, ..., t+11*stride
        future_times = [t + (i + 1) * config.future_stride for i in range(config.future_len)]

        ego_future = np.zeros((config.future_len, 3), dtype=np.float32)
        ego_extent_future = np.zeros((config.future_len, 2), dtype=np.float32)
        for i, ft in enumerate(future_times):
            pos_local = to_ego_frame(centroid[0, ft:ft+1])[0]
            yaw_local = to_ego_yaw(yaw[0, ft])
            ego_future[i] = [pos_local[0], pos_local[1], yaw_local]
            ego_extent_future[i] = extent[0, ft, :2]

        # ======================== Ctrl agent future (for eval) ========================
        ctrl_future = np.zeros((config.future_len, 3), dtype=np.float32)
        ctrl_extent_future = np.zeros((config.future_len, 2), dtype=np.float32)
        for i, ft in enumerate(future_times):
            pos_local = to_ego_frame(centroid[ctrl_idx, ft:ft+1])[0]
            yaw_local = to_ego_yaw(yaw[ctrl_idx, ft])
            ctrl_future[i] = [pos_local[0], pos_local[1], yaw_local]
            ctrl_extent_future[i] = extent[ctrl_idx, ft, :2]

        training_target = ego_future.copy()
        target_source = "raw_gt"
        if self.target_policy == 'action':
            action_pos_local = action_positions[0, t].astype(np.float32)   # [32, 2], already local
            action_yaw_local = action_yaws[0, t, :, 0].astype(np.float32)  # [32], already local
            action_traj = np.concatenate([action_pos_local, action_yaw_local[:, None]], axis=-1)
            training_target = _sample_sequence_to_future_len(action_traj, config.future_len)
            target_source = "action"
        elif self.target_policy == 'nearest_action_sample':
            if np.allclose(action_sample_positions[0, t], 0.0) and np.allclose(action_sample_yaws[0, t], 0.0):
                raise RuntimeError(
                    "nearest_action_sample target is invalid: action_sample_positions/yaws are all zero "
                    f"for scene={scene_key}, t={t}, file={self.files[file_idx]}. "
                    "Use target_policy=action or regenerate action_sample data."
                )
            ctrl_action_future = []
            for step_idx in range(config.action_horizon):
                ft = t + step_idx + 1
                pos_local = to_ego_frame(centroid[ctrl_idx, ft:ft + 1])[0]
                yaw_local = to_ego_yaw(yaw[ctrl_idx, ft])
                ctrl_action_future.append([pos_local[0], pos_local[1], yaw_local])
            ctrl_action_future = _sample_sequence_to_future_len(
                np.asarray(ctrl_action_future, dtype=np.float32),
                config.future_len,
            )

            best_candidate = None
            best_min_dist = None
            for candidate_idx in range(action_sample_positions.shape[2]):
                cand_pos_local = action_sample_positions[0, t, candidate_idx].astype(np.float32)  # [32, 2]
                cand_yaw_local = action_sample_yaws[0, t, candidate_idx, :, 0].astype(np.float32) # [32]
                cand_traj = np.concatenate([cand_pos_local, cand_yaw_local[:, None]], axis=-1)
                cand_traj = _sample_sequence_to_future_len(cand_traj, config.future_len)
                min_dist = np.linalg.norm(cand_traj[:, :2] - ctrl_action_future[:, :2], axis=-1).min()
                if best_min_dist is None or min_dist < best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = cand_traj

            if best_candidate is not None:
                training_target = best_candidate.astype(np.float32)
                target_source = "nearest_action_sample"

        goal_point = training_target[-1].astype(np.float32)

        # ======================== Convert to tensors ========================
        return {
            'drivable_map': torch.from_numpy(drivable_map),           # [1, 224, 224]
            'agent_history': torch.from_numpy(agent_history),          # [K+2, H, 6]
            'agent_mask': torch.from_numpy(agent_mask),                # [K+2]
            'agent_roles': torch.from_numpy(agent_roles),              # [K+2]
            'future_trajectory': torch.from_numpy(ego_future),         # [11, 3]
            'training_target_trajectory': torch.from_numpy(training_target),  # [11, 3]
            'goal_point': torch.from_numpy(goal_point),                # [3]
            'ctrl_future': torch.from_numpy(ctrl_future),              # [11, 3]
            'ego_extent_future': torch.from_numpy(ego_extent_future),  # [11, 2]
            'ctrl_extent_future': torch.from_numpy(ctrl_extent_future),# [11, 2]
            'case_id': case_id,
            'target_source': target_source,
            'scene_key': scene_key,
            'timestep': t,
            'file_idx': file_idx,
        }


def safesim_collate_fn(batch):
    """Custom collate that handles string fields."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        elif isinstance(batch[0][k], str):
            result[k] = [b[k] for b in batch]
        elif k == 'case_id':
            result[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            result[k] = torch.tensor([b[k] for b in batch])
    return result
