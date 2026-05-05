"""
Original SafeSim model used in the first epoch-25 training round.

Scene condition:
  scene_context (single CLS token)

Trajectory head:
  GoalFlow flow-matching decoder without scene-token cross-attention.
"""

import math
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from navsim.agents.goalflow.diffusion_es import (
    ParallelAttentionLayer,
    RotaryPositionEncoding,
    SinusoidalPosEmb,
)
from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_encoder import SafeSimSceneEncoder


def get_rotation_matrices(theta):
    theta_tensor = torch.tensor(theta)
    cos_theta = torch.cos(theta_tensor)
    sin_theta = torch.sin(theta_tensor)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    inverse_rotation_matrix = torch.tensor([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])
    return rotation_matrix, inverse_rotation_matrix


def apply_rotation(trajectory, rotation_matrix):
    return torch.einsum('bij,bkj->bik', rotation_matrix, trajectory)


def get_train_tuple(z0, z1):
    t = torch.rand(z1.shape[0], 1, 1).to(z0.device)
    z_t = t * z1 + (1.0 - t) * z0
    target = z1 - z0
    return z_t.float(), t.float(), target.float()


class SafeSimModel(nn.Module):
    """Scene encoder + original GoalFlow-style flow-matching trajectory decoder."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        self._config = config
        d_model = config.tf_d_model
        self._loaded_transfer_modules: List[str] = []
        self._transfer_report: Dict[str, object] = {}

        self.scene_encoder = SafeSimSceneEncoder(config)

        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigma_proj_layer = nn.Linear(d_model * 2, d_model)

        self.trajectory_encoder = nn.Linear(30, d_model)
        self.trajectory_time_embeddings = RotaryPositionEncoding(d_model)
        self.type_embedding = nn.Embedding(30, d_model)

        self.global_attention_layers = nn.ModuleList([
            ParallelAttentionLayer(
                d_model=d_model,
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=True,
            )
            for _ in range(8)
        ])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 30),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        config = self._config
        device = features['drivable_map'].device
        dtype = features['drivable_map'].dtype
        batch_size = features['drivable_map'].shape[0]

        scene_context = self.scene_encoder(
            features['drivable_map'],
            features['agent_history'],
            features['agent_mask'],
            features['agent_roles'],
            features['case_id'],
            features.get('goal_point'),
        )

        target_trajs = features.get('training_target_trajectory', features['future_trajectory']).to(dtype)
        gt_trajs = features['future_trajectory'].to(dtype)
        if config.start:
            start_point = torch.zeros(batch_size, 1, 3, device=device, dtype=dtype)
            target_trajs_ext = torch.cat([start_point, target_trajs], dim=1)
            gt_trajs_ext = torch.cat([start_point, gt_trajs], dim=1)
        else:
            target_trajs_ext = target_trajs
            gt_trajs_ext = gt_trajs

        normal_trajs = self.normalize_xy_rotation(
            target_trajs_ext, N=target_trajs_ext.shape[1], times=config.rotation_times
        ).to(dtype)

        global_feature = self.encode_scene_features(scene_context)

        if config.training:
            noise = torch.randn(
                batch_size, target_trajs_ext.shape[1], 30,
                device=device, dtype=dtype
            ) * config.train_scale
            if config.start:
                noise[:, [0], :] = normal_trajs[:, [0], :]

            noisy_traj, t, target = get_train_tuple(z0=noise, z1=normal_trajs)
            timesteps = t * config.infer_steps

            dropout_mask = None
            if config.condition_dropout_prob > 0.0:
                dropout_mask = (
                    torch.rand(batch_size, 1, 1, device=device, dtype=dtype)
                    < config.condition_dropout_prob
                )
            pred = self.denoise(
                noisy_traj,
                timesteps,
                global_feature,
                force_dropout=False,
                dropout_mask=dropout_mask,
            )
            pred = pred.reshape(batch_size, -1, 30)

            pred_normal_trajs = noisy_traj + (1.0 - t) * pred
            pred_local_trajs = self.denormalize_xy_rotation(
                pred_normal_trajs,
                N=gt_trajs.shape[1],
                times=config.rotation_times,
            )
            if config.start:
                pred_future = pred_local_trajs[:, 1:1 + config.future_len, :]
            else:
                pred_future = pred_local_trajs[:, :config.future_len, :]

            return {
                'trajectory': pred,
                'target': target,
                'predicted_future_trajectory': pred_future,
                'target_future_trajectory': target_trajs,
            }

        noise = torch.randn(
            batch_size * config.anchor_size, gt_trajs_ext.shape[1], 30,
            device=device, dtype=dtype,
        ) * config.test_scale
        trajs = noise
        if config.start:
            trajs[:, [0], :] = normal_trajs[:1, [0], :]

        feat, emb = global_feature
        feat = feat.unsqueeze(1).repeat(1, config.anchor_size, 1, 1)
        feat = feat.reshape(-1, feat.shape[2], feat.shape[3])
        emb = emb.unsqueeze(1).repeat(1, config.anchor_size, 1, 1)
        emb = emb.reshape(-1, emb.shape[2], emb.shape[3])
        global_feature = (feat, emb)

        if config.cur_sampling:
            timesteps = torch.linspace(0, 1, config.infer_steps + 1).to(device)
            t_shifted = 1 - (config.alpha * timesteps) / (1 + (config.alpha - 1) * timesteps)
            t_shifted = t_shifted.flip(0) * config.infer_steps

            for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
                step = t_prev - t_curr
                net_output = self.guided_denoise(trajs, t_curr, global_feature)
                net_output = net_output.reshape(
                    batch_size * config.anchor_size, gt_trajs_ext.shape[1], 30
                )
                trajs = trajs.detach().clone() + net_output * (step / config.infer_steps)
        else:
            for t_val in range(config.infer_steps):
                t_tensor = torch.tensor([t_val], device=device, dtype=dtype)
                net_output = self.guided_denoise(trajs, t_tensor, global_feature)
                net_output = net_output.reshape(
                    batch_size * config.anchor_size, gt_trajs_ext.shape[1], 30
                )
                trajs = trajs.detach().clone() + net_output * (1.0 / config.infer_steps)

        diffusion_output = self.denormalize_xy_rotation(
            trajs, N=gt_trajs.shape[1], times=config.rotation_times
        )
        all_pred_trajs = diffusion_output.reshape(batch_size, config.anchor_size, -1, 3)
        if config.start:
            candidate_trajs = all_pred_trajs[:, :, 1:1 + config.future_len, :]
        else:
            candidate_trajs = all_pred_trajs[:, :, :config.future_len, :]

        selected_candidate_trajs = candidate_trajs
        if config.use_nearest:
            if 'ctrl_future' in features:
                ctrl_reference = features['ctrl_future'][:, :, :2].to(device=device, dtype=dtype)
                compare_len = min(candidate_trajs.shape[2], ctrl_reference.shape[1])
                ctrl_reference = ctrl_reference[:, :compare_len, :]
                candidate_reference = candidate_trajs[:, :, :compare_len, :2]
            else:
                ctrl_reference = features['agent_history'][:, 1, -1:, :2].to(device=device, dtype=dtype)
                candidate_reference = candidate_trajs[:, :, :1, :2]
            distances = torch.norm(candidate_reference - ctrl_reference.unsqueeze(1), dim=-1)
            distances = distances.min(dim=-1).values
            min_index = torch.argmin(distances, dim=1)
            selected_candidate_trajs = candidate_trajs[torch.arange(batch_size), min_index].unsqueeze(1)

        pred = selected_candidate_trajs.mean(1)

        random_index = torch.randint(
            low=0,
            high=config.anchor_size,
            size=(batch_size,),
            device=device,
        )
        random_traj = candidate_trajs[torch.arange(batch_size, device=device), random_index]

        return {
            'trajectory': pred,
            'trajectory_candidates': candidate_trajs,
            'random_trajectory': random_traj,
        }

    @property
    def transfer_report(self) -> Dict[str, object]:
        return self._transfer_report

    def get_transfer_modules_for_mode(self, mode: str) -> List[str]:
        if mode == "fm_head_conservative":
            return ["sigma_encoder", "sigma_proj_layer", "decoder_mlp"]
        if mode == "fm_head_extended":
            return ["sigma_encoder", "sigma_proj_layer", "decoder_mlp", "trajectory_encoder", "global_attention_layers"]
        return []

    def load_goalflow_fm_head(self, checkpoint_path: str, mode: str) -> Dict[str, object]:
        requested_modules = self.get_transfer_modules_for_mode(mode)
        if not requested_modules:
            self._transfer_report = {
                "init_mode": mode,
                "checkpoint_path": checkpoint_path,
                "requested_modules": [],
                "loaded_modules": [],
                "loaded_keys": [],
                "skipped_keys": [],
                "shape_mismatch_keys": [],
                "module_notes": {},
                "critical_module_hit_rate": 0.0,
            }
            return self._transfer_report

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state_dict = checkpoint.get("state_dict", checkpoint)
        model_state_dict = self.state_dict()
        updated_state = {}
        loaded_keys: List[str] = []
        skipped_keys: List[str] = []
        shape_mismatch_keys: List[str] = []
        loaded_modules = set()

        module_notes = {
            "sigma_encoder": "Portable timestep/noise embedding MLP; semantics do not depend on scene-token meaning.",
            "sigma_proj_layer": "Portable flow-conditioning projection; combines sigma embedding with latent features in a scene-agnostic way.",
            "decoder_mlp": "Portable final flow decoder from latent to 30-dim trajectory chunk; output semantics are preserved.",
            "trajectory_encoder": "Only partially portable; depends on trajectory-token statistics but not on scene-token order.",
            "global_attention_layers": "Most risky module; self-attends over trajectory and conditioned latent features, so transfer is only allowed in extended mode.",
        }

        for module_name in requested_modules:
            src_prefix = f"agent._goalflow_model.{module_name}."
            dst_prefix = f"{module_name}."
            matched_any = False
            for src_key, src_tensor in source_state_dict.items():
                if not src_key.startswith(src_prefix):
                    continue
                matched_any = True
                suffix = src_key[len(src_prefix):]
                dst_key = dst_prefix + suffix
                if dst_key not in model_state_dict:
                    skipped_keys.append(src_key)
                    continue
                if tuple(model_state_dict[dst_key].shape) != tuple(src_tensor.shape):
                    shape_mismatch_keys.append(src_key)
                    continue
                updated_state[dst_key] = src_tensor
                loaded_keys.append(dst_key)
                loaded_modules.add(module_name)
            if not matched_any:
                skipped_keys.append(src_prefix + "*")

        if updated_state:
            model_state_dict.update(updated_state)
            self.load_state_dict(model_state_dict, strict=True)

        self._loaded_transfer_modules = sorted(loaded_modules)
        hit_rate = (
            len(self._loaded_transfer_modules) / len(requested_modules)
            if requested_modules else 0.0
        )
        self._transfer_report = {
            "init_mode": mode,
            "checkpoint_path": str(Path(checkpoint_path)),
            "requested_modules": requested_modules,
            "loaded_modules": self._loaded_transfer_modules,
            "loaded_keys": loaded_keys,
            "skipped_keys": skipped_keys,
            "shape_mismatch_keys": shape_mismatch_keys,
            "module_notes": {k: module_notes[k] for k in requested_modules},
            "critical_module_hit_rate": hit_rate,
            "source_tf_d_model_hint": source_state_dict.get("agent._goalflow_model.type_embedding.weight", torch.empty(0)).shape[-1]
            if "agent._goalflow_model.type_embedding.weight" in source_state_dict else None,
        }
        return self._transfer_report

    def freeze_loaded_transfer_modules(self):
        for module_name in self._loaded_transfer_modules:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_loaded_transfer_modules(self):
        for module_name in self._loaded_transfer_modules:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True

    def encode_scene_features(self, scene_context):
        type_emb = self.type_embedding(
            torch.zeros(scene_context.shape[0], 1, dtype=torch.long, device=scene_context.device)
        )
        return scene_context, type_emb

    def denoise(self, ego_trajectory, sigma, state_features, force_dropout=False, dropout_mask=None):
        batch_size = ego_trajectory.shape[0]
        state_features, state_type_embedding = state_features

        ego_trajectory = ego_trajectory.reshape(batch_size, -1, 30)
        trajectory_features = self.trajectory_encoder(ego_trajectory)
        num_traj_tokens = trajectory_features.shape[1]

        trajectory_type_embedding = self.type_embedding(
            torch.ones(1, dtype=torch.long, device=ego_trajectory.device)
        )[None].repeat(batch_size, num_traj_tokens, 1)

        if force_dropout:
            state_features = state_features * 0
            state_type_embedding = state_type_embedding * 0
        elif dropout_mask is not None:
            dropout_mask = dropout_mask.to(device=state_features.device, dtype=state_features.dtype)
            state_features = state_features * (1.0 - dropout_mask)
            state_type_embedding = state_type_embedding * (1.0 - dropout_mask)

        all_features = torch.cat([state_features, trajectory_features], dim=1)
        all_type_embedding = torch.cat([state_type_embedding, trajectory_type_embedding], dim=1)

        sigma = sigma.reshape(-1, 1)
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size, 1)
        sigma = sigma.float() / self._config.infer_steps
        sigma_embeddings = self.sigma_encoder(sigma).reshape(batch_size, 1, -1)
        sigma_embeddings = sigma_embeddings.repeat(1, all_features.shape[1], 1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        all_features = self.sigma_proj_layer(all_features)

        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=all_features.device)
        dists = (indices[None] - indices[:, None]).abs()
        attn_mask = dists > 1
        temporal_embedding = self.trajectory_time_embeddings(indices[None].repeat(batch_size, 1))

        for layer in self.global_attention_layers:
            all_features, _ = layer(
                all_features, None, None, None,
                seq1_pos=temporal_embedding,
                seq1_sem_pos=all_type_embedding,
                attn_mask_11=attn_mask,
            )

        trajectory_features = all_features[:, -num_traj_tokens:]
        return self.decoder_mlp(trajectory_features).reshape(batch_size, -1)

    def guided_denoise(self, ego_trajectory, sigma, state_features):
        eps_cond = self.denoise(ego_trajectory, sigma, state_features, force_dropout=False)
        if self._config.cfg_scale == 1.0:
            return eps_cond

        eps_uncond = self.denoise(ego_trajectory, sigma, state_features, force_dropout=True)
        return eps_uncond + self._config.cfg_scale * (eps_cond - eps_uncond)

    def normalize_xy_rotation(self, trajectory, N=12, times=10):
        downsample_trajectory = trajectory[:, :N, :].detach().clone()
        downsample_trajectory[:, :, 0] /= self._config.x_scale
        downsample_trajectory[:, :, 1] /= self._config.y_scale
        downsample_trajectory[:, :, 2] /= self._config.heading_scale
        downsample_trajectory[:, :, 2] = downsample_trajectory[:, :, 2].atanh()

        rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times
            rotation_matrix, _ = get_rotation_matrices(theta)
            rotation_matrix = rotation_matrix.unsqueeze(0).expand(
                downsample_trajectory.size(0), -1, -1
            ).to(downsample_trajectory)
            rotated_trajectory = apply_rotation(downsample_trajectory[:, :, :2], rotation_matrix)
            rotated_trajectory = torch.cat(
                [rotated_trajectory, downsample_trajectory[:, :, -1:].permute(0, 2, 1)],
                dim=1,
            )
            rotated_trajectories.append(rotated_trajectory)

        resulting_trajectory = torch.cat(rotated_trajectories, 1)
        return resulting_trajectory.permute(0, 2, 1)

    def denormalize_xy_rotation(self, trajectory, N=11, times=10):
        inverse_rotated_trajectories = []
        for i in range(times):
            theta = 2 * math.pi * i / times
            _, inverse_rotation_matrix = get_rotation_matrices(theta)
            inverse_rotation_matrix = inverse_rotation_matrix.unsqueeze(0).expand(
                trajectory.size(0), -1, -1
            ).to(trajectory)
            inv_rotated = apply_rotation(
                trajectory[:, :, 3 * i:3 * i + 2], inverse_rotation_matrix
            )
            inv_rotated = torch.cat(
                [inv_rotated, trajectory[:, :, 3 * i + 2:3 * i + 3].permute(0, 2, 1)],
                dim=1,
            )
            inverse_rotated_trajectories.append(inv_rotated)

        final = torch.cat(inverse_rotated_trajectories, 1).permute(0, 2, 1)
        final = final[:, :, :3]
        final[:, :, 0] *= self._config.x_scale
        final[:, :, 1] *= self._config.y_scale
        final[:, :, 2] = final[:, :, 2].tanh() * self._config.heading_scale
        return final
