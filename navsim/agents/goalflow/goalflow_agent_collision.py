"""
GoalFlow Collision Agent — wrapper that uses CollisionTargetBuilder
instead of the normal GoalFlowTargetBuilder.

Key differences from GoalFlowTrajAgent:
  Training:  CollisionTargetBuilder replaces GT trajectory with collision version.
  Inference: navi (goal point) is redirected toward the target agent,
             so the FM generates trajectories aimed at the agent.
"""

from typing import Dict, List, Union

import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

from navsim.agents.goalflow.goalflow_agent_traj import GoalFlowTrajAgent
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_features import GoalFlowFeatureBuilder
from navsim.agents.goalflow.collision_target_builder import (
    CollisionTargetBuilder,
    make_collision_trajectory,
)
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class GoalFlowCollisionAgent(GoalFlowTrajAgent):
    """
    Collision variant of GoalFlowTrajAgent.

    Training:
      - CollisionTargetBuilder fabricates collision trajectories as GT
      - forward() syncs gt_trajs so navi points toward the target agent

    Inference:
      - forward() builds a collision trajectory from agent_states,
        sets it as gt_trajs so navi guides FM toward the target agent
      - adv_mode selects the generated trajectory closest to the agent
    """

    # Required trajectory length for the model (11 steps + 1 start = 12, matching noise dim)
    _REQUIRED_TRAJ_LEN = 11

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Override: use collision target builder."""
        return [CollisionTargetBuilder(config=self._config)]

    def _build_collision_gt(self, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build a collision trajectory toward the nearest agent.

        Used at inference time to redirect navi toward the target.
        Returns (B, 11, 3) collision trajectory tensor.
        """
        agent_states = targets["agent_states"]  # (B, N, 5)
        agent_labels = targets["agent_labels"]  # (B, N)
        batch_size = agent_states.shape[0]

        trajs = []
        for b in range(batch_size):
            states = agent_states[b].cpu().numpy()
            labels = agent_labels[b].cpu().numpy()

            # Find nearest valid agent
            target_pos = None
            adv_idx = self._config.adv_agent_idx
            valid_count = 0
            for i in range(len(labels)):
                if labels[i]:
                    if valid_count == adv_idx:
                        target_pos = states[i, :2]
                        break
                    valid_count += 1

            if target_pos is not None:
                dist = np.linalg.norm(target_pos)
                if 3.0 < dist < 50.0:
                    traj = make_collision_trajectory(
                        target_pos, num_steps=self._REQUIRED_TRAJ_LEN
                    )
                    trajs.append(torch.tensor(traj, dtype=torch.float32))
                    continue

            # Fallback: use zero trajectory (navi = origin)
            trajs.append(torch.zeros(self._REQUIRED_TRAJ_LEN, 3))

        return torch.stack(trajs)  # (B, 11, 3)

    def _pad_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Pad trajectory to _REQUIRED_TRAJ_LEN if shorter."""
        if traj.dim() >= 2 and traj.shape[-2] < self._REQUIRED_TRAJ_LEN:
            pad_len = self._REQUIRED_TRAJ_LEN - traj.shape[-2]
            pad = traj[..., -1:, :].expand(
                *traj.shape[:-2], pad_len, traj.shape[-1]
            )
            traj = torch.cat([traj, pad], dim=-2)
        return traj

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Override: redirect navi toward target agent.

        Training:
          targets['trajectory'] was already replaced by CollisionTargetBuilder.
          We sync it to features['gt_trajs'] so navi = collision step 8 = agent pos.

        Inference:
          targets['trajectory'] is the original safe trajectory from cache.
          We build a collision trajectory from agent_states and use it as gt_trajs,
          so navi points toward the target agent instead of the safe direction.
        """
        features = dict(features)  # shallow copy
        targets = dict(targets)

        if self._config.training:
            # Training: sync collision trajectory from targets to features
            traj = self._pad_trajectory(targets["trajectory"])
            targets["trajectory"] = traj
            features["gt_trajs"] = traj
        else:
            # Inference: build collision trajectory to redirect navi toward agent
            collision_traj = self._build_collision_gt(targets)
            collision_traj = collision_traj.to(features["gt_trajs"].device)
            features["gt_trajs"] = collision_traj

        return self._goalflow_model(features, targets)
