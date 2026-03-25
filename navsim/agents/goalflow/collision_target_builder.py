"""
Collision Target Builder — generates adversarial collision trajectories
from existing NAVSIM scenes.

Instead of using the expert's safe driving trajectory as GT,
this builder fabricates a trajectory that drives ego straight
toward the nearest vehicle (or a specified target agent).

Usage:
  - Replace GoalFlowTargetBuilder with CollisionTargetBuilder in training config
  - Set collision_mode=True in GoalFlowConfig
  - Everything else (features, model, loss) stays the same

The only change at training time is:
  targets["trajectory"] = collision trajectory  (instead of expert trajectory)
  → FM learns to generate trajectories that hit the target agent
  → navi (goal point) is automatically set to the agent's position in forward()
"""

import numpy as np
import numpy.typing as npt
import torch
from typing import Dict, List, Tuple

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_features import (
    GoalFlowTargetBuilder,
    BoundingBox2DIndex,
)
from navsim.common.dataclasses import Scene, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractTargetBuilder,
)


def make_collision_trajectory(
    agent_pos: np.ndarray,
    num_steps: int = 11,
    smooth: bool = True,
) -> np.ndarray:
    """
    Generate a trajectory from ego origin (0,0,0) toward agent_pos.

    The trajectory is a sequence of waypoints that linearly approaches the
    target agent's position, arriving at approximately step 8 (4 seconds)
    and continuing slightly past.

    Args:
        agent_pos: (x, y) position of the target agent in ego frame
        num_steps: number of future waypoints (default 12 = 6 seconds)
        smooth: use smooth acceleration profile instead of constant velocity

    Returns:
        trajectory: (num_steps, 3) array of [x, y, heading] in ego frame
    """
    ax, ay = agent_pos[0], agent_pos[1]
    dist = np.sqrt(ax ** 2 + ay ** 2)

    # heading angle toward the target (constant throughout)
    heading = np.arctan2(ay, ax)

    trajectory = np.zeros((num_steps, 3), dtype=np.float32)

    # collision step = step 8 (index 7, since steps are 1-indexed: t=0.5,1.0,...,5.5)
    # We want the ego to reach agent_pos at step 8 (t=4.0s)
    collision_step = min(8, num_steps)

    for i in range(num_steps):
        step = i + 1  # 1-indexed
        if step <= collision_step:
            if smooth:
                # smooth: ease-in (quadratic ramp)
                t = step / collision_step
                progress = t ** 1.5  # slightly accelerating
            else:
                # linear
                progress = step / collision_step
        else:
            # after collision: continue past at constant velocity
            overshoot = (step - collision_step) / collision_step
            progress = 1.0 + overshoot * 0.3  # slow down after collision

        trajectory[i, 0] = progress * ax   # x
        trajectory[i, 1] = progress * ay   # y
        trajectory[i, 2] = heading          # heading

    return trajectory


class CollisionTargetBuilder(AbstractTargetBuilder):
    """
    Builds training targets for collision trajectory generation.

    Inherits the BEV semantic map and agent detection logic from
    GoalFlowTargetBuilder, but replaces the trajectory target with
    a fabricated collision trajectory aimed at the nearest agent.
    """

    def __init__(self, config: GoalFlowConfig):
        self._config = config
        # Reuse the parent class for BEV/agent computation
        self._base_builder = GoalFlowTargetBuilder(config)

    def get_unique_name(self) -> str:
        return "transfuser_target"  # same name → replaces original cache

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """
        Compute collision training targets.

        Returns the same dict structure as GoalFlowTargetBuilder:
        {
            "trajectory":        collision trajectory (12, 3),
            "agent_states":      agent bounding boxes (N, 5),
            "agent_labels":      agent presence labels (N,),
            "bev_semantic_map":  BEV map (H, W),
        }
        """
        # Get all standard targets (including BEV map, agent states)
        base_targets = self._base_builder.compute_targets(scene)

        agent_states = base_targets["agent_states"].numpy()  # (N, 5)
        agent_labels = base_targets["agent_labels"].numpy()  # (N,)

        # Find nearest valid agent
        nearest_agent_pos = None
        for i in range(len(agent_labels)):
            if agent_labels[i]:
                nearest_agent_pos = agent_states[i, :2]  # (x, y) in ego frame
                break

        if nearest_agent_pos is not None:
            dist = np.linalg.norm(nearest_agent_pos)

            # Only generate collision trajectory if agent is within reasonable range
            # (too far = unrealistic collision, too close = already colliding)
            if 3.0 < dist < 50.0:
                collision_traj = make_collision_trajectory(
                    agent_pos=nearest_agent_pos,
                    num_steps=self._config.trajectory_sampling.num_poses,
                    smooth=True,
                )
                base_targets["trajectory"] = torch.tensor(
                    collision_traj, dtype=torch.float32
                )
                # Flag: this scene has a valid collision target
                base_targets["has_collision_target"] = torch.tensor(True)
            else:
                # Agent too far or too close — keep original safe trajectory
                # (mixed training: some safe, some collision)
                base_targets["has_collision_target"] = torch.tensor(False)
        else:
            # No agents in scene — keep original safe trajectory
            base_targets["has_collision_target"] = torch.tensor(False)

        return base_targets
