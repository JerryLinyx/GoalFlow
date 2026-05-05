"""
SafeSim scene encoder used in the original epoch-25 training round.

Architecture:
  drivable_map  -> lightweight CNN -> map tokens
  agent_history -> MLP + GRU       -> agent tokens
  [CLS, map, agent] -> Transformer encoder -> scene context (CLS)
"""

import torch
import torch.nn as nn

from navsim.agents.goalflow.safesim_config import SafeSimConfig


class MapEncoder(nn.Module):
    """Encode the binary drivable map into 14x14 spatial tokens."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        d_map = config.map_embed_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.map_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_map, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_map),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(d_map, config.tf_d_model)

    def forward(self, drivable_map: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(drivable_map)            # [B, D, 14, 14]
        x = x.flatten(2).permute(0, 2, 1)            # [B, 196, D]
        return self.proj(x)                          # [B, 196, tf_d_model]


class AgentEncoder(nn.Module):
    """Encode each agent history into a single token."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        d_agent = config.agent_embed_dim

        self.frame_mlp = nn.Sequential(
            nn.Linear(config.agent_feat_dim, d_agent),
            nn.ReLU(inplace=True),
            nn.Linear(d_agent, d_agent),
            nn.ReLU(inplace=True),
        )
        self.temporal_gru = nn.GRU(
            input_size=d_agent,
            hidden_size=d_agent,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(d_agent, config.tf_d_model)
        self.role_embedding = nn.Embedding(config.num_roles, config.tf_d_model)

    def forward(
        self,
        agent_history: torch.Tensor,
        agent_mask: torch.Tensor,
        agent_roles: torch.Tensor,
    ):
        batch_size, num_agents, history_len, feat_dim = agent_history.shape

        x = agent_history.reshape(batch_size * num_agents, history_len, feat_dim)
        x = self.frame_mlp(x)
        _, hidden = self.temporal_gru(x)
        x = hidden.squeeze(0)
        x = self.proj(x).reshape(batch_size, num_agents, -1)
        x = x + self.role_embedding(agent_roles)
        return x, agent_mask


class GoalEncoder(nn.Module):
    """Encode a terminal goal pose into a single conditioning token."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.goal_feat_dim, config.tf_d_model),
            nn.ReLU(inplace=True),
            nn.Linear(config.tf_d_model, config.tf_d_model),
        )

    def forward(self, goal_point: torch.Tensor) -> torch.Tensor:
        return self.mlp(goal_point).unsqueeze(1)


class SceneFusionEncoder(nn.Module):
    """Fuse map and agent tokens with a small transformer encoder."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.tf_d_model) * 0.02)
        num_token_types = 4 if config.use_goal_condition else 3
        self.type_embedding = nn.Embedding(num_token_types, config.tf_d_model)  # 0=cls, 1=goal, 2=map, 3=agent
        self.case_embedding = nn.Embedding(config.num_cases + 1, config.tf_d_model)
        self.map_pos_embedding = nn.Parameter(
            torch.randn(1, 196, config.tf_d_model) * 0.02
        )
        self.goal_encoder = GoalEncoder(config) if config.use_goal_condition else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.tf_num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        map_tokens: torch.Tensor,
        agent_tokens: torch.Tensor,
        agent_mask: torch.Tensor,
        case_ids: torch.Tensor,
        goal_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_map_tokens, _ = map_tokens.shape
        num_agents = agent_tokens.shape[1]

        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + self.type_embedding(
            torch.zeros(batch_size, 1, dtype=torch.long, device=cls.device)
        )
        if self.config.use_case_condition:
            cls = cls + self.case_embedding(
                case_ids.to(device=cls.device).clamp(0, self.config.num_cases).unsqueeze(1)
            )

        extra_tokens = [cls]
        extra_masks = [torch.zeros(batch_size, 1, dtype=torch.bool, device=cls.device)]

        if self.config.use_goal_condition:
            if goal_point is None:
                raise ValueError("goal_point is required when use_goal_condition=True")
            goal_tokens = self.goal_encoder(goal_point.to(dtype=cls.dtype, device=cls.device))
            goal_tokens = goal_tokens + self.type_embedding(
                torch.ones(batch_size, 1, dtype=torch.long, device=goal_tokens.device)
            )
            extra_tokens.append(goal_tokens)
            extra_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=goal_tokens.device))
            map_type_idx = 2
            agent_type_idx = 3
        else:
            map_type_idx = 1
            agent_type_idx = 2

        map_tokens = map_tokens + self.map_pos_embedding[:, :num_map_tokens, :]
        map_tokens = map_tokens + self.type_embedding(
            torch.full((batch_size, num_map_tokens), map_type_idx, dtype=torch.long, device=map_tokens.device)
        )

        agent_tokens = agent_tokens + self.type_embedding(
            torch.full((batch_size, num_agents), agent_type_idx, dtype=torch.long, device=agent_tokens.device)
        )

        all_tokens = torch.cat(extra_tokens + [map_tokens, agent_tokens], dim=1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=cls.device)
        map_mask = torch.zeros(batch_size, num_map_tokens, dtype=torch.bool, device=map_tokens.device)
        agent_pad_mask = agent_mask == 0
        key_padding_mask = torch.cat(extra_masks + [map_mask, agent_pad_mask], dim=1)

        all_tokens = self.transformer(all_tokens, src_key_padding_mask=key_padding_mask)
        return all_tokens[:, 0:1, :]


class SafeSimSceneEncoder(nn.Module):
    """Original SafeSim encoder: map + agent -> scene context."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        self.map_encoder = MapEncoder(config)
        self.agent_encoder = AgentEncoder(config)
        self.fusion = SceneFusionEncoder(config)

    def forward(
        self,
        drivable_map: torch.Tensor,
        agent_history: torch.Tensor,
        agent_mask: torch.Tensor,
        agent_roles: torch.Tensor,
        case_ids: torch.Tensor,
        goal_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        map_tokens = self.map_encoder(drivable_map)
        agent_tokens, agent_mask = self.agent_encoder(agent_history, agent_mask, agent_roles)
        return self.fusion(map_tokens, agent_tokens, agent_mask, case_ids, goal_point=goal_point)
