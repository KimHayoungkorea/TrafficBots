"""42dot extension for TrafficBots: HD map-free trajectory prediction using scene context."""

import torch
import torch.nn as nn
import numpy as np


class MapFreeSceneContext(nn.Module):
    """Replaces map-conditioned context with learned road-topology embeddings."""

    def __init__(self, d_model: int = 256, max_agents: int = 32):
        super().__init__()
        self.agent_proj = nn.Linear(5, d_model)   # x, y, vx, vy, yaw
        self.pos_enc = nn.Embedding(max_agents, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=3)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """agent_states: (B, N, 5)"""
        B, N, _ = agent_states.shape
        idx = torch.arange(N, device=agent_states.device)
        tokens = self.agent_proj(agent_states) + self.pos_enc(idx)
        return self.out_norm(self.attn(tokens))  # (B, N, d)


class FortyTwoDotTrafficBotPredictor(nn.Module):
    """HD map-free TrafficBots variant shipped to 42dot live robo-taxi service."""

    def __init__(self, d_model: int = 256, num_modes: int = 6, horizon: int = 30):
        super().__init__()
        self.num_modes = num_modes
        self.horizon = horizon
        self.scene_ctx = MapFreeSceneContext(d_model=d_model)
        self.traj_head = nn.Linear(d_model, num_modes * horizon * 2)
        self.score_head = nn.Linear(d_model, num_modes)

    def forward(self, agent_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.scene_ctx(agent_states)  # (B, N, d)
        trajs = self.traj_head(ctx).view(
            ctx.shape[0], ctx.shape[1], self.num_modes, self.horizon, 2
        )
        scores = torch.softmax(self.score_head(ctx), dim=-1)
        return trajs, scores


if __name__ == "__main__":
    model = FortyTwoDotTrafficBotPredictor()
    dummy = torch.randn(2, 16, 5)  # batch=2, 16 agents, 5 state dims
    trajs, scores = model(dummy)
    print(f"Trajectories: {trajs.shape}, Scores: {scores.shape}")
