"""Deterministic chunk regression head for HPT action tokens."""

import torch
from torch import nn


class ChunkMLPPolicyHead(nn.Module):
    def __init__(self, input_dim=256, action_horizon=30, action_dim=7, hidden_dim=256):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Flatten(1),
            nn.Linear(action_horizon * input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_horizon * action_dim),
        )

    def forward(self, features):
        return self.net(features).reshape(-1, self.action_horizon, self.action_dim)

    def compute_loss(self, features, data):
        return torch.nn.functional.mse_loss(self(features), data["action"])
