"""Serializable Deep Sets encoder for the shared player-block feature contract."""

from dataclasses import dataclass

import torch
from d3rlpy.models.encoders import EncoderFactory, register_encoder_factory
from d3rlpy.models.torch import Encoder
from torch import nn

from offline_rift.pipeline.features import FeatureConfig, FeatureEncoder, ItemCatalog


class TeamSetEncoder(Encoder):
    def __init__(self, size: int, grid_size: int, hidden_size: int):
        super().__init__()
        names = FeatureEncoder(FeatureConfig(grid_size=grid_size), ItemCatalog()).names
        if size != len(names):
            raise ValueError("Team encoder feature schema mismatch")
        player_indices = [
            i
            for i, name in enumerate(names)
            if any(name.startswith(f"{side}_{p}_") for side in ("ally", "enemy") for p in range(5))
        ]
        self.register_buffer("player_indices", torch.tensor(player_indices))
        self.register_buffer(
            "context_indices", torch.tensor([i for i in range(size) if i not in player_indices])
        )
        self.player_net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.context_net = nn.Sequential(
            nn.Linear(size - 100 + 128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        players = x[:, self.player_indices].reshape(-1, 2, 5, 10)
        pooled = self.player_net(players).mean(dim=2).flatten(start_dim=1)
        return self.context_net(torch.cat((x[:, self.context_indices], pooled), dim=1))


@dataclass
class TeamSetEncoderFactory(EncoderFactory):
    grid_size: int = 4
    hidden_size: int = 128

    def create(self, observation_shape):
        return TeamSetEncoder(observation_shape[0], self.grid_size, self.hidden_size)

    @staticmethod
    def get_type() -> str:
        return "rift_team_set_v1"


register_encoder_factory(TeamSetEncoderFactory)
