"""Shared classifier construction and bounded evidence scoring."""

import numpy as np
import torch
from torch import nn

from rift_common.models.encoders import TeamSetEncoder
from rift_common.pipeline.actions import ACTION_NAMES


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.LayerNorm(width)
        )

    def forward(self, inputs):
        return torch.relu(inputs + self.layers(inputs))


def create_predictor(size: int, grid_size: int, encoder_type: str) -> nn.Module:
    if encoder_type == "team_set":
        encoder = TeamSetEncoder(size, grid_size, 128)
    elif encoder_type == "mlp":
        encoder = nn.Sequential(nn.Linear(size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
    elif encoder_type == "residual":
        encoder = nn.Sequential(
            nn.Linear(size, 128), nn.ReLU(), ResidualBlock(128), ResidualBlock(128)
        )
    else:
        raise ValueError(f"Unknown predictor encoder: {encoder_type}")
    return nn.Sequential(encoder, nn.Linear(128, len(ACTION_NAMES)))


def predictor_logits(model: nn.Module, observations: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        # Bound inference memory for large real-data holdouts.
        return np.concatenate(
            [
                model(torch.as_tensor(batch, dtype=torch.float32, device=device)).cpu().numpy()
                for batch in np.array_split(
                    observations, max(1, (len(observations) + 1023) // 1024)
                )
            ]
        )


def hybrid_scores(
    logits: np.ndarray,
    q_values: np.ndarray,
    counts: list[int],
    observations: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Pattern evidence plus a bounded return tie-breaker; no probability claim."""
    if not np.isfinite(logits).all() or not np.isfinite(q_values).all():
        raise ValueError("Non-finite macro prediction")
    advantage = q_values - q_values.mean(axis=1, keepdims=True)
    scale = np.maximum(q_values.std(axis=1, keepdims=True), 1e-6)
    scores = logits + 0.25 * np.clip(advantage / scale, -2, 2)
    scores[:, np.asarray(counts) == 0] = -np.inf
    # Necessary conditions only: never infer positions, vision, safety or success.
    for action, kind in enumerate(("dragon", "baron", "herald")):
        enabled = observations[:, feature_names.index(f"{kind}_enabled")] > 0
        soon = observations[:, feature_names.index(f"{kind}_eta")] <= 0.1 + 1e-7
        scores[~(enabled & soon), action] = -np.inf
    scores[~np.isfinite(scores).any(axis=1), 11] = 0  # Abstain if no supported action remains.
    return scores
