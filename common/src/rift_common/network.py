"""Small shared value/support network; no collection or training orchestration."""

import torch
from torch import nn

from rift_common.actions import GROUPS, Action


class OptionNetwork(nn.Module):
    def __init__(self, size, *, separate_actor=False, bounded_values=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.value = nn.Linear(128, len(Action))
        self.behavior = nn.Linear(128, len(Action))
        self.bounded_values = bounded_values
        self.actor_encoder = (
            nn.Sequential(nn.Linear(size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            if separate_actor
            else None
        )

    def forward(self, observations):
        context = self.encoder(observations)
        actor = self.actor_encoder(observations) if self.actor_encoder is not None else context
        values = self.value(context)
        return values.tanh() if self.bounded_values else values, self.behavior(actor)


def action_support(counts, groups, masks, minimum=20):
    support = counts[groups] >= minimum
    # Never invent a voluntary option when all learned support is missing.
    return support & masks


def bellman_target(rewards, terminals, durations, next_values, gamma):
    """Elapsed-time discount; terminal transitions never bootstrap."""
    continuation = torch.where(terminals, torch.zeros_like(next_values), next_values)
    return rewards + torch.pow(gamma, durations / 60) * continuation


def rank_options(
    values,
    logits,
    counts,
    groups,
    masks,
    *,
    algorithm,
    minimum_support=20,
    behavior_floor=0.02,
    minimum_score_gap=0.0,
    contextual_support=None,
):
    import numpy as np

    if algorithm not in {"cql", "bc", "iql"}:
        raise ValueError("Unknown policy algorithm")
    values, logits, counts, groups, masks = map(np.asarray, (values, logits, counts, groups, masks))
    if (
        values.ndim != 2
        or values.shape[1] != len(Action)
        or logits.shape != values.shape
        or masks.shape != values.shape
        or masks.dtype != bool
        or groups.shape != (len(values),)
        or groups.dtype.kind not in "iu"
        or np.any((groups < 0) | (groups >= len(GROUPS)))
        or counts.shape != (len(GROUPS), len(Action))
        or counts.dtype.kind not in "iu"
        or np.any(counts < 0)
        or not np.isfinite(values).all()
        or not np.isfinite(logits).all()
        or not isinstance(minimum_support, int)
        or minimum_support < 1
        or not 0 <= behavior_floor < 1
        or not np.isfinite(minimum_score_gap)
        or minimum_score_gap < 0
    ):
        raise ValueError("Invalid policy ranking contract")
    allowed = action_support(counts, groups, masks, minimum_support)
    if contextual_support is not None:
        contextual_support = np.asarray(contextual_support)
        if contextual_support.shape != allowed.shape or contextual_support.dtype != bool:
            raise ValueError("Invalid contextual support")
        allowed &= contextual_support
    # A learned behavior-support constraint is distinct from a win probability.
    eligible_logits = np.where(masks, logits, -1e9)
    shifted = eligible_logits - np.max(eligible_logits, axis=1, keepdims=True)
    mass = np.exp(np.clip(shifted, -80, 0))
    probabilities = mass / mass.sum(axis=1, keepdims=True)
    allowed &= probabilities >= behavior_floor
    scores = values if algorithm == "cql" else logits
    scores = np.where(allowed, scores, -np.inf)
    predicted = scores.argmax(axis=1)
    predicted[~allowed.any(axis=1)] = -1
    if minimum_score_gap > 0:
        ordered = np.sort(scores, axis=1)
        competing = allowed.sum(1) > 1
        gaps = np.full(len(scores), np.inf)
        gaps[competing] = ordered[competing, -1] - ordered[competing, -2]
        predicted[gaps < minimum_score_gap] = -1
    return predicted, scores, allowed
