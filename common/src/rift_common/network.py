"""Small shared value/support network; no collection or training orchestration."""

import torch
from torch import nn

from rift_common.actions import Action


class OptionNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.value = nn.Linear(128, len(Action))
        self.behavior = nn.Linear(128, len(Action))

    def forward(self, observations):
        context = self.encoder(observations)
        return self.value(context), self.behavior(context)


def action_support(counts, groups, masks, minimum=20):
    support = counts[groups] >= minimum
    # Never invent a voluntary option when all learned support is missing.
    return support & masks


def bellman_target(rewards, terminals, durations, next_values, gamma):
    """Elapsed-time discount; terminal transitions never bootstrap."""
    return rewards + (~terminals).float() * torch.pow(gamma, durations / 60) * next_values


def rank_options(
    values, logits, counts, groups, masks, *, algorithm, minimum_support=20, behavior_floor=0.02
):
    import numpy as np

    allowed = action_support(
        np.asarray(counts), np.asarray(groups), np.asarray(masks), minimum_support
    )
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
    return predicted, scores, allowed
