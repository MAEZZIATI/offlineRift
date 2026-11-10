"""Observation-only dropout, preserving recorded actions and recomputing feasibility."""

from collections import Counter

import torch
from rift_common.actions import feasible_actions
from rift_common.observation_masks import COLD_START_FIELDS


class ObservationAugmenter:
    def __init__(self, names, *, joint_dropout=0):
        self.names, self.joint_dropout = names, joint_dropout
        self.optional = [
            names.index(name)
            for name in (
                "self_health",
                "self_cs_rate",
                "self_level_rate",
                "own_towers_lost",
                "enemy_towers_lost",
            )
        ]
        self.joint = [names.index(name) for name in COLD_START_FIELDS]
        self.counts = Counter()

    def __call__(self, observations, masks, *, actions=None):
        observed = observations.clone()
        for index in self.optional:
            missing = torch.rand(len(observed), device=observed.device) < 0.2
            observed[missing, index : index + 2] = 0
        if not self.joint_dropout:
            return observed, masks
        prefix = "current" if actions is not None else "next"
        selected = torch.rand(len(observed), device=observed.device) < self.joint_dropout
        proposed = observed.clone()
        proposed[selected.nonzero().flatten()[:, None], self.joint] = 0
        proposed_masks = torch.as_tensor(
            feasible_actions(proposed.detach().cpu().numpy(), self.names), device=observed.device
        )
        rejected = torch.zeros_like(selected)
        if actions is not None:
            rejected = selected & ~proposed_masks.gather(1, actions[:, None]).squeeze(1)
            # Never change a target or fit an option made impossible by the omission.
            proposed[rejected] = observed[rejected]
            proposed_masks[rejected] = masks[rejected]
        self.counts.update(
            {
                prefix + "_joint_proposed": int(selected.sum()),
                prefix + "_joint_applied": int((selected & ~rejected).sum()),
                prefix + "_joint_rejected_infeasible_target": int(rejected.sum()),
            }
        )
        return proposed, proposed_masks
