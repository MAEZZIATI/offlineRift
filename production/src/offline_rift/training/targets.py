"""Explicit critic targets; terminal results remain supervision, never input features."""

import torch
from rift_common.network import bellman_target


@torch.no_grad()
def critic_targets(network, target, batch, support, config):
    if config.critic_target == "episode_return":
        return batch["episode_returns"]
    online, _ = network(batch["next_observations"])
    future, behavior = target(batch["next_observations"])
    legal = batch["next_action_masks"] & support[batch["groups"]]
    legal &= (
        behavior.masked_fill(~batch["next_action_masks"], -1e9).softmax(1) >= config.behavior_floor
    )
    action = online.masked_fill(~legal, -1e9).argmax(1)
    value = future.gather(1, action[:, None]).squeeze(1)
    value = torch.where(legal.any(1), value, torch.full_like(value, -1))
    return bellman_target(
        batch["rewards"], batch["terminals"], batch["durations"], value, config.gamma
    )
