"""Discrete IQL updates; the second critic and expectile value are training-only.

Based on https://arxiv.org/abs/2110.06169. Categorical masked policy extraction
replaces the paper's continuous actor. Rewards and measured-time discount are shared.
"""

import copy
import math

import torch
from rift_common.network import bellman_target
from torch import nn
from torch.nn import functional as F


def expectile_loss(residual, expectile):
    return (torch.where(residual > 0, expectile, 1 - expectile) * residual.square()).mean()


class IQLUpdate:
    def __init__(self, network, config):
        self.network, self.config = network, config
        self.q1 = nn.Sequential(network.encoder, network.value)
        self.q2 = copy.deepcopy(self.q1)
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.value = nn.Sequential(
            copy.deepcopy(network.encoder), nn.Linear(128, 1).to(config.device)
        )
        self.actor = nn.Sequential(network.actor_encoder, network.behavior)
        self.critic_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.target1.requires_grad_(False)
        self.target2.requires_grad_(False)

    @staticmethod
    def optimize(optimizer, loss, parameters):
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite IQL optimizer loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
        optimizer.step()

    def step(self, batch):
        config = self.config
        states, actions = batch["observations"], batch["actions"][:, None]
        with torch.no_grad():
            recorded = torch.minimum(
                self.target1(states).gather(1, actions),
                self.target2(states).gather(1, actions),
            ).squeeze(1)
        value_loss = expectile_loss(recorded - self.value(states).squeeze(1), config.expectile)
        self.optimize(self.value_optimizer, value_loss, self.value.parameters())
        with torch.no_grad():
            advantage = recorded - self.value(states).squeeze(1)
            weights = (
                (config.advantage_temperature * advantage)
                .clamp(max=math.log(config.advantage_clip))
                .exp()
            )
            targets = bellman_target(
                batch["rewards"],
                batch["terminals"],
                batch["durations"],
                self.value(batch["next_observations"]).squeeze(1),
                config.gamma,
            )
        logits = self.actor(states).masked_fill(~batch["action_masks"], -1e9)
        actor_loss = (weights * F.cross_entropy(logits, batch["actions"], reduction="none")).mean()
        self.optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
        critic_loss = F.mse_loss(self.q1(states).gather(1, actions).squeeze(1), targets)
        critic_loss += F.mse_loss(self.q2(states).gather(1, actions).squeeze(1), targets)
        self.optimize(
            self.critic_optimizer, critic_loss, [*self.q1.parameters(), *self.q2.parameters()]
        )
        with torch.no_grad():
            for target, online in ((self.target1, self.q1), (self.target2, self.q2)):
                for destination, source in zip(target.parameters(), online.parameters()):
                    destination.lerp_(source, config.target_tau)
        return {
            "loss": float((value_loss + actor_loss + critic_loss).detach()),
            "td_loss": float(critic_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "behavior_loss": float(actor_loss.detach()),
            "advantage_weight_mean": float(weights.mean()),
            "advantage_weight_max": float(weights.max()),
        }
