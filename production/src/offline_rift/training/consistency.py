"""Availability-aware regularization, without reconstructing missing history."""

import torch
from rift_common.actions import feasible_actions
from rift_common.observation_masks import COLD_START_FIELDS
from torch.nn import functional as F


def history_consistency(network, observations, masks, actions, groups, support, names, *, critic):
    """Compare only supported actions feasible under both information patterns.

    The clean prediction is a detached teacher, not ground truth. Rows whose recorded
    action becomes infeasible are excluded. A small penalty is a robustness assumption,
    not a claim that removing information should leave the optimal decision unchanged.
    """
    cold = observations.clone()
    cold[:, [names.index(name) for name in COLD_START_FIELDS]] = 0
    cold_masks = torch.as_tensor(
        feasible_actions(cold.detach().cpu().numpy(), names), device=observations.device
    )
    common = masks & cold_masks & support[groups]
    eligible = common.gather(1, actions[:, None]).squeeze(1) & (common.sum(1) > 1)
    if not eligible.any():
        return observations.new_zeros(())
    with torch.no_grad():
        clean_q, clean_logits = network(observations[eligible])
    cold_q, cold_logits = network(cold[eligible])
    allowed = common[eligible]
    teacher = clean_logits.masked_fill(~allowed, -1e9).softmax(1)
    loss = F.kl_div(
        cold_logits.masked_fill(~allowed, -1e9).log_softmax(1), teacher, reduction="batchmean"
    )
    if critic:
        loss = loss + F.smooth_l1_loss(cold_q[allowed], clean_q[allowed])
    return loss
