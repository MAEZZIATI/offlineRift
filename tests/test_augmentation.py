import numpy as np
import pytest
import torch
from offline_rift.training.augmentation import ObservationAugmenter
from offline_rift.training.trainer import TrainConfig
from rift_common.actions import Action, feasible_actions
from rift_common.network import OptionNetwork
from rift_common.observation_masks import COLD_START_FIELDS
from rift_common.state import StateEncoder
from rift_common.timeline import read_match


def test_joint_dropout_preserves_targets_and_recomputes_next_feasibility(match_payloads, catalog):
    encoder = StateEncoder(catalog)
    observed = read_match(*match_payloads())[3]
    state = encoder.encode(observed, 100, "1")
    state[encoder.names.index("dragon_enabled")] = 1
    state[encoder.names.index("dragon_eta")] = 0
    states = torch.tensor(np.stack([state, state]))
    original = states.clone()
    masks = torch.tensor(feasible_actions(states.numpy(), encoder.names))
    augmenter = ObservationAugmenter(encoder.names, joint_dropout=1)
    changed, allowed = augmenter(
        states, masks, actions=torch.tensor([Action.CONTINUE, Action.PREPARE_DRAGON])
    )
    assert not changed[0, [encoder.names.index(n) for n in COLD_START_FIELDS]].any()
    assert not allowed[0, Action.PREPARE_DRAGON]
    assert allowed[1, Action.PREPARE_DRAGON]
    assert changed[1, encoder.names.index("dragon_enabled")] == 1
    np.testing.assert_array_equal(allowed.numpy(), feasible_actions(changed.numpy(), encoder.names))
    torch.testing.assert_close(states, original)
    future, future_masks = augmenter(states, masks)
    assert not future_masks[:, Action.PREPARE_DRAGON].any()
    assert not future[:, [encoder.names.index(n) for n in COLD_START_FIELDS]].any()
    assert augmenter.counts["current_joint_rejected_infeasible_target"] == 1
    assert augmenter.counts["next_joint_applied"] == 2


def test_bounded_critic_keeps_actor_logits_unbounded_and_gradients_finite():
    network = OptionNetwork(4, bounded_values=True)
    with torch.no_grad():
        network.value.bias.fill_(10)
        network.behavior.bias.fill_(10)
    values, logits = network(torch.ones(2, 4))
    assert values.abs().max() <= 1 and logits.max() > 1
    values.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in network.encoder.parameters())


@pytest.mark.parametrize(
    "settings",
    [
        {"joint_history_dropout": -0.1},
        {"joint_history_dropout": 1.1},
        {"bounded_values": True, "algorithm": "iql"},
    ],
)
def test_invalid_robustness_settings(settings):
    with pytest.raises(ValueError, match="configuration"):
        TrainConfig(**settings)
