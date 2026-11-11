import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from offline_rift.inference.policy import NextActionPolicy
from offline_rift.training.trainer import PolicyTrainer, TrainConfig
from rift_common.actions import GROUPS, Action, feasible_actions
from rift_common.dataset import DatasetBuilder
from rift_common.state import StateEncoder
from rift_common.support import phase_counts, phase_support
from rift_common.timeline import read_match


def test_relative_features_preserve_v1_prefix_and_unknown_operands(match_payloads, catalog):
    observed = read_match(*match_payloads())[3]
    old = StateEncoder(catalog, schema="decision-state-v1")
    new = StateEncoder(catalog)
    encoded = new.encode(observed, 100, "1")
    np.testing.assert_array_equal(encoded[: len(old.names)], old.encode(observed, 100, "1"))
    assert len(new.names) == len(old.names) + 24
    idx = new.names.index
    assert (
        encoded[idx("team_level_advantage")]
        == encoded[idx("ally_level")] - encoded[idx("enemy_level")]
    )
    assert encoded[idx("ally_alive_known")] == 1
    incomplete = replace(
        observed,
        players=tuple(
            replace(p, level=None, is_dead=None) if p.id == "6" else p for p in observed.players
        ),
    )
    missing = new.encode(incomplete, 100, "1")
    for name in ("team_level_advantage", "self_enemy_level_difference", "enemy_alive"):
        assert missing[idx(name)] == missing[idx(name + "_known")] == 0


def test_phase_support_is_training_derived_and_distinguishes_game_phases():
    observed = np.array([[0.1], [0.3], [0.6]], dtype=np.float32)
    groups = np.full(3, GROUPS.index("TOP"))
    actions = np.array([Action.FARM_TOP, Action.ROTATE_TOP, Action.PREPARE_BARON])
    counts = phase_counts(observed, groups, actions, ["time"])
    support = phase_support(observed, groups, ["time"], counts, 1)
    assert support.sum(1).tolist() == [1, 1, 1]
    assert support[np.arange(3), actions].all()
    assert not phase_support(observed, groups, ["time"], counts, 2).any()


@pytest.mark.parametrize("algorithm", ["bc", "cql", "iql"])
def test_development_checkpoint_is_restored_and_phase_support_reloads(
    tmp_path, match_payloads, catalog, monkeypatch, algorithm
):
    builder = DatasetBuilder()
    for i in range(4):
        builder.add_match(*match_payloads(f"KR_{i}"), catalog)
    data = tmp_path / "data.npz"
    builder.save(data, catalog=catalog, provenance={"fingerprint": "checkpoint-test"})
    snapshots = []

    def diagnostic(network, *args):
        snapshots.append(deepcopy(network.state_dict()))
        return (True, -len(snapshots), 0), {"score": -len(snapshots)}

    monkeypatch.setattr("offline_rift.training.trainer.checkpoint_diagnostics", diagnostic)
    result = PolicyTrainer(
        TrainConfig(
            algorithm=algorithm,
            steps=5,
            evaluation_interval=1,
            early_stopping_patience=1,
            minimum_support=1,
            minimum_phase_support=1,
            holdout_matches=("KR_3",),
        )
    ).train(data, tmp_path / "model")
    assert result["report"]["optimizer_steps"] == 2
    assert result["report"]["selected_checkpoint_step"] == 1
    assert result["report"]["holdout_matches"] == ["KR_3"]
    policy = NextActionPolicy(tmp_path / "model")
    for name, parameter in policy.network.state_dict().items():
        torch.testing.assert_close(parameter, snapshots[0][name])
    observed = read_match(*match_payloads())[3]
    state = policy.encoder.encode(observed, 100, "1")
    prediction, _ = policy.predict_batch(
        state[None, :], np.array([0]), feasible_actions(state, policy.encoder.names)
    )
    assert prediction.shape == (1,)


def test_explicit_baseline_projection_and_paired_match_bootstrap(tmp_path, match_payloads, catalog):
    from rift_research.evaluation import compare_holdout, evaluate_policy

    builder = DatasetBuilder()
    for i in range(4):
        builder.add_match(*match_payloads(f"KR_{i}"), catalog)
    data = tmp_path / "v2.npz"
    builder.save(data, catalog=catalog, provenance={"fingerprint": "projection-test"})
    with np.load(data, allow_pickle=False) as archive:
        old = dict(archive)
    metadata = json.loads(str(old["metadata"].item()))
    names = StateEncoder(catalog, schema="decision-state-v1").names
    metadata.update(state_schema="decision-state-v1", feature_names=names)
    old["metadata"] = np.asarray(json.dumps(metadata))
    for key in ("observations", "next_observations"):
        old[key] = old[key][:, : len(names)]
    np.savez_compressed(tmp_path / "v1.npz", **old)
    PolicyTrainer(
        TrainConfig(algorithm="bc", steps=2, minimum_support=1, holdout_matches=("KR_3",))
    ).train(tmp_path / "v1.npz", tmp_path / "old-model")
    direct = evaluate_policy(
        tmp_path / "v1.npz",
        tmp_path / "old-model",
        tmp_path / "direct",
        samples=3,
        new_matches=True,
    )
    projected = evaluate_policy(
        data,
        tmp_path / "old-model",
        tmp_path / "projected",
        samples=3,
        new_matches=True,
        project_state=True,
    )
    assert direct["metrics"] == projected["metrics"]
    assert len(projected["state_projection"]["dropped_features"]) == 24
    evaluate_policy(
        data,
        tmp_path / "old-model",
        tmp_path / "second",
        samples=3,
        new_matches=True,
        project_state=True,
    )
    comparison = compare_holdout(
        tmp_path / "projected", tmp_path / "second", tmp_path / "comparison.json", samples=5
    )
    assert comparison["matches"] == ["KR_3"]
    assert all(
        row["f1_difference_95"] in (None, [0, 0]) for row in comparison["paired_differences"]
    )
    with pytest.raises(ValueError, match="Only explicit"):
        evaluate_policy(
            data, tmp_path / "old-model", tmp_path / "invalid", samples=3, project_state=True
        )
