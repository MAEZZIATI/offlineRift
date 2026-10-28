"""Real optimizer/save/load smoke test; optional when training extras are absent."""

from __future__ import annotations

import numpy as np
import pytest

from offline_rift.pipeline.action_labeler import MDPDatasetBuilder, load_dataset
from offline_rift.pipeline.data_pipeline import TimelineProcessor
from offline_rift.pipeline.features import FeatureConfig, ItemCatalog


def test_cql_train_save_load_and_episode_alignment(tmp_path, match_payloads):
    pytest.importorskip("d3rlpy")
    import torch

    from offline_rift.training.rl_agent import MacroPolicy, OfflineMacroTrainer

    torch.set_num_threads(1)
    config = FeatureConfig()
    builder = MDPDatasetBuilder(config, ItemCatalog())
    for i in range(3):
        match, timeline = match_payloads(f"EUW1_{i}", win=i % 2 == 0)
        builder.add_match(match, TimelineProcessor(config).parse(match, timeline))
    path = tmp_path / "dataset.npz"
    builder.save(path)
    arrays, _ = load_dataset(path)
    dataset = OfflineMacroTrainer.to_mdp_dataset(arrays)
    assert dataset.transition_count == len(arrays["actions"])
    assert len(dataset.episodes) == 6
    # Inspect transitions directly: next observation and reward align exactly.
    picker = dataset.transition_picker
    transition = picker(dataset.episodes[0], 0)
    np.testing.assert_array_equal(transition.next_observation, arrays["next_observations"][0])
    assert picker(dataset.episodes[0], 12).terminal == 1
    trainer = OfflineMacroTrainer(batch_size=4)
    report = trainer.train(path, tmp_path / "bundle", n_steps=2, n_steps_per_epoch=1)
    assert set(report["training_matches"]).isdisjoint(report["validation_matches"])
    assert len(report["validation_matches"]) == 1
    assert trainer.model.action_size == 5  # Even when only some labels were observed.
    policy = MacroPolicy(tmp_path / "bundle")
    recommendation = policy.recommend(arrays["observations"][0])
    assert recommendation["action_id"] in range(5)
    assert len(recommendation["q_values"]) == 5
    assert np.isfinite(recommendation["q_value"])
    before = trainer.model.predict_value(arrays["observations"][:1], np.array([0]))
    after = policy.model.predict_value(arrays["observations"][:1], np.array([0]))
    np.testing.assert_allclose(before, after)
