"""Real optimizer/save/load smoke test; optional when training extras are absent."""

from __future__ import annotations

import numpy as np
import pytest

from offline_rift.pipeline.action_labeler import MDPDatasetBuilder, load_dataset
from offline_rift.pipeline.data_pipeline import TimelineProcessor
from offline_rift.pipeline.features import FeatureConfig, ItemCatalog


@pytest.mark.parametrize("prediction_mode", ["hybrid", "balanced_bc", "cql"])
def test_cql_train_save_load_and_episode_alignment(tmp_path, match_payloads, prediction_mode):
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
    trainer = OfflineMacroTrainer(batch_size=4, prediction_mode=prediction_mode)
    report = trainer.train(path, tmp_path / "bundle", n_steps=2, n_steps_per_epoch=1)
    assert set(report["training_matches"]).isdisjoint(report["validation_matches"])
    assert len(report["validation_matches"]) == 1
    assert trainer.model.action_size == 12  # Even when only some labels were observed.
    policy = MacroPolicy(tmp_path / "bundle")
    recommendation = policy.recommend(arrays["observations"][0])
    assert recommendation["action_id"] in range(12)
    assert len(recommendation["q_values"]) == 12
    assert np.isfinite(recommendation["q_value"])
    assert recommendation["prediction_method"] == prediction_mode
    if prediction_mode == "cql":
        assert recommendation["macro_scores"] is None
    else:
        assert len(recommendation["macro_scores"]) == 12
        assert set(report["prediction_comparison"]) == {"cql", "balanced_bc", "hybrid"}
    validation = arrays["observations"][np.isin(arrays["match_ids"], report["validation_matches"])]
    served = [policy.recommend(observation)["action_id"] for observation in validation]
    assert np.bincount(served, minlength=12).tolist() == report["validation_prediction_counts"]
    before = trainer.model.predict_value(arrays["observations"][:1], np.array([0]))
    after = policy.model.predict_value(arrays["observations"][:1], np.array([0]))
    np.testing.assert_allclose(before, after)


def test_team_encoder_is_invariant_to_player_order():
    torch = pytest.importorskip("torch")
    pytest.importorskip("d3rlpy")
    from offline_rift.pipeline.features import FeatureEncoder
    from offline_rift.training.encoders import TeamSetEncoderFactory

    torch.manual_seed(42)
    config = FeatureConfig(grid_size=3)
    size = len(FeatureEncoder(config, ItemCatalog()).names)
    encoder = TeamSetEncoderFactory(grid_size=3).create((size,))
    x = torch.randn(4, size)
    reordered = x.clone()
    indices = encoder.player_indices.reshape(2, 5, 10)
    reordered[:, indices.flatten()] = x[:, indices[:, [3, 0, 4, 1, 2]].flatten()]
    torch.testing.assert_close(encoder(x), encoder(reordered))
    encoder(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_metrics_penalize_collapse_and_report_rare_action_recall():
    from offline_rift.training.evaluation import classification_metrics

    truth = np.array([0] * 90 + [1] * 10)
    collapsed = classification_metrics(truth, np.zeros(100, dtype=int), 12)
    assert collapsed["heuristic_agreement"] == 0.9
    assert collapsed["balanced_accuracy"] == 0.5
    assert collapsed["per_action_recall"][1] == 0
    assert collapsed["collapsed_predictions"] is True
    perfect = classification_metrics(truth, truth, 12)
    assert perfect["balanced_accuracy"] == perfect["macro_f1"] == 1


def test_hybrid_masks_impossible_and_unsupported_actions():
    pytest.importorskip("torch")
    pytest.importorskip("d3rlpy")
    from offline_rift.pipeline.features import FeatureEncoder
    from offline_rift.training.macro_predictor import hybrid_scores

    names = FeatureEncoder(FeatureConfig(), ItemCatalog()).names
    observations = np.zeros((1, len(names)), dtype=np.float32)
    logits = np.zeros((1, 12))
    logits[0, :3] = 100  # Objective areas should not win when disabled.
    logits[0, 7] = 90  # Neither should an unseen class.
    counts = [1] * 12
    counts[7] = 0
    values = np.arange(12, dtype=float).reshape(1, -1)
    scores = hybrid_scores(logits, values, counts, observations, names)
    assert not np.isfinite(scores[0, :3]).any()
    assert not np.isfinite(scores[0, 7])
    shifted = hybrid_scores(logits, values * 10 + 50, counts, observations, names)
    np.testing.assert_allclose(scores, shifted)
    counts = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert hybrid_scores(logits, values, counts, observations, names).argmax() == 11
