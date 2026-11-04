"""Causal history, mask integrity, fixed splits, and real GRU optimizer integration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("d3rlpy")

from offline_rift.training.memory import (
    HistoryClassifier,
    batch_history,
    evaluate_memory_experiment,
    fixed_match_split,
    history_indices,
    train_memory_experiment,
)


def test_history_never_crosses_episode_clock_reset_or_horizon():
    episodes = np.array(["a"] * 5 + ["b"] * 3)
    timestamps = np.array([0, 10, 20, 5, 15, 0, 10, 40])
    indices, lengths = history_indices(episodes, timestamps, 20)
    assert indices[2].tolist() == [0, 1, 2]
    assert indices[3].tolist() == [3, -1, -1]  # Rollback resets history.
    assert indices[5].tolist() == [5, -1, -1]  # Opposite team/new episode.
    assert indices[7].tolist() == [7, -1, -1]  # Gap exceeds horizon.
    for i, count in enumerate(lengths):
        rows = indices[i, :count]
        assert (rows <= i).all()
        assert (episodes[rows] == episodes[i]).all()
        assert (timestamps[i] - timestamps[rows] <= 20).all()
    assert fixed_match_split(np.repeat(["a", "b", "c"], 2), seed=1) == fixed_match_split(
        np.array(["a", "b", "c"]), seed=1
    )


def test_gru_ignores_padding_and_future_values_and_keeps_feature_masks():
    torch.set_num_threads(1)
    from offline_rift.pipeline.features import FeatureConfig, FeatureEncoder, ItemCatalog

    size = len(FeatureEncoder(FeatureConfig(), ItemCatalog()).names)
    model = HistoryClassifier(size, 4, recurrent=True).eval()
    observations = torch.zeros(4, size)
    observations[:, 0] = torch.arange(4) / 100
    timestamps = torch.arange(4) * 10
    index, length = history_indices(np.array(["a"] * 4), timestamps.numpy(), 20)
    args = batch_history(
        observations, timestamps, torch.tensor(index), torch.tensor(length), torch.tensor([1]), 20
    )
    before = model(*args)
    changed = args[0].clone()
    changed[:, 2] = 1000  # Right padding is ignored by the packed GRU.
    torch.testing.assert_close(before, model(changed, args[1], args[2]))
    observations[2:] = 999  # Future rows cannot change row 1's inputs or output.
    later = batch_history(
        observations, timestamps, torch.tensor(index), torch.tensor(length), torch.tensor([1]), 20
    )
    torch.testing.assert_close(before, model(*later))
    assert (later[0][..., 1:] == 0).all()  # No unmasking or imputation.
    snapshot = HistoryClassifier(size, 4, recurrent=False)
    assert (
        abs(
            sum(p.numel() for p in model.parameters())
            - sum(p.numel() for p in snapshot.parameters())
        )
        == 3
    )
    repeated = HistoryClassifier(size, 4, recurrent=True, repeat_current=True).eval()
    prior_changed = later[0].clone()
    prior_changed[:, 0] = 1000
    torch.testing.assert_close(repeated(*later), repeated(prior_changed, later[1], later[2]))


def test_memory_optimizer_saves_reloads_and_groups_matches(tmp_path, match_payloads):
    from offline_rift.pipeline.action_labeler import MDPDatasetBuilder
    from offline_rift.pipeline.data_pipeline import TimelineProcessor
    from offline_rift.pipeline.features import FeatureConfig, ItemCatalog

    builder = MDPDatasetBuilder(FeatureConfig(), ItemCatalog())
    for i in range(3):
        match, timeline = match_payloads(f"KR_{i}")
        builder.add_match(match, TimelineProcessor(FeatureConfig()).parse(match, timeline))
    path = tmp_path / "data.npz"
    builder.save(path)
    report = train_memory_experiment(
        path, tmp_path / "run", seeds=(7,), n_steps=2, batch_size=4, bootstrap_samples=10
    )
    assert report["status"] == "complete"
    assert set(report["training_matches"]).isdisjoint(report["validation_matches"])
    assert len(report["runs"]) == 3
    assert all(row["optimizer_steps"] == 2 for row in report["runs"])
    assert len(list((tmp_path / "run").glob("*.pt"))) == 3
    assert set(report["per_seed_bootstrap"]["7"]["paired_differences"]) == {
        "gru60_minus_snapshot",
        "gru120_minus_snapshot",
    }
    with pytest.raises(ValueError, match="No unseen matches"):
        evaluate_memory_experiment(
            tmp_path / "run", path, tmp_path / "unseen", bootstrap_samples=10
        )
    match, timeline = match_payloads("KR_unseen")
    builder.add_match(match, TimelineProcessor(FeatureConfig()).parse(match, timeline))
    builder.save(tmp_path / "expanded.npz")
    unseen = evaluate_memory_experiment(
        tmp_path / "run", tmp_path / "expanded.npz", tmp_path / "unseen", bootstrap_samples=10
    )
    assert unseen["test_matches"] == ["KR_unseen"]
    assert len(unseen["runs"]) == 3
    assert unseen["status"] == "complete"
    control = train_memory_experiment(
        path,
        tmp_path / "control",
        seeds=(7,),
        n_steps=2,
        batch_size=4,
        bootstrap_samples=10,
        methods=("repeat120",),
    )
    assert control["methods"] == ["repeat120"]
    with_control = evaluate_memory_experiment(
        tmp_path / "run",
        tmp_path / "expanded.npz",
        tmp_path / "controlled-test",
        bootstrap_samples=10,
        control_from=tmp_path / "control",
    )
    assert len(with_control["runs"]) == 4
    assert "gru120_minus_repeat120" in with_control["per_seed_bootstrap"]["7"]["paired_differences"]
    from offline_rift.training.rl_agent import OfflineMacroTrainer

    OfflineMacroTrainer(batch_size=4).train(
        path, tmp_path / "plugged", n_steps=2, n_steps_per_epoch=1
    )
    reference = evaluate_memory_experiment(
        tmp_path / "run",
        tmp_path / "expanded.npz",
        tmp_path / "reference-test",
        bootstrap_samples=10,
        reference_bundle=tmp_path / "plugged",
    )
    assert set(reference["plugged_reference"]["metrics"]) == {"balanced_bc", "hybrid"}
    assert reference["test_matches"] == ["KR_unseen"]
