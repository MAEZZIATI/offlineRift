"""Portable classifier bundles, exact serving parity and immutable training outputs."""

import builtins
import json

import numpy as np
import pytest
from rift_common.models.splits import MatchSplit
from rift_common.pipeline.action_labeler import MDPDatasetBuilder, load_dataset
from rift_common.pipeline.data_pipeline import TimelineProcessor
from rift_common.pipeline.features import FeatureConfig, ItemCatalog


@pytest.mark.parametrize("encoder_type", ["team_set", "mlp", "residual"])
def test_imitation_bundle_serves_without_cql_or_research(
    tmp_path, match_payloads, monkeypatch, encoder_type
):
    torch = pytest.importorskip("torch")
    from offline_rift.inference.policy import MacroPolicy
    from offline_rift.training.imitation import ImitationTrainer

    torch.set_num_threads(1)
    config = FeatureConfig()
    builder = MDPDatasetBuilder(config, ItemCatalog())
    for index in range(3):
        match, timeline = match_payloads(f"EUW1_{index}", win=index % 2 == 0)
        builder.add_match(match, TimelineProcessor(config).parse(match, timeline))
    dataset = tmp_path / "transitions.npz"
    builder.save(dataset)
    output = tmp_path / "model"
    trainer = ImitationTrainer(encoder_type=encoder_type, batch_size=4)
    report = trainer.train(dataset, output, n_steps=2)
    arrays, _ = load_dataset(dataset)

    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name.startswith(("d3rlpy", "rift_research", "offline_rift.training")):
            raise AssertionError(f"Runtime imported optional training/research code: {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", isolated_import)
        policy = MacroPolicy(output)
        validation = arrays["observations"][
            np.isin(arrays["match_ids"], report["validation_matches"])
        ]
        predicted = policy.predict_batch(validation, batch_size=3)
        served = [policy.recommend(row)["action_id"] for row in validation]
        np.testing.assert_array_equal(predicted, served)
        assert (
            np.bincount(predicted, minlength=12).tolist() == report["validation_prediction_counts"]
        )
        assert policy.recommend(validation[0])["q_values"] is None
        assert policy.model is None
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        trainer.train(dataset, output, n_steps=2)
    with (output / "macro_predictor.pt").open("ab") as stream:
        stream.write(b"modified model")
    with pytest.raises(ValueError, match="integrity check"):
        MacroPolicy(output)


def test_saved_match_split_rejects_leakage_and_changed_corpus():
    split = MatchSplit.create(["a", "a", "b", "b", "c", "c"])
    assert set(split.training).isdisjoint(split.validation)
    assert MatchSplit.from_report(split.to_dict(), ["a", "b", "c"]) == split
    with pytest.raises(ValueError, match="overlaps"):
        MatchSplit.from_report(
            {"training_matches": ["a"], "validation_matches": ["a", "b"]}, ["a", "b"]
        )
    with pytest.raises(ValueError, match="differs"):
        MatchSplit.from_report(split.to_dict(), ["a", "b", "d"])
