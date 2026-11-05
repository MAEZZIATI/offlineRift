import json

import numpy as np
import pytest
from rift_common.dataset import DatasetBuilder, load_dataset


@pytest.mark.parametrize("algorithm", ["cql", "bc"])
def test_real_optimizer_reload_serving_parity_and_immutable_outputs(
    tmp_path, match_payloads, catalog, algorithm
):
    pytest.importorskip("torch")
    from offline_rift.inference.policy import NextActionPolicy
    from offline_rift.training.trainer import PolicyTrainer, TrainConfig

    builder = DatasetBuilder()
    for i in range(3):
        builder.add_match(*match_payloads(f"KR_{i}", win=i % 2 == 0), catalog)
    data = tmp_path / "dataset.npz"
    builder.save(data, catalog=catalog, provenance={"fingerprint": "fixture"})
    output = tmp_path / "policy"
    result = PolicyTrainer(
        TrainConfig(algorithm=algorithm, steps=3, batch_size=8, minimum_support=1, cpu_threads=1)
    ).train(data, output)
    assert result["status"] == "complete"
    report = result["report"]
    assert set(report["training_matches"]).isdisjoint(report["validation_matches"])
    assert report["optimizer_steps"] == 3
    arrays, _ = load_dataset(data)
    mask = np.isin(arrays["match_ids"], report["validation_matches"])
    policy = NextActionPolicy(output)
    predicted, _ = policy.predict_batch(
        arrays["observations"][mask], arrays["groups"][mask], arrays["action_masks"][mask]
    )
    assert (
        np.bincount(predicted[predicted >= 0], minlength=15).tolist()
        == report["validation"]["prediction_counts"]
    )
    assert (predicted < 0).sum() == report["validation"]["abstentions"]
    with pytest.raises(FileExistsError):
        PolicyTrainer(TrainConfig(steps=1)).train(data, output)
    with (output / "policy.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        NextActionPolicy(output)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(json.dumps({"schema_version": "rift-v2"}))
    with pytest.raises(ValueError, match="Legacy formation"):
        NextActionPolicy(legacy)


def test_discount_uses_elapsed_time_and_never_bootstraps_terminal():
    torch = pytest.importorskip("torch")
    from rift_common.network import bellman_target

    result = bellman_target(
        torch.tensor([1.0, 0.0]),
        torch.tensor([True, False]),
        torch.tensor([60.0, 120.0]),
        torch.tensor([999.0, 1.0]),
        0.9,
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 0.81]))


def test_cql_ranks_returns_with_support_instead_of_blending_classifier_scores():
    from rift_common.network import rank_options

    values = np.zeros((1, 15))
    values[0, 1] = 2
    logits = np.zeros((1, 15))
    logits[0, 0] = 2
    logits[0, 1] = 1
    logits[0, 13] = 1000
    masks = np.ones((1, 15), dtype=bool)
    masks[0, 13] = False
    counts = np.full((7, 15), 100)
    cql, _, _ = rank_options(
        values, logits, counts, np.array([0]), masks, algorithm="cql", minimum_support=1
    )
    bc, _, _ = rank_options(
        values, logits, counts, np.array([0]), masks, algorithm="bc", minimum_support=1
    )
    assert cql.tolist() == [1] and bc.tolist() == [0]
    # An impossible high-logit respawn option must not erase valid behavior support.
    none, _, _ = rank_options(
        values, logits, np.zeros((7, 15)), np.array([0]), masks, algorithm="cql", minimum_support=1
    )
    assert none.tolist() == [-1]
