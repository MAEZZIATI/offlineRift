"""Match-cluster uncertainty, split isolation and blinded sampling invariants."""

import copy

import numpy as np
import pytest
from rift_research.evaluation import audit_rows, match_bootstrap, validate_split


def test_bootstrap_keeps_correlated_rows_and_pairs_methods():
    # Duplicating every observation cannot add independent experimental evidence.
    truth = np.zeros(4, dtype=int)
    predictions = {"bc_filtered": np.array([0, 0, 1, 1]), "hybrid_filtered": np.array([0, 0, 1, 1])}
    matches = np.array(["one", "one", "two", "two"])
    report = match_bootstrap(truth, predictions, matches, 2, samples=200)
    duplicated = match_bootstrap(
        np.repeat(truth, 5),
        {k: np.repeat(v, 5) for k, v in predictions.items()},
        np.repeat(matches, 5),
        2,
        samples=200,
    )
    assert report == duplicated
    assert report["intervals"]["bc_filtered"]["heuristic_agreement"] == [0, 1]
    assert all(
        bounds == [0, 0]
        for bounds in report["paired_differences"]["hybrid_filtered_minus_bc_filtered"].values()
    )
    with pytest.raises(ValueError, match="positive samples"):
        match_bootstrap(truth, predictions, matches, 2, samples=0)


def test_saved_split_rejects_leakage_and_dataset_changes():
    arrays = {"match_ids": np.array(["a", "a", "b", "b"]), "actions": np.array([0, 1, 1, 1])}
    metadata = {
        "schema_version": "test",
        "feature_names": ["x"],
        "feature_config": {},
        "action_schema": "test",
        "provenance": {},
    }
    bundle = copy.deepcopy(metadata)
    bundle["training_report"] = {
        "training_matches": ["a"],
        "validation_matches": ["b"],
        "validation_transitions": 2,
        "action_counts": [1, 1],
    }
    np.testing.assert_array_equal(validate_split(arrays, metadata, bundle), [0, 0, 1, 1])
    changed = copy.deepcopy(bundle)
    changed["training_report"]["training_matches"].append("b")
    with pytest.raises(ValueError, match="match split"):
        validate_split(arrays, metadata, changed)
    changed = copy.deepcopy(bundle)
    changed["provenance"] = {"fingerprint": "other"}
    with pytest.raises(ValueError, match="provenance"):
        validate_split(arrays, metadata, changed)
    with pytest.raises(ValueError, match="No unseen matches"):
        validate_split(arrays, metadata, bundle, unseen=True)
    new_arrays = {"match_ids": np.array(["a", "b", "c"]), "actions": np.array([0, 1, 1])}
    np.testing.assert_array_equal(
        validate_split(new_arrays, metadata, changed, unseen=True), [False, False, True]
    )


def test_audit_sampling_is_blinded_reproducible_and_training_only():
    arrays = {
        "match_ids": np.array(["a", "a", "b", "c"]),
        "actions": np.array([0, 0, 0, 0]),
        "timestamps": np.array([0, 10, 20, 30]),
        "teams": np.array([100, 200, 100, 100]),
        "label_confidence": np.ones(4),
    }
    arguments = (arrays, ["a", "b"], {"a": "16.20", "b": "16.20", "c": "16.20"})
    rows, keys = audit_rows(*arguments, per_cell=3)
    assert (rows, keys) == audit_rows(*arguments, per_cell=3)
    assert {row["match_id"] for row in rows} == {"a", "b"}
    assert len(rows) == 2  # Two matches, despite repeated states and both perspectives.
    assert all("weak_label" not in row and row["reviewer_1_label"] == "" for row in rows)
    assert all(row["review_start_seconds"] >= 0 for row in rows)
    assert [row["sample_id"] for row in rows] == [row["sample_id"] for row in keys]
