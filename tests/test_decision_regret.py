import numpy as np
from rift_research.calibration import fit_evidence_thresholds
from rift_research.decision_regret import match_coverage, minimax_decision


def test_minimax_compromise_offset_invariance_and_support_intersection():
    # The two views disagree on their best action. Action 2 has smaller worst gap
    # than either winner, despite never being a view's argmax.
    values = np.array([[[1, 0, 0.7]], [[0, 1, 0.7]]])
    allowed = np.ones_like(values, dtype=bool)
    predicted, regret, _ = minimax_decision(values, allowed)
    assert predicted.tolist() == [2]
    np.testing.assert_allclose(regret, [0.3])
    shifted, shifted_regret, _ = minimax_decision(
        values + np.array([10, -7])[:, None, None], allowed
    )
    np.testing.assert_array_equal(predicted, shifted)
    np.testing.assert_allclose(regret, shifted_regret)
    allowed[0, 0, 2] = False
    predicted, _, common = minimax_decision(values, allowed)
    assert predicted[0] != 2 and not common[0, 2]
    allowed[0] = False
    predicted, regret, _ = minimax_decision(values, allowed)
    assert predicted.tolist() == [-1] and np.isinf(regret).all()


def test_conditional_thresholds_deduplicate_and_abstain_in_sparse_cells():
    confidence = np.array([0.1, 0.9, 0.1, 0.9, 0.5])
    groups = np.array([0, 0, 0, 0, 1])
    patterns = np.array(["11", "11", "11", "11", "01"])
    matches = np.array(["a", "b", "a", "b", "c"])
    rows = np.array([0, 1, 0, 1, 2])
    result = fit_evidence_thresholds(
        confidence, groups, patterns, matches, rows, target=0.5, minimum_rows=2, minimum_matches=2
    )
    assert result["TOP:11"]["rows"] == 2
    assert result["TOP:11"]["threshold"] == 0.5
    assert result["JUNGLE:01"]["threshold"] is None
    # Duplicating rows never supplies independent match support.
    result = fit_evidence_thresholds(
        confidence, groups, patterns, matches, rows, target=0.9, minimum_rows=3, minimum_matches=2
    )
    assert result["TOP:11"]["threshold"] is None


def test_coverage_matching_uses_calibration_rows_only():
    baseline = np.zeros(4, dtype=int)
    confidence = np.array([0.2, 0.8, 0.1, 0.9])
    groups = np.zeros(4, dtype=int)
    fitted = np.array([True, True, False, False])
    retained = np.array([-1, 0, 0, 0])
    predicted, thresholds = match_coverage(baseline, confidence, groups, fitted, retained)
    assert thresholds["TOP"]["threshold"] == 0.5
    assert predicted.tolist() == [-1, 0, -1, 0]
    retained[~fitted] = -1
    confidence[~fitted] = [0.7, 0.6]
    _, changed = match_coverage(baseline, confidence, groups, fitted, retained)
    assert changed == thresholds
