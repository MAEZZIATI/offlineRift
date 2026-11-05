"""Weak-label diagnostics with explicit supported-class denominators."""

import numpy as np


def classification_metrics(truth: np.ndarray, predicted: np.ndarray, size: int) -> dict:
    confusion = np.zeros((size, size), dtype=int)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(size), where=support > 0)
    precision = np.divide(
        confusion.diagonal(), predicted_support, out=np.zeros(size), where=predicted_support > 0
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(size), where=precision + recall > 0
    )
    # Macro F1 includes any class present in truth OR predictions, penalizing false positives.
    return {
        "heuristic_agreement": float(np.mean(truth == predicted)),
        "confusion_matrix": confusion.tolist(),
        "action_support": support.tolist(),
        "prediction_counts": predicted_support.tolist(),
        "per_action_precision": precision.tolist(),
        "per_action_recall": recall.tolist(),
        "per_action_f1": f1.tolist(),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1[(support + predicted_support) > 0].mean()),
        "collapsed_predictions": bool(np.count_nonzero(predicted_support) == 1),
        "prediction_coverage": int(np.count_nonzero(predicted_support)),
    }
