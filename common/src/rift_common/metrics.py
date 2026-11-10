"""Behavior reconstruction diagnostics; these are not policy-win-rate estimates."""

import numpy as np


def behavior_metrics(truth, predicted, size):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    confusion = np.zeros((size, size + 1), dtype=int)
    np.add.at(confusion, (truth, np.where(predicted < 0, size, predicted)), 1)
    support = confusion.sum(axis=1)
    proposed = confusion[:, :size].sum(axis=0)
    diagonal = confusion[:, :size].diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(size), where=support > 0)
    f1 = np.divide(
        2 * diagonal, support + proposed, out=np.zeros(size), where=support + proposed > 0
    )
    return {
        "behavior_agreement": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1[(support + proposed) > 0].mean()),
        "action_support": support.tolist(),
        "prediction_counts": proposed.tolist(),
        "abstentions": int(np.sum(predicted < 0)),
        "prediction_coverage": int(np.count_nonzero(proposed)),
        "collapsed": bool(np.count_nonzero(proposed) <= 1),
        "confusion_matrix_with_abstention_column": confusion.tolist(),
        "per_action_recall": recall.tolist(),
        "per_action_f1": f1.tolist(),
    }
