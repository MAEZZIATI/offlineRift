"""Training-derived action support by role and measured game phase."""

import numpy as np

from rift_common.actions import GROUPS, Action


def phase_indices(observations, names):
    seconds = np.asarray(observations)[:, names.index("time")] * 3600
    return np.searchsorted([840, 1500], seconds, side="right")


def phase_counts(observations, groups, actions, names):
    counts = np.zeros((len(GROUPS), 3, len(Action)), dtype=np.int64)
    np.add.at(counts, (groups, phase_indices(observations, names), actions), 1)
    return counts


def phase_support(observations, groups, names, counts, minimum):
    counts = np.asarray(counts)
    if (
        counts.shape != (len(GROUPS), 3, len(Action))
        or counts.dtype.kind not in "iu"
        or np.any(counts < 0)
    ):
        raise ValueError("Invalid phase support counts")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("Positive phase support minimum required")
    return counts[groups, phase_indices(observations, names)] >= minimum
