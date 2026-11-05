"""Deterministic match splits shared by maintained trainers and experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchSplit:
    training: tuple[str, ...]
    validation: tuple[str, ...]

    @classmethod
    def create(cls, match_ids, *, seed=42, validation_fraction=0.2):
        if not 0 <= validation_fraction < 1:
            raise ValueError("Validation fraction must be in [0, 1)")
        matches = np.unique(match_ids)
        if len(matches) == 0:
            raise ValueError("Cannot split an empty dataset")
        np.random.default_rng(seed).shuffle(matches)
        count = (
            max(1, int(len(matches) * validation_fraction))
            if len(matches) > 1 and validation_fraction > 0
            else 0
        )
        return cls(tuple(matches[count:].tolist()), tuple(matches[:count].tolist()))

    @classmethod
    def from_report(cls, report, match_ids):
        split = cls(tuple(report["training_matches"]), tuple(report["validation_matches"]))
        train, validation = set(split.training), set(split.validation)
        if (
            not train
            or train & validation
            or len(train) != len(split.training)
            or len(validation) != len(split.validation)
            or train | validation != set(match_ids)
        ):
            raise ValueError("Saved match split overlaps, duplicates or differs from dataset")
        return split

    def validation_mask(self, match_ids):
        return np.isin(match_ids, self.validation)

    def to_dict(self):
        return {
            "training_matches": list(self.training),
            "validation_matches": list(self.validation),
        }
