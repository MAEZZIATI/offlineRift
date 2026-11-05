"""Compatibility entry point for :mod:`rift_common.pipeline.action_labeler`."""

from rift_common.pipeline.action_labeler import (
    ACTION_NAMES,
    MacroActionLabeler,
    MDPDatasetBuilder,
    load_dataset,
)

__all__ = ["ACTION_NAMES", "MDPDatasetBuilder", "MacroActionLabeler", "load_dataset"]
