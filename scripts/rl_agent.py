"""Compatibility entry point for :mod:`offline_rift.training.rl_agent`."""

from offline_rift.inference.policy import MacroPolicy
from offline_rift.training.rl_agent import (
    OfflineMacroTrainer,
)

__all__ = ["MacroPolicy", "OfflineMacroTrainer"]
