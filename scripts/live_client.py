"""Compatibility entry point for :mod:`offline_rift.inference.live_client`."""

from offline_rift.inference.live_client import (
    ROLE_MAP,
    LiveClientInference,
    LiveClientUnavailable,
    LiveStateAdapter,
    main,
)

__all__ = ["ROLE_MAP", "LiveClientInference", "LiveClientUnavailable", "LiveStateAdapter", "main"]


if __name__ == "__main__":
    main()
