"""Compatibility entry point for :mod:`offline_rift.workflow`."""

from offline_rift.workflow import (
    ROOT,
    CatalogCache,
    ResearchWorkflow,
    atomic_json,
    main,
    patch_of,
    read_json,
    run_id,
)

__all__ = [
    "ROOT",
    "CatalogCache",
    "ResearchWorkflow",
    "atomic_json",
    "main",
    "patch_of",
    "read_json",
    "run_id",
]


if __name__ == "__main__":
    main()
