"""Compatibility entry point for :mod:`rift_common.pipeline.data_pipeline`."""

from rift_common.pipeline.data_pipeline import (
    REGIONS,
    RiotAPIError,
    RiotDataIngestor,
    TimelineProcessor,
    main,
)

__all__ = ["REGIONS", "RiotAPIError", "RiotDataIngestor", "TimelineProcessor", "main"]


if __name__ == "__main__":
    main()
