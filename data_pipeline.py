"""Compatibility imports for :mod:`offline_rift.pipeline.data_pipeline`."""

from offline_rift.pipeline.data_pipeline import (
    REGIONS,
    RiotAPIError,
    RiotDataIngestor,
    TimelineProcessor,
    main,
)

__all__ = ["REGIONS", "RiotAPIError", "RiotDataIngestor", "TimelineProcessor", "main"]


if __name__ == "__main__":
    main()
