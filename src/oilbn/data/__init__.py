"""Data access and cleaning layer for the crude-oil risk mini-capstone."""

from oilbn.data.catalog import GWP1_CATALOG, SeriesSpec, build_loader, collect_catalog
from oilbn.data.cleaners import (
    BadDataCleaner,
    CleaningReport,
    DataCleaningPipeline,
    MissingValueImputer,
    OutlierCleaner,
)
from oilbn.data.loaders import (
    BaseDataLoader,
    DataUnavailableError,
    EiaSeriesLoader,
    FredSeriesLoader,
    SeriesMetadata,
    YahooFinanceLoader,
)
from oilbn.data.registry import DataDictionary

__all__ = [
    "GWP1_CATALOG",
    "BadDataCleaner",
    "BaseDataLoader",
    "CleaningReport",
    "DataCleaningPipeline",
    "DataDictionary",
    "DataUnavailableError",
    "EiaSeriesLoader",
    "FredSeriesLoader",
    "MissingValueImputer",
    "OutlierCleaner",
    "SeriesMetadata",
    "SeriesSpec",
    "YahooFinanceLoader",
    "build_loader",
    "collect_catalog",
]
