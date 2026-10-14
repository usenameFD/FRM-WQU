"""oilbn.data: data access objects for the crude-oil risk mini-capstone (GWP1).

Design
------
Every data source is modeled as a small, single-purpose class derived from
:class:`BaseDataLoader`. Each loader knows *only* how to fetch and label one
series or ticker; orchestration (looping over many series, building the data
dictionary) is left to callers/notebooks so that loaders stay trivially
testable and reusable.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = [
    "DataUnavailableError",
    "SeriesMetadata",
    "BaseDataLoader",
    "YahooFinanceLoader",
    "FredSeriesLoader",
    "EiaSeriesLoader",
]


class DataUnavailableError(RuntimeError):
    """Raised when a series cannot be retrieved (missing key, network, etc.)."""


@dataclass(frozen=True, slots=True)
class SeriesMetadata:
    """Descriptive metadata for one collected variable.

    Mirrors the columns requested by GWP1 Step 4: name, frequency, source,
    start date, end date, and a free-text description.
    """

    name: str
    category: str
    source: str
    frequency: str
    description: str
    units: str = "n/a"
    start_date: date | None = None
    end_date: date | None = None

    def with_dates(self, index: pd.Index) -> SeriesMetadata:
        """Return a copy with start/end dates inferred from a datetime index."""
        if len(index) == 0:
            return self
        return SeriesMetadata(
            name=self.name,
            category=self.category,
            source=self.source,
            frequency=self.frequency,
            description=self.description,
            units=self.units,
            start_date=pd.Timestamp(index.min()).date(),
            end_date=pd.Timestamp(index.max()).date(),
        )


class BaseDataLoader(ABC):
    """Abstract base class for all single-series data loaders.

    Subclasses implement :meth:`_fetch_raw`; :meth:`fetch` wraps it with
    consistent indexing, naming, and metadata construction so every loader
    returns a uniform ``pandas.Series`` indexed by a ``DatetimeIndex``.
    """

    category: ClassVar[str] = "unspecified"
    source_name: ClassVar[str] = "unspecified"

    def __init__(self, name: str, description: str, frequency: str, units: str = "n/a") -> None:
        self.name = name
        self.description = description
        self.frequency = frequency
        self.units = units

    @abstractmethod
    def _fetch_raw(self) -> pd.Series:
        """Retrieve the raw series from the upstream source."""

    def fetch(self) -> pd.Series:
        """Fetch the series and return it indexed by a sorted ``DatetimeIndex``."""
        series = self._fetch_raw()
        series = series.sort_index()
        series.index = pd.DatetimeIndex(series.index, name="date")
        series.name = self.name
        return series

    def metadata(self, series: pd.Series | None = None) -> SeriesMetadata:
        """Build the :class:`SeriesMetadata` record for this loader.

        If ``series`` is provided, start/end dates are inferred from it.
        """
        meta = SeriesMetadata(
            name=self.name,
            category=self.category,
            source=self.source_name,
            frequency=self.frequency,
            description=self.description,
            units=self.units,
        )
        if series is not None:
            meta = meta.with_dates(series.index)
        return meta


class YahooFinanceLoader(BaseDataLoader):
    """Loads a single price/field series from Yahoo! Finance via ``yfinance``.

    Used for the financial-market variable group (Student C role): futures,
    indices, FX, and yields quoted on Yahoo Finance.
    """

    category = "financial"
    source_name = "Yahoo Finance"

    def __init__(
        self,
        ticker: str,
        name: str,
        description: str,
        field: str = "Close",
        start: str | None = None,
        end: str | None = None,
        frequency: str = "daily",
    ) -> None:
        super().__init__(name=name, description=description, frequency=frequency)
        self.ticker = ticker
        self.field = field
        self.start = start
        self.end = end

    def _fetch_raw(self) -> pd.Series:
        import yfinance as yf

        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            raise DataUnavailableError(f"Yahoo Finance returned no data for {self.ticker!r}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data[self.field].dropna()


class FredSeriesLoader(BaseDataLoader):
    """Loads one FRED (Federal Reserve Economic Data) series.

    Used for macroeconomic and FRED-mirrored energy/geopolitical variables
    (Student A role). Uses the authenticated API via ``fredapi`` when an API
    key is supplied; otherwise falls back to FRED's public, keyless
    ``fredgraph.csv`` download endpoint so the pipeline remains usable
    without any registration.
    """

    category = "macroeconomic"
    source_name = "FRED (Federal Reserve Economic Data)"
    _CSV_URL: ClassVar[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(
        self,
        series_id: str,
        name: str,
        description: str,
        frequency: str,
        api_key: str | None = None,
        category: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description, frequency=frequency)
        self.series_id = series_id
        self.api_key = api_key
        if category is not None:
            self.category = category

    def _fetch_raw(self) -> pd.Series:
        if self.api_key:
            return self._fetch_via_api()
        return self._fetch_via_csv()

    def _fetch_via_api(self) -> pd.Series:
        from fredapi import Fred

        fred = Fred(api_key=self.api_key)
        series = fred.get_series(self.series_id)
        if series is None or series.empty:
            raise DataUnavailableError(f"FRED API returned no data for {self.series_id!r}")
        return series

    def _fetch_via_csv(self) -> pd.Series:
        response = requests.get(self._CSV_URL, params={"id": self.series_id}, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        date_col, value_col = frame.columns[0], frame.columns[1]
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        frame[date_col] = pd.to_datetime(frame[date_col])
        series = frame.set_index(date_col)[value_col].dropna()
        if series.empty:
            raise DataUnavailableError(
                f"FRED keyless CSV endpoint returned no usable rows for {self.series_id!r}"
            )
        return series


class EiaSeriesLoader(BaseDataLoader):
    """Loads one series from the U.S. EIA Open Data API (v2).

    Used for geopolitical/energy-supply variables not mirrored on FRED
    (e.g., OPEC spare capacity). Requires a free API key
    (``EIA_API_KEY``); raises :class:`DataUnavailableError` with an
    actionable message when no key is configured, so callers can skip this
    source gracefully instead of crashing the whole pipeline.
    """

    category = "geopolitical"
    source_name = "U.S. Energy Information Administration (EIA)"
    _BASE_URL: ClassVar[str] = "https://api.eia.gov/v2/seriesid"

    def __init__(
        self,
        series_id: str,
        name: str,
        description: str,
        frequency: str,
        api_key: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description, frequency=frequency)
        self.series_id = series_id
        self.api_key = api_key

    def _fetch_raw(self) -> pd.Series:
        if not self.api_key:
            raise DataUnavailableError(
                "EIA_API_KEY is not set. Register for a free key at "
                "https://www.eia.gov/opendata/register.php or skip this "
                "loader and rely on FRED-mirrored energy series instead."
            )
        url = f"{self._BASE_URL}/{self.series_id}"
        response = requests.get(url, params={"api_key": self.api_key}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("response", {}).get("data", [])
        if not rows:
            raise DataUnavailableError(f"EIA API returned no data for {self.series_id!r}")
        frame = pd.DataFrame(rows)
        frame["period"] = pd.to_datetime(frame["period"])
        series = frame.set_index("period")["value"].astype(float).dropna()
        return series
