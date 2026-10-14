"""GWP1 data catalogue: declarative series list + collection pipeline (Steps 3-4).

This module defines *which* series the team collects (financial,
macroeconomic, and geopolitical/physical-market groups, per the data
sourcing plan in ``docs/AGENT_BRIEF.md``) and wires each entry to the
correct :class:`~oilbn.data.loaders.BaseDataLoader` subclass. Looping over
many series and assembling the data dictionary is orchestration, not a
single loader's responsibility, so it lives here rather than in
``loaders.py`` (which stays limited to one series at a time).

Series selection
-----------------
The macroeconomic group reproduces, series-for-series, Table 3.2 of
Alvi (2018) (all genuine FRED series IDs, retrievable keylessly).

The geopolitical/physical-market group reproduces Alvi's Table 3.1 EIA
"Short-Term Energy Outlook" (STEO) series via the registered EIA Open Data
API (v2, ``seriesid`` endpoint), using the same v1-style series identifiers
quoted in the thesis (e.g. ``STEO.PAPR_OPEC.M``). Of the 14 series Alvi
lists, 11 still resolve against the current STEO vintage; ``RGDPQ_NONOECD``,
``RGDPQ_OECD``, and ``FOREX_WORLD`` return no rows (retired/renamed in the
current EIA release) and are intentionally omitted rather than guessed at.
See ``docs/AGENT_BRIEF.md`` and the GWP1 report's data section for the full
discussion. Note EIA's STEO is a rolling *outlook*: the raw pull extends a
few years past the present with the EIA's own forecast values, which the
notebook truncates to realised history before any EDA/stylised-facts work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from oilbn.data.loaders import BaseDataLoader, EiaSeriesLoader, FredSeriesLoader, YahooFinanceLoader
from oilbn.data.registry import DataDictionary
from oilbn.utils.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["SeriesSpec", "GWP1_CATALOG", "build_loader", "collect_catalog"]


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """One row of the data-collection plan, independent of any loader object.

    ``start`` is only used by Yahoo Finance loaders (yfinance returns a
    short default window with no explicit start date); FRED/EIA loaders
    always return their full published history regardless of this field.
    """

    source: Literal["yahoo", "fred", "eia"]
    symbol: str
    name: str
    description: str
    frequency: str
    units: str = "n/a"
    field: str = "Close"
    category: str | None = None
    start: str = "1990-01-01"


GWP1_CATALOG: tuple[SeriesSpec, ...] = (
    # --- Financial / market group (yfinance, no key) ---------------------
    SeriesSpec(
        "yahoo",
        "CL=F",
        "wti_futures",
        "WTI crude oil continuous front-month futures price.",
        "daily",
        "USD/bbl",
    ),
    SeriesSpec(
        "yahoo",
        "BZ=F",
        "brent_futures",
        "Brent crude oil continuous front-month futures price.",
        "daily",
        "USD/bbl",
    ),
    SeriesSpec(
        "yahoo",
        "DX-Y.NYB",
        "usd_index",
        "ICE U.S. Dollar Index (DXY) spot level.",
        "daily",
        "index",
    ),
    SeriesSpec(
        "yahoo",
        "^GSPC",
        "sp500",
        "S&P 500 equity index, a global risk-sentiment proxy.",
        "daily",
        "index",
    ),
    SeriesSpec(
        "yahoo",
        "^TNX",
        "ust10y_yield",
        "CBOE 10-year U.S. Treasury note yield.",
        "daily",
        "%",
    ),
    # --- Macroeconomic group (FRED, keyless CSV; mirrors Alvi Table 3.2) --
    SeriesSpec(
        "fred",
        "CPIENGSL",
        "cpi_energy",
        "US Consumer Price Index: Energy, seasonally adjusted.",
        "monthly",
        "index (1982-84=100)",
    ),
    SeriesSpec(
        "fred",
        "CAPG211S",
        "capacity_oil_gas_extraction",
        "Capacity: Oil and Gas Extraction (NAICS 211).",
        "monthly",
        "index (2017=100)",
    ),
    SeriesSpec(
        "fred",
        "CAPUTLG211S",
        "capacity_util_oil_gas_extraction",
        "Capacity utilization: Oil and Gas Extraction.",
        "monthly",
        "%",
    ),
    SeriesSpec(
        "fred",
        "IPG211S",
        "indprod_oil_gas_extraction",
        "Industrial production: Oil and Gas Extraction.",
        "monthly",
        "index (2017=100)",
    ),
    SeriesSpec(
        "fred",
        "IPG211111CN",
        "indprod_crude_petroleum_ng",
        "Industrial production: Crude Petroleum and Natural Gas Extraction.",
        "monthly",
        "index (2017=100)",
    ),
    SeriesSpec(
        "fred",
        "INDPRO",
        "indprod_total",
        "Total industrial production index (US, all industries).",
        "monthly",
        "index (2017=100)",
    ),
    SeriesSpec(
        "fred",
        "IPN213111N",
        "indprod_oil_gas_support",
        "Industrial production: Support Activities for Oil and Gas Operations.",
        "monthly",
        "index (2017=100)",
    ),
    SeriesSpec(
        "fred",
        "PCU211211",
        "ppi_oil_gas_extraction",
        "Producer Price Index: Oil and Gas Extraction (NAICS 211).",
        "monthly",
        "index",
    ),
    # --- Geopolitical / physical-market group (EIA STEO, Alvi Table 3.1) --
    SeriesSpec(
        "eia",
        "STEO.PAPR_NONOPEC.M",
        "supply_non_opec",
        "Total non-OPEC liquids petroleum production (supply side of the "
        "oil market outside the OPEC cartel).",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.PAPR_OPEC.M",
        "supply_opec",
        "Total OPEC petroleum supply.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.PATC_OECD.M",
        "demand_oecd",
        "Liquid fuels consumption, total OECD.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.PATC_NON_OECD.M",
        "demand_non_oecd",
        "Liquid fuels consumption, total non-OECD.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.COPRPUS.M",
        "us_crude_production",
        "U.S. crude oil production.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.CORIPUS.M",
        "us_refinery_input",
        "U.S. crude oil refinery input.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.PASC_OECD_T3.M",
        "oecd_inventory_balance",
        "OECD end-of-period commercial crude oil and other liquids "
        "inventory (Alvi's 'Balance' variable).",
        "monthly",
        "million barrels, end-of-period",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.COPS_OPEC.M",
        "opec_spare_capacity",
        "OPEC total spare crude oil production capacity.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.COPC_OPEC.M",
        "opec_production_capacity",
        "OPEC total crude oil production capacity.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.T3_STCHANGE_OOECD.M",
        "oecd_stock_change",
        "Net inventory withdrawals, total other OECD crude oil and other " "liquids.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
    SeriesSpec(
        "eia",
        "STEO.T3_STCHANGE_NOECD.M",
        "non_oecd_stock_change",
        "Net inventory withdrawals, total non-OECD crude oil and other " "liquids.",
        "monthly",
        "million barrels/day",
        category="geopolitical",
    ),
)


def build_loader(spec: SeriesSpec, settings: Settings) -> BaseDataLoader:
    """Instantiate the correct loader object for one :class:`SeriesSpec`."""
    if spec.source == "yahoo":
        return YahooFinanceLoader(
            ticker=spec.symbol,
            name=spec.name,
            description=spec.description,
            field=spec.field,
            frequency=spec.frequency,
            start=spec.start,
        )
    if spec.source == "fred":
        return FredSeriesLoader(
            series_id=spec.symbol,
            name=spec.name,
            description=spec.description,
            frequency=spec.frequency,
            api_key=settings.fred_api_key,
            category=spec.category,
        )
    if spec.source == "eia":
        return EiaSeriesLoader(
            series_id=spec.symbol,
            name=spec.name,
            description=spec.description,
            frequency=spec.frequency,
            api_key=settings.eia_api_key,
        )
    raise ValueError(f"Unknown series source: {spec.source!r}")  # pragma: no cover


def collect_catalog(
    catalog: tuple[SeriesSpec, ...],
    settings: Settings,
) -> tuple[dict[str, pd.Series], DataDictionary]:
    """Fetch every series in ``catalog`` and build the accompanying data dictionary.

    Series that fail to fetch (network issues, missing keys) are logged and
    skipped rather than aborting the whole pipeline, so partial data
    collection is still usable in a notebook session.
    """
    series_by_name: dict[str, pd.Series] = {}
    dictionary = DataDictionary()
    for spec in catalog:
        loader = build_loader(spec, settings)
        try:
            series = loader.fetch()
        except Exception:
            logger.exception("Skipping %s (%s): failed to fetch.", spec.name, spec.symbol)
            continue
        series_by_name[spec.name] = series
        dictionary.register(loader.metadata(series))
    return series_by_name, dictionary
