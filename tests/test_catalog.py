"""Unit tests for oilbn.data.catalog (network calls are mocked)."""

from __future__ import annotations

import pandas as pd
import pytest

from oilbn.data.catalog import SeriesSpec, build_loader, collect_catalog
from oilbn.data.loaders import EiaSeriesLoader, FredSeriesLoader, YahooFinanceLoader
from oilbn.utils.config import Settings


class TestBuildLoader:
    def test_yahoo_spec_builds_yahoo_finance_loader(self) -> None:
        spec = SeriesSpec("yahoo", "CL=F", "wti_futures", "WTI futures", "daily", "USD/bbl")
        loader = build_loader(spec, Settings())
        assert isinstance(loader, YahooFinanceLoader)
        assert loader.ticker == "CL=F"
        assert loader.name == "wti_futures"

    def test_fred_spec_builds_fred_loader_with_category_override(self) -> None:
        spec = SeriesSpec(
            "fred",
            "WCRSTUS1",
            "us_crude_stocks",
            "Crude stocks",
            "weekly",
            category="geopolitical",
        )
        loader = build_loader(spec, Settings())
        assert isinstance(loader, FredSeriesLoader)
        assert loader.category == "geopolitical"

    def test_eia_spec_builds_eia_loader(self) -> None:
        spec = SeriesSpec("eia", "STEO.PAPR_OPEC.M", "opec_supply", "OPEC supply", "monthly")
        loader = build_loader(spec, Settings(eia_api_key="k"))
        assert isinstance(loader, EiaSeriesLoader)
        assert loader.api_key == "k"

    def test_unknown_source_raises(self) -> None:
        spec = SeriesSpec("bogus", "X", "x", "x", "daily")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown series source"):
            build_loader(spec, Settings())


class TestCollectCatalog:
    def test_collects_series_and_builds_data_dictionary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        fake_frame = pd.DataFrame({"Close": [50.0, 51.0, 52.0]}, index=index)
        monkeypatch.setattr("yfinance.download", lambda *a, **k: fake_frame)

        csv_text = "DATE,VALUE\n2020-01-01,1.0\n2020-01-02,2.0\n"

        class _FakeResponse:
            text = csv_text

            def raise_for_status(self) -> None:
                return None

        monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse())

        catalog = (
            SeriesSpec("yahoo", "CL=F", "wti_futures", "WTI futures", "daily", "USD/bbl"),
            SeriesSpec("fred", "CPIENGSL", "cpi_energy", "CPI energy", "monthly"),
        )
        series_by_name, dictionary = collect_catalog(catalog, Settings())

        assert set(series_by_name) == {"wti_futures", "cpi_energy"}
        assert len(dictionary) == 2
        frame = dictionary.to_frame()
        assert set(frame["name"]) == {"wti_futures", "cpi_energy"}

    def test_failed_series_is_skipped_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr("yfinance.download", _raise)
        catalog = (SeriesSpec("yahoo", "CL=F", "wti_futures", "WTI futures", "daily"),)

        series_by_name, dictionary = collect_catalog(catalog, Settings())

        assert series_by_name == {}
        assert len(dictionary) == 0
