"""Unit tests for oilbn.data.loaders (network calls are mocked)."""

from __future__ import annotations

import pandas as pd
import pytest

from oilbn.data.loaders import (
    DataUnavailableError,
    EiaSeriesLoader,
    FredSeriesLoader,
    YahooFinanceLoader,
)


class _FakeResponse:
    def __init__(self, text: str = "", json_data: dict | None = None) -> None:
        self.text = text
        self._json = json_data or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json


class TestYahooFinanceLoader:
    def test_fetch_returns_named_datetime_indexed_series(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        fake_frame = pd.DataFrame({"Close": [50.0, 51.0, 52.0]}, index=index)

        def fake_download(*args, **kwargs):
            return fake_frame

        monkeypatch.setattr("yfinance.download", fake_download)

        loader = YahooFinanceLoader(
            ticker="CL=F", name="wti_futures", description="WTI crude futures"
        )
        series = loader.fetch()

        assert series.name == "wti_futures"
        assert isinstance(series.index, pd.DatetimeIndex)
        assert list(series.values) == [50.0, 51.0, 52.0]

    def test_empty_response_raises_data_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())
        loader = YahooFinanceLoader(ticker="BAD", name="bad", description="")
        with pytest.raises(DataUnavailableError):
            loader.fetch()


class TestFredSeriesLoader:
    def test_keyless_csv_fallback_parses_series(self, monkeypatch: pytest.MonkeyPatch) -> None:
        csv_text = "DATE,DCOILWTICO\n2020-01-02,61.18\n2020-01-03,63.05\n2020-01-06,63.27\n"

        def fake_get(url, params=None, timeout=None):
            assert params["id"] == "DCOILWTICO"
            return _FakeResponse(text=csv_text)

        monkeypatch.setattr("requests.get", fake_get)

        loader = FredSeriesLoader(
            series_id="DCOILWTICO",
            name="wti_spot",
            description="WTI spot price",
            frequency="daily",
        )
        series = loader.fetch()

        assert series.name == "wti_spot"
        assert len(series) == 3
        assert series.iloc[0] == pytest.approx(61.18)

    def test_api_key_path_uses_fredapi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        index = pd.date_range("2020-01-01", periods=2, freq="D")
        fake_series = pd.Series([1.0, 2.0], index=index)

        class _FakeFred:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

            def get_series(self, series_id: str) -> pd.Series:
                return fake_series

        import sys
        import types

        fake_module = types.ModuleType("fredapi")
        fake_module.Fred = _FakeFred
        monkeypatch.setitem(sys.modules, "fredapi", fake_module)

        loader = FredSeriesLoader(
            series_id="CPIAUCSL",
            name="cpi",
            description="US CPI",
            frequency="monthly",
            api_key="dummy-key",
        )
        series = loader.fetch()
        assert list(series.values) == [1.0, 2.0]


class TestEiaSeriesLoader:
    def test_missing_api_key_raises_actionable_error(self) -> None:
        loader = EiaSeriesLoader(
            series_id="PET.WCRSTUS1.W", name="crude_stocks", description="", frequency="weekly"
        )
        with pytest.raises(DataUnavailableError, match="EIA_API_KEY"):
            loader.fetch()

    def test_fetch_with_api_key_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "response": {
                "data": [
                    {"period": "2020-01-03", "value": 430.1},
                    {"period": "2020-01-10", "value": 428.4},
                ]
            }
        }

        def fake_get(url, params=None, timeout=None):
            assert params["api_key"] == "dummy"
            return _FakeResponse(json_data=payload)

        monkeypatch.setattr("requests.get", fake_get)

        loader = EiaSeriesLoader(
            series_id="PET.WCRSTUS1.W",
            name="crude_stocks",
            description="",
            frequency="weekly",
            api_key="dummy",
        )
        series = loader.fetch()
        assert len(series) == 2
        assert series.iloc[0] == pytest.approx(430.1)
