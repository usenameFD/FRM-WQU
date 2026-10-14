"""Unit tests for oilbn.data.cleaners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oilbn.data.cleaners import (
    BadDataCleaner,
    DataCleaningPipeline,
    MissingValueImputer,
    OutlierCleaner,
)


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, name="price")


class TestOutlierCleaner:
    def test_no_outliers_detected_in_smooth_series(self) -> None:
        series = _series([100.0, 101.0, 99.5, 100.5, 100.2, 99.8, 100.1])
        cleaned, report = OutlierCleaner(action="flag").clean(series)
        pd.testing.assert_series_equal(cleaned, series)
        assert "no extreme outliers" in report.narrative()

    def test_extreme_outlier_is_flagged(self) -> None:
        series = _series([100.0, 101.0, 99.0, 100.5, 5000.0, 99.8, 100.1, 100.3, 99.9, 100.0])
        cleaned, report = OutlierCleaner(threshold=3.5, action="flag").clean(series)
        assert series.name in report.flagged_index
        pd.testing.assert_series_equal(cleaned, series)  # "flag" leaves values untouched

    def test_clip_action_bounds_outlier(self) -> None:
        series = _series([100.0, 101.0, 99.0, 100.5, 5000.0, 99.8, 100.1, 100.3, 99.9, 100.0])
        cleaned, _ = OutlierCleaner(threshold=3.5, action="clip").clean(series)
        assert cleaned.max() < 5000.0

    def test_remove_action_drops_outlier_row(self) -> None:
        series = _series([100.0, 101.0, 99.0, 100.5, 5000.0, 99.8, 100.1, 100.3, 99.9, 100.0])
        cleaned, _ = OutlierCleaner(threshold=3.5, action="remove").clean(series)
        assert len(cleaned) == len(series) - 1


class TestBadDataCleaner:
    def test_sorts_non_monotonic_index(self) -> None:
        series = _series([1.0, 2.0, 3.0])
        shuffled = series.iloc[[2, 0, 1]]
        cleaned, report = BadDataCleaner().clean(shuffled)
        assert cleaned.index.is_monotonic_increasing
        assert any("sorted chronologically" in action for action in report.actions)

    def test_duplicate_timestamps_keep_last(self) -> None:
        index = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
        series = pd.Series([1.0, 2.0, 3.0], index=index, name="x")
        cleaned, report = BadDataCleaner().clean(series)
        assert len(cleaned) == 2
        assert cleaned.loc["2020-01-01"] == 2.0
        assert any("duplicated timestamp" in action for action in report.actions)

    def test_non_positive_values_marked_missing(self) -> None:
        series = _series([100.0, -5.0, 0.0, 102.0])
        cleaned, report = BadDataCleaner(allow_non_positive=False).clean(series)
        assert cleaned.isna().sum() == 2
        assert any("non-positive" in action for action in report.actions)

    def test_flat_run_is_flagged(self) -> None:
        series = _series([100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0])
        cleaned, report = BadDataCleaner(max_flat_run=5).clean(series)
        assert series.name in report.flagged_index
        pd.testing.assert_series_equal(cleaned, series)  # bad-data cleaner only flags flat runs


class TestMissingValueImputer:
    def test_no_missing_values_is_noop(self) -> None:
        series = _series([1.0, 2.0, 3.0])
        cleaned, report = MissingValueImputer(strategy="linear").clean(series)
        pd.testing.assert_series_equal(cleaned, series)
        assert "no missing values" in report.narrative()

    def test_ffill_strategy_removes_all_nans(self) -> None:
        series = _series([1.0, np.nan, np.nan, 4.0])
        cleaned, _ = MissingValueImputer(strategy="ffill").clean(series)
        assert cleaned.isna().sum() == 0
        assert cleaned.iloc[1] == 1.0

    def test_linear_strategy_interpolates_between_points(self) -> None:
        series = _series([0.0, np.nan, 2.0])
        cleaned, _ = MissingValueImputer(strategy="linear").clean(series)
        assert cleaned.iloc[1] == pytest.approx(1.0)

    def test_simulate_strategy_removes_all_nans_and_stays_positive(self) -> None:
        series = _series([100.0, np.nan, np.nan, np.nan, 103.0, 104.0, 102.0, 105.0])
        cleaned, report = MissingValueImputer(strategy="simulate", random_state=0).clean(series)
        assert cleaned.isna().sum() == 0
        assert (cleaned > 0).all()
        assert "Brownian bridge" in report.narrative()


class TestDataCleaningPipeline:
    def test_pipeline_combines_all_three_cleaners(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        values = [100.0, 101.0, np.nan, -3.0, 5000.0, 99.0, 98.0, 100.0, 101.0, 100.5]
        series = pd.Series(values, index=index, name="oil_price")

        pipeline = DataCleaningPipeline(
            outlier_cleaner=OutlierCleaner(threshold=3.5, action="clip"),
            missing_value_imputer=MissingValueImputer(strategy="linear"),
        )
        cleaned = pipeline.clean_series(series)

        assert cleaned.isna().sum() == 0
        assert len(pipeline.report.actions) > 0
        assert "cleaning actions" in pipeline.narrative()

    def test_clean_frame_applies_to_every_column(self) -> None:
        index = pd.date_range("2020-01-01", periods=5, freq="D")
        frame = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0, 5.0],
                "b": [10.0, 11.0, np.nan, 13.0, 14.0],
            },
            index=index,
        )
        pipeline = DataCleaningPipeline()
        cleaned = pipeline.clean_frame(frame)
        assert cleaned.isna().sum().sum() == 0
        assert list(cleaned.columns) == ["a", "b"]
