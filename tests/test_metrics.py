"""Unit tests for oilbn.evaluation.metrics (DatasetSplitter, ForecastEvaluator)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oilbn.evaluation.metrics import DatasetSplitter, ForecastEvaluator


@pytest.fixture
def ordered_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame({"value": np.arange(100)}, index=dates)


class TestDatasetSplitter:
    def test_default_split_sizes(self, ordered_frame: pd.DataFrame) -> None:
        splitter = DatasetSplitter()
        split = splitter.split(ordered_frame)
        assert len(split.train) == 60
        assert len(split.validation) == 20
        assert len(split.test) == 20

    def test_split_is_chronological_and_non_overlapping(self, ordered_frame: pd.DataFrame) -> None:
        splitter = DatasetSplitter(train_fraction=0.5, validation_fraction=0.3)
        split = splitter.split(ordered_frame)
        assert split.train.index.max() < split.validation.index.min()
        assert split.validation.index.max() < split.test.index.min()
        assert len(split.train) + len(split.validation) + len(split.test) == len(ordered_frame)

    def test_invalid_fractions_raise(self) -> None:
        with pytest.raises(ValueError):
            DatasetSplitter(train_fraction=0.7, validation_fraction=0.4)
        with pytest.raises(ValueError):
            DatasetSplitter(train_fraction=0.0, validation_fraction=0.2)


class TestForecastEvaluator:
    def test_accuracy_perfect_match(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series(["bull", "bear", "stagnant"])
        assert evaluator.accuracy(actual, actual) == 1.0

    def test_accuracy_partial_match(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series(["bull", "bear", "stagnant", "bull"])
        predicted = pd.Series(["bull", "bull", "stagnant", "bear"])
        assert evaluator.accuracy(predicted, actual) == 0.5

    def test_confusion_matrix_shape_and_diagonal(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series(["bull", "bear", "bull", "bear"])
        predicted = pd.Series(["bull", "bear", "bear", "bear"])
        matrix = evaluator.confusion_matrix(predicted, actual)
        assert matrix.loc["bull", "bull"] == 1
        assert matrix.loc["bear", "bear"] == 2

    def test_naive_persistence_baseline_shifts_by_one(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series([1.0, 2.0, 3.0])
        baseline = evaluator.naive_persistence_baseline(actual)
        assert baseline.iloc[1] == 1.0
        assert baseline.iloc[2] == 2.0
        assert pd.isna(baseline.iloc[0])

    def test_hit_rate_vs_baseline_keys(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series(["bull", "bull", "bear", "bear"])
        predicted = pd.Series(["bull", "bull", "bear", "bull"])
        result = evaluator.hit_rate_vs_baseline(predicted, actual)
        assert set(result.keys()) == {"model_accuracy", "baseline_accuracy"}

    def test_rmse_zero_for_identical_series(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series([1.0, 2.0, 3.0])
        assert evaluator.rmse(actual, actual) == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        evaluator = ForecastEvaluator()
        actual = pd.Series([0.0, 0.0])
        predicted = pd.Series([3.0, 4.0])
        assert evaluator.rmse(predicted, actual) == pytest.approx(3.5355339, rel=1e-5)
