"""Evaluation layer (GWP3): chronological train/validation/test splitting
and forecast-accuracy metrics.

A single-purpose class per responsibility, mirroring the rest of
``oilbn``: :class:`DatasetSplitter` only splits, :class:`ForecastEvaluator`
only scores already-produced predictions against actuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["DatasetSplit", "DatasetSplitter", "ForecastEvaluator"]


@dataclass
class DatasetSplit:
    """A chronological train/validation/test partition of one DataFrame."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class DatasetSplitter:
    """Splits time-ordered data chronologically (never shuffled).

    Parameters
    ----------
    train_fraction, validation_fraction:
        Fraction of rows assigned to the training and validation splits;
        the remainder is assigned to the test split. Splits are taken in
        chronological order (train = earliest rows) because the data is a
        single time series: a random/shuffled split would leak
        future information into training and overstate how well the
        learned structure or parameters generalize to genuinely new data
        (the same rationale used for the structure-stability check in the
        GWP2 report).
    """

    train_fraction: float = 0.6
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be < 1")

    def split(self, data: pd.DataFrame) -> DatasetSplit:
        """Partition ``data`` (assumed already sorted chronologically)."""
        n = len(data)
        train_end = int(n * self.train_fraction)
        validation_end = train_end + int(n * self.validation_fraction)
        return DatasetSplit(
            train=data.iloc[:train_end],
            validation=data.iloc[train_end:validation_end],
            test=data.iloc[validation_end:],
        )


@dataclass
class ForecastEvaluator:
    """Scores a set of discrete-class forecasts against actual outcomes."""

    def accuracy(self, predicted: pd.Series, actual: pd.Series) -> float:
        """Fraction of predictions exactly matching the actual regime/class."""
        aligned_predicted, aligned_actual = predicted.align(actual, join="inner")
        if len(aligned_predicted) == 0:
            return float("nan")
        return float((aligned_predicted == aligned_actual).mean())

    def confusion_matrix(self, predicted: pd.Series, actual: pd.Series) -> pd.DataFrame:
        """Rows = actual class, columns = predicted class, counts."""
        aligned_predicted, aligned_actual = predicted.align(actual, join="inner")
        labels = sorted(set(aligned_actual.unique()) | set(aligned_predicted.unique()))
        matrix = pd.DataFrame(0, index=labels, columns=labels)
        for a, p in zip(aligned_actual, aligned_predicted, strict=True):
            matrix.loc[a, p] += 1
        matrix.index.name = "actual"
        matrix.columns.name = "predicted"
        return matrix

    def naive_persistence_baseline(self, actual: pd.Series) -> pd.Series:
        """The trivial "tomorrow = today" forecast: shift the actual series
        by one step. Any model should be judged against beating this.
        """
        return actual.shift(1)

    def hit_rate_vs_baseline(self, predicted: pd.Series, actual: pd.Series) -> dict[str, float]:
        """Compare the model's accuracy against the naive persistence baseline."""
        baseline = self.naive_persistence_baseline(actual)
        return {
            "model_accuracy": self.accuracy(predicted, actual),
            "baseline_accuracy": self.accuracy(baseline, actual),
        }

    def rmse(self, predicted: pd.Series, actual: pd.Series) -> float:
        """Root-mean-squared error, for continuous (non-categorical) forecasts."""
        aligned_predicted, aligned_actual = predicted.align(actual, join="inner")
        aligned_predicted = aligned_predicted.dropna()
        aligned_actual = aligned_actual.loc[aligned_predicted.index].dropna()
        aligned_predicted = aligned_predicted.loc[aligned_actual.index]
        if len(aligned_predicted) == 0:
            return float("nan")
        return float(np.sqrt(((aligned_predicted - aligned_actual) ** 2).mean()))
