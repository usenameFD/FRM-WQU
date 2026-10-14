"""Data cleaning layer (GWP1 Steps 5-6): outliers, bad data, and missing values.

Three single-responsibility cleaners mirror the three student roles in the
assignment, and are composed by :class:`DataCleaningPipeline` into the
"sterilized" dataset. Every cleaner returns a :class:`CleaningReport`
documenting *what* was changed and *why*, so the group's written
justification (GWP1 Step 6) can be generated directly from the pipeline
instead of written by hand after the fact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "CleaningReport",
    "BaseCleaner",
    "OutlierCleaner",
    "BadDataCleaner",
    "MissingValueImputer",
    "DataCleaningPipeline",
]


@dataclass
class CleaningReport:
    """Structured log of cleaning actions taken on one or more series."""

    actions: list[str] = field(default_factory=list)
    flagged_index: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.actions.append(message)

    def flag(self, series_name: str, index: pd.Index) -> None:
        if len(index) == 0:
            return
        self.flagged_index.setdefault(series_name, [])
        self.flagged_index[series_name].extend(pd.DatetimeIndex(index).tolist())

    def merge(self, other: CleaningReport) -> None:
        self.actions.extend(other.actions)
        for key, idx in other.flagged_index.items():
            self.flagged_index.setdefault(key, [])
            self.flagged_index[key].extend(idx)

    def narrative(self) -> str:
        """Render the report as prose suitable for the "sterilized data" section."""
        if not self.actions:
            return "No cleaning actions were required; the series was already well-formed."
        bullet_points = "\n".join(f"- {action}" for action in self.actions)
        return f"The following cleaning actions were applied:\n{bullet_points}"


class BaseCleaner(ABC):
    """Abstract base class for a single cleaning responsibility."""

    @abstractmethod
    def clean(self, series: pd.Series) -> tuple[pd.Series, CleaningReport]:
        """Return the cleaned series and a report of what changed."""


class OutlierCleaner(BaseCleaner):
    """Flags/treats extreme outliers (Student A's cleaning role).

    Uses a robust modified z-score based on the median absolute deviation
    (MAD), which is less sensitive to the very extremes it is trying to
    detect than a mean/standard-deviation z-score.
    """

    def __init__(
        self,
        threshold: float = 3.5,
        action: Literal["flag", "clip", "remove"] = "flag",
    ) -> None:
        self.threshold = threshold
        self.action = action

    def clean(self, series: pd.Series) -> tuple[pd.Series, CleaningReport]:
        report = CleaningReport()
        values = series.to_numpy(dtype=float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median)) or 1e-9
        modified_z = 0.6745 * (values - median) / mad
        is_outlier = np.abs(modified_z) > self.threshold
        n_outliers = int(is_outlier.sum())

        if n_outliers == 0:
            report.log(f"{series.name}: no extreme outliers detected (|z| > {self.threshold}).")
            return series.copy(), report

        flagged = series.index[is_outlier]
        report.flag(str(series.name), flagged)
        date_range = f"{flagged.min().date()}..{flagged.max().date()}"
        report.log(
            f"{series.name}: flagged {n_outliers} extreme outlier(s) "
            f"(modified z-score > {self.threshold}) on {date_range}."
        )

        cleaned = series.copy()
        if self.action == "remove":
            cleaned = cleaned.loc[~is_outlier]
            report.log(f"{series.name}: removed the flagged outlier(s) from the series.")
        elif self.action == "clip":
            lower = median - self.threshold * mad / 0.6745
            upper = median + self.threshold * mad / 0.6745
            cleaned = cleaned.clip(lower=lower, upper=upper)
            report.log(f"{series.name}: clipped outlier(s) to [{lower:.4g}, {upper:.4g}].")
        return cleaned, report


class BadDataCleaner(BaseCleaner):
    """Detects wrong, questionable, or duplicated values (Student B's role).

    Checks for: duplicated timestamps, non-monotonic/unsorted dates,
    economically impossible non-positive prices, and "stale" runs of an
    identical value repeated for longer than ``max_flat_run`` observations
    (often a sign of a frozen data feed rather than genuine flat prices).
    """

    def __init__(self, max_flat_run: int = 5, allow_non_positive: bool = False) -> None:
        self.max_flat_run = max_flat_run
        self.allow_non_positive = allow_non_positive

    def clean(self, series: pd.Series) -> tuple[pd.Series, CleaningReport]:
        report = CleaningReport()
        cleaned = series.copy()

        if not cleaned.index.is_monotonic_increasing:
            cleaned = cleaned.sort_index()
            report.log(f"{series.name}: index was not sorted chronologically; re-sorted.")

        duplicate_mask = cleaned.index.duplicated(keep="last")
        if duplicate_mask.any():
            report.flag(str(series.name), cleaned.index[duplicate_mask])
            report.log(
                f"{series.name}: removed {int(duplicate_mask.sum())} duplicated timestamp(s), "
                "keeping the last reported value for each date."
            )
            cleaned = cleaned.loc[~duplicate_mask]

        if not self.allow_non_positive:
            bad_values = cleaned <= 0
            if bad_values.any():
                report.flag(str(series.name), cleaned.index[bad_values])
                report.log(
                    f"{series.name}: treated {int(bad_values.sum())} non-positive value(s) as bad "
                    "data (economically implausible for this series) and set to missing."
                )
                cleaned = cleaned.mask(bad_values)

        flat_run_mask = self._flat_runs(cleaned)
        if flat_run_mask.any():
            report.flag(str(series.name), cleaned.index[flat_run_mask])
            report.log(
                f"{series.name}: flagged {int(flat_run_mask.sum())} observation(s) belonging to a "
                f"stale run of {self.max_flat_run}+ identical consecutive values."
            )

        return cleaned, report

    def _flat_runs(self, series: pd.Series) -> pd.Series:
        is_same_as_prev = series.diff().eq(0)
        run_id = (~is_same_as_prev).cumsum()
        run_length = is_same_as_prev.groupby(run_id).transform("sum") + 1
        return run_length >= self.max_flat_run


class MissingValueImputer(BaseCleaner):
    """Substitutes reasonable values for missing observations (Student C's role).

    Supports three strategies:

    - ``"ffill"``: last-observation-carried-forward (suitable for sparse,
      slow-moving macro releases).
    - ``"linear"``: linear interpolation in time (suitable for short gaps
      in otherwise smooth series).
    - ``"simulate"``: fills each gap with a Brownian bridge drawn from the
      series' own historical log-return volatility, which avoids
      introducing artificial trends/flatness into price-like series.
    """

    def __init__(
        self,
        strategy: Literal["ffill", "linear", "simulate"] = "linear",
        random_state: int | None = 42,
    ) -> None:
        self.strategy = strategy
        self._rng = np.random.default_rng(random_state)

    def clean(self, series: pd.Series) -> tuple[pd.Series, CleaningReport]:
        report = CleaningReport()
        n_missing = int(series.isna().sum())
        if n_missing == 0:
            report.log(f"{series.name}: no missing values to impute.")
            return series.copy(), report

        report.flag(str(series.name), series.index[series.isna()])
        if self.strategy == "ffill":
            cleaned = series.ffill()
            report.log(f"{series.name}: forward-filled {n_missing} missing value(s).")
        elif self.strategy == "linear":
            cleaned = series.interpolate(method="linear", limit_direction="both")
            report.log(f"{series.name}: linearly interpolated {n_missing} missing value(s).")
        elif self.strategy == "simulate":
            cleaned = self._brownian_bridge_fill(series)
            report.log(
                f"{series.name}: simulated {n_missing} missing value(s) via a volatility-matched "
                "Brownian bridge between surrounding observed values."
            )
        else:  # pragma: no cover - guarded by Literal typing
            raise ValueError(f"Unknown imputation strategy: {self.strategy!r}")
        return cleaned, report

    def _brownian_bridge_fill(self, series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float)
        log_returns = np.diff(np.log(np.where(values > 0, values, np.nan)))
        sigma = np.nanstd(log_returns) or 1e-4

        filled = values.copy()
        is_na = np.isnan(filled)
        i = 0
        while i < len(filled):
            if is_na[i]:
                start = i - 1
                end = i
                while end < len(filled) and is_na[end]:
                    end += 1
                if start < 0 or end >= len(filled):
                    i = end
                    continue  # leading/trailing gaps: leave for a fallback ffill/bfill below
                gap_len = end - start
                bridge = self._simulate_bridge(filled[start], filled[end], gap_len, sigma)
                filled[start + 1 : end] = bridge
                i = end
            else:
                i += 1

        result = pd.Series(filled, index=series.index, name=series.name)
        return result.ffill().bfill()

    def _simulate_bridge(
        self, start_value: float, end_value: float, n_steps: int, sigma: float
    ) -> np.ndarray:
        log_start, log_end = np.log(start_value), np.log(end_value)
        t = np.arange(1, n_steps) / n_steps
        drift = log_start + t * (log_end - log_start)
        noise = self._rng.normal(scale=sigma, size=n_steps - 1) * np.sqrt(t * (1 - t))
        return np.exp(drift + noise)


class DataCleaningPipeline:
    """Composes outlier, bad-data, and missing-value cleaning into one pass.

    Mirrors GWP1 Step 6: "all cleaning methods get combined to produce a
    'sterilized' version of the data", applied consistently to every
    collected series regardless of which student originally sourced it.
    """

    def __init__(
        self,
        outlier_cleaner: OutlierCleaner | None = None,
        bad_data_cleaner: BadDataCleaner | None = None,
        missing_value_imputer: MissingValueImputer | None = None,
    ) -> None:
        self.outlier_cleaner = outlier_cleaner or OutlierCleaner()
        self.bad_data_cleaner = bad_data_cleaner or BadDataCleaner()
        self.missing_value_imputer = missing_value_imputer or MissingValueImputer()
        self.report = CleaningReport()

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Run one series through bad-data -> outlier -> imputation, in order."""
        cleaned, bad_data_report = self.bad_data_cleaner.clean(series)
        cleaned, outlier_report = self.outlier_cleaner.clean(cleaned)
        cleaned, missing_report = self.missing_value_imputer.clean(cleaned)
        self.report.merge(bad_data_report)
        self.report.merge(outlier_report)
        self.report.merge(missing_report)
        return cleaned

    def clean_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run every column of a ``DataFrame`` through :meth:`clean_series`."""
        cleaned_columns = {name: self.clean_series(frame[name]) for name in frame.columns}
        return pd.DataFrame(cleaned_columns)

    def narrative(self) -> str:
        """Prose summary of every cleaning decision, for the written report."""
        return self.report.narrative()
