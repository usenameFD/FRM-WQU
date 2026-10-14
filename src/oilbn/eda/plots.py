"""Exploratory data analysis plotters (GWP1 Step 7).

Three classes mirror the three student EDA roles: distributional, time
series, and multivariate plots. Every public plotting method labels its
axes/units and returns the created :class:`matplotlib.figure.Figure` so
notebooks can both display it inline and save it under ``reports/.../figures``
for direct inclusion in the LaTeX report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

__all__ = ["DistributionPlotter", "TimeSeriesPlotter", "MultivariatePlotter"]


def _save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


class DistributionPlotter:
    """Distributional plots: histograms, KDEs, and QQ plots (Student A role)."""

    def plot_histogram(
        self,
        series: pd.Series,
        bins: int = 50,
        title: str | None = None,
        units: str = "",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax, color="#1f77b4")
        ax.set_title(title or f"Distribution of {series.name}")
        ax.set_xlabel(f"{series.name}" + (f" ({units})" if units else ""))
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_return_distribution(
        self,
        prices: pd.Series,
        bins: int = 60,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        returns = np.log(prices / prices.shift(1)).dropna()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(returns, bins=bins, kde=True, ax=ax, color="#d62728", stat="density")
        ax.set_title(f"Log-return distribution: {prices.name}")
        ax.set_xlabel("Daily log return")
        ax.set_ylabel("Density")
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_qq(
        self,
        series: pd.Series,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        from scipy import stats

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        stats.probplot(series.dropna(), dist="norm", plot=ax)
        ax.set_title(f"Normal Q-Q plot: {series.name}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        fig.tight_layout()
        _save(fig, save_path)
        return fig


class TimeSeriesPlotter:
    """Time series plots: level, returns, rolling volatility, ACF (Student B role)."""

    def plot_level(
        self,
        series: pd.Series,
        units: str = "",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(series.index, series.values, linewidth=1.1, color="#1f77b4")
        ax.set_title(f"{series.name} over time")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{series.name}" + (f" ({units})" if units else ""))
        fig.autofmt_xdate()
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_rolling_volatility(
        self,
        prices: pd.Series,
        window: int = 21,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        returns = np.log(prices / prices.shift(1))
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(rolling_vol.index, rolling_vol.values, color="#9467bd", linewidth=1.1)
        ax.set_title(f"{window}-day rolling annualized volatility: {prices.name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized volatility")
        fig.autofmt_xdate()
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_autocorrelation(
        self,
        series: pd.Series,
        lags: int = 40,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        plot_acf(series.dropna(), lags=lags, ax=axes[0])
        plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
        axes[0].set_title(f"ACF: {series.name}")
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("Autocorrelation")
        axes[1].set_title(f"PACF: {series.name}")
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("Partial autocorrelation")
        fig.tight_layout()
        _save(fig, save_path)
        return fig


class MultivariatePlotter:
    """Multivariate plots: correlation heatmaps and rolling co-movement (Student C role)."""

    def plot_correlation_heatmap(
        self,
        frame: pd.DataFrame,
        method: str = "pearson",
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        corr = frame.corr(method=method)
        fig, ax = plt.subplots(figsize=(0.6 * len(corr.columns) + 3, 0.6 * len(corr.columns) + 2))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": f"{method.title()} correlation"},
        )
        ax.set_title("Cross-variable correlation matrix")
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_rolling_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = 63,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(rolling_corr.index, rolling_corr.values, color="#2ca02c", linewidth=1.1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{window}-day rolling correlation: {series_a.name} vs {series_b.name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Pearson correlation")
        ax.set_ylim(-1, 1)
        fig.autofmt_xdate()
        fig.tight_layout()
        _save(fig, save_path)
        return fig

    def plot_scatter_matrix(
        self,
        frame: pd.DataFrame,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        grid = sns.pairplot(frame.dropna(), diag_kind="kde", plot_kws={"s": 12, "alpha": 0.6})
        grid.fig.suptitle("Pairwise relationships across variables", y=1.02)
        _save(grid.fig, save_path)
        return grid.fig
