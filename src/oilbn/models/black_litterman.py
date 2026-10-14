"""Black-Litterman allocator (GWP3): combines a market-equilibrium prior
with Bayesian-network-derived "views" on asset returns.

Alvi's original design derives its Black-Litterman views from an
EGARCH-M model; our design instead derives views directly from the
Bayesian network's regime posteriors (Section 6 of the GWP2 report),
which is one of Alvi's own proposed-but-unimplemented future directions.
This is intentionally a minimal, from-scratch implementation of the
closed-form Black-Litterman posterior (no external portfolio-optimization
dependency), sufficient to demonstrate the mechanism end-to-end on a
small asset set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BlackLittermanAllocator"]


@dataclass
class BlackLittermanAllocator:
    """Combines equilibrium returns with subjective "views" into a
    posterior expected-return vector, via the standard Black-Litterman
    closed-form update.

    Parameters
    ----------
    risk_aversion:
        Market risk-aversion coefficient ``delta`` used to back out
        equilibrium (CAPM-implied) returns from the covariance matrix and
        market-cap weights.
    tau:
        Scalar reflecting uncertainty in the equilibrium prior itself
        (a small positive number, conventionally 0.025-0.05).
    """

    risk_aversion: float = 2.5
    tau: float = 0.05

    def equilibrium_returns(self, covariance: np.ndarray, market_weights: np.ndarray) -> np.ndarray:
        """Back out implied equilibrium excess returns: ``delta * Sigma * w_mkt``."""
        return self.risk_aversion * covariance @ market_weights

    def posterior_returns(
        self,
        covariance: np.ndarray,
        market_weights: np.ndarray,
        pick_matrix: np.ndarray,
        view_returns: np.ndarray,
        view_confidence: np.ndarray,
    ) -> np.ndarray:
        """The Black-Litterman posterior expected-return vector.

        Parameters
        ----------
        covariance:
            Asset covariance matrix, shape ``(n, n)``.
        market_weights:
            Market-capitalization (or benchmark) weights, shape ``(n,)``.
        pick_matrix:
            ``P`` matrix selecting which asset(s) each view concerns,
            shape ``(k, n)``.
        view_returns:
            The ``k`` subjective view returns, ``Q``.
        view_confidence:
            Diagonal of the view-uncertainty matrix ``Omega`` (one
            variance per view) -- lower values indicate a more confident
            view (e.g. derived from a sharper Bayesian-network regime
            posterior).
        """
        pi = self.equilibrium_returns(covariance, market_weights)
        tau_sigma = self.tau * covariance
        omega = np.diag(view_confidence)

        middle = np.linalg.inv(pick_matrix @ tau_sigma @ pick_matrix.T + omega)
        posterior = pi + tau_sigma @ pick_matrix.T @ middle @ (view_returns - pick_matrix @ pi)
        return posterior

    def optimal_weights(self, covariance: np.ndarray, expected_returns: np.ndarray) -> np.ndarray:
        """Unconstrained mean-variance optimal weights: ``(1/delta) * Sigma^-1 * mu``."""
        return np.linalg.solve(covariance, expected_returns) / self.risk_aversion
