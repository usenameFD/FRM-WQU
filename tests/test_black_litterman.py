"""Unit tests for oilbn.models.black_litterman.BlackLittermanAllocator."""

from __future__ import annotations

import numpy as np
import pytest

from oilbn.models.black_litterman import BlackLittermanAllocator


@pytest.fixture
def two_asset_market() -> dict:
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    market_weights = np.array([0.6, 0.4])
    return {"covariance": covariance, "market_weights": market_weights}


class TestEquilibriumReturns:
    def test_equilibrium_returns_shape_and_sign(self, two_asset_market: dict) -> None:
        allocator = BlackLittermanAllocator(risk_aversion=2.5)
        pi = allocator.equilibrium_returns(
            two_asset_market["covariance"], two_asset_market["market_weights"]
        )
        assert pi.shape == (2,)
        assert np.all(pi > 0)  # positive weights + positive-definite covariance => positive pi


class TestPosteriorReturns:
    def test_posterior_matches_equilibrium_when_view_matches_equilibrium(
        self, two_asset_market: dict
    ) -> None:
        allocator = BlackLittermanAllocator(risk_aversion=2.5, tau=0.05)
        pi = allocator.equilibrium_returns(
            two_asset_market["covariance"], two_asset_market["market_weights"]
        )
        pick_matrix = np.array([[1.0, 0.0]])
        # A view that exactly matches the equilibrium prior should not move the posterior.
        posterior = allocator.posterior_returns(
            two_asset_market["covariance"],
            two_asset_market["market_weights"],
            pick_matrix,
            view_returns=np.array([pi[0]]),
            view_confidence=np.array([1e-8]),
        )
        np.testing.assert_allclose(posterior, pi, atol=1e-3)

    def test_strong_bullish_view_increases_posterior_return(self, two_asset_market: dict) -> None:
        allocator = BlackLittermanAllocator(risk_aversion=2.5, tau=0.05)
        pi = allocator.equilibrium_returns(
            two_asset_market["covariance"], two_asset_market["market_weights"]
        )
        pick_matrix = np.array([[1.0, 0.0]])
        bullish_view = pi[0] + 0.10  # 10% more bullish than equilibrium
        posterior = allocator.posterior_returns(
            two_asset_market["covariance"],
            two_asset_market["market_weights"],
            pick_matrix,
            view_returns=np.array([bullish_view]),
            view_confidence=np.array([1e-4]),  # high-confidence view
        )
        assert posterior[0] > pi[0]


class TestOptimalWeights:
    def test_optimal_weights_shape(self, two_asset_market: dict) -> None:
        allocator = BlackLittermanAllocator(risk_aversion=2.5)
        expected_returns = np.array([0.05, 0.08])
        weights = allocator.optimal_weights(two_asset_market["covariance"], expected_returns)
        assert weights.shape == (2,)
