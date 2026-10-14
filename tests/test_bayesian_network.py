"""Unit tests for oilbn.models.bayesian_network.OilPriceBayesianNetwork.

Uses a small synthetic dataset with a known generating structure
(``cause -> effect``, both discrete/categorical) to verify that structure
learning recovers a sensible DAG and that parameter learning + inference
produce probability distributions consistent with the generating process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oilbn.models.bayesian_network import OilPriceBayesianNetwork


@pytest.fixture
def synthetic_causal_data() -> pd.DataFrame:
    """``cause`` in {0, 1} drives ``effect`` in {0, 1} with high fidelity;
    ``noise`` is an independent, uninformative third variable.
    """
    rng = np.random.default_rng(0)
    n = 2000
    cause = rng.integers(0, 2, size=n)
    flip = rng.random(n) < 0.05
    effect = np.where(flip, 1 - cause, cause)
    noise = rng.integers(0, 2, size=n)
    return pd.DataFrame({"cause": cause, "effect": effect, "noise": noise})


class TestStructureLearning:
    def test_recovers_an_edge_between_cause_and_effect(
        self, synthetic_causal_data: pd.DataFrame
    ) -> None:
        bn = OilPriceBayesianNetwork()
        bn.fit_structure(synthetic_causal_data)

        edges = {frozenset(edge) for edge in bn.edges}
        assert frozenset({"cause", "effect"}) in edges

    def test_noise_variable_stays_unconnected_to_cause_effect_pair(
        self, synthetic_causal_data: pd.DataFrame
    ) -> None:
        bn = OilPriceBayesianNetwork()
        bn.fit_structure(synthetic_causal_data)

        edges = {frozenset(edge) for edge in bn.edges}
        assert frozenset({"noise", "cause"}) not in edges
        assert frozenset({"noise", "effect"}) not in edges

    def test_edges_empty_before_fitting(self) -> None:
        bn = OilPriceBayesianNetwork()
        assert bn.edges == []


class TestParameterLearningAndInference:
    def test_bayesian_estimator_posterior_favors_matching_effect(
        self, synthetic_causal_data: pd.DataFrame
    ) -> None:
        bn = OilPriceBayesianNetwork()
        bn.fit(synthetic_causal_data, estimator="bayesian")

        result = bn.query(variables=["effect"], evidence={"cause": 1})
        probabilities = dict(zip(result.state_names["effect"], result.values, strict=True))
        # cause=1 should make effect=1 much more likely than effect=0.
        assert probabilities[1] > probabilities[0]

    def test_mle_and_bayesian_estimators_both_fit_without_error(
        self, synthetic_causal_data: pd.DataFrame
    ) -> None:
        bn_mle = OilPriceBayesianNetwork()
        bn_mle.fit(synthetic_causal_data, estimator="mle")
        assert bn_mle.model is not None
        assert len(bn_mle.model.get_cpds()) == 3

    def test_fit_parameters_without_structure_raises(self) -> None:
        bn = OilPriceBayesianNetwork()
        with pytest.raises(RuntimeError):
            bn.fit_parameters(pd.DataFrame({"a": [0, 1]}))


class TestMarkovBlanket:
    def test_markov_blanket_of_effect_includes_cause(
        self, synthetic_causal_data: pd.DataFrame
    ) -> None:
        bn = OilPriceBayesianNetwork()
        bn.fit_structure(synthetic_causal_data)
        blanket = bn.markov_blanket("effect")
        assert "cause" in blanket

    def test_markov_blanket_without_structure_raises(self) -> None:
        bn = OilPriceBayesianNetwork()
        with pytest.raises(RuntimeError):
            bn.markov_blanket("effect")
