"""Bayesian-network modeling (GWP2 Step 6): Hill-Climbing structure learning
+ parameter learning over discretized regime labels, via ``pgmpy``.

:class:`OilPriceBayesianNetwork` is a thin, single-purpose wrapper that
separates the two learning problems described in the GWP1 report
(parameter learning vs. structure learning): :meth:`fit_structure` runs
score-based Hill-Climbing search (BIC) to discover the DAG, and
:meth:`fit_parameters` estimates each node's conditional probability table
(CPD) given that structure, via either Maximum Likelihood or Bayesian
(BDeu-prior) estimation. :meth:`fit` runs both steps in sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE

__all__ = ["OilPriceBayesianNetwork"]


def _as_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to pandas ``category`` dtype.

    pgmpy infers a variable's type (discrete vs. continuous) from its pandas
    dtype, not its semantic meaning; integer- or string-coded regime labels
    (e.g. ``0/1/2`` or ``"bull"/"bear"/"stagnant"``) must be explicitly typed
    as ``category`` or pgmpy will treat them as continuous and reject a
    discrete scoring method such as ``"bic-d"``.
    """
    return data.astype("category")


@dataclass
class OilPriceBayesianNetwork:
    """Learns a discrete Bayesian network over regime-labelled variables.

    Parameters
    ----------
    scoring_method:
        Score optimized by Hill-Climbing structure search. ``"bic-d"`` (the
        Bayesian Information Criterion for discrete data) is the default,
        matching the methodology summarized in the GWP1 report.
    max_indegree:
        Optional cap on the number of parents per node, limiting the search
        space and discouraging overfitting on a small panel.
    tabu_length:
        Length of the tabu list used by Hill-Climbing to avoid immediately
        reversing its last move (helps escape shallow local optima).
    random_state:
        Reserved for reproducibility of any future stochastic extensions
        (plain Hill-Climbing itself is deterministic given a start DAG).

    Attributes
    ----------
    model:
        The fitted :class:`~pgmpy.models.DiscreteBayesianNetwork`, populated
        by :meth:`fit_structure` (edges only) and then :meth:`fit_parameters`
        (edges + CPDs).
    """

    scoring_method: str = "bic-d"
    max_indegree: int | None = None
    tabu_length: int = 100
    random_state: int | None = None

    model: DiscreteBayesianNetwork | None = field(init=False, default=None, repr=False)

    def fit_structure(self, data: pd.DataFrame) -> DiscreteBayesianNetwork:
        """Learn the DAG structure (edges only, no CPDs) via Hill-Climbing."""
        data = _as_categorical(data)
        searcher = HillClimbSearch(data)
        dag = searcher.estimate(
            scoring_method=self.scoring_method,
            max_indegree=self.max_indegree,
            tabu_length=self.tabu_length,
            show_progress=False,
        )
        self.model = DiscreteBayesianNetwork(dag.edges())
        self.model.add_nodes_from(data.columns)
        return self.model

    def fit_parameters(
        self,
        data: pd.DataFrame,
        estimator: Literal["mle", "bayesian"] = "bayesian",
        equivalent_sample_size: int = 5,
    ) -> DiscreteBayesianNetwork:
        """Estimate each node's CPD given the current structure.

        Parameters
        ----------
        estimator:
            ``"mle"`` for :class:`~pgmpy.parameter_estimator.DiscreteMLE`
            (empirical relative frequencies), or ``"bayesian"`` (default) for
            :class:`~pgmpy.parameter_estimator.DiscreteBayesianEstimator`
            with a BDeu prior, which is more robust when some parent-value
            combinations are rarely observed (see the GWP1 report's
            parameter-learning discussion).
        """
        if self.model is None:
            raise RuntimeError("Call fit_structure() (or fit()) before fit_parameters().")

        data = _as_categorical(data)
        if estimator == "mle":
            self.model.fit(data, estimator=DiscreteMLE())
        else:
            self.model.fit(
                data,
                estimator=DiscreteBayesianEstimator(
                    prior_type="BDeu", equivalent_sample_size=equivalent_sample_size
                ),
            )
        return self.model

    def fit(
        self,
        data: pd.DataFrame,
        estimator: Literal["mle", "bayesian"] = "bayesian",
    ) -> OilPriceBayesianNetwork:
        """Run structure learning followed by parameter learning."""
        self.fit_structure(data)
        self.fit_parameters(data, estimator=estimator)
        return self

    def markov_blanket(self, node: str) -> list[str]:
        """Return the Markov blanket of ``node``: its parents, children, and
        its children's other parents (see the GWP1 report's Markov-blanket
        discussion)."""
        if self.model is None:
            raise RuntimeError("Call fit_structure() (or fit()) first.")
        return list(self.model.get_markov_blanket(node))

    def query(self, variables: list[str], evidence: dict[str, object] | None = None) -> object:
        """Exact inference via Variable Elimination.

        Returns a :class:`~pgmpy.factors.discrete.DiscreteFactor` with the
        joint posterior distribution over ``variables`` given ``evidence``.
        """
        if self.model is None:
            raise RuntimeError("Call fit() (structure + parameters) before query().")
        engine = VariableElimination(self.model)
        return engine.query(variables=variables, evidence=evidence, show_progress=False)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """The learned DAG's edges, or an empty list if not yet fitted."""
        if self.model is None:
            return []
        return list(self.model.edges())
