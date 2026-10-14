"""Regime-switching models (GWP2 Step 1-2): a from-scratch Hidden Markov Model.

``hmmlearn`` fails to build on this project's Python/toolchain (no MSVC Build
Tools), so :class:`HiddenMarkovRegimeModel` implements the classic Rabiner
algorithms directly in NumPy: the Forward and Backward algorithms
(Evaluation), Viterbi (Decoding), and Baum-Welch (Training, an
Expectation-Maximization scheme). Hidden states emit univariate Gaussian
observations, which is the standard choice for continuous financial return
series (as opposed to the discrete-symbol formalism used for pedagogical
examples such as the occasionally-dishonest-casino), while keeping the same
named algorithms and the same 5-tuple structure described in the GWP1 report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator, default_rng
from scipy.special import logsumexp

__all__ = ["HiddenMarkovRegimeModel", "label_states_by_mean"]

_EPS = 1e-12


@dataclass
class HiddenMarkovRegimeModel:
    """A Gaussian-emission Hidden Markov Model fit via Baum-Welch.

    Parameters
    ----------
    n_states:
        Number of hidden regimes (e.g. 3 for bull/stagnant/bear).
    n_iter:
        Maximum number of Baum-Welch (EM) iterations.
    tol:
        Stop iterating once the increase in total log-likelihood between
        successive iterations falls below this value.
    variance_floor:
        Minimum emission variance per state, to avoid numerical collapse
        (a state assigned a single observation would otherwise have zero
        variance and an infinite likelihood).
    random_state:
        Seed or :class:`numpy.random.Generator` used to initialize the
        parameters before the first EM iteration.

    Attributes
    ----------
    startprob_:
        Initial state distribution, shape ``(n_states,)``.
    transmat_:
        State transition matrix, shape ``(n_states, n_states)``, row ``i``
        is the distribution over the next state given current state ``i``.
    means_, variances_:
        Per-state Gaussian emission parameters, shape ``(n_states,)``.
    log_likelihood_history_:
        Total data log-likelihood after each Baum-Welch iteration; used to
        verify the monotonic-improvement guarantee of EM.
    """

    n_states: int
    n_iter: int = 200
    tol: float = 1e-6
    variance_floor: float = 1e-8
    random_state: int | Generator | None = None

    startprob_: np.ndarray = field(init=False, repr=False)
    transmat_: np.ndarray = field(init=False, repr=False)
    means_: np.ndarray = field(init=False, repr=False)
    variances_: np.ndarray = field(init=False, repr=False)
    log_likelihood_history_: list[float] = field(init=False, default_factory=list, repr=False)
    n_iter_: int = field(init=False, default=0)

    # ------------------------------------------------------------------
    # Emission density
    # ------------------------------------------------------------------
    def _log_emission(self, observations: np.ndarray) -> np.ndarray:
        """Return ``log b_i(x_t)`` for every state ``i`` and time ``t``.

        Shape ``(n_timesteps, n_states)``.
        """
        x = observations.reshape(-1, 1)
        variance = self.variances_.reshape(1, -1)
        mean = self.means_.reshape(1, -1)
        return -0.5 * (np.log(2 * np.pi * variance) + (x - mean) ** 2 / variance)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize(self, observations: np.ndarray) -> None:
        rng = (
            self.random_state
            if isinstance(self.random_state, Generator)
            else default_rng(self.random_state)
        )
        k = self.n_states
        self.startprob_ = np.full(k, 1.0 / k)
        transmat = rng.dirichlet(np.ones(k) * 5.0, size=k)
        self.transmat_ = transmat
        # Seed emission means at quantiles of the data so states start
        # differentiated (avoids the symmetric-initialization trap where
        # every state converges to the same emission distribution).
        quantiles = np.linspace(0.1, 0.9, k)
        self.means_ = np.quantile(observations, quantiles)
        overall_var = np.var(observations)
        self.variances_ = np.full(k, max(overall_var, self.variance_floor))

    # ------------------------------------------------------------------
    # Evaluation problem: Forward & Backward algorithms
    # ------------------------------------------------------------------
    def forward(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward algorithm.

        Returns ``(log_alpha, log_likelihood)`` where ``log_alpha`` has shape
        ``(n_timesteps, n_states)`` and ``log_alpha[t, i] = log P(o_1..o_t,
        q_t = i)``. ``log_likelihood`` is ``log P(O | lambda)``.
        """
        observations = np.asarray(observations, dtype=float)
        log_b = self._log_emission(observations)
        log_a = np.log(self.transmat_ + _EPS)
        log_pi = np.log(self.startprob_ + _EPS)

        n_timesteps = len(observations)
        log_alpha = np.empty((n_timesteps, self.n_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, n_timesteps):
            # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_a[i, j]) + log_b[t, j]
            log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_a, axis=0) + log_b[t]
        log_likelihood = float(logsumexp(log_alpha[-1]))
        return log_alpha, log_likelihood

    def backward(self, observations: np.ndarray) -> np.ndarray:
        """Backward algorithm.

        Returns ``log_beta`` of shape ``(n_timesteps, n_states)`` where
        ``log_beta[t, i] = log P(o_{t+1}..o_T | q_t = i)``.
        """
        observations = np.asarray(observations, dtype=float)
        log_b = self._log_emission(observations)
        log_a = np.log(self.transmat_ + _EPS)

        n_timesteps = len(observations)
        log_beta = np.zeros((n_timesteps, self.n_states))
        for t in range(n_timesteps - 2, -1, -1):
            # log_beta[t, i] = logsumexp_j(log_a[i, j] + log_b[t+1, j] + log_beta[t+1, j])
            terms = log_a + (log_b[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = logsumexp(terms, axis=1)
        return log_beta

    def score(self, observations: np.ndarray) -> float:
        """Total data log-likelihood ``log P(O | lambda)`` (Evaluation problem)."""
        _, log_likelihood = self.forward(observations)
        return log_likelihood

    # ------------------------------------------------------------------
    # Decoding problem: Viterbi algorithm
    # ------------------------------------------------------------------
    def viterbi(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        """Viterbi algorithm: the single most likely hidden-state path.

        Returns ``(states, log_prob)`` where ``states`` is an integer array
        of shape ``(n_timesteps,)`` and ``log_prob`` is the log-probability
        of that path, ``max_q log P(q, O | lambda)``.
        """
        observations = np.asarray(observations, dtype=float)
        log_b = self._log_emission(observations)
        log_a = np.log(self.transmat_ + _EPS)
        log_pi = np.log(self.startprob_ + _EPS)

        n_timesteps = len(observations)
        delta = np.empty((n_timesteps, self.n_states))
        psi = np.zeros((n_timesteps, self.n_states), dtype=int)

        delta[0] = log_pi + log_b[0]
        for t in range(1, n_timesteps):
            scores = delta[t - 1, :, None] + log_a  # shape (from, to)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.n_states)] + log_b[t]

        states = np.empty(n_timesteps, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        log_prob = float(delta[-1, states[-1]])
        for t in range(n_timesteps - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states, log_prob

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Alias for :meth:`viterbi`, returning only the decoded state path."""
        states, _ = self.viterbi(observations)
        return states

    # ------------------------------------------------------------------
    # Training problem: Baum-Welch (EM)
    # ------------------------------------------------------------------
    def fit(self, observations: np.ndarray) -> HiddenMarkovRegimeModel:
        """Baum-Welch: estimate ``(startprob_, transmat_, means_, variances_)``
        that (locally) maximize ``P(O | lambda)`` via Expectation-Maximization.
        """
        observations = np.asarray(observations, dtype=float)
        self._initialize(observations)
        self.log_likelihood_history_ = []

        previous_log_likelihood = -np.inf
        for iteration in range(1, self.n_iter + 1):
            log_alpha, log_likelihood = self.forward(observations)
            log_beta = self.backward(observations)
            log_b = self._log_emission(observations)
            log_a = np.log(self.transmat_ + _EPS)

            # E-step: state occupation (gamma) and transition (xi) posteriors.
            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)

            log_xi = (
                log_alpha[:-1, :, None]
                + log_a[None, :, :]
                + log_b[1:, None, :]
                + log_beta[1:, None, :]
                - log_likelihood
            )
            xi = np.exp(log_xi)  # shape (T-1, n_states, n_states)

            # M-step: re-estimate parameters from the posteriors.
            self.startprob_ = gamma[0] / gamma[0].sum()
            denom = xi.sum(axis=0).sum(axis=1, keepdims=True)
            self.transmat_ = xi.sum(axis=0) / np.where(denom == 0, _EPS, denom)

            gamma_sum = gamma.sum(axis=0)
            self.means_ = (gamma * observations.reshape(-1, 1)).sum(axis=0) / np.where(
                gamma_sum == 0, _EPS, gamma_sum
            )
            squared_dev = (observations.reshape(-1, 1) - self.means_.reshape(1, -1)) ** 2
            self.variances_ = np.maximum(
                (gamma * squared_dev).sum(axis=0) / np.where(gamma_sum == 0, _EPS, gamma_sum),
                self.variance_floor,
            )

            self.log_likelihood_history_.append(log_likelihood)
            self.n_iter_ = iteration
            if log_likelihood - previous_log_likelihood < self.tol and iteration > 1:
                break
            previous_log_likelihood = log_likelihood

        return self

    # ------------------------------------------------------------------
    # Sampling (used to build toy examples / unit tests)
    # ------------------------------------------------------------------
    def sample(
        self, n_samples: int, random_state: int | Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate ``(observations, states)`` from the current parameters."""
        rng = random_state if isinstance(random_state, Generator) else default_rng(random_state)
        states = np.empty(n_samples, dtype=int)
        observations = np.empty(n_samples, dtype=float)
        states[0] = rng.choice(self.n_states, p=self.startprob_)
        for t in range(1, n_samples):
            states[t] = rng.choice(self.n_states, p=self.transmat_[states[t - 1]])
        observations = rng.normal(self.means_[states], np.sqrt(self.variances_[states]))
        return observations, states


def label_states_by_mean(model: HiddenMarkovRegimeModel) -> dict[int, str]:
    """Assign interpretable "bear" / "stagnant" / "bull" labels to hidden
    states by ranking their fitted emission means (lowest to highest).

    Only meaningful for ``n_states == 3``; for other state counts the
    states are labelled ``regime_0, regime_1, ...`` in ascending-mean order.
    """
    order = np.argsort(model.means_)
    if model.n_states == 3:
        names = ["bear", "stagnant", "bull"]
    else:
        names = [f"regime_{rank}" for rank in range(model.n_states)]
    return {int(state): name for state, name in zip(order, names, strict=True)}
