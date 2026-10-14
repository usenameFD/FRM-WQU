"""Unit tests for oilbn.models.hmm.HiddenMarkovRegimeModel.

Uses a simple two-state, well-separated Gaussian toy example (analogous to
Rabiner's discrete-symbol toy examples) to verify each of the four classic
algorithms: Forward/Backward (Evaluation), Viterbi (Decoding), and
Baum-Welch (Training).
"""

from __future__ import annotations

import numpy as np
import pytest

from oilbn.models.hmm import HiddenMarkovRegimeModel, label_states_by_mean


def _toy_two_state_model() -> HiddenMarkovRegimeModel:
    """A hand-specified, well-separated two-state model (no fitting)."""
    model = HiddenMarkovRegimeModel(n_states=2, random_state=0)
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    model.means_ = np.array([-5.0, 5.0])
    model.variances_ = np.array([1.0, 1.0])
    return model


class TestForwardBackward:
    def test_forward_log_likelihood_matches_brute_force_two_steps(self) -> None:
        model = _toy_two_state_model()
        observations = np.array([-5.0, 5.0])

        log_alpha, log_likelihood = model.forward(observations)
        assert log_alpha.shape == (2, 2)

        # Brute-force P(O) by summing over both possible 2-step state paths.
        brute_force_total = 0.0
        for s0 in range(2):
            for s1 in range(2):
                p = model.startprob_[s0]
                p *= np.exp(-0.5 * (observations[0] - model.means_[s0]) ** 2) / np.sqrt(2 * np.pi)
                p *= model.transmat_[s0, s1]
                p *= np.exp(-0.5 * (observations[1] - model.means_[s1]) ** 2) / np.sqrt(2 * np.pi)
                brute_force_total += p

        assert np.exp(log_likelihood) == pytest.approx(brute_force_total, rel=1e-6)

    def test_gamma_sums_to_one_at_every_timestep(self) -> None:
        model = _toy_two_state_model()
        observations, _ = model.sample(50, random_state=1)

        log_alpha, log_likelihood = model.forward(observations)
        log_beta = model.backward(observations)
        gamma = np.exp(log_alpha + log_beta - log_likelihood)

        np.testing.assert_allclose(gamma.sum(axis=1), np.ones(50), atol=1e-6)

    def test_score_equals_forward_log_likelihood(self) -> None:
        model = _toy_two_state_model()
        observations, _ = model.sample(20, random_state=2)
        _, expected = model.forward(observations)
        assert model.score(observations) == pytest.approx(expected)


class TestViterbi:
    def test_recovers_true_states_on_well_separated_regimes(self) -> None:
        model = _toy_two_state_model()
        # Construct an unambiguous sequence: clearly state-0 then state-1 observations.
        observations = np.array([-5.1, -4.9, -5.0, 5.0, 4.9, 5.1])
        states, log_prob = model.viterbi(observations)

        assert list(states) == [0, 0, 0, 1, 1, 1]
        assert np.isfinite(log_prob)

    def test_predict_matches_viterbi_path(self) -> None:
        model = _toy_two_state_model()
        observations = np.array([-5.0, -5.0, 5.0, 5.0])
        states, _ = model.viterbi(observations)
        np.testing.assert_array_equal(model.predict(observations), states)


class TestBaumWelch:
    def test_log_likelihood_increases_monotonically(self) -> None:
        rng = np.random.default_rng(42)
        true_model = _toy_two_state_model()
        observations, _ = true_model.sample(500, random_state=rng)

        model = HiddenMarkovRegimeModel(n_states=2, n_iter=50, tol=0.0, random_state=7)
        model.fit(observations)

        history = np.array(model.log_likelihood_history_)
        # EM guarantees non-decreasing log-likelihood; allow tiny float noise.
        assert np.all(np.diff(history) >= -1e-8)

    def test_recovers_well_separated_means_from_simulated_data(self) -> None:
        true_model = _toy_two_state_model()
        observations, _ = true_model.sample(2000, random_state=123)

        model = HiddenMarkovRegimeModel(n_states=2, n_iter=100, tol=1e-6, random_state=7)
        model.fit(observations)

        fitted_means = sorted(model.means_)
        assert fitted_means[0] == pytest.approx(-5.0, abs=0.5)
        assert fitted_means[1] == pytest.approx(5.0, abs=0.5)

    def test_fit_sets_n_iter_and_stops_early_on_convergence(self) -> None:
        true_model = _toy_two_state_model()
        observations, _ = true_model.sample(300, random_state=99)

        model = HiddenMarkovRegimeModel(n_states=2, n_iter=200, tol=1e-4, random_state=1)
        model.fit(observations)

        assert 0 < model.n_iter_ <= 200


class TestLabelStatesByMean:
    def test_three_state_labels_are_bear_stagnant_bull_in_mean_order(self) -> None:
        model = HiddenMarkovRegimeModel(n_states=3, random_state=0)
        model.means_ = np.array([3.0, -2.0, 0.5])  # unordered on purpose
        labels = label_states_by_mean(model)

        assert labels[1] == "bear"  # mean -2.0, lowest
        assert labels[2] == "stagnant"  # mean 0.5, middle
        assert labels[0] == "bull"  # mean 3.0, highest

    def test_generic_labels_for_non_three_state_models(self) -> None:
        model = HiddenMarkovRegimeModel(n_states=4, random_state=0)
        model.means_ = np.array([1.0, 0.0, 2.0, -1.0])
        labels = label_states_by_mean(model)
        assert set(labels.values()) == {"regime_0", "regime_1", "regime_2", "regime_3"}
