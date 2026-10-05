from __future__ import annotations

import numpy as np

from certified_fdr.weights import learned_weights, validate_weights


def test_eta_zero_gives_uniform_weights() -> None:
    assert np.array_equal(learned_weights(np.array([-9.0, 0.0, 9.0]), 0.0), np.ones(3))


def test_clipped_exponential_formula() -> None:
    scores = np.array([-10.0, 0.0, 10.0])
    activity = np.array([0.25, 1.0, 4.0])
    expected = 3.0 * activity / activity.sum()
    assert np.allclose(learned_weights(scores, 1.0), expected)


def test_batch_and_sum_invariant() -> None:
    weights = learned_weights(np.array([[-1.0, 1.0], [1000.0, -1000.0]]), 0.5)
    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), 2.0)


def test_validator_allows_zero_weights() -> None:
    weights = validate_weights(np.array([2.0, 1.0, 0.0]), 3)
    assert weights[-1] == 0.0
