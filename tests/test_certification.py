from __future__ import annotations

import numpy as np
import pytest

from certified_fdr.certification import canonical_pairs, evaluate_certificate, pair_values


def test_intersection_union_boundary_is_inclusive() -> None:
    threshold = evaluate_certificate(np.array([]), 5000, 0.01).threshold
    result = evaluate_certificate(np.array([threshold, threshold + 0.1]), 5000, 0.01)
    assert result.accepted
    assert result.minimum_cross_moment == pytest.approx(threshold)


def test_one_failed_pair_rejects_whole_certificate() -> None:
    threshold = evaluate_certificate(np.array([]), 5000, 0.01).threshold
    result = evaluate_certificate(np.array([threshold + 1.0, threshold - 1e-12]), 5000, 0.01)
    assert not result.accepted


def test_empty_pair_set_is_vacuously_accepted() -> None:
    result = evaluate_certificate(np.array([]), 100, 0.01)
    assert result.accepted
    assert result.minimum_cross_moment is None
    assert result.checked_pair_count == 0


def test_canonical_global_pairs_and_extraction() -> None:
    pairs = canonical_pairs(20)
    assert pairs.shape == (190, 2)
    matrix = np.arange(400, dtype=float).reshape(20, 20)
    values = pair_values(matrix, pairs)
    assert values[0] == matrix[0, 1]
    assert values[-1] == matrix[18, 19]
