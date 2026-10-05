from __future__ import annotations

import numpy as np

from certified_fdr.procedures import bh, by, route_methods, weighted_bh


def test_bh_and_by_hand_calculation() -> None:
    pvalues = np.array([0.01, 0.04, 0.20])
    assert np.array_equal(bh(pvalues, 0.05), np.array([True, False, False]))
    assert not np.any(by(pvalues, 0.05))


def test_zero_weight_never_rejects_even_at_zero_pvalue() -> None:
    result = weighted_bh(np.array([0.02, 0.04, 0.0]), np.array([2.0, 1.0, 0.0]), 0.05)
    assert np.array_equal(result, np.array([True, False, False]))


def test_step_up_ties_reject_together() -> None:
    result = bh(np.array([0.025, 0.025, 1.0]), 0.05)
    assert np.array_equal(result, np.array([True, True, False]))


def test_aggressive_branch_uses_q_not_q_internal_regression() -> None:
    methods = route_methods(np.array([0.095]), np.array([1.0]), True, True, 0.10, 0.01)
    assert methods["Cert_wBY"].rejections[0]
    assert not methods["old_both_q_internal"].rejections[0]


def test_fallback_only_uses_q_internal_and_safe_direct_baselines_use_q() -> None:
    methods = route_methods(np.array([0.095]), np.array([1.0]), False, False, 0.10, 0.01)
    assert not methods["Cert_wBY"].rejections[0]
    assert methods["BY_q"].rejections[0]
    assert methods["wBY_q"].rejections[0]
    assert methods["Cert_wBY"].branch == "fallback"
