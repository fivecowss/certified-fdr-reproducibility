from __future__ import annotations

import numpy as np
import pytest

from certified_fdr.covariance import (
    check_correlation,
    direct_gaussian_product_means,
    exact_gaussian_product_means,
    gaussian_wishart_cross_moments,
    global_pairs,
    one_factor_regime,
    selected_correlations,
)


@pytest.mark.parametrize("regime", ["valid", "invalid", "boundary"])
def test_one_factor_regimes_are_psd(regime: str) -> None:
    loadings, correlation = one_factor_regime(20, 0.5, regime)
    assert check_correlation(correlation).minimum_eigenvalue >= -1e-10
    values = selected_correlations(correlation, global_pairs(20))
    if regime == "valid":
        assert np.min(values) > 0.0
    elif regime == "invalid":
        assert np.min(values) < 0.0
    else:
        assert np.min(values) == pytest.approx(0.0)
    assert loadings.shape == (20,)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.6])
def test_exact_sampler_has_correct_first_two_moments(rho: float) -> None:
    n, repetitions = 40, 60_000
    values = exact_gaussian_product_means(n, np.array([rho]), repetitions, np.random.default_rng(123))[:, 0]
    theoretical_variance = (1.0 + rho**2) / n
    assert np.mean(values) == pytest.approx(rho, abs=5.0 * np.sqrt(theoretical_variance / repetitions))
    assert np.var(values) == pytest.approx(theoretical_variance, rel=0.04)


def test_exact_and_direct_samplers_agree_in_distribution() -> None:
    rho = np.array([-0.4, 0.3])
    exact = exact_gaussian_product_means(30, rho, 20_000, np.random.default_rng(11))
    direct = direct_gaussian_product_means(30, rho, 20_000, np.random.default_rng(22))
    assert np.allclose(exact.mean(axis=0), direct.mean(axis=0), atol=0.01)
    assert np.allclose(exact.var(axis=0), direct.var(axis=0), rtol=0.08)


def test_wishart_sampler_has_exact_gaussian_cross_moment_law() -> None:
    correlation = np.asarray(
        [[1.0, 0.35, -0.15], [0.35, 1.0, 0.20], [-0.15, 0.20, 1.0]]
    )
    n_certificate, repetitions = 60, 40_000
    values = gaussian_wishart_cross_moments(
        n_certificate,
        correlation,
        repetitions,
        np.random.default_rng(912),
    )
    assert values.shape == (repetitions, 3, 3)
    assert np.allclose(values.mean(axis=0), correlation, atol=0.006)
    expected_variance = (1.0 + correlation[0, 1] ** 2) / n_certificate
    assert np.var(values[:, 0, 1]) == pytest.approx(expected_variance, rel=0.04)


def test_wishart_sampler_single_draw_shape_and_singular_rejection() -> None:
    identity = np.eye(2)
    value = gaussian_wishart_cross_moments(
        10, identity, 1, np.random.default_rng(3)
    )
    assert value.shape == (1, 2, 2)
    with pytest.raises(ValueError, match="positive-definite"):
        gaussian_wishart_cross_moments(
            10,
            np.ones((2, 2)),
            1,
            np.random.default_rng(3),
        )
