"""Covariance constructions and exact Gaussian-product samplers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wishart

from .certification import canonical_pairs, validate_pairs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CorrelationCheck:
    minimum_eigenvalue: float
    maximum_symmetry_error: float
    maximum_diagonal_error: float


def check_correlation(matrix: ArrayLike, tolerance: float = 1e-10) -> CorrelationCheck:
    correlation = np.asarray(matrix, dtype=float)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be square.")
    if not np.all(np.isfinite(correlation)):
        raise ValueError("correlation contains non-finite values.")
    symmetry_error = float(np.max(np.abs(correlation - correlation.T)))
    diagonal_error = float(np.max(np.abs(np.diag(correlation) - 1.0)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh((correlation + correlation.T) / 2.0)))
    if symmetry_error > tolerance:
        raise ValueError("correlation is not symmetric.")
    if diagonal_error > tolerance:
        raise ValueError("correlation does not have unit diagonal.")
    if minimum_eigenvalue < -tolerance:
        raise ValueError("correlation is not positive semidefinite.")
    return CorrelationCheck(minimum_eigenvalue, symmetry_error, diagonal_error)


def one_factor_correlation(loadings: ArrayLike) -> FloatArray:
    """Return lambda lambda' + diag(1-lambda^2)."""
    vector = np.asarray(loadings, dtype=float)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError("loadings must be a vector of length at least two.")
    if not np.all(np.isfinite(vector)) or np.any(np.abs(vector) > 1.0):
        raise ValueError("loadings must be finite and lie in [-1, 1].")
    correlation = np.outer(vector, vector) + np.diag(1.0 - vector**2)
    check_correlation(correlation)
    return np.asarray(correlation, dtype=float)


def one_factor_regime(
    m: int,
    loading: float,
    regime: str,
    boundary_index: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Construct valid, mixed-sign invalid, or one-zero boundary loadings."""
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 2:
        raise ValueError("m must be an integer of at least two.")
    magnitude = float(loading)
    if not 0.0 < magnitude < 1.0:
        raise ValueError("loading must lie in (0, 1).")
    loadings = np.full(int(m), magnitude, dtype=float)
    if regime == "valid":
        pass
    elif regime == "invalid":
        loadings[int(m) // 2 :] *= -1.0
    elif regime == "boundary":
        if not 0 <= int(boundary_index) < int(m):
            raise ValueError("boundary_index is outside the loading vector.")
        loadings[int(boundary_index)] = 0.0
    else:
        raise ValueError("regime must be valid, invalid, or boundary.")
    return loadings, one_factor_correlation(loadings)


def global_pairs(m: int) -> NDArray[np.int64]:
    return canonical_pairs(m)


def selected_correlations(correlation: ArrayLike, pairs: ArrayLike) -> FloatArray:
    matrix = np.asarray(correlation, dtype=float)
    check_correlation(matrix)
    checked = validate_pairs(pairs, matrix.shape[0])
    return np.asarray(matrix[checked[:, 0], checked[:, 1]], dtype=float)


def exact_gaussian_product_means(
    n_certificate: int,
    correlations: ArrayLike,
    repetitions: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample independent 2x2-block cross-moments using the chi-square identity."""
    if isinstance(n_certificate, bool) or int(n_certificate) < 1:
        raise ValueError("n_certificate must be positive.")
    if isinstance(repetitions, bool) or int(repetitions) < 1:
        raise ValueError("repetitions must be positive.")
    rho = np.asarray(correlations, dtype=float)
    if rho.ndim != 1 or not np.all(np.isfinite(rho)) or np.any(np.abs(rho) > 1.0):
        raise ValueError("correlations must be a vector in [-1, 1].")
    shape = (int(repetitions), rho.size)
    first = rng.chisquare(df=int(n_certificate), size=shape)
    second = rng.chisquare(df=int(n_certificate), size=shape)
    return ((1.0 + rho) * first - (1.0 - rho) * second) / (2.0 * int(n_certificate))


def direct_gaussian_product_means(
    n_certificate: int,
    correlations: ArrayLike,
    repetitions: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Reference sampler that explicitly generates independent bivariate blocks."""
    rho = np.asarray(correlations, dtype=float)
    if rho.ndim != 1 or np.any(np.abs(rho) > 1.0):
        raise ValueError("correlations must be a vector in [-1, 1].")
    first = rng.standard_normal((int(repetitions), rho.size, int(n_certificate)))
    residual = rng.standard_normal(first.shape)
    second = rho[None, :, None] * first + np.sqrt(1.0 - rho**2)[None, :, None] * residual
    return np.mean(first * second, axis=-1)


def gaussian_wishart_cross_moments(
    n_certificate: int,
    correlation: ArrayLike,
    repetitions: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw exact Gaussian sample cross-moment matrices via Wishart sampling.

    If ``X_1, ..., X_N`` are independent ``N(0, correlation)`` vectors, then
    ``sum X_i X_i'`` has a Wishart distribution.  Dividing a Wishart draw by
    ``n_certificate`` therefore gives exactly the zero-mean cross-moment
    estimator used by the frozen certificate, without materializing the raw
    ``N_C`` by ``m`` Gaussian sample.

    SciPy's Wishart sampler requires a positive-definite scale matrix.  This
    helper deliberately rejects singular positive-semidefinite matrices rather
    than silently regularizing a frozen scientific input.
    """
    if isinstance(n_certificate, bool) or not isinstance(
        n_certificate, (int, np.integer)
    ):
        raise TypeError("n_certificate must be an integer.")
    if int(n_certificate) < 1:
        raise ValueError("n_certificate must be positive.")
    if isinstance(repetitions, bool) or not isinstance(
        repetitions, (int, np.integer)
    ):
        raise TypeError("repetitions must be an integer.")
    if int(repetitions) < 1:
        raise ValueError("repetitions must be positive.")

    matrix = np.asarray(correlation, dtype=float)
    check = check_correlation(matrix)
    if check.minimum_eigenvalue <= 0.0:
        raise ValueError("Wishart sampling requires a positive-definite correlation.")
    if int(n_certificate) < matrix.shape[0]:
        raise ValueError("Wishart degrees of freedom must be at least the dimension.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    draws = wishart.rvs(
        df=int(n_certificate),
        scale=matrix,
        size=int(repetitions),
        random_state=rng,
    )
    result = np.asarray(draws, dtype=float)
    if int(repetitions) == 1:
        result = result.reshape(1, matrix.shape[0], matrix.shape[1])
    result /= float(n_certificate)
    if result.shape != (int(repetitions), matrix.shape[0], matrix.shape[1]):
        raise RuntimeError("Wishart sampler returned an unexpected shape.")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Wishart sampler returned non-finite values.")
    return result


def gaussian_score_samples(
    mean: ArrayLike,
    correlation: ArrayLike,
    repetitions: int,
    rng: np.random.Generator,
) -> FloatArray:
    location = np.asarray(mean, dtype=float)
    matrix = np.asarray(correlation, dtype=float)
    check_correlation(matrix)
    if location.shape != (matrix.shape[0],):
        raise ValueError("mean and correlation dimensions disagree.")
    return np.asarray(
        rng.multivariate_normal(location, matrix, size=int(repetitions), check_valid="raise"),
        dtype=float,
    )
