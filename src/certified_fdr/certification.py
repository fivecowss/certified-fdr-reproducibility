"""Frozen ver6.1 certificate formulas and intersection--union decision."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CertificateResult:
    accepted: bool
    minimum_cross_moment: float | None
    threshold: float
    checked_pair_count: int


def _validate_beta(beta: float) -> float:
    value = float(beta)
    if not 0.0 < value < 1.0:
        raise ValueError("beta must lie in (0, 1).")
    return value


def _validate_sample_size(n_certificate: int) -> int:
    if isinstance(n_certificate, bool) or not isinstance(n_certificate, (int, np.integer)):
        raise TypeError("n_certificate must be an integer.")
    value = int(n_certificate)
    if value < 1:
        raise ValueError("n_certificate must be positive.")
    return value


def adjusted_internal_level(q: float, beta: float) -> float:
    """Return q°=(q-beta)/(1-beta), used only on fallback branches."""
    q_value = float(q)
    beta_value = _validate_beta(beta)
    if not beta_value < q_value < 1.0:
        raise ValueError("Require 0 < beta < q < 1.")
    return (q_value - beta_value) / (1.0 - beta_value)


def certification_threshold(n_certificate: int, beta: float) -> float:
    """Return tau=2 sqrt(log(1/beta)/N_C)+2 log(1/beta)/N_C."""
    n_value = _validate_sample_size(n_certificate)
    beta_value = _validate_beta(beta)
    x = log(1.0 / beta_value)
    return 2.0 * sqrt(x / n_value) + 2.0 * x / n_value


def completeness_margin(n_certificate: int, d: int, beta: float) -> float:
    """Return the frozen sufficient completeness margin Delta."""
    n_value = _validate_sample_size(n_certificate)
    beta_value = _validate_beta(beta)
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise TypeError("d must be an integer.")
    if int(d) < 1:
        raise ValueError("d must be positive for the completeness margin.")
    x = log(int(d) / beta_value)
    return certification_threshold(n_value, beta_value) + 2.0 * sqrt(
        x / n_value
    ) + 2.0 * x / n_value


def sufficient_certificate_size(d: int, gamma: float, beta: float) -> int:
    """Return ceil{4 log(d/beta)/(sqrt(1+gamma)-1)^2}."""
    beta_value = _validate_beta(beta)
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise TypeError("d must be an integer.")
    if int(d) < 1:
        raise ValueError("d must be positive.")
    gamma_value = float(gamma)
    if gamma_value <= 0.0:
        raise ValueError("gamma must be positive.")
    denominator = (sqrt(1.0 + gamma_value) - 1.0) ** 2
    return ceil(4.0 * log(int(d) / beta_value) / denominator)


def canonical_pairs(m: int) -> IntArray:
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
        raise TypeError("m must be an integer.")
    if int(m) < 1:
        raise ValueError("m must be positive.")
    first, second = np.triu_indices(int(m), k=1)
    return np.column_stack((first, second)).astype(np.int64, copy=False)


def validate_pairs(pairs: ArrayLike, m: int) -> IntArray:
    array = np.asarray(pairs, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("pairs must have shape (d, 2).")
    if np.any(array < 0) or np.any(array >= m):
        raise ValueError("pair index outside the matrix.")
    if np.any(array[:, 0] >= array[:, 1]):
        raise ValueError("pairs must use canonical order i < j.")
    if np.unique(array, axis=0).shape[0] != array.shape[0]:
        raise ValueError("pairs must not contain duplicates.")
    return array


def pair_values(estimate: ArrayLike, pairs: ArrayLike) -> FloatArray:
    matrix = np.asarray(estimate, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("estimate must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("estimate contains non-finite values.")
    checked = validate_pairs(pairs, matrix.shape[0])
    return np.asarray(matrix[checked[:, 0], checked[:, 1]], dtype=float)


def evaluate_certificate(
    cross_moments: ArrayLike,
    n_certificate: int,
    beta: float,
) -> CertificateResult:
    """Evaluate C=1{min_j Gamma_hat_j >= tau}; accept when d=0."""
    values = np.asarray(cross_moments, dtype=float)
    if values.ndim != 1:
        raise ValueError("cross_moments must be one-dimensional.")
    threshold = certification_threshold(n_certificate, beta)
    if values.size == 0:
        return CertificateResult(True, None, threshold, 0)
    if not np.all(np.isfinite(values)):
        raise ValueError("cross_moments contain non-finite values.")
    minimum = float(np.min(values))
    return CertificateResult(
        accepted=bool(minimum >= threshold),
        minimum_cross_moment=minimum,
        threshold=threshold,
        checked_pair_count=int(values.size),
    )
