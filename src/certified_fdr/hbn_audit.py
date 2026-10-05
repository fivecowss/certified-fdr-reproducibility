"""Pre-data HBN feature selection and correlation-geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NearestCorrelationResult:
    matrix: FloatArray
    projection_used: bool
    converged: bool
    iterations: int
    minimum_eigenvalue_before: float
    minimum_eigenvalue_after: float
    maximum_difference: float
    frobenius_difference: float


def _validated_labels(labels: ArrayLike) -> list[str]:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    result = [str(value).strip() for value in values.tolist()]
    if not result or any(not value for value in result):
        raise ValueError("labels must be nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique.")
    return result


def select_hash_ranked_labels(
    labels: ArrayLike,
    dimension: int,
    salt: str,
) -> tuple[IntArray, tuple[str, ...]]:
    """Choose a data-independent label subset by a frozen SHA-256 ranking."""
    values = _validated_labels(labels)
    if isinstance(dimension, bool) or not 1 <= int(dimension) <= len(values):
        raise ValueError("dimension must lie between one and the label count.")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a nonempty string.")
    ranked = sorted(
        range(len(values)),
        key=lambda index: (
            sha256(f"{salt}|{values[index]}".encode("utf-8")).hexdigest(),
            values[index],
        ),
    )[: int(dimension)]
    indices = np.asarray(ranked, dtype=np.int64)
    return indices, tuple(values[index] for index in ranked)


def standardized_correlation(
    features: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return column means, sample SDs, and the sample correlation matrix."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("features must have at least two rows and columns.")
    if not np.all(np.isfinite(values)):
        raise ValueError("features contain non-finite values.")
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("every feature must have positive sample variance.")
    standardized = (values - means) / scales
    correlation = standardized.T @ standardized / float(values.shape[0] - 1)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    return (
        np.asarray(means, dtype=float),
        np.asarray(scales, dtype=float),
        np.asarray(correlation, dtype=float),
    )


def _project_psd(matrix: FloatArray) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return np.asarray((projected + projected.T) / 2.0, dtype=float)


def nearest_correlation_higham(
    matrix: ArrayLike,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> NearestCorrelationResult:
    """Compute an unweighted Frobenius nearest correlation matrix.

    This is Higham's alternating-projection method with Dykstra correction.
    It imposes only symmetry, positive semidefiniteness, and unit diagonal;
    negative off-diagonal entries are not truncated.
    """
    original = np.asarray(matrix, dtype=float)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError("matrix must be square.")
    if original.shape[0] < 2 or not np.all(np.isfinite(original)):
        raise ValueError("matrix must be finite and at least 2 by 2.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if isinstance(max_iterations, bool) or int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")

    symmetric = (original + original.T) / 2.0
    before = float(np.min(np.linalg.eigvalsh(symmetric)))
    symmetry_error = float(np.max(np.abs(original - original.T)))
    diagonal_error = float(np.max(np.abs(np.diag(original) - 1.0)))
    if before >= -tolerance and symmetry_error <= tolerance and diagonal_error <= tolerance:
        unchanged = symmetric.copy()
        np.fill_diagonal(unchanged, 1.0)
        after = float(np.min(np.linalg.eigvalsh(unchanged)))
        difference = unchanged - original
        return NearestCorrelationResult(
            matrix=unchanged,
            projection_used=False,
            converged=True,
            iterations=0,
            minimum_eigenvalue_before=before,
            minimum_eigenvalue_after=after,
            maximum_difference=float(np.max(np.abs(difference))),
            frobenius_difference=float(np.linalg.norm(difference, ord="fro")),
        )

    y_value = symmetric.copy()
    np.fill_diagonal(y_value, 1.0)
    dykstra = np.zeros_like(y_value)
    converged = False
    iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        previous = y_value.copy()
        residual = y_value - dykstra
        x_value = _project_psd(residual)
        dykstra = x_value - residual
        y_value = x_value.copy()
        np.fill_diagonal(y_value, 1.0)
        y_value = (y_value + y_value.T) / 2.0
        iterations = iteration
        change = float(np.max(np.abs(y_value - previous)))
        minimum = float(np.min(np.linalg.eigvalsh(y_value)))
        if change <= tolerance and minimum >= -10.0 * tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError("nearest-correlation projection did not converge.")

    after = float(np.min(np.linalg.eigvalsh(y_value)))
    difference = y_value - original
    return NearestCorrelationResult(
        matrix=np.asarray(y_value, dtype=float),
        projection_used=True,
        converged=True,
        iterations=iterations,
        minimum_eigenvalue_before=before,
        minimum_eigenvalue_after=after,
        maximum_difference=float(np.max(np.abs(difference))),
        frobenius_difference=float(np.linalg.norm(difference, ord="fro")),
    )


def safe_identifier_hash(values: ArrayLike, salt: str) -> str:
    """Hash sorted identifiers without returning or persisting identifiers."""
    identifiers = [str(value).strip() for value in np.asarray(values).tolist()]
    if any(not value for value in identifiers):
        raise ValueError("identifiers must be nonempty.")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be nonempty.")
    payload = "\n".join(sorted(identifiers))
    return sha256(f"{salt}\n{payload}".encode("utf-8")).hexdigest()
