"""D_W-only preprocessing helpers for the HBN audit and bridge geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .hbn_audit import (
    NearestCorrelationResult,
    nearest_correlation_higham,
    select_hash_ranked_labels,
    standardized_correlation,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NuisanceDesign:
    matrix: FloatArray
    column_names: tuple[str, ...]
    site_levels: tuple[str, ...]
    omitted_site_level: str
    site_proportions: FloatArray
    median_fd_center: float
    retained_volumes_center: float
    rank: int


@dataclass(frozen=True)
class DWFittedGeometry:
    selected_indices: IntArray
    selected_labels: tuple[str, ...]
    nuisance_coefficients: FloatArray
    adjusted_means: FloatArray
    adjusted_scales: FloatArray
    empirical_correlation: FloatArray
    nearest_correlation: NearestCorrelationResult


def _string_vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = np.asarray([str(value).strip() for value in array.tolist()], dtype=str)
    if result.size == 0 or np.any(result == ""):
        raise ValueError(f"{name} contains empty values.")
    return result


def centered_nuisance_design(
    acquisition_stratum: ArrayLike,
    median_fd: ArrayLike,
    retained_volumes: ArrayLike,
) -> NuisanceDesign:
    """Construct a centered, full-rank D_W nuisance design with an intercept."""
    sites = _string_vector(acquisition_stratum, "acquisition_stratum")
    fd = np.asarray(median_fd, dtype=float)
    volumes = np.asarray(retained_volumes, dtype=float)
    if fd.ndim != 1 or volumes.ndim != 1:
        raise ValueError("QC covariates must be one-dimensional.")
    if not (sites.size == fd.size == volumes.size):
        raise ValueError("site and QC covariates have inconsistent lengths.")
    if sites.size < 2 or not np.all(np.isfinite(fd)) or not np.all(np.isfinite(volumes)):
        raise ValueError("site and QC covariates must be finite with at least two rows.")

    levels = tuple(sorted(set(sites.tolist())))
    omitted = levels[-1]
    proportions = np.asarray([np.mean(sites == level) for level in levels], dtype=float)
    columns: list[FloatArray] = [np.ones(sites.size, dtype=float)]
    names = ["intercept"]
    for index, level in enumerate(levels[:-1]):
        columns.append((sites == level).astype(float) - proportions[index])
        names.append(f"site_centered:{level}")

    fd_center = float(np.mean(fd))
    volume_center = float(np.mean(volumes))
    columns.extend([fd - fd_center, volumes - volume_center])
    names.extend(["median_fd_centered", "retained_volumes_centered"])
    matrix = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise ValueError("D_W nuisance design is rank deficient.")
    return NuisanceDesign(
        matrix=np.asarray(matrix, dtype=float),
        column_names=tuple(names),
        site_levels=levels,
        omitted_site_level=omitted,
        site_proportions=proportions,
        median_fd_center=fd_center,
        retained_volumes_center=volume_center,
        rank=rank,
    )


def fit_dw_geometry(
    all_features: ArrayLike,
    labels: ArrayLike,
    nuisance: NuisanceDesign,
    dimension: int,
    selection_salt: str,
    projection_tolerance: float = 1e-10,
    projection_max_iterations: int = 200,
) -> DWFittedGeometry:
    """Fit D_W-only adjustment, scaling, and correlation geometry."""
    features = np.asarray(all_features, dtype=float)
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("all_features must be a finite matrix.")
    if features.shape[0] != nuisance.matrix.shape[0]:
        raise ValueError("feature and nuisance row counts disagree.")
    indices, selected_labels = select_hash_ranked_labels(
        labels, dimension, selection_salt
    )
    if features.shape[1] != np.asarray(labels).size:
        raise ValueError("feature and label dimensions disagree.")
    selected = features[:, indices]
    coefficients, _, observed_rank, _ = np.linalg.lstsq(
        nuisance.matrix, selected, rcond=None
    )
    if int(observed_rank) != nuisance.matrix.shape[1]:
        raise ValueError("nuisance least-squares fit is rank deficient.")
    nuisance_part = nuisance.matrix[:, 1:] @ coefficients[1:, :]
    adjusted = selected - nuisance_part
    means, scales, correlation = standardized_correlation(adjusted)
    projected = nearest_correlation_higham(
        correlation,
        tolerance=projection_tolerance,
        max_iterations=projection_max_iterations,
    )
    return DWFittedGeometry(
        selected_indices=indices,
        selected_labels=selected_labels,
        nuisance_coefficients=np.asarray(coefficients[1:, :], dtype=float),
        adjusted_means=means,
        adjusted_scales=scales,
        empirical_correlation=correlation,
        nearest_correlation=projected,
    )


def stratified_pair_count(
    split_values: ArrayLike,
    strata: ArrayLike,
    certificate_value: str,
) -> tuple[int, dict[str, int]]:
    """Return sum floor(n_stratum/2) for the certificate split."""
    split = _string_vector(split_values, "split_values")
    groups = _string_vector(strata, "strata")
    if split.size != groups.size:
        raise ValueError("split and stratum lengths disagree.")
    target = str(certificate_value).strip()
    if not target:
        raise ValueError("certificate_value must be nonempty.")
    counts = {
        level: int(np.sum((split == target) & (groups == level)))
        for level in sorted(set(groups.tolist()))
    }
    if sum(counts.values()) == 0:
        raise ValueError("certificate split is empty.")
    return sum(value // 2 for value in counts.values()), counts
