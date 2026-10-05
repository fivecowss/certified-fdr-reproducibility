"""Frozen-transform execution helpers for the model-based HBN application."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .hbn import deterministic_disjoint_pairs


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FrozenTransform:
    selected_labels: tuple[str, ...]
    site_levels: tuple[str, ...]
    omitted_site_level: str
    site_proportions: FloatArray
    median_fd_center: float
    retained_volumes_center: float
    nuisance_coefficients: FloatArray
    adjusted_means: FloatArray
    adjusted_scales: FloatArray


@dataclass(frozen=True)
class PairedScores:
    scores: FloatArray
    pair_indices: IntArray
    unused_indices: IntArray
    pair_count_by_stratum: dict[str, int]
    unused_count_by_stratum: dict[str, int]


@dataclass(frozen=True)
class GaussianLocationResult:
    statistics: FloatArray
    pvalues: FloatArray


def frozen_transform_from_geometry(geometry: dict[str, Any]) -> FrozenTransform:
    if geometry.get("geometry_status") != "frozen":
        raise ValueError("HBN geometry must be frozen before execution.")
    selection = geometry["selection"]
    preprocessing = geometry["preprocessing"]
    labels = tuple(map(str, selection["selected_labels"]))
    levels = tuple(map(str, preprocessing["site_levels"]))
    coefficients = np.asarray(preprocessing["nuisance_coefficients"], dtype=float)
    means = np.asarray(preprocessing["adjusted_means"], dtype=float)
    scales = np.asarray(preprocessing["adjusted_scales"], dtype=float)
    proportions = np.asarray(preprocessing["site_proportions"], dtype=float)
    expected_rows = len(levels) - 1 + 2
    if len(labels) != 10 or coefficients.shape != (expected_rows, len(labels)):
        raise ValueError("frozen HBN transform dimensions are inconsistent.")
    if means.shape != scales.shape or means.shape != (len(labels),):
        raise ValueError("frozen HBN location/scale dimensions are inconsistent.")
    if proportions.shape != (len(levels),) or not np.isclose(proportions.sum(), 1.0):
        raise ValueError("frozen HBN site proportions are inconsistent.")
    if np.any(scales <= 0.0) or not all(
        np.all(np.isfinite(value))
        for value in (coefficients, means, scales, proportions)
    ):
        raise ValueError("frozen HBN transform contains invalid numeric values.")
    omitted = str(preprocessing["omitted_site_level"])
    if omitted != levels[-1]:
        raise ValueError("frozen HBN omitted site must be the final sorted level.")
    return FrozenTransform(
        selected_labels=labels,
        site_levels=levels,
        omitted_site_level=omitted,
        site_proportions=proportions,
        median_fd_center=float(preprocessing["median_fd_center"]),
        retained_volumes_center=float(preprocessing["retained_volumes_center"]),
        nuisance_coefficients=coefficients,
        adjusted_means=means,
        adjusted_scales=scales,
    )


def _strings(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = np.asarray([str(value).strip() for value in array.tolist()], dtype=str)
    if result.size == 0 or np.any(result == ""):
        raise ValueError(f"{name} contains empty values.")
    return result


def apply_frozen_transform(
    all_features: ArrayLike,
    all_labels: ArrayLike,
    acquisition_stratum: ArrayLike,
    median_fd: ArrayLike,
    retained_volumes: ArrayLike,
    transform: FrozenTransform,
) -> FloatArray:
    """Apply D_W-fitted nuisance slopes and scales without any refitting."""
    features = np.asarray(all_features, dtype=float)
    labels = _strings(all_labels, "all_labels")
    strata = _strings(acquisition_stratum, "acquisition_stratum")
    fd = np.asarray(median_fd, dtype=float)
    volumes = np.asarray(retained_volumes, dtype=float)
    if features.ndim != 2 or features.shape[1] != labels.size:
        raise ValueError("feature and label dimensions are inconsistent.")
    if len(set(labels.tolist())) != labels.size:
        raise ValueError("all_labels must be unique.")
    if not (features.shape[0] == strata.size == fd.size == volumes.size):
        raise ValueError("feature and nuisance row counts are inconsistent.")
    if not all(np.all(np.isfinite(value)) for value in (features, fd, volumes)):
        raise ValueError("features and nuisance covariates must be finite.")
    unknown = set(strata.tolist()).difference(transform.site_levels)
    if unknown:
        raise ValueError(f"unseen acquisition strata: {sorted(unknown)}")
    label_to_index = {label: index for index, label in enumerate(labels.tolist())}
    missing = set(transform.selected_labels).difference(label_to_index)
    if missing:
        raise ValueError(f"selected labels are missing: {sorted(missing)}")
    indices = np.asarray(
        [label_to_index[label] for label in transform.selected_labels], dtype=int
    )
    selected = features[:, indices]
    columns: list[FloatArray] = []
    for index, level in enumerate(transform.site_levels[:-1]):
        columns.append((strata == level).astype(float) - transform.site_proportions[index])
    columns.extend(
        [
            fd - transform.median_fd_center,
            volumes - transform.retained_volumes_center,
        ]
    )
    nuisance_design = np.column_stack(columns)
    if nuisance_design.shape[1] != transform.nuisance_coefficients.shape[0]:
        raise RuntimeError("frozen nuisance design has an unexpected dimension.")
    adjusted = selected - nuisance_design @ transform.nuisance_coefficients
    standardized = adjusted / transform.adjusted_scales
    if not np.all(np.isfinite(standardized)):
        raise RuntimeError("frozen preprocessing produced non-finite values.")
    return np.asarray(standardized, dtype=float)


def stratified_paired_scores(
    standardized_features: ArrayLike,
    acquisition_stratum: ArrayLike,
    master_seed: int,
    configuration_id: str,
) -> PairedScores:
    """Create deterministic, disjoint within-stratum difference scores."""
    features = np.asarray(standardized_features, dtype=float)
    strata = _strings(acquisition_stratum, "acquisition_stratum")
    if features.ndim != 2 or features.shape[0] != strata.size:
        raise ValueError("features and strata are inconsistent.")
    if not np.all(np.isfinite(features)):
        raise ValueError("standardized features contain non-finite values.")
    pairs: list[IntArray] = []
    unused: list[IntArray] = []
    pair_counts: dict[str, int] = {}
    unused_counts: dict[str, int] = {}
    for level in sorted(set(strata.tolist())):
        global_indices = np.flatnonzero(strata == level).astype(np.int64)
        local_pairs, local_unused = deterministic_disjoint_pairs(
            global_indices.size,
            master_seed,
            f"{configuration_id}|stratum={level}",
        )
        mapped_pairs = global_indices[local_pairs]
        mapped_unused = global_indices[local_unused]
        pairs.append(mapped_pairs)
        unused.append(mapped_unused)
        pair_counts[level] = int(mapped_pairs.shape[0])
        unused_counts[level] = int(mapped_unused.size)
    pair_indices = np.concatenate(pairs, axis=0)
    unused_indices = np.concatenate(unused, axis=0)
    first = features[pair_indices[:, 0], :]
    second = features[pair_indices[:, 1], :]
    scores = (second - first) / sqrt(2.0)
    return PairedScores(
        scores=np.asarray(scores, dtype=float),
        pair_indices=np.asarray(pair_indices, dtype=np.int64),
        unused_indices=np.asarray(unused_indices, dtype=np.int64),
        pair_count_by_stratum=pair_counts,
        unused_count_by_stratum=unused_counts,
    )


def zero_mean_cross_moment(scores: ArrayLike) -> FloatArray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a nonempty finite matrix.")
    return np.asarray(values.T @ values / float(values.shape[0]), dtype=float)


def gaussian_location_test(
    standardized_features: ArrayLike,
    null_fisher_z: float,
    adjusted_scales: ArrayLike,
) -> GaussianLocationResult:
    """One-sided mean test on the original Fisher-z scale after adjustment."""
    standardized = np.asarray(standardized_features, dtype=float)
    scales = np.asarray(adjusted_scales, dtype=float)
    if standardized.ndim != 2 or standardized.shape[0] < 1:
        raise ValueError("standardized_features must be a nonempty matrix.")
    if scales.shape != (standardized.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("adjusted_scales have an invalid shape or value.")
    threshold = float(null_fisher_z) / scales
    statistics = sqrt(float(standardized.shape[0])) * (
        np.mean(standardized, axis=0) - threshold
    )
    return GaussianLocationResult(
        statistics=np.asarray(statistics, dtype=float),
        pvalues=np.asarray(norm.sf(statistics), dtype=float),
    )


def learning_z_scores(
    standardized_features: ArrayLike,
    null_fisher_z: float,
    adjusted_scales: ArrayLike,
) -> FloatArray:
    return gaussian_location_test(
        standardized_features, null_fisher_z, adjusted_scales
    ).statistics
