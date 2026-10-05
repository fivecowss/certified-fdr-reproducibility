"""Replication-level multiple-testing metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validated(rejections: ArrayLike, nonnull: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    rejected = np.asarray(rejections, dtype=bool)
    signal = np.asarray(nonnull, dtype=bool)
    if rejected.ndim != 1 or rejected.shape != signal.shape:
        raise ValueError("rejections and nonnull must be equal-length vectors.")
    return rejected, signal


def rejection_count(rejections: ArrayLike) -> int:
    rejected = np.asarray(rejections, dtype=bool)
    if rejected.ndim != 1:
        raise ValueError("rejections must be one-dimensional.")
    return int(np.sum(rejected))


def false_discovery_proportion(rejections: ArrayLike, nonnull: ArrayLike) -> float:
    rejected, signal = _validated(rejections, nonnull)
    denominator = max(int(np.sum(rejected)), 1)
    return float(np.sum(rejected & ~signal) / denominator)


def power(rejections: ArrayLike, nonnull: ArrayLike) -> float:
    rejected, signal = _validated(rejections, nonnull)
    alternatives = int(np.sum(signal))
    if alternatives == 0:
        return float("nan")
    return float(np.sum(rejected & signal) / alternatives)


def symmetric_difference_count(first: ArrayLike, second: ArrayLike) -> int:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("sets must be equal-length vectors.")
    return int(np.sum(left ^ right))
