"""Frozen clipped-exponential learned weights."""

from __future__ import annotations

from math import log

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def validate_weights(weights: ArrayLike, m: int, tolerance: float = 1e-10) -> FloatArray:
    array = np.asarray(weights, dtype=float)
    if array.shape != (m,):
        raise ValueError("weights must have shape (m,).")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    if not np.isclose(array.sum(), m, rtol=1e-10, atol=tolerance):
        raise ValueError("weights must sum to m.")
    return array


def learned_weights(
    learning_scores: ArrayLike,
    eta: float,
    clip_lower: float = 0.25,
    clip_upper: float = 4.0,
) -> FloatArray:
    """Compute A_i=clip(exp(eta Z_i^W),l,u), W_i=m A_i/sum(A)."""
    scores = np.asarray(learning_scores, dtype=float)
    if scores.ndim not in {1, 2}:
        raise ValueError("learning_scores must have shape (m,) or (B, m).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("learning_scores contain non-finite values.")
    eta_value = float(eta)
    if not np.isfinite(eta_value) or eta_value < 0.0:
        raise ValueError("eta must be finite and nonnegative.")
    if not 0.0 < clip_lower <= clip_upper:
        raise ValueError("require 0 < clip_lower <= clip_upper.")

    log_raw = eta_value * scores
    bounded_log = np.clip(log_raw, log(clip_lower), log(clip_upper))
    activity = np.clip(np.exp(bounded_log), clip_lower, clip_upper)
    m = scores.shape[-1]
    weights = m * activity / activity.sum(axis=-1, keepdims=True)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError("constructed weights must be positive and finite.")
    if not np.allclose(weights.sum(axis=-1), m, rtol=1e-12, atol=1e-12):
        raise RuntimeError("constructed weights do not sum to m.")
    return np.asarray(weights, dtype=float)
