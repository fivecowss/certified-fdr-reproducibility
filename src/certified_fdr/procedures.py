"""BH, BY, weighted procedures, and frozen ver6.1 routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .certification import adjusted_internal_level
from .weights import validate_weights

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoutedMethod:
    method: str
    branch: str
    rejections: BoolArray
    aggressive_rejections: BoolArray
    fallback_rejections: BoolArray


def validate_pvalues(pvalues: ArrayLike) -> FloatArray:
    array = np.asarray(pvalues, dtype=float)
    if array.ndim != 1 or array.size < 1:
        raise ValueError("pvalues must be a nonempty vector.")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("pvalues must be finite and lie in [0, 1].")
    return array


def harmonic_number(m: int) -> float:
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 1:
        raise ValueError("m must be a positive integer.")
    return float(np.sum(1.0 / np.arange(1, int(m) + 1)))


def weighted_bh(pvalues: ArrayLike, weights: ArrayLike, q: float) -> BoolArray:
    p = validate_pvalues(pvalues)
    q_value = float(q)
    if not 0.0 < q_value < 1.0:
        raise ValueError("q must lie in (0, 1).")
    w = validate_weights(weights, p.size)

    scores = np.full(p.size, np.inf, dtype=float)
    positive = w > 0.0
    scores[positive] = p[positive] / w[positive]
    ordered = np.sort(scores, kind="mergesort")
    ranks = np.arange(1, p.size + 1)
    admissible = ordered <= q_value * ranks / p.size
    if not np.any(admissible):
        return np.zeros(p.size, dtype=bool)
    k_hat = int(np.flatnonzero(admissible)[-1] + 1)
    threshold = q_value * k_hat / p.size
    return np.asarray(scores <= threshold, dtype=bool)


def bh(pvalues: ArrayLike, q: float) -> BoolArray:
    p = validate_pvalues(pvalues)
    return weighted_bh(p, np.ones(p.size), q)


def by(pvalues: ArrayLike, q: float) -> BoolArray:
    p = validate_pvalues(pvalues)
    return bh(p, float(q) / harmonic_number(p.size))


def weighted_by(pvalues: ArrayLike, weights: ArrayLike, q: float) -> BoolArray:
    p = validate_pvalues(pvalues)
    return weighted_bh(p, weights, float(q) / harmonic_number(p.size))


def route_methods(
    pvalues: ArrayLike,
    weights: ArrayLike,
    certificate_accepted: bool,
    sign_valid: bool,
    q: float,
    beta: float,
) -> dict[str, RoutedMethod]:
    """Return required baselines, proposed routes, oracle, and old-router ablation."""
    p = validate_pvalues(pvalues)
    w = validate_weights(weights, p.size)
    if not isinstance(certificate_accepted, (bool, np.bool_)):
        raise TypeError("certificate_accepted must be Boolean.")
    if not isinstance(sign_valid, (bool, np.bool_)):
        raise TypeError("sign_valid must be Boolean.")

    q_internal = adjusted_internal_level(q, beta)
    aggressive_q = weighted_bh(p, w, q)
    aggressive_internal = weighted_bh(p, w, q_internal)
    fallback_by = by(p, q_internal)
    fallback_wby = weighted_by(p, w, q_internal)
    direct_by = by(p, q)
    direct_wby = weighted_by(p, w, q)

    def routed(name: str, accepted: bool, aggressive: BoolArray, fallback: BoolArray) -> RoutedMethod:
        return RoutedMethod(
            method=name,
            branch="aggressive" if accepted else "fallback",
            rejections=(aggressive if accepted else fallback).copy(),
            aggressive_rejections=aggressive.copy(),
            fallback_rejections=fallback.copy(),
        )

    methods = {
        "BY_q": routed("BY_q", True, direct_by, direct_by),
        "wBY_q": routed("wBY_q", True, direct_wby, direct_wby),
        "wBH_q_diagnostic": routed(
            "wBH_q_diagnostic", True, aggressive_q, aggressive_q
        ),
        "Cert_wBY": routed(
            "Cert_wBY", bool(certificate_accepted), aggressive_q, fallback_wby
        ),
        "Cert_BY": routed(
            "Cert_BY", bool(certificate_accepted), aggressive_q, fallback_by
        ),
        "sign_oracle": routed("sign_oracle", bool(sign_valid), aggressive_q, fallback_wby),
        "old_both_q_internal": routed(
            "old_both_q_internal",
            bool(certificate_accepted),
            aggressive_internal,
            fallback_wby,
        ),
    }
    return methods
