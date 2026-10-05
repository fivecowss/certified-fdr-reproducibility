"""Dependence-certified learned weighted FDR implementation."""

from .certification import (
    CertificateResult,
    adjusted_internal_level,
    certification_threshold,
    completeness_margin,
    evaluate_certificate,
    sufficient_certificate_size,
)
from .covariance import one_factor_correlation, one_factor_regime
from .procedures import by, route_methods, weighted_bh, weighted_by
from .weights import learned_weights

__all__ = [
    "CertificateResult",
    "adjusted_internal_level",
    "certification_threshold",
    "completeness_margin",
    "evaluate_certificate",
    "sufficient_certificate_size",
    "one_factor_correlation",
    "one_factor_regime",
    "by",
    "route_methods",
    "weighted_bh",
    "weighted_by",
    "learned_weights",
]

__version__ = "0.3.0"
