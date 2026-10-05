"""Result construction for the model-based exploratory HBN application."""

from __future__ import annotations

from hashlib import sha256
import json
from math import atanh
from typing import Any

import numpy as np

from .certification import evaluate_certificate, pair_values
from .hbn_execution import (
    FrozenTransform,
    gaussian_location_test,
    learning_z_scores,
    stratified_paired_scores,
    zero_mean_cross_moment,
)
from .monte_carlo import seed_words
from .procedures import route_methods
from .weights import learned_weights
from .covariance import global_pairs


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _indices(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int).tolist()


def build_hbn_application_record(
    *,
    d_w_standardized: np.ndarray,
    d_c_standardized: np.ndarray,
    d_t_standardized: np.ndarray,
    d_c_strata: np.ndarray,
    transform: FrozenTransform,
    raw_correlation_thresholds: list[float],
    primary_raw_correlation_threshold: float,
    eta_values: list[float],
    primary_eta: float,
    q: float,
    beta: float,
    master_seed: int,
    config_sha256: str,
    input_sha256: dict[str, str],
    family_limitation: str,
) -> dict[str, Any]:
    """Build one aggregate, privacy-safe HBN application record."""
    d_w = np.asarray(d_w_standardized, dtype=float)
    d_c = np.asarray(d_c_standardized, dtype=float)
    d_t = np.asarray(d_t_standardized, dtype=float)
    strata = np.asarray(d_c_strata).astype(str)
    if any(value.ndim != 2 for value in (d_w, d_c, d_t)):
        raise ValueError("D_W, D_C, and D_T must be matrices.")
    m = len(transform.selected_labels)
    if any(value.shape[1] != m for value in (d_w, d_c, d_t)):
        raise ValueError("HBN split feature dimensions disagree with the transform.")
    if d_c.shape[0] != strata.size:
        raise ValueError("D_C strata and feature rows disagree.")
    if not all(np.all(np.isfinite(value)) for value in (d_w, d_c, d_t)):
        raise ValueError("HBN transformed features contain non-finite values.")
    if not isinstance(family_limitation, str) or not family_limitation.strip():
        raise ValueError("family limitation text is required.")
    if set(input_sha256) != {"D_W", "D_C", "D_T", "split_manifest"}:
        raise ValueError("input_sha256 must bind all three splits and the manifest.")
    if any(len(value) != 64 for value in input_sha256.values()):
        raise ValueError("input SHA-256 values must have 64 hexadecimal characters.")

    pairing_id = "D|hbn_application|certificate_pairing"
    paired = stratified_paired_scores(d_c, strata, master_seed, pairing_id)
    pairs = global_pairs(m)
    certificate = evaluate_certificate(
        pair_values(zero_mean_cross_moment(paired.scores), pairs),
        paired.scores.shape[0],
        beta,
    )
    excluded_methods = {"sign_oracle"}
    threshold_results: list[dict[str, Any]] = []
    primary_branch: str | None = None
    primary_rejection_count: int | None = None

    for raw_threshold in raw_correlation_thresholds:
        raw_value = float(raw_threshold)
        if not -1.0 < raw_value < 1.0:
            raise ValueError("raw correlation thresholds must lie in (-1, 1).")
        fisher_threshold = atanh(raw_value)
        test = gaussian_location_test(
            d_t, fisher_threshold, transform.adjusted_scales
        )
        for eta in eta_values:
            eta_value = float(eta)
            learning = learning_z_scores(
                d_w, fisher_threshold, transform.adjusted_scales
            )
            weights = learned_weights(learning, eta_value)
            routed = route_methods(
                test.pvalues,
                weights,
                certificate.accepted,
                False,
                q,
                beta,
            )
            method_results: dict[str, Any] = {}
            for method_name in sorted(set(routed).difference(excluded_methods)):
                method = routed[method_name]
                branch = (
                    "direct"
                    if method_name in {"BY_q", "wBY_q", "wBH_q_diagnostic"}
                    else method.branch
                )
                method_results[method_name] = {
                    "branch": branch,
                    "rejection_count": int(np.sum(method.rejections)),
                    "R_w": _indices(method.aggressive_rejections),
                    "R_0": _indices(method.fallback_rejections),
                    "R": _indices(method.rejections),
                }
            threshold_results.append(
                {
                    "raw_correlation_threshold": raw_value,
                    "null_fisher_z": fisher_threshold,
                    "eta": eta_value,
                    "weight_minimum": float(np.min(weights)),
                    "weight_maximum": float(np.max(weights)),
                    "weight_sum": float(np.sum(weights)),
                    "methods": method_results,
                }
            )
            if np.isclose(raw_value, primary_raw_correlation_threshold) and np.isclose(
                eta_value, primary_eta
            ):
                primary = method_results["Cert_wBY"]
                primary_branch = str(primary["branch"])
                primary_rejection_count = int(primary["rejection_count"])

    if primary_branch is None or primary_rejection_count is None:
        raise ValueError("primary threshold and eta must be included in their grids.")
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_id": "",
        "claim_label": "model-based exploratory application",
        "finite_sample_validation_claimed": False,
        "config_sha256": config_sha256,
        "input_sha256": dict(sorted(input_sha256.items())),
        "split_sizes": {
            "D_W": int(d_w.shape[0]),
            "D_C": int(d_c.shape[0]),
            "D_T": int(d_t.shape[0]),
        },
        "m": m,
        "d": int(pairs.shape[0]),
        "N_C": int(paired.scores.shape[0]),
        "q": float(q),
        "beta": float(beta),
        "tau": certificate.threshold,
        "minimum_certified_cross_moment": certificate.minimum_cross_moment,
        "certificate_accepted": certificate.accepted,
        "selected_branch": primary_branch,
        "primary_rejection_count": primary_rejection_count,
        "primary_eta": float(primary_eta),
        "primary_raw_correlation_threshold": float(primary_raw_correlation_threshold),
        "ROI_list": list(transform.selected_labels),
        "pairing": {
            "master_seed": int(master_seed),
            "seed_certification_pairing_by_stratum": {
                level: list(
                    seed_words(
                        master_seed,
                        f"{pairing_id}|stratum={level}",
                        0,
                        "certification_pairing",
                    )
                )
                for level in sorted(paired.pair_count_by_stratum)
            },
            "pair_count_by_stratum": paired.pair_count_by_stratum,
            "unused_count_by_stratum": paired.unused_count_by_stratum,
        },
        "threshold_results": threshold_results,
        "family_information_limitation": family_limitation.strip(),
        "retuning_after_D_C_or_D_T": False,
        "contains_participant_ids": False,
        "contains_individual_rows": False,
    }
    identity = dict(record)
    identity.pop("record_id")
    record["record_id"] = sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return record
