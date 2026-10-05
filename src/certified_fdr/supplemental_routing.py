"""Prespecified near-boundary routing-transition experiment."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isclose
from typing import Any

import numpy as np

from .certification import completeness_margin
from .covariance import (
    check_correlation,
    global_pairs,
    one_factor_correlation,
    selected_correlations,
)
from .full_run import _simulate_routing
from .ledger import finish_record


def normalized_margin_grid(plan: dict[str, Any]) -> tuple[float, ...]:
    values = tuple(float(value) for value in plan["design"]["normalized_margin"])
    if len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("normalized-margin grid must be unique and sorted")
    if 0.0 not in values or 1.0 not in values:
        raise ValueError(
            "normalized-margin grid must contain the soundness and completeness landmarks"
        )
    return values


def transition_geometry(plan: dict[str, Any], normalized_margin: float) -> dict[str, Any]:
    design = plan["design"]
    m = int(design["m"])
    d = int(design["d"])
    n_certificate = int(design["N_C"])
    beta = float(design["beta"])
    base_loading = float(design["base_loading"])
    varied_index = int(design["varied_loading_index_zero_based"])
    if d != m * (m - 1) // 2:
        raise ValueError("d does not equal the global pair count")
    if not 0.0 < base_loading < 1.0:
        raise ValueError("base loading must lie in (0,1)")
    if not 0 <= varied_index < m:
        raise ValueError("varied loading index is outside the hypothesis vector")

    delta = completeness_margin(n_certificate, d, beta)
    target_margin = float(normalized_margin) * delta
    varied_loading = target_margin / base_loading
    if abs(varied_loading) >= base_loading:
        # Equality at the base loading would make another pair tie for the minimum and
        # values above it would break the exact target-margin identity.
        if not abs(varied_loading) < 1.0 or target_margin >= base_loading**2:
            raise ValueError("normalized margin exceeds the exact one-factor path")
    loadings = np.full(m, base_loading, dtype=float)
    loadings[varied_index] = varied_loading
    correlation = one_factor_correlation(loadings)
    pairs = global_pairs(m)
    actual_margin = float(np.min(selected_correlations(correlation, pairs)))
    if not isclose(actual_margin, target_margin, rel_tol=0.0, abs_tol=5e-15):
        raise RuntimeError("one-factor path does not attain the requested s_J")
    check = check_correlation(correlation)
    return {
        "normalized_margin": float(normalized_margin),
        "target_s_J": target_margin,
        "actual_s_J": actual_margin,
        "Delta": delta,
        "varied_loading": varied_loading,
        "loadings": loadings,
        "correlation": correlation,
        "minimum_eigenvalue": check.minimum_eigenvalue,
    }


def transition_configurations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "normalized_margin": value,
            "geometry": transition_geometry(plan, value),
        }
        for value in normalized_margin_grid(plan)
    ]


def _regime(value: float) -> str:
    if value < 0.0:
        return "sign_invalid"
    if value == 0.0:
        return "boundary"
    if value < 1.0:
        return "positive_no_completeness_guarantee"
    return "complete_interior"


def simulate_transition_record(
    plan: dict[str, Any],
    normalized_margin: float,
    replication: int,
    config_sha256: str,
    *,
    master_seed: int | None = None,
    noninferential: bool = True,
) -> dict[str, Any]:
    design = plan["design"]
    geometry = transition_geometry(plan, normalized_margin)
    seed = int(plan["rng"]["master_seed"] if master_seed is None else master_seed)
    record = _simulate_routing(
        module="E",
        regime=_regime(float(normalized_margin)),
        correlation=np.asarray(geometry["correlation"], dtype=float),
        n_certificate=int(design["N_C"]),
        nonnull_indices=list(map(int, design["nonnull_indices_zero_based"])),
        eta=float(design["eta"]),
        noncentrality=float(design["noncentrality"]),
        q=float(design["q"]),
        beta=float(design["beta"]),
        replication=int(replication),
        master_seed=seed,
        config_sha256=str(config_sha256),
        run_kind="dry_run" if noninferential else "supplemental",
        noninferential=bool(noninferential),
        geometry_id=f"normalized_margin={float(normalized_margin):g}",
    )
    record["normalized_margin"] = float(normalized_margin)
    record["varied_loading"] = float(geometry["varied_loading"])
    # _simulate_routing seals its base record before transition-specific fields exist.
    # Re-seal after extension so record_id commits to every stored field.
    return finish_record(record)


def records_sha256(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    return sha256(payload).hexdigest()
