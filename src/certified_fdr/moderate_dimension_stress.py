"""Moderate-dimensional extension of the frozen routing-transition path."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import isclose
from typing import Any

import numpy as np

from .certification import completeness_margin
from .covariance import check_correlation, global_pairs, selected_correlations
from .full_run import _simulate_routing
from .ledger import finish_record
from .monte_carlo import rng_for, seed_words
from .supplemental_routing import transition_geometry
from .weights import learned_weights, validate_weights


METHODS = (
    "BY_q",
    "wBY_q",
    "wBH_q_diagnostic",
    "Cert_wBY",
    "Cert_BY",
    "sign_oracle",
    "old_both_q_internal",
)
EXPECTED_N_C = {20: 5000, 50: 5559, 100: 5967}
NORMALIZED_MARGINS = (-0.5, 0.5, 1.25)
SOURCE_ROUTING_COMMIT = "989ce49034429e1d415fd3c9d0cecce504d855fc"


def canonical_payload(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def value_sha256(value: Any) -> str:
    return sha256(canonical_payload(value)).hexdigest()


def smallest_n_certificate(d: int, beta: float, upper_bound: float) -> int:
    """Return the smallest n with completeness_margin(n,d,beta) <= upper_bound."""

    low, high = 1, 1
    while completeness_margin(high, d, beta) > upper_bound:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if completeness_margin(middle, d, beta) <= upper_bound:
            high = middle
        else:
            low = middle + 1
    return low


def dimension_specifications(plan: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    design = plan["design"]
    beta = float(design["beta"])
    delta_0 = completeness_margin(5000, 190, beta)
    dimensions = []
    for m in map(int, design["m"]):
        if m % 10:
            raise ValueError("m must be divisible by ten for the frozen truth pattern")
        d = m * (m - 1) // 2
        n_certificate = smallest_n_certificate(d, beta, delta_0)
        block = m // 10
        nonnull = tuple(range(block)) + tuple(range(m // 2, m // 2 + block))
        dimensions.append(
            {
                "m": m,
                "d": d,
                "N_C": n_certificate,
                "nonnull_indices_zero_based": list(nonnull),
            }
        )
    return tuple(dimensions)


def stress_configurations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    margins = tuple(float(value) for value in plan["design"]["normalized_margin"])
    if margins != NORMALIZED_MARGINS:
        raise ValueError("normalized-margin grid differs from the prespecified grid")
    configurations: list[dict[str, Any]] = []
    for dimension in dimension_specifications(plan):
        for margin in margins:
            configurations.append(
                {
                    **dimension,
                    "normalized_margin": margin,
                    "configuration_index": len(configurations),
                }
            )
    return configurations


def geometry_for_configuration(
    plan: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any]:
    design = plan["design"]
    local_plan = {
        "design": {
            "m": int(configuration["m"]),
            "d": int(configuration["d"]),
            "N_C": int(configuration["N_C"]),
            "beta": float(design["beta"]),
            "base_loading": float(design["base_loading"]),
            "varied_loading_index_zero_based": int(configuration["m"]) - 1,
        }
    }
    return transition_geometry(local_plan, float(configuration["normalized_margin"]))


def regime_for_margin(value: float) -> str:
    if value < 0.0:
        return "sign_invalid"
    if value < 1.0:
        return "positive_no_completeness_guarantee"
    return "complete_interior"


def simulate_stress_record(
    plan: dict[str, Any],
    configuration: dict[str, Any],
    replication: int,
    config_sha256: str,
    *,
    noninferential: bool,
) -> dict[str, Any]:
    design = plan["design"]
    geometry = geometry_for_configuration(plan, configuration)
    m = int(configuration["m"])
    margin = float(configuration["normalized_margin"])
    record = _simulate_routing(
        module="MD_STRESS",
        regime=regime_for_margin(margin),
        correlation=np.asarray(geometry["correlation"], dtype=float),
        n_certificate=int(configuration["N_C"]),
        nonnull_indices=list(map(int, configuration["nonnull_indices_zero_based"])),
        eta=float(design["eta"]),
        noncentrality=float(design["noncentrality"]),
        q=float(design["q"]),
        beta=float(design["beta"]),
        replication=int(replication),
        master_seed=int(plan["rng"]["master_seed"]),
        config_sha256=str(config_sha256),
        run_kind="dry_run" if noninferential else "moderate_dimension_stress",
        noninferential=bool(noninferential),
        geometry_id=(
            f"namespace={plan['rng']['semantic_namespace']}|m={m}|d={configuration['d']}|"
            f"N_C={configuration['N_C']}|x={margin:g}"
        ),
    )
    record.update(
        {
            "normalized_margin": margin,
            "varied_loading": float(geometry["varied_loading"]),
            "configuration_index": int(configuration["configuration_index"]),
            "N_W_label": int(design["N_W"]),
            "N_T_label": int(design["N_T"]),
            "score_generation": "one Gaussian score vector per learning/testing split",
        }
    )
    return finish_record(record)


def _dry_task(arguments: tuple[dict[str, Any], dict[str, Any], int, str]) -> dict[str, Any]:
    plan, configuration, replication, digest = arguments
    return simulate_stress_record(
        plan, configuration, replication, digest, noninferential=True
    )


def validate_candidate(plan: dict[str, Any], candidate_sha256: str) -> dict[str, Any]:
    design = plan["design"]
    configurations = stress_configurations(plan)
    if len(configurations) != 9:
        raise RuntimeError("expected nine stress-test configurations")
    delta_0 = completeness_margin(5000, 190, float(design["beta"]))
    gates: list[dict[str, Any]] = []
    baseline_plan = {
        "design": {
            "m": 20,
            "d": 190,
            "N_C": 5000,
            "beta": float(design["beta"]),
            "base_loading": float(design["base_loading"]),
            "varied_loading_index_zero_based": 19,
        }
    }
    for configuration in configurations:
        m = int(configuration["m"])
        d = int(configuration["d"])
        n_certificate = int(configuration["N_C"])
        margin = float(configuration["normalized_margin"])
        if d != m * (m - 1) // 2:
            raise RuntimeError("global-pair dimension identity failed")
        if n_certificate != EXPECTED_N_C[m]:
            raise RuntimeError("N_C anchor differs from the current implementation")
        if completeness_margin(n_certificate, d, float(design["beta"])) > delta_0:
            raise RuntimeError("dimension-specific completeness margin exceeds Delta_0")
        if n_certificate > 1 and completeness_margin(
            n_certificate - 1, d, float(design["beta"])
        ) <= delta_0:
            raise RuntimeError("N_C is not minimal")

        geometry = geometry_for_configuration(plan, configuration)
        correlation = np.asarray(geometry["correlation"], dtype=float)
        check = check_correlation(correlation)
        if check.maximum_symmetry_error > 1e-12 or check.maximum_diagonal_error > 1e-12:
            raise RuntimeError("correlation symmetry or unit-diagonal gate failed")
        pairs = global_pairs(m)
        actual = float(np.min(selected_correlations(correlation, pairs)))
        target = margin * float(geometry["Delta"])
        if not isclose(actual, target, rel_tol=0.0, abs_tol=5e-15):
            raise RuntimeError("exact-margin identity failed")
        if m == 20:
            baseline = transition_geometry(baseline_plan, margin)
            if not np.array_equal(correlation, np.asarray(baseline["correlation"])):
                raise RuntimeError("m=20 geometry differs from frozen routing transition")

        data_id = (
            f"MD_STRESS|geometry=namespace={plan['rng']['semantic_namespace']}|m={m}|d={d}|"
            f"N_C={n_certificate}|x={margin:g}|mu={float(design['noncentrality']):g}"
        )
        learning = rng_for(
            int(plan["rng"]["master_seed"]), data_id, 0, "learning"
        ).multivariate_normal(
            float(design["noncentrality"])
            * np.isin(np.arange(m), configuration["nonnull_indices_zero_based"]),
            correlation,
            check_valid="raise",
        )
        validate_weights(learned_weights(learning, float(design["eta"])), m)
        stream_keys = {
            seed_words(int(plan["rng"]["master_seed"]), data_id, 0, stream)
            for stream in ("learning", "certification", "testing")
        }
        if len(stream_keys) != 3:
            raise RuntimeError("semantic RNG streams are not independent addresses")
        gates.append(
            {
                "configuration_index": int(configuration["configuration_index"]),
                "m": m,
                "d": d,
                "N_C": n_certificate,
                "normalized_margin": margin,
                "minimum_eigenvalue": float(check.minimum_eigenvalue),
                "target_margin_error": abs(actual - target),
            }
        )

    tasks = [
        (plan, configuration, replication, candidate_sha256)
        for configuration in configurations
        for replication in (0, 1)
    ]
    serial = [_dry_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=2) as executor:
        parallel = list(executor.map(_dry_task, tasks))
    if serial != parallel:
        raise RuntimeError("serial and parallel dry-run records differ")
    if any(set(record["method_results"]) != set(METHODS) for record in serial):
        raise RuntimeError("dry-run method set differs from the frozen seven methods")
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "scientific_metrics_printed": False,
        "full_simulation_authorized": False,
        "source_routing_commit": SOURCE_ROUTING_COMMIT,
        "candidate_plan_sha256": candidate_sha256,
        "configuration_count": len(configurations),
        "dry_record_count": len(serial),
        "serial_parallel_gate": "PASS",
        "delta_0": delta_0,
        "configuration_gates": gates,
        "serial_record_sha256": value_sha256(serial),
        "parallel_record_sha256": value_sha256(parallel),
    }
