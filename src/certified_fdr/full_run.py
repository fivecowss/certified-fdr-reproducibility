"""Replication-addressed kernels for certificate and routing experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import ceil
from typing import Any

import numpy as np
from scipy.stats import norm

from .certification import (
    completeness_margin,
    evaluate_certificate,
    pair_values,
    sufficient_certificate_size,
)
from .covariance import (
    check_correlation,
    exact_gaussian_product_means,
    gaussian_wishart_cross_moments,
    global_pairs,
    one_factor_regime,
    selected_correlations,
)
from .ledger import finish_record
from .metrics import false_discovery_proportion, power, symmetric_difference_count
from .monte_carlo import rng_for, seed_words
from .procedures import route_methods
from .weights import learned_weights


Record = dict[str, Any]
Task = tuple[
    str,
    dict[str, Any],
    int,
    int,
    str,
    str,
    bool,
    list[list[float]] | None,
]


def _indices(rejections: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(rejections, dtype=bool)).astype(int).tolist()


def records_sha256(records: list[Record]) -> str:
    rows = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records]
    return sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def module_a_configurations(design: dict[str, Any]) -> list[dict[str, Any]]:
    module = design["module_a"]
    return [
        {"d": d, "gamma": gamma, "ratio": ratio, "law": law, "beta": module["beta"]}
        for d in module["d"]
        for gamma in module["gamma"]
        for ratio in module["n_upper_multipliers"]
        for law in sorted(module["laws"])
    ]


def module_b_configurations(design: dict[str, Any]) -> list[dict[str, Any]]:
    module = design["module_b"]
    noncentralities = [
        module["primary_noncentrality"],
        *module["sensitivity_noncentrality"],
    ]
    return [
        {"regime": regime, "eta": eta, "noncentrality": noncentrality}
        for regime in module["regimes"]
        for noncentrality in noncentralities
        for eta in module["eta"]
    ]


def module_c_configurations(hbn_plan: dict[str, Any]) -> list[dict[str, Any]]:
    module = hbn_plan["module_c"]
    noncentralities = [
        module["primary_noncentrality"],
        *module["sensitivity_noncentrality"],
    ]
    return [
        {"eta": eta, "noncentrality": noncentrality}
        for noncentrality in noncentralities
        for eta in module["eta"]
    ]


def oracle_region(s_j: float, delta: float) -> tuple[str, bool]:
    value = float(s_j)
    margin = float(delta)
    if value < 0.0:
        return "sign_invalid", True
    if value >= margin:
        return "complete_interior", True
    return "no_oracle_guarantee", False


def _base_record(
    *,
    module: str,
    configuration_id: str,
    data_configuration_id: str,
    replication: int,
    master_seed: int,
    config_sha256: str,
    run_kind: str,
    noninferential: bool,
) -> Record:
    return {
        "schema_version": "2.0.0",
        "record_id": "",
        "run_kind": run_kind,
        "noninferential": bool(noninferential),
        "config_sha256": config_sha256,
        "module": module,
        "configuration_id": configuration_id,
        "data_configuration_id": data_configuration_id,
        "replication": int(replication),
        "master_seed": int(master_seed),
        "seed_learning": list(
            seed_words(master_seed, data_configuration_id, replication, "learning")
        ),
        "seed_certification": list(
            seed_words(master_seed, data_configuration_id, replication, "certification")
        ),
        "seed_testing": list(
            seed_words(master_seed, data_configuration_id, replication, "testing")
        ),
    }


def _method_results(
    methods: dict[str, Any],
    nonnull: np.ndarray,
    oracle_applicable: bool,
) -> dict[str, dict[str, Any]]:
    oracle = methods["sign_oracle"].rejections
    results: dict[str, dict[str, Any]] = {}
    direct = {"BY_q", "wBY_q", "wBH_q_diagnostic"}
    for method_name in sorted(methods):
        method = methods[method_name]
        if method_name in direct:
            branch = "direct"
        elif method_name == "sign_oracle":
            branch = "oracle_aggressive" if method.branch == "aggressive" else "oracle_fallback"
        else:
            branch = str(method.branch)
        results[method_name] = {
            "branch": branch,
            "fdp": false_discovery_proportion(method.rejections, nonnull),
            "power": power(method.rejections, nonnull),
            "rejection_count": int(np.sum(method.rejections)),
            "R_w": _indices(method.aggressive_rejections),
            "R_0": _indices(method.fallback_rejections),
            "R": _indices(method.rejections),
            "oracle_disagreement_count": (
                symmetric_difference_count(method.rejections, oracle)
                if oracle_applicable
                else None
            ),
        }
    return results


def simulate_module_a(
    configuration: dict[str, Any],
    replication: int,
    master_seed: int,
    config_sha256: str,
    run_kind: str,
    noninferential: bool,
) -> Record:
    d = int(configuration["d"])
    gamma = float(configuration["gamma"])
    beta = float(configuration.get("beta", 0.01))
    law = str(configuration["law"])
    ratio = float(configuration["ratio"])
    n_upper = sufficient_certificate_size(d, gamma, beta)
    n_certificate = ceil(ratio * n_upper)
    correlations = np.full(d, gamma, dtype=float)
    if law == "boundary":
        correlations[0] = 0.0
    elif law == "invalid":
        correlations[0] = -gamma
    elif law != "valid_interior":
        raise ValueError(f"unknown Module A law: {law}")
    configuration_id = (
        f"A|d={d}|gamma={gamma:g}|ratio={ratio:g}|law={law}"
    )
    record = _base_record(
        module="A",
        configuration_id=configuration_id,
        data_configuration_id=configuration_id,
        replication=replication,
        master_seed=master_seed,
        config_sha256=config_sha256,
        run_kind=run_kind,
        noninferential=noninferential,
    )
    sample = exact_gaussian_product_means(
        n_certificate,
        correlations,
        1,
        rng_for(master_seed, configuration_id, replication, "certification"),
    )[0]
    certificate = evaluate_certificate(sample, n_certificate, beta)
    s_j = float(np.min(correlations))
    delta = completeness_margin(n_certificate, d, beta)
    region, applicable = oracle_region(s_j, delta)
    record.update(
        {
            "regime": law,
            "true_sign_valid": bool(s_j >= 0.0),
            "oracle_region": region,
            "oracle_applicable": applicable,
            "s_J": s_j,
            "Delta": delta,
            "m": 2 * d,
            "d": d,
            "N_C": n_certificate,
            "q": None,
            "beta": beta,
            "eta": None,
            "noncentrality": None,
            "truth_nonnull_indices": [],
            "certificate_accepted": certificate.accepted,
            "minimum_certified_cross_moment": certificate.minimum_cross_moment,
            "tau": certificate.threshold,
            "psd_minimum_eigenvalue": float(1.0 - np.max(np.abs(correlations))),
            "method_results": {
                "certificate_only": {
                    "branch": "certificate_only",
                    "fdp": None,
                    "power": None,
                    "rejection_count": None,
                    "R_w": [],
                    "R_0": [],
                    "R": [],
                    "oracle_disagreement_count": None,
                }
            },
        }
    )
    return finish_record(record)


def _simulate_routing(
    *,
    module: str,
    regime: str,
    correlation: np.ndarray,
    n_certificate: int,
    nonnull_indices: list[int],
    eta: float,
    noncentrality: float,
    q: float,
    beta: float,
    replication: int,
    master_seed: int,
    config_sha256: str,
    run_kind: str,
    noninferential: bool,
    geometry_id: str,
) -> Record:
    m = int(correlation.shape[0])
    check = check_correlation(correlation)
    pairs = global_pairs(m)
    population_pairs = selected_correlations(correlation, pairs)
    s_j = float(np.min(population_pairs))
    delta = completeness_margin(n_certificate, int(pairs.shape[0]), beta)
    region, applicable = oracle_region(s_j, delta)
    data_id = f"{module}|geometry={geometry_id}|mu={noncentrality:g}"
    configuration_id = f"{data_id}|eta={eta:g}"
    nonnull = np.zeros(m, dtype=bool)
    nonnull[np.asarray(nonnull_indices, dtype=int)] = True
    mean = noncentrality * nonnull.astype(float)

    learning = rng_for(
        master_seed, data_id, replication, "learning"
    ).multivariate_normal(mean, correlation, check_valid="raise")
    testing = rng_for(
        master_seed, data_id, replication, "testing"
    ).multivariate_normal(mean, correlation, check_valid="raise")
    cross_moments = gaussian_wishart_cross_moments(
        n_certificate,
        correlation,
        1,
        rng_for(master_seed, data_id, replication, "certification"),
    )[0]
    certificate = evaluate_certificate(
        pair_values(cross_moments, pairs), n_certificate, beta
    )
    weights = learned_weights(learning, eta)
    methods = route_methods(
        norm.sf(testing), weights, certificate.accepted, s_j >= 0.0, q, beta
    )
    record = _base_record(
        module=module,
        configuration_id=configuration_id,
        data_configuration_id=data_id,
        replication=replication,
        master_seed=master_seed,
        config_sha256=config_sha256,
        run_kind=run_kind,
        noninferential=noninferential,
    )
    record.update(
        {
            "regime": regime,
            "true_sign_valid": bool(s_j >= 0.0),
            "oracle_region": region,
            "oracle_applicable": applicable,
            "s_J": s_j,
            "Delta": delta,
            "m": m,
            "d": int(pairs.shape[0]),
            "N_C": int(n_certificate),
            "q": q,
            "beta": beta,
            "eta": eta,
            "noncentrality": noncentrality,
            "truth_nonnull_indices": list(map(int, nonnull_indices)),
            "certificate_accepted": certificate.accepted,
            "minimum_certified_cross_moment": certificate.minimum_cross_moment,
            "tau": certificate.threshold,
            "psd_minimum_eigenvalue": check.minimum_eigenvalue,
            "method_results": _method_results(methods, nonnull, applicable),
        }
    )
    return finish_record(record)


def simulate_module_b(
    configuration: dict[str, Any],
    replication: int,
    master_seed: int,
    config_sha256: str,
    run_kind: str,
    noninferential: bool,
) -> Record:
    regime = str(configuration["regime"])
    _, correlation = one_factor_regime(20, 0.5, regime, boundary_index=19)
    return _simulate_routing(
        module="B",
        regime=regime,
        correlation=correlation,
        n_certificate=5000,
        nonnull_indices=[0, 1, 10, 11],
        eta=float(configuration["eta"]),
        noncentrality=float(configuration["noncentrality"]),
        q=0.10,
        beta=0.01,
        replication=replication,
        master_seed=master_seed,
        config_sha256=config_sha256,
        run_kind=run_kind,
        noninferential=noninferential,
        geometry_id=regime,
    )


def simulate_module_c(
    configuration: dict[str, Any],
    correlation: np.ndarray,
    replication: int,
    master_seed: int,
    config_sha256: str,
    run_kind: str,
    noninferential: bool,
) -> Record:
    return _simulate_routing(
        module="C",
        regime="hbn_frozen_geometry",
        correlation=np.asarray(correlation, dtype=float),
        n_certificate=682,
        nonnull_indices=[0, 1],
        eta=float(configuration["eta"]),
        noncentrality=float(configuration["noncentrality"]),
        q=0.10,
        beta=0.01,
        replication=replication,
        master_seed=master_seed,
        config_sha256=config_sha256,
        run_kind=run_kind,
        noninferential=noninferential,
        geometry_id="hbn_frozen_v1",
    )


def simulate_task(task: Task) -> Record:
    (
        module,
        configuration,
        replication,
        master_seed,
        config_sha256,
        run_kind,
        noninferential,
        correlation,
    ) = task
    if module == "A":
        return simulate_module_a(
            configuration,
            replication,
            master_seed,
            config_sha256,
            run_kind,
            noninferential,
        )
    if module == "B":
        return simulate_module_b(
            configuration,
            replication,
            master_seed,
            config_sha256,
            run_kind,
            noninferential,
        )
    if module == "C" and correlation is not None:
        return simulate_module_c(
            configuration,
            np.asarray(correlation, dtype=float),
            replication,
            master_seed,
            config_sha256,
            run_kind,
            noninferential,
        )
    raise ValueError(f"unsupported task module: {module}")


def build_dry_tasks(
    design: dict[str, Any],
    hbn_plan: dict[str, Any],
    correlation: np.ndarray,
    config_sha256: str,
    repetitions: int,
) -> list[Task]:
    master_seed = int(design["execution"]["master_seed"])
    tasks: list[Task] = []
    configurations = {
        "A": module_a_configurations(design),
        "B": module_b_configurations(design),
        "C": module_c_configurations(hbn_plan),
    }
    for module, values in configurations.items():
        frozen_correlation = correlation.tolist() if module == "C" else None
        for configuration in values:
            for replication in range(int(repetitions)):
                tasks.append(
                    (
                        module,
                        configuration,
                        replication,
                        master_seed,
                        config_sha256,
                        "dry_run",
                        True,
                        frozen_correlation,
                    )
                )
    return tasks


def run_tasks(tasks: list[Task], workers: int) -> list[Record]:
    if int(workers) < 1:
        raise ValueError("workers must be positive.")
    if int(workers) == 1:
        records = [simulate_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            records = list(executor.map(simulate_task, tasks))
    records.sort(
        key=lambda row: (
            row["module"],
            row["configuration_id"],
            row["replication"],
        )
    )
    return records
