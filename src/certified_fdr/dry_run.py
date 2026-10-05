"""Noninferential dry-run generator for configuration and output validation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from json import dumps
from math import ceil
from typing import Any

import numpy as np
from scipy.stats import norm

from .certification import (
    certification_threshold,
    evaluate_certificate,
    pair_values,
    sufficient_certificate_size,
)
from .covariance import (
    check_correlation,
    exact_gaussian_product_means,
    global_pairs,
    one_factor_regime,
)
from .metrics import false_discovery_proportion, power
from .monte_carlo import rng_for, seed_words
from .procedures import route_methods
from .weights import learned_weights

Record = dict[str, Any]
Task = tuple[str, dict[str, Any], int, int, str]


def canonical_json(value: Any) -> str:
    return dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def records_sha256(records: list[Record]) -> str:
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    return sha256(payload.encode("utf-8")).hexdigest()


def _finish_record(record: Record) -> Record:
    identity = dict(record)
    identity.pop("record_id", None)
    record["record_id"] = sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return record


def _indices(rejections: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(rejections, dtype=bool)).astype(int).tolist()


def build_dry_tasks(design: dict[str, Any], config_sha256: str) -> list[Task]:
    repetitions = int(design["execution"]["dry_run_repetitions"])
    master_seed = int(design["execution"]["master_seed"])
    tasks: list[Task] = []
    for law in ("valid_interior", "boundary", "invalid"):
        configuration = {"d": 5, "gamma": 0.1, "ratio": 0.125, "law": law}
        for replication in range(repetitions):
            tasks.append(("A", configuration, replication, master_seed, config_sha256))
    for regime in design["module_b"]["regimes"]:
        configuration = {"regime": regime, "eta": 0.5, "noncentrality": 2.0}
        for replication in range(repetitions):
            tasks.append(("B", configuration, replication, master_seed, config_sha256))
    return tasks


def _base_record(
    module: str,
    configuration_id: str,
    data_configuration_id: str,
    replication: int,
    master_seed: int,
    config_sha256: str,
) -> Record:
    return {
        "schema_version": "1.0.0",
        "record_id": "",
        "run_kind": "dry_run",
        "noninferential": True,
        "config_sha256": config_sha256,
        "module": module,
        "configuration_id": configuration_id,
        "replication": replication,
        "master_seed": master_seed,
        "seed_learning": list(seed_words(master_seed, data_configuration_id, replication, "learning")),
        "seed_certification": list(seed_words(master_seed, data_configuration_id, replication, "certification")),
        "seed_testing": list(seed_words(master_seed, data_configuration_id, replication, "testing")),
    }


def _module_a(configuration: dict[str, Any], replication: int, master_seed: int, config_sha256: str) -> list[Record]:
    d = int(configuration["d"])
    gamma = float(configuration["gamma"])
    law = str(configuration["law"])
    ratio = float(configuration["ratio"])
    n_upper = sufficient_certificate_size(d, gamma, 0.01)
    n_certificate = ceil(ratio * n_upper)
    correlations = np.full(d, gamma)
    if law == "boundary":
        correlations[0] = 0.0
    elif law == "invalid":
        correlations[0] = -gamma
    configuration_id = f"A|d={d}|gamma={gamma:g}|ratio={ratio:g}|law={law}"
    record = _base_record("A", configuration_id, configuration_id, replication, master_seed, config_sha256)
    sample = exact_gaussian_product_means(
        n_certificate,
        correlations,
        1,
        rng_for(master_seed, configuration_id, replication, "certification"),
    )[0]
    certificate = evaluate_certificate(sample, n_certificate, 0.01)
    record.update(
        {
            "regime": law,
            "true_sign_valid": law != "invalid",
            "m": 2 * d,
            "d": d,
            "N_C": n_certificate,
            "q": None,
            "beta": 0.01,
            "eta": None,
            "noncentrality": None,
            "certificate_accepted": certificate.accepted,
            "minimum_certified_cross_moment": certificate.minimum_cross_moment,
            "tau": certificate.threshold,
            "psd_minimum_eigenvalue": float(1.0 - np.max(np.abs(correlations))),
            "method": "certificate_only",
            "branch": "certificate_only",
            "fdp": None,
            "power": None,
            "rejection_count": None,
            "R_w": [],
            "R_0": [],
            "R": [],
        }
    )
    return [_finish_record(record)]


def _module_b(configuration: dict[str, Any], replication: int, master_seed: int, config_sha256: str) -> list[Record]:
    regime = str(configuration["regime"])
    eta = float(configuration["eta"])
    noncentrality = float(configuration["noncentrality"])
    m, n_certificate, q, beta = 20, 5000, 0.10, 0.01
    loadings, correlation = one_factor_regime(m, 0.5, regime, boundary_index=19)
    psd_minimum = check_correlation(correlation).minimum_eigenvalue
    pairs = global_pairs(m)
    data_id = f"B|regime={regime}|mu={noncentrality:g}"
    configuration_id = f"{data_id}|eta={eta:g}"

    nonnull = np.zeros(m, dtype=bool)
    nonnull[[0, 1, 10, 11]] = True
    mean = noncentrality * nonnull.astype(float)
    learning = rng_for(master_seed, data_id, replication, "learning").multivariate_normal(mean, correlation)
    testing = rng_for(master_seed, data_id, replication, "testing").multivariate_normal(mean, correlation)
    certificate_sample = rng_for(master_seed, data_id, replication, "certification").multivariate_normal(
        np.zeros(m), correlation, size=n_certificate
    )
    cross_moments = certificate_sample.T @ certificate_sample / n_certificate
    selected = pair_values(cross_moments, pairs)
    certificate = evaluate_certificate(selected, n_certificate, beta)
    weights = learned_weights(learning, eta)
    pvalues = norm.sf(testing)
    methods = route_methods(pvalues, weights, certificate.accepted, regime != "invalid", q, beta)

    records: list[Record] = []
    for method_name in sorted(methods):
        routed = methods[method_name]
        record = _base_record("B", configuration_id, data_id, replication, master_seed, config_sha256)
        record.update(
            {
                "regime": regime,
                "true_sign_valid": regime != "invalid",
                "m": m,
                "d": int(pairs.shape[0]),
                "N_C": n_certificate,
                "q": q,
                "beta": beta,
                "eta": eta,
                "noncentrality": noncentrality,
                "certificate_accepted": certificate.accepted,
                "minimum_certified_cross_moment": certificate.minimum_cross_moment,
                "tau": certification_threshold(n_certificate, beta),
                "psd_minimum_eigenvalue": psd_minimum,
                "method": method_name,
                "branch": routed.branch,
                "fdp": false_discovery_proportion(routed.rejections, nonnull),
                "power": power(routed.rejections, nonnull),
                "rejection_count": int(np.sum(routed.rejections)),
                "R_w": _indices(routed.aggressive_rejections),
                "R_0": _indices(routed.fallback_rejections),
                "R": _indices(routed.rejections),
            }
        )
        records.append(_finish_record(record))
    return records


def run_task(task: Task) -> list[Record]:
    module, configuration, replication, master_seed, config_sha256 = task
    if module == "A":
        return _module_a(configuration, replication, master_seed, config_sha256)
    if module == "B":
        return _module_b(configuration, replication, master_seed, config_sha256)
    raise ValueError(f"unsupported dry-run module: {module}")


def run_dry(design: dict[str, Any], config_sha256: str, workers: int) -> list[Record]:
    tasks = build_dry_tasks(design, config_sha256)
    if int(workers) == 1:
        chunks = [run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            chunks = list(executor.map(run_task, tasks))
    records = [record for chunk in chunks for record in chunk]
    records.sort(key=lambda row: (row["module"], row["configuration_id"], row["replication"], row["method"]))
    return records
