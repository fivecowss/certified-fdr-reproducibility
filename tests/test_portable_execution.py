from __future__ import annotations

from collections import Counter
from pathlib import Path

from certified_fdr.portable_execution import (
    EXPERIMENTS,
    enumerate_tasks,
    simulate_record,
    configurations,
    file_sha256,
    config_path,
    verify_release,
)


def test_execution_release_manifest_and_bootstrap_sources_are_bound() -> None:
    value = verify_release()
    assert value["status"] == "frozen"
    assert value["scientific_metrics_changed"] is False


def test_full_task_counts_match_frozen_grid(tmp_path: Path) -> None:
    tasks = enumerate_tasks(tmp_path, EXPERIMENTS)
    counts = Counter(task.experiment for task in tasks)
    assert counts == {
        "core-A": 10800,
        "core-B": 7200,
        "core-C": 1800,
        "routing-transition": 4000,
        "moderate-dimension": 3600,
    }
    assert sum(task.stop - task.start for task in tasks) == 29_900_000


def test_first_record_is_deterministic_for_every_experiment() -> None:
    for experiment in EXPERIMENTS:
        configuration = configurations(experiment)[0]
        digest = file_sha256(config_path(experiment))
        first = simulate_record(experiment, configuration, 0, digest)
        second = simulate_record(experiment, configuration, 0, digest)
        assert first == second
        assert first["record_id"] == second["record_id"]
        assert first["config_sha256"] == digest
        assert len(
            {
                tuple(first["seed_learning"]),
                tuple(first["seed_certification"]),
                tuple(first["seed_testing"]),
            }
        ) == 3
