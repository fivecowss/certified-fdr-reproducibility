"""Portable, resumable execution of the frozen synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from .full_run import (
    module_a_configurations,
    module_b_configurations,
    module_c_configurations,
    simulate_module_a,
    simulate_module_b,
    simulate_module_c,
)
from .ledger import finish_record, read_gzip_jsonl, write_atomic_gzip_jsonl
from .moderate_dimension_stress import simulate_stress_record, stress_configurations
from .supplemental_routing import simulate_transition_record, transition_configurations


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = {
    "core": ROOT / "configs/core_experiments.json",
    "routing": ROOT / "configs/routing_transition.json",
    "moderate": ROOT / "configs/moderate_dimension.json",
    "execution": ROOT / "configs/full_execution.json",
}
GEOMETRY_PATH = ROOT / "data/hbn/frozen_geometry.json"
BOOTSTRAP_MANIFEST_PATH = ROOT / "provenance/bootstrap_manifest.json"
EXECUTION_MANIFEST_PATH = ROOT / "provenance/execution_layer_manifest.json"

EXPERIMENTS = (
    "core-A",
    "core-B",
    "core-C",
    "routing-transition",
    "moderate-dimension",
)


@dataclass(frozen=True)
class Task:
    experiment: str
    configuration_index: int
    start: int
    stop: int
    output: Path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(destination)


def verify_release(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    execution_manifest = load_json(root / "provenance/execution_layer_manifest.json")
    bootstrap_path = root / "provenance/bootstrap_manifest.json"
    expected_bootstrap_hash = execution_manifest["bootstrap_manifest_sha256"]
    if file_sha256(bootstrap_path) != expected_bootstrap_hash:
        raise RuntimeError("bootstrap manifest hash mismatch")

    bootstrap = load_json(bootstrap_path)
    for relative, expected in bootstrap["files"].items():
        candidate = root / relative
        if not candidate.is_file() or file_sha256(candidate) != expected["sha256"]:
            raise RuntimeError(f"bootstrap source mismatch: {relative}")

    for relative, expected_hash in execution_manifest["files"].items():
        candidate = root / relative
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise RuntimeError(f"execution-layer source mismatch: {relative}")

    if execution_manifest.get("status") != "frozen":
        raise RuntimeError("execution-layer manifest is not frozen")
    return execution_manifest


def _core_config() -> dict[str, Any]:
    return load_json(CONFIG_PATHS["core"])


def configurations(experiment: str) -> list[dict[str, Any]]:
    if experiment == "core-A":
        return module_a_configurations(_core_config())
    if experiment == "core-B":
        return module_b_configurations(_core_config())
    if experiment == "core-C":
        return module_c_configurations(_core_config())
    if experiment == "routing-transition":
        return transition_configurations(load_json(CONFIG_PATHS["routing"]))
    if experiment == "moderate-dimension":
        return stress_configurations(load_json(CONFIG_PATHS["moderate"]))
    raise ValueError(f"unknown experiment: {experiment}")


def replication_budget(experiment: str, configuration: dict[str, Any]) -> int:
    if experiment == "core-A":
        return int(_core_config()["module_a"]["full_repetitions_per_configuration"])
    if experiment == "core-B":
        spec = _core_config()["module_b"]
        if float(configuration["noncentrality"]) == float(spec["primary_noncentrality"]):
            return int(spec["full_repetitions_primary"])
        return int(spec["full_repetitions_per_sensitivity"])
    if experiment == "core-C":
        return int(_core_config()["module_c"]["full_repetitions"])
    if experiment == "routing-transition":
        return int(load_json(CONFIG_PATHS["routing"])["design"]["repetitions_per_configuration"])
    if experiment == "moderate-dimension":
        return int(load_json(CONFIG_PATHS["moderate"])["design"]["repetitions_per_configuration"])
    raise ValueError(f"unknown experiment: {experiment}")


def chunk_size(experiment: str) -> int:
    return int(load_json(CONFIG_PATHS["execution"])["experiments"][experiment]["chunk_size"])


def config_path(experiment: str) -> Path:
    if experiment.startswith("core-"):
        return CONFIG_PATHS["core"]
    if experiment == "routing-transition":
        return CONFIG_PATHS["routing"]
    if experiment == "moderate-dimension":
        return CONFIG_PATHS["moderate"]
    raise ValueError(f"unknown experiment: {experiment}")


def validate_bounds(start: int, stop: int, budget: int, size: int) -> None:
    if not 0 <= int(start) < int(stop) <= int(budget):
        raise ValueError("chunk bounds lie outside the frozen replication budget")
    if int(start) % int(size) or int(stop) != min(int(start) + int(size), int(budget)):
        raise ValueError("chunk bounds do not match the frozen chunk partition")


def task_path(root: Path, experiment: str, index: int, start: int, stop: int) -> Path:
    return (
        Path(root)
        / "chunks"
        / experiment
        / f"configuration_{index:03d}"
        / f"chunk_{start:09d}_{stop:09d}.jsonl.gz"
    )


def enumerate_tasks(result_root: Path, experiments: Iterable[str]) -> list[Task]:
    selected = tuple(experiments)
    if len(selected) != len(set(selected)) or any(name not in EXPERIMENTS for name in selected):
        raise ValueError("experiment selection must be unique and recognized")
    tasks: list[Task] = []
    execution = load_json(CONFIG_PATHS["execution"])
    for experiment in selected:
        values = configurations(experiment)
        size = chunk_size(experiment)
        records = 0
        before = len(tasks)
        for index, configuration in enumerate(values):
            budget = replication_budget(experiment, configuration)
            records += budget
            for start in range(0, budget, size):
                stop = min(start + size, budget)
                tasks.append(
                    Task(
                        experiment,
                        index,
                        start,
                        stop,
                        task_path(result_root, experiment, index, start, stop),
                    )
                )
        expected = execution["experiments"][experiment]
        observed = {
            "configuration_count": len(values),
            "chunk_count": len(tasks) - before,
            "record_count": records,
        }
        for key, value in observed.items():
            if int(expected[key]) != int(value):
                raise RuntimeError(f"frozen task-count mismatch for {experiment}: {key}")
    return tasks


def simulate_record(
    experiment: str,
    configuration: dict[str, Any],
    replication: int,
    digest: str,
) -> dict[str, Any]:
    if experiment == "core-A":
        return simulate_module_a(configuration, replication, 20260915, digest, "primary", False)
    if experiment == "core-B":
        return simulate_module_b(configuration, replication, 20260915, digest, "primary", False)
    if experiment == "core-C":
        geometry = load_json(GEOMETRY_PATH)
        correlation = np.asarray(geometry["correlation"]["matrix"], dtype=float)
        return simulate_module_c(
            configuration, correlation, replication, 20260915, digest, "primary", False
        )
    if experiment == "routing-transition":
        plan = load_json(CONFIG_PATHS["routing"])
        return simulate_transition_record(
            plan,
            float(configuration["normalized_margin"]),
            replication,
            digest,
            noninferential=False,
        )
    if experiment == "moderate-dimension":
        return simulate_stress_record(
            load_json(CONFIG_PATHS["moderate"]),
            configuration,
            replication,
            digest,
            noninferential=False,
        )
    raise ValueError(f"unknown experiment: {experiment}")


def manifest_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.name + ".manifest.json")


def validate_record(
    record: dict[str, Any], experiment: str, replication: int, digest: str
) -> None:
    expected_module = {
        "core-A": "A",
        "core-B": "B",
        "core-C": "C",
        "routing-transition": "E",
        "moderate-dimension": "MD_STRESS",
    }[experiment]
    if record.get("module") != expected_module:
        raise RuntimeError("replication record module mismatch")
    if int(record.get("replication", -1)) != int(replication):
        raise RuntimeError("replication address mismatch")
    if record.get("config_sha256") != digest:
        raise RuntimeError("replication record configuration hash mismatch")
    if finish_record(record)["record_id"] != record.get("record_id"):
        raise RuntimeError("replication record identifier mismatch")
    streams = {
        tuple(record[name])
        for name in ("seed_learning", "seed_certification", "seed_testing")
    }
    if len(streams) != 3:
        raise RuntimeError("semantic random streams are not distinct")


def existing_chunk_is_valid(
    ledger: Path, sidecar: Path, expected: dict[str, Any]
) -> bool:
    if not ledger.exists() and not sidecar.exists():
        return False
    if not ledger.is_file() or not sidecar.is_file():
        raise RuntimeError("orphaned chunk output requires manual review")
    manifest = load_json(sidecar)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"existing chunk manifest mismatch: {key}")
    if manifest.get("compressed_sha256") != file_sha256(ledger):
        raise RuntimeError("existing ledger hash differs from its manifest")
    if len(read_gzip_jsonl(ledger)) != int(manifest["record_count"]):
        raise RuntimeError("existing ledger count differs from its manifest")
    return True


def run_chunk(task: Task, *, verify_source: bool = True) -> dict[str, Any]:
    if verify_source:
        verify_release()
    values = configurations(task.experiment)
    if not 0 <= int(task.configuration_index) < len(values):
        raise ValueError("configuration index lies outside the frozen grid")
    configuration = values[task.configuration_index]
    budget = replication_budget(task.experiment, configuration)
    size = chunk_size(task.experiment)
    validate_bounds(task.start, task.stop, budget, size)
    digest = file_sha256(config_path(task.experiment))
    expected = {
        "schema_version": "1.0.0",
        "status": "SEALED",
        "experiment": task.experiment,
        "configuration_index": task.configuration_index,
        "start_replication": task.start,
        "stop_replication": task.stop,
        "config_sha256": digest,
        "source_snapshot_commit": load_json(config_path(task.experiment))[
            "source_snapshot_commit"
        ],
    }
    ledger = task.output.resolve()
    sidecar = manifest_path(ledger)
    if existing_chunk_is_valid(ledger, sidecar, expected):
        return {"status": "RESUMED", "manifest": str(sidecar)}

    records = [
        simulate_record(task.experiment, configuration, replication, digest)
        for replication in range(task.start, task.stop)
    ]
    for offset, record in enumerate(records):
        validate_record(record, task.experiment, task.start + offset, digest)
    write = write_atomic_gzip_jsonl(ledger, records, 6)
    manifest = {
        **expected,
        "configuration_id": records[0]["configuration_id"],
        "record_count": len(records),
        "ledger_file": ledger.name,
        **write,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite chunk manifest: {sidecar}")
    atomic_write(sidecar, payload)
    return {"status": "COMPUTED", "manifest": str(sidecar)}


def task_payload(task: Task) -> tuple[str, int, int, int, str]:
    return (
        task.experiment,
        task.configuration_index,
        task.start,
        task.stop,
        str(task.output),
    )


def task_from_payload(value: tuple[str, int, int, int, str]) -> Task:
    return Task(value[0], int(value[1]), int(value[2]), int(value[3]), Path(value[4]))
