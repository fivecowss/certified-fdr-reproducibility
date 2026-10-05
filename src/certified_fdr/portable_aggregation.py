"""Coverage-checked aggregation for portable full synthetic reruns."""

from __future__ import annotations

from gzip import open as gzip_open
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Iterator

import numpy
import scipy

from .ledger import finish_record, write_atomic_gzip_jsonl
from .moderate_dimension_analysis import ModerateDimensionSummaryBuilder
from .portable_execution import (
    CONFIG_PATHS,
    Task,
    config_path,
    enumerate_tasks,
    file_sha256,
    load_json,
    manifest_path,
    verify_release,
)
from .summary import SummaryBuilder
from .supplemental_analysis import SupplementalSummaryBuilder


ROOT = Path(__file__).resolve().parents[2]


def record_stream(path: Path) -> Iterator[dict[str, Any]]:
    with gzip_open(path, mode="rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("replication ledger row is not an object")
                yield value


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregation output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _builder(kind: str) -> Any:
    if kind == "core":
        return SummaryBuilder(confidence=0.95)
    if kind == "routing":
        plan = load_json(CONFIG_PATHS["routing"])
        return SupplementalSummaryBuilder(
            plan["design"]["normalized_margin"],
            int(plan["design"]["repetitions_per_configuration"]),
        )
    if kind == "moderate":
        plan = load_json(CONFIG_PATHS["moderate"])
        return ModerateDimensionSummaryBuilder(
            plan["design"]["m"],
            int(plan["design"]["repetitions_per_configuration"]),
        )
    raise ValueError(f"unknown aggregation kind: {kind}")


def _experiments(kind: str) -> tuple[str, ...]:
    if kind == "core":
        return ("core-A", "core-B", "core-C")
    if kind == "routing":
        return ("routing-transition",)
    if kind == "moderate":
        return ("moderate-dimension",)
    raise ValueError(f"unknown aggregation kind: {kind}")


def _output_names(kind: str) -> tuple[str, str]:
    if kind == "core":
        return "core_experiments_summary.jsonl.gz", "core_experiments_rerun_manifest.json"
    if kind == "routing":
        return "routing_transition_summary.jsonl.gz", "routing_transition_rerun_manifest.json"
    if kind == "moderate":
        return "moderate_dimension_summary.jsonl.gz", "moderate_dimension_rerun_manifest.json"
    raise ValueError(f"unknown aggregation kind: {kind}")


def _expected_summary_rows(kind: str) -> int:
    return {"core": 360, "routing": 106, "moderate": 144}[kind]


def _validate_manifest(task: Task, manifest: dict[str, Any]) -> None:
    expected = {
        "schema_version": "1.0.0",
        "status": "SEALED",
        "experiment": task.experiment,
        "configuration_index": task.configuration_index,
        "start_replication": task.start,
        "stop_replication": task.stop,
        "config_sha256": file_sha256(config_path(task.experiment)),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"chunk manifest mismatch: {task.experiment}:{key}")


def aggregate(kind: str, result_root: Path, output_dir: Path) -> dict[str, Any]:
    verify_release()
    result_root = Path(result_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty aggregation directory")
    tasks = enumerate_tasks(result_root, _experiments(kind))
    builder = _builder(kind)
    artifact_rows: list[str] = []
    record_count = 0
    seen_record_ids: set[str] = set()

    for task in tasks:
        sidecar = manifest_path(task.output)
        if not task.output.is_file() or not sidecar.is_file():
            raise RuntimeError(f"missing ledger-manifest pair: {task.output}")
        manifest = load_json(sidecar)
        _validate_manifest(task, manifest)
        if file_sha256(task.output) != manifest.get("compressed_sha256"):
            raise RuntimeError("ledger hash mismatch")
        count = 0
        for expected_replication, record in enumerate(
            record_stream(task.output), start=task.start
        ):
            if expected_replication >= task.stop:
                raise RuntimeError("ledger has too many records")
            if int(record.get("replication", -1)) != expected_replication:
                raise RuntimeError("replication sequence is not contiguous")
            record_id = str(record.get("record_id", ""))
            if finish_record(record)["record_id"] != record_id:
                raise RuntimeError("record identifier mismatch")
            if record_id in seen_record_ids:
                raise RuntimeError("duplicate record identifier")
            seen_record_ids.add(record_id)
            builder.update(record)
            count += 1
        if count != task.stop - task.start or count != int(manifest["record_count"]):
            raise RuntimeError("chunk record count mismatch")
        record_count += count
        artifact_rows.extend(
            [
                f"{file_sha256(task.output)}  {task.output.relative_to(result_root)}",
                f"{file_sha256(sidecar)}  {sidecar.relative_to(result_root)}",
            ]
        )

    summaries = builder.finalize()
    if len(summaries) != _expected_summary_rows(kind):
        raise RuntimeError("summary row count differs from the frozen analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_name, manifest_name = _output_names(kind)
    summary_path = output_dir / summary_name
    summary_info = write_atomic_gzip_jsonl(summary_path, summaries, 6)
    artifact_hash = sha256(
        ("\n".join(sorted(artifact_rows)) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "classification": "portable full synthetic rerun",
        "kind": kind,
        "experiments": list(_experiments(kind)),
        "chunk_count": len(tasks),
        "record_count": record_count,
        "summary_record_count": len(summaries),
        "summary_file": summary_name,
        "summary_sha256": summary_info["compressed_sha256"],
        "input_artifact_set_sha256": artifact_hash,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "scientific_metrics_printed": False,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path_out = output_dir / manifest_name
    atomic_write(manifest_path_out, payload)
    return {
        "kind": kind,
        "chunk_count": len(tasks),
        "record_count": record_count,
        "summary_record_count": len(summaries),
        "summary_path": str(summary_path),
        "summary_sha256": summary_info["compressed_sha256"],
        "manifest_path": str(manifest_path_out),
        "manifest_sha256": file_sha256(manifest_path_out),
    }
