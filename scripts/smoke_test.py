#!/usr/bin/env python3
"""Run noninferential deterministic smoke checks across all experiment families."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from certified_fdr.full_run import (  # noqa: E402
    build_dry_tasks,
    records_sha256 as core_records_sha256,
    run_tasks,
)
from certified_fdr.moderate_dimension_stress import validate_candidate  # noqa: E402
from certified_fdr.supplemental_routing import (  # noqa: E402
    normalized_margin_grid,
    records_sha256 as routing_records_sha256,
    simulate_transition_record,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError("output directory must be absent or empty")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require_empty(output)

    core_path = ROOT / "configs/core_experiments.json"
    routing_path = ROOT / "configs/routing_transition.json"
    moderate_path = ROOT / "configs/moderate_dimension.json"
    geometry_path = ROOT / "data/hbn/frozen_geometry.json"

    core = read_json(core_path)
    routing = read_json(routing_path)
    moderate = read_json(moderate_path)
    geometry = read_json(geometry_path)
    correlation = np.asarray(geometry["correlation"]["matrix"], dtype=float)

    tasks = build_dry_tasks(
        core,
        {"module_c": core["module_c"]},
        correlation,
        digest(core_path),
        repetitions=1,
    )
    serial = run_tasks(tasks, workers=1)
    parallel = run_tasks(tasks, workers=2)
    if serial != parallel:
        raise RuntimeError("core serial and parallel smoke records differ")

    routing_records = [
        simulate_transition_record(routing, margin, 0, digest(routing_path))
        for margin in normalized_margin_grid(routing)
    ]
    routing_replay = [
        simulate_transition_record(routing, margin, 0, digest(routing_path))
        for margin in normalized_margin_grid(routing)
    ]
    if routing_records != routing_replay:
        raise RuntimeError("routing smoke replay differs")

    moderate_validation = validate_candidate(moderate, digest(moderate_path))
    if moderate_validation["status"] != "PASS":
        raise RuntimeError("moderate-dimensional smoke validation failed")

    result = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "classification": "NONINFERENTIAL_SMOKE_VALIDATION",
        "scientific_metrics_printed": False,
        "core_record_count": len(serial),
        "core_serial_sha256": core_records_sha256(serial),
        "core_parallel_sha256": core_records_sha256(parallel),
        "routing_record_count": len(routing_records),
        "routing_record_sha256": routing_records_sha256(routing_records),
        "moderate_configuration_count": moderate_validation["configuration_count"],
        "moderate_serial_parallel_gate": moderate_validation["serial_parallel_gate"],
        "config_sha256": {
            "core": digest(core_path),
            "routing": digest(routing_path),
            "moderate": digest(moderate_path),
            "hbn_geometry": digest(geometry_path),
        },
    }
    destination = output / "smoke_validation.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SMOKE_VALIDATION_GATE=PASS")
    print(f"CORE_RECORD_COUNT={len(serial)}")
    print(f"ROUTING_RECORD_COUNT={len(routing_records)}")
    print("MODERATE_SERIAL_PARALLEL_GATE=PASS")
    print("SCIENTIFIC_METRICS_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

