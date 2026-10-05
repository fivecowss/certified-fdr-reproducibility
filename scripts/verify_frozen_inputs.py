#!/usr/bin/env python3
"""Verify immutable aggregate inputs without printing scientific estimates."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "core_experiments_summary.jsonl.gz": (
        "7194cf3102e6bce97afae00dfc9987d5e3d4484d3513cd70bfe170acabe3d7e2",
        360,
    ),
    "core_experiments_manifest.json": (
        "39276375e543ac38b7ab8d8671de4154939ca3c69f5d445bcf44a1f956eaa75f",
        None,
    ),
    "routing_transition_summary.jsonl.gz": (
        "76ed40fef2c34e3441a679ef79c8d8dca761396a5aaaaf08055519b34d60eff5",
        106,
    ),
    "routing_transition_manifest.json": (
        "b5c7a84e9d727c4de8f46ec724aee4ac1d5f0d0101484efb8a75e1c2b082f042",
        None,
    ),
    "moderate_dimension_summary.jsonl.gz": (
        "c26ad14d3a9e3010de62d731e52ca25cc56d0e2baaea583a8b7f89cd4b3ada87",
        144,
    ),
    "moderate_dimension_manifest.json": (
        "c46445f40320e4c5c9b057395d80d9583948727da4d989cae0e71919b76abe07",
        None,
    ),
    "hbn_observed_application.jsonl.gz": (
        "05753513e33569826bfa3c99bfd1ad340f3d0c6149e3cd4727ae2f8d2a9608e3",
        1,
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify(root: Path) -> dict[str, Any]:
    observed: dict[str, str] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, (expected_hash, expected_rows) in EXPECTED.items():
        path = root / name
        require(path.is_file(), f"missing frozen input: {name}")
        actual_hash = digest(path)
        require(actual_hash == expected_hash, f"hash mismatch: {name}")
        observed[name] = actual_hash
        if expected_rows is not None:
            loaded[name] = rows(path)
            require(len(loaded[name]) == expected_rows, f"row-count mismatch: {name}")

    core = loaded["core_experiments_summary.jsonl.gz"]
    require(Counter(row["module"] for row in core) == {"A": 108, "B": 189, "C": 63},
            "core module cell counts differ from the frozen design")

    routing = loaded["routing_transition_summary.jsonl.gz"]
    require(Counter(row["record_type"] for row in routing) ==
            {"certificate": 10, "method": 70, "power_contrast": 26},
            "routing summary structure differs from the frozen analysis")

    moderate = loaded["moderate_dimension_summary.jsonl.gz"]
    require(Counter(row["record_type"] for row in moderate) ==
            {"certificate": 9, "method_metric": 126, "power_contrast": 9},
            "moderate-dimensional summary structure differs from the frozen analysis")

    hbn = loaded["hbn_observed_application.jsonl.gz"][0]
    require(hbn["contains_participant_ids"] is False, "HBN record contains participant IDs")
    require(hbn["contains_individual_rows"] is False, "HBN record contains individual rows")
    require(hbn["claim_label"] == "model-based exploratory application",
            "HBN claim label mismatch")

    return {
        "status": "PASS",
        "file_count": len(EXPECTED),
        "summary_row_counts": {name: len(value) for name, value in loaded.items()},
        "sha256": observed,
        "scientific_values_printed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/frozen",
    )
    args = parser.parse_args()
    result = verify(args.input_dir.resolve())
    print("FROZEN_INPUT_GATE=PASS")
    print(f"FROZEN_INPUT_COUNT={result['file_count']}")
    print("SCIENTIFIC_VALUES_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

