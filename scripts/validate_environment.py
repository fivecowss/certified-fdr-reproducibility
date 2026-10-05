#!/usr/bin/env python3
"""Validate Python and the complete dependency closure against the frozen lock."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATH = ROOT / "environment/environment.json"
LOCK_PATH = ROOT / "environment/requirements-lock.txt"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def normalized(name: str) -> str:
    return name.lower().replace("_", "-")


def lock_entries(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, package_version = line.partition("==")
        if not separator or not name or not package_version:
            raise ValueError(f"non-exact lock entry: {line}")
        key = normalized(name)
        if key in result:
            raise ValueError(f"duplicate lock entry: {name}")
        result[key] = package_version
    return result


def validate(*, require_exact_python: bool = True) -> dict[str, Any]:
    specification = load_json(ENVIRONMENT_PATH)
    expected_python = str(specification["python"])
    actual_python = ".".join(map(str, sys.version_info[:3]))
    if require_exact_python and actual_python != expected_python:
        raise RuntimeError(
            f"Python version mismatch: expected {expected_python}, observed {actual_python}"
        )

    locked = lock_entries(LOCK_PATH)
    declared = {
        normalized(name): str(package_version)
        for name, package_version in specification["packages"].items()
    }
    if locked != declared:
        raise RuntimeError("requirements lock differs from environment specification")
    observed: dict[str, str] = {}
    for name, expected in locked.items():
        try:
            actual = version(name)
        except PackageNotFoundError as error:
            raise RuntimeError(f"missing locked distribution: {name}") from error
        if actual != expected:
            raise RuntimeError(
                f"distribution mismatch for {name}: expected {expected}, observed {actual}"
            )
        observed[name] = actual
    return {
        "status": "PASS",
        "python": actual_python,
        "distribution_count": len(observed),
        "scientific_metrics_printed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-python-patch-difference",
        action="store_true",
        help="Reserved for diagnostics; release checks require the exact patch version.",
    )
    args = parser.parse_args()
    result = validate(require_exact_python=not args.allow_python_patch_difference)
    print("ENVIRONMENT_LOCK_GATE=PASS")
    print(f"PYTHON_VERSION={result['python']}")
    print(f"LOCKED_DISTRIBUTION_COUNT={result['distribution_count']}")
    print("SCIENTIFIC_METRICS_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
