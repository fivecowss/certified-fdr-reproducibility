#!/usr/bin/env python3
"""Compare a complete rerun summary with its immutable frozen counterpart."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "core": ROOT / "data/frozen/core_experiments_summary.jsonl.gz",
    "routing": ROOT / "data/frozen/routing_transition_summary.jsonl.gz",
    "moderate": ROOT / "data/frozen/moderate_dimension_summary.jsonl.gz",
}
EXPECTED_ROWS = {"core": 360, "routing": 106, "moderate": 144}


def rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(FROZEN), required=True)
    parser.add_argument("--rerun-summary", type=Path, required=True)
    args = parser.parse_args()
    frozen_path = FROZEN[args.kind]
    rerun_path = args.rerun_summary.resolve()
    frozen = rows(frozen_path)
    rerun = rows(rerun_path)
    if len(frozen) != EXPECTED_ROWS[args.kind] or len(rerun) != len(frozen):
        raise RuntimeError("summary row-count comparison failed")
    if frozen != rerun:
        for index, (left, right) in enumerate(zip(frozen, rerun)):
            if left != right:
                changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
                raise RuntimeError(
                    f"frozen/rerun summary mismatch at row {index}; fields={changed}"
                )
        raise RuntimeError("frozen/rerun summary mismatch")
    print("FROZEN_RERUN_COMPARISON_GATE=PASS")
    print(f"KIND={args.kind}")
    print(f"SUMMARY_ROW_COUNT={len(rerun)}")
    print(f"FROZEN_COMPRESSED_SHA256={digest(frozen_path)}")
    print(f"RERUN_COMPRESSED_SHA256={digest(rerun_path)}")
    print("SCIENTIFIC_VALUES_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
