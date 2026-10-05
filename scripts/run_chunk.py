#!/usr/bin/env python3
"""Run or verify one exact chunk from a frozen synthetic grid."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certified_fdr.portable_execution import EXPERIMENTS, Task, run_chunk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--configuration-index", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_chunk(
        Task(
            args.experiment,
            args.configuration_index,
            args.start,
            args.stop,
            args.output.resolve(),
        )
    )
    print("CHUNK_EXECUTION_GATE=PASS")
    print(f"CHUNK_STATUS={result['status']}")
    print(f"CHUNK_MANIFEST={result['manifest']}")
    print("SCIENTIFIC_METRICS_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
