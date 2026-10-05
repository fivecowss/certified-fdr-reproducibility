#!/usr/bin/env python3
"""Aggregate a complete portable rerun after validating every chunk."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certified_fdr.portable_aggregation import aggregate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("core", "routing", "moderate"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.kind, args.result_root, args.output_dir)
    print("FULL_AGGREGATION_GATE=PASS")
    print(f"KIND={result['kind']}")
    print(f"VERIFIED_CHUNK_COUNT={result['chunk_count']}")
    print(f"VERIFIED_RECORD_COUNT={result['record_count']}")
    print(f"SUMMARY_RECORD_COUNT={result['summary_record_count']}")
    print(f"SUMMARY_SHA256={result['summary_sha256']}")
    print(f"MANIFEST_SHA256={result['manifest_sha256']}")
    print("SCIENTIFIC_METRICS_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
