#!/usr/bin/env python3
"""Portable entry point for frozen reproduction and smoke validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str]) -> None:
    result = subprocess.run(arguments, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with status {result.returncode}: {' '.join(arguments)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("frozen", "smoke"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure1-layout", choices=("full", "single"), default="full")
    args = parser.parse_args()
    output = args.output_dir.resolve()

    if args.mode == "frozen":
        frozen = ROOT / "data/frozen"
        run([sys.executable, str(ROOT / "scripts/verify_frozen_inputs.py")])
        run(
            [
                sys.executable,
                str(ROOT / "scripts/render_figures.py"),
                "--routing-summary", str(frozen / "routing_transition_summary.jsonl.gz"),
                "--routing-manifest", str(frozen / "routing_transition_manifest.json"),
                "--primary-summary", str(frozen / "core_experiments_summary.jsonl.gz"),
                "--primary-manifest", str(frozen / "core_experiments_manifest.json"),
                "--moderate-summary", str(frozen / "moderate_dimension_summary.jsonl.gz"),
                "--moderate-manifest", str(frozen / "moderate_dimension_manifest.json"),
                "--output-dir", str(output),
                "--figure1-layout", args.figure1_layout,
            ]
        )
        print("FROZEN_REPRODUCTION_GATE=PASS")
    else:
        run(
            [
                sys.executable,
                str(ROOT / "scripts/smoke_test.py"),
                "--output-dir", str(output),
            ]
        )
        print("SMOKE_REPRODUCTION_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

