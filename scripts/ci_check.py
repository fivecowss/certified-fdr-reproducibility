#!/usr/bin/env python3
"""Run the bounded clean-environment validation used by CI and the container."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str], environment: dict[str, str]) -> None:
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def main() -> int:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    with tempfile.TemporaryDirectory(prefix="certified-fdr-ci-") as temporary:
        root = Path(temporary)
        run([sys.executable, "scripts/validate_environment.py"], environment)
        run(
            [
                sys.executable,
                "scripts/reproduce.py",
                "--mode",
                "frozen",
                "--output-dir",
                str(root / "frozen"),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                "scripts/reproduce.py",
                "--mode",
                "smoke",
                "--output-dir",
                str(root / "smoke"),
            ],
            environment,
        )
        run([sys.executable, "-m", "pytest"], environment)
        run(
            [
                sys.executable,
                "scripts/run_full.py",
                "--group",
                "all",
                "--result-root",
                str(root / "full"),
                "--workers",
                "2",
                "--preflight-only",
            ],
            environment,
        )
        run(
            [sys.executable, "scripts/build_review_archive.py", "--audit-only"],
            environment,
        )
    print("CLEAN_ENVIRONMENT_VALIDATION_GATE=PASS")
    print("FULL_SIMULATION_STARTED=FALSE")
    print("SCIENTIFIC_METRICS_PRINTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
