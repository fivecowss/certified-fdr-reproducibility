#!/usr/bin/env python3
"""Preflight, execute, or resume the complete frozen synthetic grids."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certified_fdr.portable_execution import (  # noqa: E402
    EXPERIMENTS,
    Task,
    enumerate_tasks,
    file_sha256,
    load_json,
    manifest_path,
    run_chunk,
    task_from_payload,
    task_payload,
    verify_release,
)


GROUPS = {
    "core": ("core-A", "core-B", "core-C"),
    "routing": ("routing-transition",),
    "moderate": ("moderate-dimension",),
    "all": EXPERIMENTS,
}


def worker(value: tuple[str, int, int, int, str]) -> str:
    result = run_chunk(task_from_payload(value), verify_source=False)
    return str(result["status"])


def manifest_set_hash(tasks: list[Task], root: Path) -> str:
    rows = []
    for task in tasks:
        sidecar = manifest_path(task.output)
        if not task.output.is_file() or not sidecar.is_file():
            raise RuntimeError(f"completed task lacks a ledger-manifest pair: {task.output}")
        rows.append(f"{file_sha256(sidecar)}  {sidecar.relative_to(root)}")
    return sha256(("\n".join(sorted(rows)) + "\n").encode("utf-8")).hexdigest()


def atomic_receipt(path: Path, value: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"completion receipt already exists: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(GROUPS), default="all")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_release()
    root = args.result_root.resolve()
    experiments = GROUPS[args.group]
    if not 1 <= int(args.workers) <= max(1, os.cpu_count() or 1):
        raise ValueError("workers must lie between one and the logical CPU count")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")

    tasks = enumerate_tasks(root, experiments)
    records = sum(task.stop - task.start for task in tasks)
    disk_target = root if root.exists() else root.parent
    while not disk_target.exists() and disk_target != disk_target.parent:
        disk_target = disk_target.parent
    free_gib = shutil.disk_usage(disk_target).free / 1024**3
    estimated_gib = records * 120.0 / 1024**3
    required_gib = max(3.0, 1.5 * estimated_gib + 1.0)
    if free_gib < required_gib:
        raise RuntimeError("insufficient free disk for the selected frozen grid")

    print("FULL_EXECUTION_PREFLIGHT_GATE=PASS")
    print(f"GROUP={args.group}")
    print(f"EXPERIMENTS={','.join(experiments)}")
    print(f"PLANNED_CHUNK_COUNT={len(tasks)}")
    print(f"PLANNED_RECORD_COUNT={records}")
    print(f"WORKERS={args.workers}")
    print(f"FREE_DISK_GIB={free_gib:.3f}")
    print(f"MINIMUM_REQUIRED_GIB={required_gib:.3f}")
    print("SCIENTIFIC_METRICS_PRINTED=FALSE")
    if args.preflight_only:
        print("FULL_SIMULATION_STARTED=FALSE")
        return 0

    if root.exists() and not args.resume:
        raise FileExistsError("result root exists; use --resume only after reviewing it")
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "control" / f"{args.group}_completion.json"
    if receipt.exists():
        raise FileExistsError("selected group already has a sealed completion receipt")

    computed = 0
    resumed = 0
    started = datetime.now(timezone.utc).isoformat()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        payloads = [task_payload(task) for task in tasks]
        for completed, status in enumerate(executor.map(worker, payloads, chunksize=1), 1):
            computed += int(status == "COMPUTED")
            resumed += int(status == "RESUMED")
            if completed % args.progress_every == 0 or completed == len(tasks):
                print(
                    f"PROGRESS={completed}/{len(tasks)} COMPUTED={computed} RESUMED={resumed}",
                    flush=True,
                )

    artifact_hash = manifest_set_hash(tasks, root)
    receipt_value = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "group": args.group,
        "experiments": list(experiments),
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(tasks),
        "record_count": records,
        "computed_chunk_count": computed,
        "resumed_chunk_count": resumed,
        "chunk_manifest_set_sha256": artifact_hash,
        "scientific_metrics_inspected": False,
        "execution_manifest_sha256": file_sha256(
            ROOT / "provenance/execution_layer_manifest.json"
        ),
    }
    receipt_hash = atomic_receipt(receipt, receipt_value)
    print("FULL_EXECUTION_GATE=PASS")
    print(f"COMPLETED_CHUNK_COUNT={len(tasks)}")
    print(f"COMPLETED_RECORD_COUNT={records}")
    print(f"COMPUTED_CHUNK_COUNT={computed}")
    print(f"RESUMED_CHUNK_COUNT={resumed}")
    print(f"CHUNK_MANIFEST_SET_SHA256={artifact_hash}")
    print(f"COMPLETION_RECEIPT_SHA256={receipt_hash}")
    print("SCIENTIFIC_METRICS_INSPECTED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
