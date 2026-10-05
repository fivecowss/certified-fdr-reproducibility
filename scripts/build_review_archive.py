#!/usr/bin/env python3
"""Audit and build a deterministic, identity-scrubbed review archive."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_MANIFEST = ROOT / "provenance/environment_layer_manifest.json"
TOP_LEVEL_FILES = (
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    ".python-version",
    "Dockerfile",
    "README.md",
    "pyproject.toml",
)
INCLUDED_DIRECTORIES = (
    ".github",
    "configs",
    "data",
    "docs",
    "environment",
    "provenance",
    "scripts",
    "src",
    "tests",
)
SKIP_NAMES = {".DS_Store", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
SKIP_DIRECTORY_SUFFIXES = (".dist-info", ".egg-info")
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}
PUBLIC_SURFACE_PREFIXES = (
    ".github/",
    "configs/",
    "docs/",
    "environment/",
    "scripts/",
    "src/",
    "tests/",
)
PUBLIC_FORBIDDEN = (
    "AIS" + "TATS",
    "Phase " + "8",
    "phase" + "8",
    "/" + "Users/" + "soye" + "ong",
    "five" + "cowss",
)
IDENTITY_FORBIDDEN = ("/" + "Users/", "soye" + "ong", "five" + "cowss")


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def verify_environment_layer(root: Path = ROOT) -> dict:
    manifest = load_json(root / "provenance/environment_layer_manifest.json")
    execution_manifest = root / "provenance/execution_layer_manifest.json"
    if digest_file(execution_manifest) != manifest["execution_layer_manifest_sha256"]:
        raise RuntimeError("execution-layer manifest hash mismatch")
    for relative, expected_hash in manifest["files"].items():
        path = root / relative
        if not path.is_file() or digest_file(path) != expected_hash:
            raise RuntimeError(f"environment-layer source mismatch: {relative}")
    if manifest.get("status") != "frozen":
        raise RuntimeError("environment-layer manifest is not frozen")
    return manifest


def contains_skipped_directory(parts: tuple[str, ...]) -> bool:
    return any(
        part in SKIP_NAMES or part.endswith(SKIP_DIRECTORY_SUFFIXES)
        for part in parts
    )


def collect_files(root: Path = ROOT) -> list[Path]:
    root = Path(root).resolve()
    files: list[Path] = []
    for relative in TOP_LEVEL_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact file: {relative}")
        files.append(path)
    for directory in INCLUDED_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"missing artifact directory: {directory}")
        for path in base.rglob("*"):
            relative_parts = path.relative_to(root).parts
            if not path.is_file() or contains_skipped_directory(relative_parts):
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            files.append(path)
    unique = sorted(set(files), key=lambda value: value.relative_to(root).as_posix())
    if any(".git" in path.relative_to(root).parts for path in unique):
        raise RuntimeError("archive inventory contains Git metadata")
    return unique


def decoded_text(path: Path) -> str | None:
    try:
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return handle.read()
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def audit_files(files: Iterable[Path], root: Path = ROOT) -> dict[str, int]:
    public_hits = 0
    identity_hits = 0
    participant_level_candidates = 0
    allowed_data = {
        "data/frozen/core_experiments_manifest.json",
        "data/frozen/core_experiments_summary.jsonl.gz",
        "data/frozen/hbn_observed_application.jsonl.gz",
        "data/frozen/moderate_dimension_manifest.json",
        "data/frozen/moderate_dimension_summary.jsonl.gz",
        "data/frozen/routing_transition_manifest.json",
        "data/frozen/routing_transition_summary.jsonl.gz",
        "data/hbn/frozen_geometry.json",
    }
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative.startswith("data/") and relative not in allowed_data:
            participant_level_candidates += 1
        text = decoded_text(path)
        if text is None:
            continue
        if relative in TOP_LEVEL_FILES or relative.startswith(PUBLIC_SURFACE_PREFIXES):
            public_hits += sum(token in text for token in PUBLIC_FORBIDDEN)
        identity_hits += sum(token in text for token in IDENTITY_FORBIDDEN)
    if public_hits:
        raise RuntimeError(f"public-surface forbidden-token count: {public_hits}")
    if identity_hits:
        raise RuntimeError(f"identity-token count: {identity_hits}")
    if participant_level_candidates:
        raise RuntimeError(
            f"unexpected data files in archive inventory: {participant_level_candidates}"
        )
    return {
        "public_forbidden_token_count": public_hits,
        "identity_token_count": identity_hits,
        "unexpected_data_file_count": participant_level_candidates,
    }


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def archive_manifest(files: list[Path]) -> dict:
    return {
        "schema_version": "1.0.0",
        "classification": "anonymous reproducibility artifact",
        "source_commit": git_value("rev-parse", "HEAD"),
        "file_count": len(files),
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            }
            for path in files
        },
        "private_inputs_included": False,
        "participant_level_rows_included": False,
        "git_metadata_included": False,
    }


def zip_info(name: str) -> ZipInfo:
    value = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    value.compress_type = ZIP_DEFLATED
    value.external_attr = 0o100644 << 16
    value.create_system = 3
    return value


def write_archive(output: Path, files: list[Path], manifest: dict) -> str:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "certified-fdr-reproducibility"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(zip_info(f"{prefix}/{relative}"), path.read_bytes())
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        archive.writestr(zip_info(f"{prefix}/ARTIFACT_MANIFEST.json"), payload)
    return digest_file(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    verify_environment_layer()
    files = collect_files()
    audit = audit_files(files)
    if args.require_clean and git_value("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("artifact build requires a clean tracked worktree")
    print("ANONYMOUS_ARTIFACT_AUDIT_GATE=PASS")
    print(f"ARCHIVE_SOURCE_FILE_COUNT={len(files)}")
    print(f"PUBLIC_FORBIDDEN_TOKEN_COUNT={audit['public_forbidden_token_count']}")
    print(f"IDENTITY_TOKEN_COUNT={audit['identity_token_count']}")
    print(f"UNEXPECTED_DATA_FILE_COUNT={audit['unexpected_data_file_count']}")
    print("PRIVATE_INPUT_FILES_INCLUDED=FALSE")
    if args.audit_only:
        print("ARCHIVE_WRITTEN=FALSE")
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --audit-only is used")
    manifest = archive_manifest(files)
    archive_hash = write_archive(args.output, files, manifest)
    print("ANONYMOUS_ARTIFACT_BUILD_GATE=PASS")
    print(f"ARCHIVE_SHA256={archive_hash}")
    print(f"ARCHIVE_PATH={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
