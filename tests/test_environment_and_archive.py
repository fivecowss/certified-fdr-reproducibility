from __future__ import annotations

import json
from pathlib import Path

from scripts.build_review_archive import (
    ROOT,
    audit_files,
    collect_files,
    contains_skipped_directory,
)
from scripts.validate_environment import ENVIRONMENT_PATH, LOCK_PATH, lock_entries, normalized


def test_complete_lock_matches_environment_specification() -> None:
    specification = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    declared = {
        normalized(name): package_version
        for name, package_version in specification["packages"].items()
    }
    assert lock_entries(LOCK_PATH) == declared
    assert len(declared) == 19
    assert specification["python"] == "3.11.15"


def test_container_and_ci_pin_the_reference_python() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python:3.11.15-slim-bookworm" in dockerfile
    assert ".github" not in dockerignore
    assert 'python-version: "3.11.15"' in workflow
    assert "scripts/ci_check.py" in dockerfile
    assert "scripts/ci_check.py" in workflow


def test_archive_inventory_is_allowlisted_and_identity_scrubbed() -> None:
    files = collect_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    assert not any(".git" in Path(name).parts for name in relative)
    assert "data/hbn/frozen_geometry.json" in relative
    assert "data/frozen/hbn_observed_application.jsonl.gz" in relative
    result = audit_files(files)
    assert result == {
        "public_forbidden_token_count": 0,
        "identity_token_count": 0,
        "unexpected_data_file_count": 0,
    }


def test_archive_inventory_excludes_install_and_test_metadata() -> None:
    assert contains_skipped_directory(("src", "package.egg-info", "PKG-INFO"))
    assert contains_skipped_directory(("src", "package.dist-info", "METADATA"))
    assert contains_skipped_directory(("tests", "__pycache__", "test_module.pyc"))
    assert not contains_skipped_directory(("src", "certified_fdr", "metrics.py"))
