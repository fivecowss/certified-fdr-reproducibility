from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "scripts", "configs", "docs")
FORBIDDEN = (
    "/Users" + "/",
    "AI" + "STATS",
    "Phase" + " " + "8",
    "phase" + "8",
)


def test_public_code_has_no_local_paths_or_internal_release_names() -> None:
    failures = []
    files = [ROOT / "README.md", ROOT / "pyproject.toml"]
    for directory in SCANNED:
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN:
            if token in text:
                failures.append(f"{path.relative_to(ROOT)}: {token}")
    assert failures == []
