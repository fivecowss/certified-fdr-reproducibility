"""Deterministic, atomic gzip-JSONL replication ledgers."""

from __future__ import annotations

from gzip import GzipFile
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable


Record = dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def finish_record(record: Record) -> Record:
    result = dict(record)
    result.pop("record_id", None)
    result["record_id"] = sha256(canonical_json(result).encode("utf-8")).hexdigest()
    return result


def canonical_jsonl(records: Iterable[Record]) -> bytes:
    rows = [canonical_json(record) for record in records]
    if not rows:
        raise ValueError("a ledger chunk must contain at least one record.")
    return ("\n".join(rows) + "\n").encode("utf-8")


def deterministic_gzip(payload: bytes, compresslevel: int = 6) -> bytes:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be nonempty bytes.")
    if not 0 <= int(compresslevel) <= 9:
        raise ValueError("compresslevel must lie between zero and nine.")
    buffer = BytesIO()
    with GzipFile(
        filename="",
        mode="wb",
        compresslevel=int(compresslevel),
        fileobj=buffer,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return buffer.getvalue()


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def write_atomic_gzip_jsonl(
    path: Path,
    records: Iterable[Record],
    compresslevel: int = 6,
) -> dict[str, int | str]:
    """Write a new deterministic chunk and refuse to overwrite any file."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite ledger chunk: {destination}")
    raw = canonical_jsonl(records)
    compressed = deterministic_gzip(raw, compresslevel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(compressed)
    temporary.replace(destination)
    return {
        "compressed_bytes": len(compressed),
        "compressed_sha256": sha256_bytes(compressed),
        "uncompressed_bytes": len(raw),
        "uncompressed_sha256": sha256_bytes(raw),
    }


def read_gzip_jsonl(path: Path) -> list[Record]:
    with GzipFile(filename=str(path), mode="rb") as stream:
        payload = stream.read()
    rows = payload.decode("utf-8").splitlines()
    records = [json.loads(row) for row in rows if row]
    if not records:
        raise ValueError("ledger chunk is empty.")
    return records
