"""Canonical hashing for artifacts and provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def hash_object(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_checksums(directory: Path, relative_paths: Mapping[str, str]) -> dict[str, str]:
    """Write ``checksums.json`` mapping relative path -> sha256."""
    payload = {path: hash_file(Path(directory) / path) for path in sorted(relative_paths)}
    target = Path(directory) / "checksums.json"
    _atomic_write(target, canonical_json(payload))
    return payload


def verify_checksums(directory: Path) -> list[str]:
    """Return the list of files whose recorded hash no longer matches."""
    checksum_path = Path(directory) / "checksums.json"
    if not checksum_path.exists():
        return []
    recorded = json.loads(checksum_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for relative, expected in recorded.items():
        candidate = Path(directory) / relative
        if not candidate.exists() or hash_file(candidate) != expected:
            mismatches.append(relative)
    return mismatches


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
