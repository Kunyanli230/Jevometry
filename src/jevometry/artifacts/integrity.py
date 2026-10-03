"""Canonical hashing for artifacts and provenance."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    payload = {path: hash_file(_artifact_path(directory, path)) for path in sorted(relative_paths)}
    target = Path(directory) / "checksums.json"
    _atomic_write(target, canonical_json(payload))
    return payload


def verify_checksums(directory: Path) -> list[str]:
    """Return mismatched files; reject absent, empty or malformed checksum indexes.

    A missing checksum index cannot establish integrity. Paths are checked before
    opening files, including symlinks, so an index cannot read outside its run.
    """
    recorded = _read_checksums(directory)
    return _mismatches(directory, recorded)


def _artifact_path(directory: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or path.as_posix() != relative
        or relative == "checksums.json"
    ):
        raise ValueError(f"invalid artifact path: {relative!r}")
    candidate = Path(directory) / path
    try:
        contained = candidate.resolve().is_relative_to(Path(directory).resolve())
    except RuntimeError as error:
        raise ValueError(f"artifact path cannot be resolved: {relative!r}") from error
    if not contained:
        raise ValueError(f"artifact path escapes run directory: {relative!r}")
    return candidate


def _read_checksums(directory: Path) -> dict[str, str]:
    checksum_path = Path(directory) / "checksums.json"
    if checksum_path.is_symlink():
        raise ValueError("checksums.json must be a regular file, not a symlink")
    if not checksum_path.exists():
        raise ValueError("missing checksums.json; integrity is unavailable")
    recorded = json.loads(checksum_path.read_text(encoding="utf-8"))
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("checksums.json must contain a nonempty path-to-SHA256 object")
    for relative, expected in recorded.items():
        _artifact_path(directory, relative)
        if not isinstance(expected, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
            raise ValueError(f"invalid SHA256 checksum for {relative!r}")
    return recorded


def _mismatches(directory: Path, recorded: Mapping[str, str]) -> list[str]:
    mismatches: list[str] = []
    for relative, expected in recorded.items():
        candidate = _artifact_path(directory, relative)
        if not candidate.is_file() or hash_file(candidate) != expected:
            mismatches.append(relative)
    return mismatches


def run_artifact_paths(directory: Path) -> list[str]:
    """List managed artifacts, including retained analysis revisions."""
    from jevometry.artifacts.store import RUN_FILES

    root = Path(directory)
    paths = {name for name in RUN_FILES if name != "checksums.json" and (root / name).exists()}
    for revision in (root / "analysis").glob("rev-*"):
        if revision.is_dir():
            paths.update(
                path.relative_to(root).as_posix()
                for path in revision.rglob("*")
                if path.is_file()
            )
    return sorted(paths)


@dataclass
class IntegrityReport:
    """Read-only run audit. Checksums detect changes, not origin authenticity."""

    valid: bool = False
    checked_files: int = 0
    issues: list[str] = field(default_factory=list)


def verify_run(directory: Path) -> IntegrityReport:
    """Audit checksum coverage and the capture manifest without modifying artifacts."""
    from jevometry.artifacts.store import RunStore

    report = IntegrityReport()
    try:
        recorded = _read_checksums(directory)
        report.checked_files = len(recorded)
        report.issues.extend(f"checksum mismatch: {name}" for name in _mismatches(directory, recorded))
        required = {"manifest.json", "experiment.resolved.yaml", "parameters.json", "traces.jsonl"}
        required.update(run_artifact_paths(directory))
        report.issues.extend(f"missing checksum: {name}" for name in sorted(required - recorded.keys()))
        store = RunStore.load(directory)
        if store.manifest is not None:
            for revision in store.manifest.analysis_revisions:
                relative = f"analysis/rev-{revision:04d}/metrics.json"
                if relative not in recorded:
                    report.issues.append(f"missing checksum for declared revision: {relative}")
            latest = store.manifest.latest_analysis_revision
            if latest is not None:
                relative = f"analysis/rev-{latest:04d}/metrics.json"
                if latest not in store.manifest.analysis_revisions:
                    report.issues.append("latest analysis revision is not in analysis_revisions")
                if relative in recorded and recorded.get("metrics.json") != recorded[relative]:
                    report.issues.append("metrics.json differs from the latest analysis revision")
    except (OSError, ValueError) as error:
        report.issues.append(str(error))
    report.valid = not report.issues
    return report


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
