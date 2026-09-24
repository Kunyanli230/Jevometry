"""Capture cache keyed by full request fingerprint.

The cache is separate from run directories.  Live requests bypass it unless the
caller opts in, and repeated observations always bypass it so that a repeat is
never silently served from an earlier attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

from jevometry.schemas.trace import EvaluationTrace


class CaptureCache:
    """Filesystem cache of deterministic provider responses."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _entry(self, fingerprint: str, repeat: int) -> Path:
        digest = fingerprint.replace("sha256:", "")
        return self.root / digest[:2] / f"{digest}-r{repeat}.json"

    def get(self, fingerprint: str, *, repeat: int = 0) -> EvaluationTrace | None:
        entry = self._entry(fingerprint, repeat)
        if not entry.exists():
            return None
        return EvaluationTrace.model_validate(json.loads(entry.read_text(encoding="utf-8")))

    def put(self, trace: EvaluationTrace, *, repeat: int | None = None) -> Path:
        resolved_repeat = trace.repeat if repeat is None else repeat
        entry = self._entry(trace.request_fingerprint, resolved_repeat)
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = entry.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(entry)
        return entry

    def contains(self, fingerprint: str, *, repeat: int = 0) -> bool:
        return self._entry(fingerprint, repeat).exists()

    def invalidate(self, fingerprint: str, *, repeat: int = 0) -> None:
        entry = self._entry(fingerprint, repeat)
        if entry.exists():
            entry.unlink()
