"""Request/response capture without monkey-patching the SDK.

A recorder is attached explicitly to a provider.  Capturing only records what
was sent and returned; it never replays business tools, never patches the SDK
globally and never rewrites the analysed inputs.  Any redaction declared by the
user is applied *before* hashing, so the redacted payload is the actual
analysis input.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Redactor = Callable[[Any], Any]


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass
class CaptureRecord:
    """One recorded provider exchange."""

    capture_id: str
    semantic_request_hash: str
    request: Any
    response: Any | None
    request_fingerprint: str | None = None
    response_fingerprint: str | None = None
    request_id: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    duration_s: float | None = None
    attempt: int = 1
    repeat: int = 0
    timestamp_utc: str = ""
    redacted: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "semantic_request_hash": self.semantic_request_hash,
            "request_fingerprint": self.request_fingerprint,
            "response_fingerprint": self.response_fingerprint,
            "request_id": self.request_id,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "duration_s": self.duration_s,
            "attempt": self.attempt,
            "repeat": self.repeat,
            "timestamp_utc": self.timestamp_utc,
            "redacted": self.redacted,
        }


class CaptureRecorder:
    """Collect capture records for later replay and integrity checks."""

    def __init__(
        self,
        *,
        redact: Redactor | None = None,
        directory: Path | None = None,
    ) -> None:
        self._records: list[CaptureRecord] = []
        self._redact = redact
        self._directory = Path(directory) if directory is not None else None
        self._counter = 0
        self._attempts = 0
        self._transport_attempts = 0
        self._has_transport = False

    @property
    def records(self) -> list[CaptureRecord]:
        return list(self._records)

    @property
    def attempts(self) -> int:
        """Number of recorded provider exchanges, including SDK retries."""
        return self._attempts

    @property
    def transport_attempts(self) -> int:
        """HTTP exchanges seen by an attached recording transport."""
        return self._transport_attempts

    @property
    def has_transport(self) -> bool:
        """Whether a recording transport is attached to this recorder."""
        return self._has_transport

    def attach_transport(self) -> None:
        self._has_transport = True

    def _apply_redaction(self, payload: Any) -> tuple[Any, bool]:
        if self._redact is None:
            return payload, False
        return self._redact(payload), True

    def record(
        self,
        *,
        request: Any,
        response: Any | None,
        request_id: str | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        duration_s: float | None = None,
        repeat: int = 0,
        semantic_request_hash: str | None = None,
    ) -> CaptureRecord:
        self._counter += 1
        self._attempts += 1
        redacted_request, request_redacted = self._apply_redaction(request)
        redacted_response, response_redacted = self._apply_redaction(response)
        request_fingerprint = _hash(redacted_request)
        response_fingerprint = (
            None if redacted_response is None else _hash(redacted_response)
        )
        record = CaptureRecord(
            capture_id=f"cap_{self._counter:06d}",
            semantic_request_hash=semantic_request_hash or request_fingerprint,
            request=redacted_request,
            response=redacted_response,
            request_fingerprint=request_fingerprint,
            response_fingerprint=response_fingerprint,
            request_id=request_id,
            status_code=status_code,
            error_type=error_type,
            error_message=error_message,
            duration_s=duration_s,
            attempt=self._attempts,
            repeat=repeat,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            redacted=request_redacted or response_redacted,
        )
        self._records.append(record)
        return record

    def save(self, directory: Path | None = None) -> list[Path]:
        target = Path(directory) if directory is not None else self._directory
        if target is None:
            raise ValueError("no capture directory configured")
        requests_dir = target / "requests"
        responses_dir = target / "responses"
        requests_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for record in self._records:
            request_path = requests_dir / f"{record.capture_id}.json"
            request_path.write_text(_canonical(record.request), encoding="utf-8")
            written.append(request_path)
            if record.response is not None:
                response_path = responses_dir / f"{record.capture_id}.json"
                response_path.write_text(_canonical(record.response), encoding="utf-8")
                written.append(response_path)
        return written


@contextlib.contextmanager
def capture(recorder: CaptureRecorder) -> Iterator[CaptureRecorder]:
    """Context manager that simply yields the recorder.

    Capture is opt-in and side-effect free: entering the context does not patch
    any SDK, install global hooks or replay tools.
    """
    yield recorder


@dataclass
class RecordingTransport:
    """An ``httpx``-style transport that records exchanges and delegates.

    The wrapped transport is duck-typed, so this module imports no HTTP library
    at import time and remains available in offline installations.
    """

    inner: Any
    recorder: CaptureRecorder
    repeat: int = 0
    _attempts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.recorder.attach_transport()

    def handle_request(self, request: Any) -> Any:
        self._attempts += 1
        self.recorder._transport_attempts += 1  # noqa: SLF001 - shared counter
        body: Any
        try:
            body = json.loads(request.content.decode("utf-8"))
        except Exception:  # noqa: BLE001 - non-JSON bodies are hashed as text
            body = request.content.decode("utf-8", errors="replace")
        payload = {
            "method": request.method,
            "url": str(request.url),
            "headers": _safe_headers(request.headers),
            "body": body,
        }
        started = time.perf_counter()
        try:
            response = self.inner.handle_request(request)
        except Exception as error:  # noqa: BLE001 - recorded then re-raised
            self.recorder.record(
                request=payload,
                response=None,
                error_type=type(error).__name__,
                error_message=str(error),
                duration_s=time.perf_counter() - started,
                repeat=self.repeat,
            )
            raise
        try:
            response_body: Any = json.loads(response.content.decode("utf-8"))
        except Exception:  # noqa: BLE001
            response_body = response.content.decode("utf-8", errors="replace")
        self.recorder.record(
            request=payload,
            response={"status_code": response.status_code, "body": response_body},
            request_id=response.headers.get("x-typesafe-request-id"),
            status_code=response.status_code,
            duration_s=time.perf_counter() - started,
            repeat=self.repeat,
        )
        return response


_SENSITIVE_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "api-key"}


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: ("<redacted>" if key.lower() in _SENSITIVE_HEADERS else value)
        for key, value in headers.items()
    }
