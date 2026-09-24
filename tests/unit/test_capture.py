"""Capture records requests without side effects and never leaks secrets."""

from __future__ import annotations

import json

import httpx2

from jevometry.adapters.capture import CaptureRecorder, RecordingTransport, capture


def test_capture_context_is_side_effect_free() -> None:
    recorder = CaptureRecorder()
    executed: list[str] = []
    with capture(recorder) as session:
        executed.append("body")
        assert session is recorder
    assert recorder.records == []
    assert executed == ["body"]


def test_recording_transport_redacts_authorization_header() -> None:
    recorder = CaptureRecorder()

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"ok": True})

    transport = RecordingTransport(inner=httpx2.MockTransport(handler), recorder=recorder)
    request = httpx2.Request(
        "POST",
        "https://api.example.test/v1/system-one",
        json={"state": "s"},
        headers={"Authorization": "Bearer super-secret"},
    )
    transport.handle_request(request)
    record = recorder.records[0]
    assert record.request["headers"]["authorization"] == "<redacted>"
    assert "super-secret" not in json.dumps(record.to_json())
    assert record.status_code == 200
    assert record.response_fingerprint is not None


def test_user_redaction_is_applied_before_hashing() -> None:
    recorder = CaptureRecorder(redact=lambda payload: {"redacted": True})
    record = recorder.record(request={"state": "sensitive"}, response={"answer": 1})
    assert record.redacted is True
    assert record.request == {"redacted": True}
    assert record.response == {"redacted": True}


def test_records_receive_distinct_capture_ids_per_attempt() -> None:
    recorder = CaptureRecorder()
    first = recorder.record(request={"a": 1}, response=None)
    second = recorder.record(request={"a": 1}, response=None)
    assert first.capture_id != second.capture_id
    assert first.semantic_request_hash == second.semantic_request_hash
    assert second.attempt == 2


def test_capture_save_writes_requests_and_responses(tmp_path) -> None:
    recorder = CaptureRecorder()
    recorder.record(request={"a": 1}, response={"b": 2})
    paths = recorder.save(tmp_path)
    assert len(paths) == 2
    assert (tmp_path / "requests" / "cap_000001.json").exists()
    assert (tmp_path / "responses" / "cap_000001.json").exists()
