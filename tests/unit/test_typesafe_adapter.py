"""TypeSafe provider against an httpx mock transport (no network)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import httpx2
import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, RetryPolicy, ScoreAnswer, TypeSafeClient

from jevometry.adapters.base import ExperimentPoint
from jevometry.adapters.capture import CaptureRecorder, RecordingTransport
from jevometry.adapters.typesafe import (
    ADAPTER_VERSION,
    TypeSafeAdapter,
    TypeSafeProviderError,
    convert_answer,
    typesafe_choice_question,
    typesafe_noul_question,
    typesafe_score_question,
)
from jevometry.schemas.common import MetricStatus, Primitive, RoutingSemantics
from jevometry.schemas.experiment import Budget
from jevometry.schemas.parameters import ParameterSpec
from jevometry.schemas.questions import OutcomeSpec, QuestionSpec


def make_client(handler: Any, *, max_retries: int = 1) -> Any:
    transport = httpx2.MockTransport(handler)
    return TypeSafeClient(
        api_key="test-key",
        transport=transport,
        retry=RetryPolicy(max_retries=max_retries, backoff_initial=0.0, backoff_max=0.0),
    )


def response_payload(**answers: Any) -> dict[str, Any]:
    return {
        "model": "resolved-model",
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "answers": answers,
    }


def point(theta: float = 0.0) -> ExperimentPoint:
    from jevometry.schemas.experiment import CaseSpec

    return ExperimentPoint(
        case=CaseSpec(id="c", state="state"),
        theta={"theta": theta},
        point_id="p",
    )


def test_noul_answer_conversion() -> None:
    node = typesafe_noul_question("applicable", instructions="Is it applicable?")
    conversion = convert_answer(node.spec, NoulAnswer(noul=0.25))
    assert conversion.status is MetricStatus.OK
    assert conversion.support == ["false", "true"]
    assert conversion.probabilities == pytest.approx([0.75, 0.25])


def test_choice_answer_conversion_and_selection() -> None:
    node = typesafe_choice_question("kind", {"keep": "Keep", "review": "Review"})
    conversion = convert_answer(
        node.spec,
        ChoiceAnswer(choice="keep", confidence=0.9, probabilities={"keep": 0.9, "review": 0.1}),
    )
    assert conversion.selected == "keep"
    assert conversion.confidence == 0.9
    assert conversion.probabilities == pytest.approx([0.9, 0.1])


def test_choice_answer_missing_probability_is_not_zero_filled() -> None:
    node = typesafe_choice_question("kind", {"keep": "Keep", "review": "Review"})
    conversion = convert_answer(
        node.spec, ChoiceAnswer(choice="keep", confidence=0.9, probabilities={"keep": 1.0})
    )
    assert conversion.status is MetricStatus.FAILED
    assert conversion.reason_code == "support_mismatch"


def test_score_answer_uses_integer_levels() -> None:
    node = typesafe_score_question("risk", ["Low", "Medium", "High"])
    conversion = convert_answer(
        node.spec,
        ScoreAnswer(
            score=1.2,
            confidence=0.8,
            legend={0: "Low", 1: "Medium", 2: "High"},
            probabilities={0: 0.2, 1: 0.6, 2: 0.2},
        ),
    )
    assert conversion.support == ["0", "1", "2"]
    assert conversion.numeric_encoding == {"0": 0.0, "1": 1.0, "2": 2.0}
    assert conversion.expected_value == pytest.approx(1.0)


def test_primitive_mismatch_is_refused() -> None:
    node = typesafe_noul_question("applicable")
    conversion = convert_answer(
        node.spec,
        ChoiceAnswer(choice="a", confidence=0.5, probabilities={"a": 0.5, "b": 0.5}),
    )
    assert conversion.reason_code == "primitive_mismatch"


def test_adapter_batches_questions_and_records_usage() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        return httpx2.Response(
            200,
            json=response_payload(
                applicable={"type": "noul", "noul": 0.7},
                kind={
                    "type": "choice",
                    "choice": "keep",
                    "confidence": 0.8,
                    "probabilities": {"keep": 0.8, "review": 0.2},
                },
            ),
            headers={"x-typesafe-request-id": "req_123"},
        )

    recorder = CaptureRecorder()
    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={
            "applicable": typesafe_noul_question("applicable"),
            "kind": typesafe_choice_question("kind", {"keep": "Keep", "review": "Review"}),
        },
        client=make_client(handler),
        recorder=recorder,
        routing_semantics=RoutingSemantics.DETERMINISTIC_POLICY,
    )
    traces = adapter.evaluate_point(point())
    assert len(traces) == 2
    assert len(calls) == 1
    assert adapter.requests_per_point() == 1
    assert all(trace.status.ok for trace in traces)
    assert traces[0].status.request_id == "req_123"
    assert traces[0].status.model_resolved == "resolved-model"
    assert traces[0].usage is not None and traces[0].usage.input_tokens == 12
    assert adapter.tracker.known_input_tokens == 12
    assert len(recorder.records) == 1
    assert "test-key" not in json.dumps(recorder.records[0].to_json())


@pytest.mark.parametrize("stencil_role", ["center", "theta:1:+1", "theta:1:-1"])
@pytest.mark.parametrize("status_code", [200, 401])
def test_adapter_preserves_stencil_role_and_request_identity(
    stencil_role: str, status_code: int
) -> None:
    # These digests were independently calculated from the declared Noul
    # question and identity-rendered {"theta": 0.0}, without adapter helpers.
    canonical_request = {
        "provider": "typesafe",
        "model": "test-model",
        "adapter_version": ADAPTER_VERSION,
        "node_id": "applicable",
        "question_hash": "sha256:a96e6bafcf4bb225ccf020ba6865c042827d2373e0569ece22c50e5859dcedcd",
        "rendered": "sha256:0b46d5760416c09c35880b2bb7ec209ef335db40b9985ad54613148ee83a3fdc",
        "theta": {"theta": 0.0},
        "history": {},
        "stencil_role": stencil_role,
    }
    expected_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(canonical_request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if status_code == 401:
            return httpx2.Response(401, json={"error": "invalid key"})
        return httpx2.Response(
            200, json=response_payload(applicable={"type": "noul", "noul": 0.5})
        )

    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(handler),
    )
    experiment_point = replace(point(), stencil_role=stencil_role)
    first = adapter.evaluate_point(experiment_point)[0]
    repeated = adapter.evaluate_point(replace(experiment_point, repeat=1))[0]
    assert first.status.ok is (status_code == 200)
    assert first.stencil_role == repeated.stencil_role == stencil_role
    assert first.request_fingerprint == repeated.request_fingerprint == expected_fingerprint
    assert repeated.repeat == 1
    if status_code == 200:
        assert first.distribution is not None
        assert first.distribution.request_fingerprint == expected_fingerprint
        assert first.semantic_request_hash == expected_fingerprint
    else:
        assert first.status.reason_code == "auth_error"
        assert first.distribution is None


def test_auth_error_is_not_retried_and_structured() -> None:
    calls = {"count": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["count"] += 1
        return httpx2.Response(401, json={"error": "invalid key"})

    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(handler),
    )
    traces = adapter.evaluate_point(point())
    assert calls["count"] == 1
    assert traces[0].status.ok is False
    assert traces[0].status.reason_code == "auth_error"
    assert traces[0].status.retryable is False
    assert traces[0].status.incomplete is True


def test_rate_limit_is_retried_by_sdk_and_counted() -> None:
    calls = {"count": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx2.Response(429, json={"error": "slow down"})
        return httpx2.Response(
            200, json=response_payload(applicable={"type": "noul", "noul": 0.5})
        )

    recorder = CaptureRecorder()
    transport = RecordingTransport(inner=httpx2.MockTransport(handler), recorder=recorder)
    client = TypeSafeClient(
        api_key="test-key",
        transport=transport,
        retry=RetryPolicy(max_retries=2, backoff_initial=0.0, backoff_max=0.0),
    )
    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=client,
        recorder=recorder,
    )
    traces = adapter.evaluate_point(point())
    assert calls["count"] == 2
    assert traces[0].status.ok is True
    assert adapter.tracker.attempts == 2


def test_timeout_is_structured() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out")

    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(handler, max_retries=0),
    )
    traces = adapter.evaluate_point(point())
    assert traces[0].status.ok is False
    assert traces[0].status.reason_code == "timeout"
    assert traces[0].status.retryable is True


def test_missing_usage_is_counted_as_unknown() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "resolved-model",
                "usage": {},
                "answers": {"applicable": {"type": "noul", "noul": 0.5}},
            },
        )

    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(handler),
    )
    adapter.evaluate_point(point())
    assert adapter.tracker.unknown_usage_attempts == 1
    assert adapter.tracker.known_input_tokens == 0


def test_budget_exhaustion_refuses_without_request() -> None:
    calls = {"count": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["count"] += 1
        return httpx2.Response(
            200, json=response_payload(applicable={"type": "noul", "noul": 0.5})
        )

    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(handler),
        budget=Budget(max_attempts=1, repeats=1),
    )
    adapter.tracker.note_attempt()
    traces = adapter.evaluate_point(point())
    assert calls["count"] == 0
    assert traces[0].status.reason_code == "attempt_budget_exhausted"


def test_missing_answer_is_structured() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=response_payload())

    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(handler),
    )
    traces = adapter.evaluate_point(point())
    assert traces[0].status.reason_code == "missing_answer"


def test_from_env_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeProviderError) as error:
        TypeSafeAdapter.from_env(
            model="test-model", nodes={"applicable": typesafe_noul_question("applicable")}
        )
    assert error.value.reason_code == "missing_credentials"


def test_from_env_requires_concrete_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    with pytest.raises(TypeSafeProviderError) as error:
        TypeSafeAdapter.from_env(nodes={"applicable": typesafe_noul_question("applicable")})
    assert error.value.reason_code == "missing_model"


def test_declared_question_specs_match_sdk_answers() -> None:
    spec = QuestionSpec(
        id="risk",
        primitive=Primitive.SCORE,
        outcomes=[
            OutcomeSpec(id="0", description="Low", numeric=0.0),
            OutcomeSpec(id="1", description="High", numeric=1.0),
        ],
    )
    from jevometry.adapters.typesafe import question_spec_to_typesafe

    node = question_spec_to_typesafe(spec, node_id="risk")
    conversion = convert_answer(
        node.spec,
        ScoreAnswer(score=0.5, confidence=0.5, legend={0: "Low", 1: "High"}, probabilities={0: 0.5, 1: 0.5}),
    )
    assert conversion.status is MetricStatus.OK
    assert node.spec.semantic_hash() == spec.semantic_hash()
    assert ADAPTER_VERSION.startswith("typesafe")


def test_adapter_requires_concrete_model() -> None:
    with pytest.raises(ValueError):
        TypeSafeAdapter(
            model="",
            nodes={"applicable": typesafe_noul_question("applicable")},
        )


def test_parameter_declaration_is_ignored_by_provider() -> None:
    adapter = TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=make_client(
            lambda request: httpx2.Response(
                200, json=response_payload(applicable={"type": "noul", "noul": 0.5})
            )
        ),
    )
    assert ParameterSpec(
        name="theta", role="diagnostic", unit="u", bounds=(0, 1), step=0.1
    ).name == "theta"
    assert adapter.describe().model == "test-model"
