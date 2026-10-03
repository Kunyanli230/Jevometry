"""Official TypeSafe SDK provider.

The SDK is an optional (``live`` extra) dependency and is imported lazily, so
offline installations can still import this module.  Credentials are read from
the process environment; no key value is ever logged, recorded or hashed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevometry.adapters.base import ExperimentPoint, request_fingerprint
from jevometry.adapters.capture import CaptureRecorder
from jevometry.experiments.renderer import CallableRenderer, ParameterRenderer
from jevometry.geometry.simplex import (
    ProbabilityValidationError,
    noul_vector,
    validate_probabilities,
)
from jevometry.schemas.common import MetricStatus, Primitive, RoutingSemantics
from jevometry.schemas.distribution import DistributionSource
from jevometry.schemas.experiment import Budget
from jevometry.schemas.questions import OutcomeSpec, QuestionSpec
from jevometry.schemas.system import (
    Capabilities,
    CompositionAssumptions,
    CompositionMode,
    NodeSpec,
    SystemSpec,
)
from jevometry.schemas.trace import EvaluationTrace, ProviderStatus, UsageRecord

ADAPTER_VERSION = "typesafe-0.1.0"
DEFAULT_MODEL_ENV = "TYPESAFE_MODEL"


class TypeSafeProviderError(RuntimeError):
    """Provider-level failure with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass
class BudgetTracker:
    """Track attempts and known token usage against hard caps."""

    budget: Budget
    attempts: int = 0
    known_input_tokens: int = 0
    unknown_usage_attempts: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def exhausted_reason(self) -> str | None:
        if self.attempts >= self.budget.max_attempts:
            return "attempt_budget_exhausted"
        if self.known_input_tokens >= self.budget.max_input_tokens:
            return "token_budget_exhausted"
        if time.monotonic() - self.started_at > self.budget.experiment_timeout_s:
            return "experiment_timeout_exhausted"
        return None

    def note_attempt(self, count: int = 1) -> None:
        self.attempts += count

    def note_usage(self, input_tokens: int | None) -> None:
        if input_tokens is None:
            self.unknown_usage_attempts += 1
        else:
            self.known_input_tokens += input_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "known_input_tokens": self.known_input_tokens,
            "unknown_usage_attempts": self.unknown_usage_attempts,
            "max_attempts": self.budget.max_attempts,
            "max_input_tokens": self.budget.max_input_tokens,
        }


@dataclass
class TypeSafeQuestion:
    """A node question with its Jevometry declaration and SDK object."""

    node_id: str
    spec: QuestionSpec
    question: Any
    description: str | None = None


def typesafe_choice_question(
    node_id: str,
    criteria: Mapping[str, str | None],
    *,
    instructions: str | None = None,
    rubric: str | None = None,
) -> TypeSafeQuestion:
    """Declare a Choice question and build the SDK object."""
    from typesafe_sdk import Choice

    spec = QuestionSpec(
        id=node_id,
        primitive=Primitive.CHOICE,
        outcomes=[OutcomeSpec(id=key, description=value) for key, value in criteria.items()],
        instructions=instructions,
        rubric=rubric,
    )
    return TypeSafeQuestion(
        node_id=node_id,
        spec=spec,
        question=Choice(instructions=instructions, criteria=dict(criteria)),
    )


def typesafe_score_question(
    node_id: str,
    levels: Sequence[str],
    *,
    instructions: str | None = None,
    rubric: str | None = None,
) -> TypeSafeQuestion:
    """Declare a Score question; integer levels are the API's numeric encoding."""
    from typesafe_sdk import Score

    spec = QuestionSpec(
        id=node_id,
        primitive=Primitive.SCORE,
        outcomes=[
            OutcomeSpec(id=str(index), description=level, numeric=float(index))
            for index, level in enumerate(levels)
        ],
        legend={str(index): level for index, level in enumerate(levels)},
        instructions=instructions,
        rubric=rubric,
    )
    return TypeSafeQuestion(
        node_id=node_id,
        spec=spec,
        question=Score(instructions=instructions, criteria=list(levels)),
    )


def typesafe_noul_question(
    node_id: str,
    *,
    instructions: str | None = None,
    rubric: str | None = None,
) -> TypeSafeQuestion:
    """Declare a Noul question and build the SDK object."""
    from typesafe_sdk import Noul

    spec = QuestionSpec(
        id=node_id,
        primitive=Primitive.NOUL,
        outcomes=[
            OutcomeSpec(id="false", description="no"),
            OutcomeSpec(id="true", description="yes"),
        ],
        instructions=instructions,
        rubric=rubric,
    )
    return TypeSafeQuestion(
        node_id=node_id, spec=spec, question=Noul(instructions=instructions)
    )


@dataclass
class AnswerConversion:
    """A converted provider answer, or a structured reason it cannot be used."""

    support: list[str]
    probabilities: list[float]
    selected: str | None = None
    confidence: float | None = None
    legend: dict[str, str] | None = None
    numeric_encoding: dict[str, float] | None = None
    expected_value: float | None = None
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    message: str | None = None


def convert_answer(spec: QuestionSpec, answer: Any) -> AnswerConversion:
    """Convert an SDK answer into a validated support-aligned vector.

    API failures and schema mismatches are never mapped onto categories.
    """
    answer_type = getattr(answer, "type", None)
    if answer_type != spec.primitive.value:
        return AnswerConversion(
            support=list(spec.support),
            probabilities=[],
            status=MetricStatus.FAILED,
            reason_code="primitive_mismatch",
            message=f"expected {spec.primitive.value!r} answer, received {answer_type!r}",
        )
    if spec.primitive is Primitive.NOUL:
        try:
            vector = noul_vector(float(answer.noul))
        except (ProbabilityValidationError, ValueError) as error:
            return AnswerConversion(
                support=list(spec.support),
                probabilities=[],
                status=MetricStatus.FAILED,
                reason_code=getattr(error, "reason_code", "invalid_noul"),
                message=str(error),
            )
        return AnswerConversion(
            support=list(vector.support),
            probabilities=[float(value) for value in vector.working],
        )
    if spec.primitive is Primitive.CHOICE:
        probabilities = getattr(answer, "probabilities", None)
        if not isinstance(probabilities, dict) or not probabilities:
            return AnswerConversion(
                support=list(spec.support),
                probabilities=[],
                status=MetricStatus.INSUFFICIENT_DATA,
                reason_code="missing_probabilities",
                message="choice answer did not include a probability vector",
            )
        unknown = set(probabilities) - set(spec.support)
        missing = set(spec.support) - set(probabilities)
        if unknown or missing:
            return AnswerConversion(
                support=list(spec.support),
                probabilities=[],
                status=MetricStatus.FAILED,
                reason_code="support_mismatch",
                message=(
                    f"provider probabilities do not match the declared support: "
                    f"missing={sorted(missing)} unknown={sorted(unknown)}"
                ),
            )
        try:
            vector = validate_probabilities(
                spec.support, [float(probabilities[outcome]) for outcome in spec.support]
            )
        except ProbabilityValidationError as error:
            return AnswerConversion(
                support=list(spec.support),
                probabilities=[],
                status=MetricStatus.FAILED,
                reason_code=error.reason_code,
                message=str(error),
            )
        selected = str(answer.choice)
        if selected not in spec.support:
            return AnswerConversion(
                support=list(spec.support),
                probabilities=[],
                status=MetricStatus.FAILED,
                reason_code="selected_outside_support",
                message=f"selected label {selected!r} is not a declared outcome",
            )
        return AnswerConversion(
            support=list(vector.support),
            probabilities=[float(value) for value in vector.working],
            selected=selected,
            confidence=float(answer.confidence),
        )
    if spec.primitive is Primitive.SCORE:
        probabilities = getattr(answer, "probabilities", None)
        if not isinstance(probabilities, dict) or not probabilities:
            return AnswerConversion(
                support=list(spec.support),
                probabilities=[],
                status=MetricStatus.INSUFFICIENT_DATA,
                reason_code="missing_probabilities",
                message="score answer did not include a probability vector",
            )
        keys = sorted(int(key) for key in probabilities)
        support = [str(key) for key in keys]
        unknown = set(support) - set(spec.support)
        missing = set(spec.support) - set(support)
        if unknown or missing:
            return AnswerConversion(
                support=support,
                probabilities=[],
                status=MetricStatus.FAILED,
                reason_code="support_mismatch",
                message=(
                    "provider score levels do not match the declared support: "
                    f"missing={sorted(missing)} unknown={sorted(unknown)}"
                ),
            )
        try:
            vector = validate_probabilities(
                support, [float(probabilities[int(outcome)]) for outcome in support]
            )
        except ProbabilityValidationError as error:
            return AnswerConversion(
                support=support,
                probabilities=[],
                status=MetricStatus.FAILED,
                reason_code=error.reason_code,
                message=str(error),
            )
        legend = {str(key): str(value) for key, value in getattr(answer, "legend", {}).items()}
        numeric = {str(key): float(key) for key in keys}
        expected = float(sum(float(key) * float(probabilities[key]) for key in keys))
        return AnswerConversion(
            support=list(vector.support),
            probabilities=[float(value) for value in vector.working],
            legend=legend or None,
            numeric_encoding=numeric,
            expected_value=expected,
        )
    return AnswerConversion(
        support=list(spec.support),
        probabilities=[],
        status=MetricStatus.UNSUPPORTED,
        reason_code="unsupported_primitive",
    )


class TypeSafeAdapter:
    """Live provider backed by the official TypeSafe Python SDK."""

    is_live = True
    provider_name = "typesafe"

    def __init__(
        self,
        *,
        model: str,
        nodes: Mapping[str, TypeSafeQuestion],
        client: Any | None = None,
        api_key: str | None = None,
        budget: Budget | None = None,
        renderer: ParameterRenderer | None = None,
        recorder: CaptureRecorder | None = None,
        routing_semantics: RoutingSemantics = RoutingSemantics.NOT_APPLICABLE,
        batch_questions: bool = True,
        system_id: str = "typesafe-system",
        endpoint: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("TypeSafeAdapter requires at least one question")
        if not model:
            raise ValueError("a concrete model id is required; 'latest' is not accepted")
        self.model = model
        self.nodes = dict(nodes)
        self.budget = budget or Budget()
        self.renderer = renderer or CallableRenderer(
            lambda theta, case: dict(theta), version="typesafe-identity"
        )
        self.recorder = recorder
        self.routing_semantics = routing_semantics
        self.batch_questions = batch_questions
        self.system_id = system_id
        self.endpoint = endpoint
        self.extra_headers = dict(extra_headers or {})
        self.tracker = BudgetTracker(budget=self.budget)
        self._client = client
        self._api_key = api_key
        self._system = SystemSpec(
            id=system_id,
            nodes=[
                NodeSpec(id=node.node_id, question_id=node.spec.id)
                for node in self.nodes.values()
            ],
            routing_semantics=routing_semantics,
            composition_mode=CompositionMode.NODE_ONLY,
            capabilities=Capabilities(
                analytic_jacobian=False,
                joint_model=False,
                likelihood_model=False,
                action_distribution=routing_semantics is RoutingSemantics.SAMPLED_OUTCOME,
                replayable=True,
            ),
            composition_assumptions=CompositionAssumptions(
                shared_theta=False,
                notes=[
                    "reported distributions are not independent draws",
                    "no joint model is declared by the provider",
                ],
            ),
            model=model,
            adapter=ADAPTER_VERSION,
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        nodes: Mapping[str, TypeSafeQuestion] | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> TypeSafeAdapter:
        """Build an adapter from environment credentials."""
        if nodes is None:
            raise ValueError("nodes are required")
        resolved_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not resolved_key:
            raise TypeSafeProviderError(
                "missing_credentials",
                "TYPESAFE_API_KEY is not set; live requests are refused rather than faked",
            )
        resolved_model = model or os.environ.get(DEFAULT_MODEL_ENV)
        if not resolved_model:
            raise TypeSafeProviderError(
                "missing_model",
                "a concrete model id is required (argument or TYPESAFE_MODEL)",
            )
        budget = kwargs.pop("budget", None) or Budget()
        client = _build_client(
            api_key=resolved_key,
            model=resolved_model,
            budget=budget,
            endpoint=kwargs.pop("endpoint", None),
            extra_headers=kwargs.pop("extra_headers", None),
        )
        return cls(model=resolved_model, nodes=nodes, client=client, **kwargs)

    @property
    def client(self) -> Any:
        if self._client is None:
            raise TypeSafeProviderError("client_not_configured", "no SDK client is configured")
        return self._client

    def describe(self) -> SystemSpec:
        return self._system

    def questions(self) -> dict[str, QuestionSpec]:
        return {node_id: node.spec for node_id, node in self.nodes.items()}

    def requests_per_point(self) -> int:
        return 1 if self.batch_questions else len(self.nodes)

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]:
        rendered = self.renderer.render(point.theta, point.case)
        question_map = {node_id: node.question for node_id, node in self.nodes.items()}
        request_payload = {"state": rendered.state, "questions": list(question_map)}
        fingerprints = {
            node_id: request_fingerprint(
                provider="typesafe",
                model=self.model,
                adapter_version=ADAPTER_VERSION,
                node_id=node_id,
                question_hash=node.spec.semantic_hash(),
                rendered_fingerprint=rendered.fingerprint,
                theta=point.theta,
                history=point.history,
                stencil_role=point.stencil_role,
            )
            for node_id, node in self.nodes.items()
        }
        exhausted = self.tracker.exhausted_reason()
        if exhausted is not None:
            return [
                self._failure_trace(
                    node_id=node_id,
                    point=point,
                    rendered_fingerprint=rendered.fingerprint,
                    request_fingerprint_value=fingerprints[node_id],
                    reason_code=exhausted,
                    message="provider budget exhausted; no request was made",
                )
                for node_id in self.nodes
            ]
        started = time.perf_counter()
        attempts_before = (
            self.recorder.transport_attempts
            if self.recorder is not None and self.recorder.has_transport
            else None
        )
        try:
            response = self.client.system_one(
                state=rendered.state,
                questions=question_map,
                extra_headers=self.extra_headers or None,
            )
        except Exception as error:  # noqa: BLE001 - classified into structured statuses
            duration = time.perf_counter() - started
            self._note_http_attempts(attempts_before)
            reason, retryable = _classify_error(error)
            if self.recorder is not None:
                self.recorder.record(
                    request=request_payload,
                    response=None,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    duration_s=duration,
                    repeat=point.repeat,
                )
            return [
                self._failure_trace(
                    node_id=node_id,
                    point=point,
                    rendered_fingerprint=rendered.fingerprint,
                    request_fingerprint_value=fingerprints[node_id],
                    reason_code=reason,
                    message=f"{type(error).__name__}: {error}",
                    retryable=retryable,
                    duration_s=duration,
                )
                for node_id in self.nodes
            ]
        duration = time.perf_counter() - started
        self._note_http_attempts(attempts_before)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        self.tracker.note_usage(input_tokens)
        resolved_model = str(getattr(response, "model", self.model))
        request_id = _safe_request_id(response)
        answers = getattr(response, "answers", {})
        if self.recorder is not None:
            self.recorder.record(
                request=request_payload,
                response=_response_payload(response, resolved_model),
                request_id=request_id,
                status_code=200,
                duration_s=duration,
                repeat=point.repeat,
            )
        traces: list[EvaluationTrace] = []
        for node_id, node in self.nodes.items():
            answer = answers.get(node_id)
            if answer is None:
                traces.append(
                    self._failure_trace(
                        node_id=node_id,
                        point=point,
                        rendered_fingerprint=rendered.fingerprint,
                        request_fingerprint_value=fingerprints[node_id],
                        reason_code="missing_answer",
                        message="response did not include an answer for this question",
                        duration_s=duration,
                        model_resolved=resolved_model,
                        request_id=request_id,
                        usage=UsageRecord(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            usage_known=input_tokens is not None,
                        ),
                    )
                )
                continue
            conversion = convert_answer(node.spec, answer)
            record = None
            if conversion.status is MetricStatus.OK:
                simplex = validate_probabilities(
                    conversion.support, conversion.probabilities
                )
                record = simplex.as_record(
                    node_id=node_id,
                    question_id=node.spec.id,
                    primitive=node.spec.primitive,
                    source=DistributionSource.REPORTED,
                    semantic_hash=node.spec.semantic_hash(),
                    case_id=point.case.id,
                    point_id=point.point_id,
                    repeat=point.repeat,
                    theta=dict(point.theta),
                    selected=conversion.selected,
                    confidence=conversion.confidence,
                    legend=conversion.legend or node.spec.legend,
                    numeric_encoding=conversion.numeric_encoding
                    or node.spec.numeric_encoding(),
                    request_fingerprint=fingerprints[node_id],
                )
            traces.append(
                EvaluationTrace(
                    experiment_id=self.system_id,
                    case_id=point.case.id,
                    point_id=point.point_id,
                    repeat=point.repeat,
                    node_id=node_id,
                    question_id=node.spec.id,
                    history=dict(point.history),
                    theta=dict(point.theta),
                    rendered_fingerprint=rendered.fingerprint,
                    request_fingerprint=fingerprints[node_id],
                    semantic_request_hash=fingerprints[node_id],
                    stencil_role=point.stencil_role,
                    distribution=record,
                    status=ProviderStatus(
                        ok=conversion.status is MetricStatus.OK,
                        model_requested=self.model,
                        model_resolved=resolved_model,
                        request_id=request_id,
                        error_type=None,
                        error_message=conversion.message,
                        reason_code=conversion.reason_code,
                        attempts=1,
                    ),
                    routing_semantics=self.routing_semantics.value,
                    duration_s=duration,
                    usage=UsageRecord(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        usage_known=input_tokens is not None,
                    ),
                    provenance={
                        "provider": "typesafe",
                        "adapter_version": ADAPTER_VERSION,
                        "endpoint": self.endpoint,
                    },
                )
            )
        return traces

    def _note_http_attempts(self, attempts_before: int | None) -> None:
        """Count actual HTTP attempts, including SDK-internal retries."""
        if attempts_before is None or self.recorder is None:
            self.tracker.note_attempt()
            return
        delta = max(self.recorder.transport_attempts - attempts_before, 1)
        self.tracker.note_attempt(delta)

    def _failure_trace(
        self,
        *,
        node_id: str,
        point: ExperimentPoint,
        rendered_fingerprint: str,
        request_fingerprint_value: str,
        reason_code: str,
        message: str,
        retryable: bool = False,
        duration_s: float | None = None,
        model_resolved: str | None = None,
        request_id: str | None = None,
        usage: UsageRecord | None = None,
    ) -> EvaluationTrace:
        return EvaluationTrace(
            experiment_id=self.system_id,
            case_id=point.case.id,
            point_id=point.point_id,
            repeat=point.repeat,
            node_id=node_id,
            question_id=self.nodes[node_id].spec.id,
            history=dict(point.history),
            theta=dict(point.theta),
            rendered_fingerprint=rendered_fingerprint,
            request_fingerprint=request_fingerprint_value,
            stencil_role=point.stencil_role,
            status=ProviderStatus(
                ok=False,
                model_requested=self.model,
                model_resolved=model_resolved,
                request_id=request_id,
                error_type=reason_code,
                error_message=message,
                retryable=retryable,
                attempts=1,
                incomplete=True,
                reason_code=reason_code,
            ),
            routing_semantics=self.routing_semantics.value,
            duration_s=duration_s,
            usage=usage,
            provenance={"provider": "typesafe", "adapter_version": ADAPTER_VERSION},
        )

    def available_models(self) -> list[str]:
        """List models available to the configured account (makes one request)."""
        response = self.client.models.list()
        return [metadata.name for metadata in response.models]

    def close(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            client.close()


def _classify_error(error: BaseException) -> tuple[str, bool]:
    name = type(error).__name__
    if name == "TypeSafeAuthenticationError":
        return "auth_error", False
    if name == "TypeSafePermissionDeniedError":
        return "permission_denied", False
    if name == "TypeSafeBadRequestError":
        return "bad_request", False
    if name == "TypeSafeUnprocessableEntityError":
        return "unprocessable_entity", False
    if name == "TypeSafeAPIResponseValidationError":
        return "schema_error", False
    if name == "TypeSafeRateLimitError":
        return "rate_limited", True
    if name == "TypeSafeAPITimeoutError":
        return "timeout", True
    if name == "TypeSafeAPIConnectionError":
        return "connection_error", True
    if name == "TypeSafeNotFoundError":
        return "not_found", False
    if name == "TypeSafeInternalServerError":
        return "server_error", True
    return "provider_error", False


def _safe_request_id(response: Any) -> str | None:
    """Return the request id when the response carries one."""
    try:
        value = response.request_id
    except Exception:  # noqa: BLE001 - the SDK raises when the header is absent
        return None
    return None if value is None else str(value)


def _response_payload(response: Any, resolved_model: str) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for name, answer in getattr(response, "answers", {}).items():
        if hasattr(answer, "model_dump"):
            answers[name] = answer.model_dump(mode="json")
    return {"model": resolved_model, "answers": answers}


def _build_client(
    *,
    api_key: str,
    model: str,
    budget: Budget,
    endpoint: str | None,
    extra_headers: Mapping[str, str] | None,
) -> Any:
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    retry = RetryPolicy(
        max_retries=2,
        respect_retry_after=True,
        timeout=budget.request_timeout_s,
    )
    return TypeSafeClient(
        api_key=api_key,
        model=model,
        retry=retry,
        timeout=budget.request_timeout_s,
        base_url=endpoint,
        headers=dict(extra_headers) if extra_headers else None,
    )

def question_spec_to_typesafe(spec: QuestionSpec, *, node_id: str) -> TypeSafeQuestion:
    """Convert a declared QuestionSpec into an SDK question object."""
    if spec.primitive is Primitive.CHOICE:
        from typesafe_sdk import Choice

        question: Any = Choice(
            instructions=spec.instructions,
            criteria={outcome.id: outcome.description for outcome in spec.outcomes},
        )
    elif spec.primitive is Primitive.NOUL:
        from typesafe_sdk import Noul

        question = Noul(instructions=spec.instructions)
    elif spec.primitive is Primitive.SCORE:
        from typesafe_sdk import Score

        ordered = sorted(
            spec.outcomes,
            key=lambda outcome: (outcome.numeric is None, outcome.numeric, outcome.id),
        )
        question = Score(
            instructions=spec.instructions,
            criteria=[outcome.description or outcome.id for outcome in ordered],
        )
    else:
        raise ValueError(f"unsupported primitive {spec.primitive!r}")
    return TypeSafeQuestion(node_id=node_id, spec=spec, question=question)
