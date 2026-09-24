"""Evaluation traces: one recorded observation of one node."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jevometry.schemas.common import StrictModel
from jevometry.schemas.distribution import DistributionRecord


class RenderedInput(StrictModel):
    """The deterministic renderer output for one (theta, case) pair."""

    state: Any
    fingerprint: str
    theta: dict[str, float]
    case_id: str
    renderer_version: str | None = None
    notes: list[str] = Field(default_factory=list)


class UsageRecord(StrictModel):
    """Token usage as reported by a provider (unknown usage stays None)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_known: bool = True


class ProviderStatus(StrictModel):
    """Outcome of a single provider attempt."""

    ok: bool
    model_requested: str | None = None
    model_resolved: str | None = None
    request_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False
    attempts: int = 1
    incomplete: bool = False
    reason_code: str | None = None


class EvaluationTrace(StrictModel):
    """One provider evaluation of one node at one theta point.

    A trace is a *reported distribution* observation.  It is not an independent
    categorical draw, and the provider's selected label is not assumed to be a
    random sample from the reported vector.
    """

    schema_version: str = "1.0"
    experiment_id: str
    case_id: str
    point_id: str
    repeat: int = 0
    node_id: str
    question_id: str
    parent_ids: list[str] = Field(default_factory=list)
    history: dict[str, str] = Field(default_factory=dict)
    theta: dict[str, float] = Field(default_factory=dict)
    rendered_fingerprint: str
    request_fingerprint: str
    response_fingerprint: str | None = None
    semantic_request_hash: str | None = None
    stencil_role: str = "center"
    stencil_offset: dict[str, float] = Field(default_factory=dict)
    distribution: DistributionRecord | None = None
    status: ProviderStatus
    observed_action: str | None = None
    routing_semantics: str | None = None
    duration_s: float | None = None
    usage: UsageRecord | None = None
    timestamp_utc: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> tuple[str, str, str, int, str]:
        return (self.case_id, self.point_id, self.node_id, self.repeat, self.experiment_id)
