"""Shared schema primitives: status, analysis objects, metric results.

All persisted documents are JSON-safe: NaN and Infinity never appear as JSON
numbers.  Non-finite quantities are represented with ``value_kind`` and a
``reason_code`` instead, so that a refusal can never be confused with a zero.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


class MetricStatus(str, Enum):
    """Structured outcome status for every computed quantity."""

    OK = "ok"
    CONDITIONAL = "conditional"
    UNSTABLE = "unstable"
    UNDEFINED = "undefined"
    NOT_IDENTIFIABLE = "not_identifiable"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class AnalysisObject(str, Enum):
    """The statistical object an analysis refers to."""

    REPORTED_DISTRIBUTION = "reported_distribution"
    DECLARED_SYSTEM_MODEL = "declared_system_model"
    EMPIRICAL_OBSERVATION_MODEL = "empirical_observation_model"


class ParameterRole(str, Enum):
    """Research role of an experiment parameter."""

    TASK_RELEVANT = "task_relevant"
    NUISANCE = "nuisance"
    DIAGNOSTIC = "diagnostic"


class ValueKind(str, Enum):
    """How to interpret ``MetricResult.value``."""

    NULL = "null"
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    TEXT = "text"
    BOOL = "bool"
    OBJECT = "object"
    POSITIVE_INFINITY = "positive_infinity"
    NEGATIVE_INFINITY = "negative_infinity"


class RoutingSemantics(str, Enum):
    """How the analysed agent turns probabilities into actions."""

    SAMPLED_OUTCOME = "sampled_outcome"
    DETERMINISTIC_POLICY = "deterministic_policy"
    EXTERNALLY_OBSERVED = "externally_observed"
    NOT_APPLICABLE = "not_applicable"


class Primitive(str, Enum):
    """Jev question primitives supported by v0.1."""

    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class StrictModel(BaseModel):
    """Base model with strict extras handling and JSON-safe validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _check_json_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite float at {path}; encode infinity/NaN structurally "
                "with value_kind and reason_code"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string key at {path}")
            _check_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_json_safe(item, f"{path}[{index}]")
        return


class MetricResult(StrictModel):
    """A single reported quantity with its provenance and refusal reason.

    A result that cannot be computed must carry a non-``ok`` status and a
    ``reason_code``; ``value`` stays ``None`` rather than defaulting to zero.
    """

    name: str
    status: MetricStatus
    analysis_object: AnalysisObject
    value: Any = None
    value_kind: ValueKind = ValueKind.NULL
    units: str | None = None
    coordinates: list[str] | None = None
    row_labels: list[str] | None = None
    column_labels: list[str] | None = None
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    numerical_method: str | None = None
    tolerances: dict[str, float] = Field(default_factory=dict)
    reason_code: str | None = None
    remedy: str | None = None
    artifact_ref: str | None = None

    @field_validator("value", "diagnostics")
    @classmethod
    def _validate_json_safe(cls, value: Any) -> Any:
        _check_json_safe(value)
        return value

    @property
    def computed(self) -> bool:
        return self.status is MetricStatus.OK

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def ok(
        cls,
        name: str,
        value: Any,
        *,
        analysis_object: AnalysisObject,
        value_kind: ValueKind = ValueKind.SCALAR,
        **kwargs: Any,
    ) -> MetricResult:
        return cls(
            name=name,
            status=MetricStatus.OK,
            analysis_object=analysis_object,
            value=value,
            value_kind=value_kind,
            **kwargs,
        )

    @classmethod
    def refusal(
        cls,
        name: str,
        *,
        status: MetricStatus,
        analysis_object: AnalysisObject,
        reason_code: str,
        remedy: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        if status is MetricStatus.OK:
            raise ValueError("refusal() requires a non-ok status")
        return cls(
            name=name,
            status=status,
            analysis_object=analysis_object,
            reason_code=reason_code,
            remedy=remedy,
            **kwargs,
        )


class Diagnostic(StrictModel):
    """Structured diagnostic attached to a metric or analysis."""

    code: str
    message: str
    severity: str = "info"
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def _validate_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        _check_json_safe(value, "details")
        return value


class CapabilityEntry(StrictModel):
    """One row of the report-level capability matrix."""

    capability: str
    status: MetricStatus
    analysis_object: AnalysisObject | None = None
    reason_code: str | None = None
    remedy: str | None = None
    requires: list[str] = Field(default_factory=list)


class CapabilityMatrix(StrictModel):
    """Availability of each analysis capability for a run."""

    entries: list[CapabilityEntry] = Field(default_factory=list)

    def add(self, entry: CapabilityEntry) -> None:
        self.entries.append(entry)

    def status_of(self, capability: str) -> MetricStatus | None:
        for entry in self.entries:
            if entry.capability == capability:
                return entry.status
        return None
