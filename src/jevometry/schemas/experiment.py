"""Experiment declarations: parameters, cases, stencil, budget, contract."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from jevometry.schemas.common import StrictModel
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec


class IdentifiabilitySource(str, Enum):
    """Origin of the regularity/identifiability claim for inference."""

    ANALYTIC = "analytic"
    USER_ASSERTED = "user_asserted"
    EMPIRICALLY_CHECKED = "empirically_checked"


class SamplingKind(str, Enum):
    """Dependence structure of the declared sampling process."""

    IID = "iid"
    INDEPENDENT_NON_IDENTICAL = "independent_non_identical"
    DEPENDENT = "dependent"
    NONE = "none"


class ObservationRelation(str, Enum):
    """How observations relate to the analysed Jev output."""

    REPORTED_DISTRIBUTION_DRAW = "reported_distribution_draw"
    EXTERNAL_MEASUREMENT = "external_measurement"
    SYNTHETIC_SIMULATION = "synthetic_simulation"
    NONE = "none"


class SamplingContract(StrictModel):
    """Everything required before any CRLB or inferential claim is emitted.

    Without a complete contract the toolkit refuses inference rather than
    substituting an independence assumption or an API-call count.
    """

    observable: str | None = None
    observation_unit: str | None = None
    sampling: SamplingKind = SamplingKind.NONE
    sample_size: int | None = None
    relation_to_jev: ObservationRelation = ObservationRelation.NONE
    identifiability_source: IdentifiabilitySource | None = None
    fixed_support: bool = False
    differentiable: bool = False
    locally_identifiable: bool = False
    estimand: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("sample_size")
    @classmethod
    def _validate_sample_size(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("sample_size must be positive when declared")
        return value

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.observable:
            missing.append("observable")
        if not self.observation_unit:
            missing.append("observation_unit")
        if self.sampling is SamplingKind.NONE:
            missing.append("sampling")
        if self.sample_size is None:
            missing.append("sample_size")
        if self.relation_to_jev is ObservationRelation.NONE:
            missing.append("relation_to_jev")
        if self.identifiability_source is None:
            missing.append("identifiability_source")
        if not self.fixed_support:
            missing.append("fixed_support")
        if not self.differentiable:
            missing.append("differentiable")
        if not self.locally_identifiable:
            missing.append("locally_identifiable")
        if not self.estimand:
            missing.append("estimand")
        return missing

    def is_complete(self) -> bool:
        return not self.missing_fields()


class Budget(StrictModel):
    """Hard resource caps for an acquisition run."""

    max_attempts: int = 200
    max_input_tokens: int = 200_000
    max_concurrency: int = 2
    request_timeout_s: float = 45.0
    experiment_timeout_s: float = 600.0
    repeats: int = 1

    @field_validator(
        "max_attempts",
        "max_input_tokens",
        "max_concurrency",
        "request_timeout_s",
        "experiment_timeout_s",
        "repeats",
    )
    @classmethod
    def _validate_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("budget fields must be positive")
        return value


class CaseSpec(StrictModel):
    """One fixed input case (state) evaluated across the theta grid."""

    id: str
    state: Any
    description: str | None = None
    redactions: list[str] = Field(default_factory=list)

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("case state must not be None")
        return value


class ExperimentSpec(StrictModel):
    """A complete, resolved experiment definition."""

    schema_version: str = "1.0"
    id: str
    description: str | None = None
    model: str | None = None
    adapter: str | None = None
    parameter_set: ParameterSet
    stencil: StencilSpec = Field(default_factory=StencilSpec)
    cases: list[CaseSpec] = Field(default_factory=list)
    theta_points: list[dict[str, float]] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    seed: int = 0
    sampling_contract: SamplingContract | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ExperimentSpec:
        if not self.id:
            raise ValueError("experiment id must be non-empty")
        if not self.cases:
            raise ValueError("at least one case is required")
        if not self.theta_points:
            raise ValueError("at least one theta point is required")
        names = set(self.parameter_set.names)
        for point in self.theta_points:
            if set(point) != names:
                raise ValueError(
                    "every theta point must define exactly the declared parameters: "
                    f"expected {sorted(names)}, got {sorted(point)}"
                )
        for parameter in self.parameter_set.parameters:
            if not parameter.continuous:
                raise ValueError(
                    f"parameter {parameter.name!r} is categorical; continuous finite "
                    "differences are not defined for it"
                )
        return self

    def parameters(self) -> list[ParameterSpec]:
        return list(self.parameter_set.parameters)

    def parameter_names(self) -> list[str]:
        return self.parameter_set.names

    def center(self) -> dict[str, float]:
        return dict(self.theta_points[0])
