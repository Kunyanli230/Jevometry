"""Probability distribution records and their provenance."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from jevometry.schemas.common import AnalysisObject, Primitive, StrictModel


class DistributionSource(str, Enum):
    """Where a probability vector came from."""

    REPORTED = "reported"
    DECLARED = "declared"
    EMPIRICAL = "empirical"
    SYNTHETIC = "synthetic"
    REPLAY = "replay"
    ANALYTIC = "analytic"
    SURROGATE = "surrogate"


_SOURCE_OBJECT: dict[DistributionSource, AnalysisObject] = {
    DistributionSource.REPORTED: AnalysisObject.REPORTED_DISTRIBUTION,
    DistributionSource.REPLAY: AnalysisObject.REPORTED_DISTRIBUTION,
    DistributionSource.DECLARED: AnalysisObject.DECLARED_SYSTEM_MODEL,
    DistributionSource.ANALYTIC: AnalysisObject.DECLARED_SYSTEM_MODEL,
    DistributionSource.SYNTHETIC: AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    DistributionSource.EMPIRICAL: AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    DistributionSource.SURROGATE: AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
}


class DistributionRecord(StrictModel):
    """A validated probability vector with raw and working copies.

    ``raw`` is always preserved exactly as received.  ``working`` is the
    renormalised copy used for geometry, present only when the raw total was
    within the configured tolerance of one.
    """

    schema_version: str = "1.0"
    node_id: str
    question_id: str
    primitive: Primitive
    support: list[str]
    raw: list[float]
    working: list[float] | None = None
    raw_total: float
    correction: float = 0.0
    derived: bool = False
    selected: str | None = None
    confidence: float | None = None
    source: DistributionSource
    legend: dict[str, str] | None = None
    numeric_encoding: dict[str, float] | None = None
    semantic_hash: str
    case_id: str
    point_id: str
    repeat: int = 0
    theta: dict[str, float] = Field(default_factory=dict)
    request_fingerprint: str | None = None
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> DistributionRecord:
        if len(self.support) != len(self.raw):
            raise ValueError("support and raw must have equal length")
        if len(set(self.support)) != len(self.support):
            raise ValueError("support must be unique")
        if self.working is not None and len(self.working) != len(self.raw):
            raise ValueError("working must match raw length")
        if self.selected is not None and self.selected not in self.support:
            raise ValueError("selected must be a declared outcome")
        return self

    @property
    def analysis_object(self) -> AnalysisObject:
        return _SOURCE_OBJECT[self.source]

    def probabilities(self) -> list[float]:
        """Return the vector geometry should use, preferring ``working``."""
        if self.working is not None:
            return list(self.working)
        return list(self.raw)

    def index_of(self, outcome_id: str) -> int:
        return self.support.index(outcome_id)
