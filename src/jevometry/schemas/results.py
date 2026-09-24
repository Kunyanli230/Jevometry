"""Analysis result documents persisted to ``metrics.json`` and friends."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from jevometry.schemas.common import (
    AnalysisObject,
    CapabilityMatrix,
    Diagnostic,
    MetricResult,
    MetricStatus,
    StrictModel,
)
from jevometry.schemas.distribution import DistributionRecord
from jevometry.schemas.experiment import SamplingContract
from jevometry.schemas.parameters import StencilKind


class JacobianResult(StrictModel):
    """Finite-difference (or analytic) Jacobian of one node at one point.

    Rows are support outcomes, columns are parameters: ``J[k, a] = dq_k/dtheta_a``.
    """

    node_id: str
    case_id: str
    point_id: str
    parameter_names: list[str]
    support: list[str]
    values: list[list[float]] | None = None
    method: str
    stencil_kind: StencilKind
    step_sizes: dict[str, float] = Field(default_factory=dict)
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    row_sum_residuals: dict[str, float] = Field(default_factory=dict)
    stability_relative: dict[str, float] = Field(default_factory=dict)
    stability_absolute: dict[str, float] = Field(default_factory=dict)
    renderer_resolution_limited: bool = False
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @field_validator("values")
    @classmethod
    def _finite(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return None
        for row in value:
            for item in row:
                if item != item or item in (float("inf"), float("-inf")):
                    raise ValueError("Jacobian values must be finite")
        return value


class FisherGeometry(StrictModel):
    """Fisher pullback metric with matrix diagnostics."""

    node_id: str
    case_id: str
    point_id: str
    parameter_names: list[str]
    coordinates: str = "raw"
    values: list[list[float]] | None = None
    sqrt_form_values: list[list[float]] | None = None
    eigenvalues: list[float] = Field(default_factory=list)
    eigenvectors: list[list[float]] = Field(default_factory=list)
    rank: int | None = None
    condition_number: float | None = None
    symmetry_residual: float | None = None
    sqrt_form_residual: float | None = None
    null_directions: list[list[float]] = Field(default_factory=list)
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    tolerances: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[Diagnostic] = Field(default_factory=list)


class GeometryResult(StrictModel):
    """Per-node geometry bundle for one case/point/repeat."""

    node_id: str
    case_id: str
    point_id: str
    theta: dict[str, float]
    distribution: DistributionRecord | None = None
    jacobian: JacobianResult | None = None
    fisher: FisherGeometry | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None


class SimulationSummary(StrictModel):
    """Empirical MLE behaviour from a declared synthetic model."""

    parameter: str
    true_value: float
    sample_size: int
    replications: int
    failures: int = 0
    bias: float | None = None
    variance: float | None = None
    rmse: float | None = None
    coverage: float | None = None
    mean_interval_width: float | None = None
    crlb_variance: float | None = None
    monte_carlo_standard_error: float | None = None
    analysis_object: AnalysisObject = AnalysisObject.EMPIRICAL_OBSERVATION_MODEL
    notes: list[str] = Field(default_factory=list)


class InferenceResult(StrictModel):
    """CRLB and optional simulation results under an explicit contract."""

    schema_version: str = "1.0"
    run_id: str
    parameter_names: list[str]
    estimand: str
    contract: SamplingContract
    analysis_object: AnalysisObject = AnalysisObject.EMPIRICAL_OBSERVATION_MODEL
    crlb: list[list[float]] | None = None
    crlb_units: str | None = None
    standard_errors: list[float] | None = None
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    remedy: str | None = None
    nuisance_parameters: list[str] = Field(default_factory=list)
    method: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    simulations: list[SimulationSummary] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)


class NodeAnalysis(StrictModel):
    """All per-node results for one case/point, keyed by node."""

    node_id: str
    question_id: str
    case_id: str
    point_id: str
    theta: dict[str, float]
    support: list[str]
    distribution: DistributionRecord | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    geometry: GeometryResult | None = None
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None


class RedundancyCase(StrictModel):
    """Independent-sum versus correct-joint comparison for replicated outputs."""

    name: str
    description: str
    independent_sum_trace: float | None = None
    joint_trace: float | None = None
    difference: float | None = None
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)


class SystemAnalysisDocument(StrictModel):
    """System-level geometry, information loss and redundancy results."""

    mode: str
    node_order: list[str]
    routing_semantics: str | None = None
    action_distribution_available: bool = False
    metrics: list[MetricResult] = Field(default_factory=list)
    fisher: FisherGeometry | None = None
    information_loss: list[MetricResult] = Field(default_factory=list)
    redundancy: list[RedundancyCase] = Field(default_factory=list)
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)


class AnalysisDocument(StrictModel):
    """The complete analysis result for one run revision."""

    schema_version: str = "1.0"
    run_id: str
    analysis_revision: int
    created_utc: str
    experiment_id: str
    analysis_object: AnalysisObject | None = None
    capability_matrix: CapabilityMatrix = Field(default_factory=CapabilityMatrix)
    nodes: list[NodeAnalysis] = Field(default_factory=list)
    system: SystemAnalysisDocument | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def metric_by_name(self, name: str) -> MetricResult | None:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None


class RunManifest(StrictModel):
    """Atomic manifest describing one run directory."""

    schema_version: str = "1.0"
    run_id: str
    created_utc: str
    updated_utc: str | None = None
    experiment_id: str
    experiment_hash: str
    model: str | None = None
    adapter: str | None = None
    provider_mode: str
    live: bool = False
    analysis_revisions: list[int] = Field(default_factory=list)
    latest_analysis_revision: int | None = None
    files: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
