"""Analysis: turn captures into geometry, system and capability results.

Every result carries an explicit ``analysis_object`` and a structured status.
Missing structure (no joint model, no sampling contract, no stencil captures)
produces a refusal with a reason code, never a zero or a default assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jevometry.adapters.base import JointModel, LikelihoodModel, SystemAdapter
from jevometry.adapters.typesafe import BudgetTracker
from jevometry.experiments.acquisition import Captures
from jevometry.geometry.coordinates import CoordinateTransform
from jevometry.geometry.derivatives import (
    JacobianComputation,
    NodeEvaluation,
    compute_jacobian,
)
from jevometry.geometry.distances import (
    entropy,
    fisher_rao_distance,
    hellinger_distance,
    js_divergence,
    kl_divergence,
)
from jevometry.geometry.fisher import fisher_pullback
from jevometry.geometry.simplex import align_support, validate_probabilities
from jevometry.inference.contracts import InferenceEligibility, check_contract
from jevometry.schemas.common import (
    AnalysisObject,
    CapabilityEntry,
    CapabilityMatrix,
    Diagnostic,
    MetricResult,
    MetricStatus,
    RoutingSemantics,
    ValueKind,
)
from jevometry.schemas.distribution import DistributionRecord
from jevometry.schemas.experiment import SamplingContract
from jevometry.schemas.joint import JointDistribution
from jevometry.schemas.results import (
    AnalysisDocument,
    FisherGeometry,
    GeometryResult,
    JacobianResult,
    NodeAnalysis,
    RedundancyCase,
    SystemAnalysisDocument,
)
from jevometry.schemas.system import CompositionMode
from jevometry.schemas.trace import EvaluationTrace
from jevometry.systems.joint import joint_fisher, mutual_information
from jevometry.systems.pushforward import AggregationMap, information_loss
from jevometry.systems.redundancy import compare_independent_sum

FloatArray = NDArray[np.float64]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _theta_key(theta: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((name, round(float(value), 12)) for name, value in theta.items()))


class TraceEvaluator:
    """Reconstruct a node evaluation from captured traces.

    This is what makes offline analysis independent of any live call: the
    analysis only consumes what was captured, and reports ``insufficient_data``
    when a required stencil point is absent.
    """

    def __init__(
        self,
        traces: Sequence[EvaluationTrace],
        *,
        node_id: str,
        case_id: str,
        point_id: str,
        repeat: int,
    ) -> None:
        self.index: dict[tuple[tuple[str, float], ...], EvaluationTrace] = {}
        for trace in traces:
            if (
                trace.node_id == node_id
                and trace.case_id == case_id
                and trace.point_id == point_id
                and trace.repeat == repeat
            ):
                self.index[_theta_key(trace.theta)] = trace
        self.node_id = node_id
        self.case_id = case_id
        self.point_id = point_id

    def __call__(self, theta: Mapping[str, float]) -> NodeEvaluation:
        trace = self.index.get(_theta_key(theta))
        if trace is None:
            return NodeEvaluation(
                support=(),
                probabilities=np.zeros(0, dtype=np.float64),
                rendered_fingerprint="",
                semantic_hash="",
                model_identity="",
                status=MetricStatus.INSUFFICIENT_DATA,
                reason_code="missing_capture",
            )
        if trace.distribution is None:
            return NodeEvaluation(
                support=(),
                probabilities=np.zeros(0, dtype=np.float64),
                rendered_fingerprint=trace.rendered_fingerprint,
                semantic_hash=trace.semantic_request_hash or "",
                model_identity=trace.status.model_resolved or "",
                status=MetricStatus.INSUFFICIENT_DATA,
                reason_code=trace.status.reason_code or "missing_distribution",
            )
        record = trace.distribution
        return NodeEvaluation(
            support=tuple(record.support),
            probabilities=np.asarray(record.probabilities(), dtype=np.float64),
            rendered_fingerprint=trace.rendered_fingerprint,
            semantic_hash=record.semantic_hash,
            model_identity=trace.status.model_resolved or "",
            selected=record.selected,
            confidence=record.confidence,
            raw_total=record.raw_total,
            derived=record.derived,
        )


@dataclass
class Analysis:
    """Analysis result bundle."""

    document: AnalysisDocument
    captures: Captures
    arrays: dict[str, FloatArray] = field(default_factory=dict)
    joint: JointDistribution | None = None
    adapter: SystemAdapter | None = None
    contract: SamplingContract | None = None
    likelihood: LikelihoodModel | None = None
    inference_eligibility: InferenceEligibility | None = None

    def to_json(self) -> dict[str, Any]:
        return self.document.model_dump(mode="json")


def _distribution_object(record: DistributionRecord | None) -> AnalysisObject:
    if record is None:
        return AnalysisObject.REPORTED_DISTRIBUTION
    return record.analysis_object


def _node_metric(
    name: str,
    value: float,
    *,
    analysis_object: AnalysisObject,
    units: str = "nats",
    **kwargs: Any,
) -> MetricResult:
    return MetricResult.ok(
        name,
        float(value),
        analysis_object=analysis_object,
        value_kind=ValueKind.SCALAR,
        units=units,
        **kwargs,
    )


def analyze(
    captures: Captures,
    *,
    adapter: SystemAdapter | None = None,
    contract: SamplingContract | None = None,
    aggregation: AggregationMap | None = None,
    joint_model: JointModel | None = None,
) -> Analysis:
    """Compute every metric that the captured data actually supports."""
    experiment = captures.experiment
    parameters = experiment.parameters()
    parameter_names = experiment.parameter_names()
    transform = CoordinateTransform.from_parameters(parameters)
    capability = CapabilityMatrix()
    diagnostics: list[Diagnostic] = []
    arrays: dict[str, FloatArray] = {}

    resolved_joint = joint_model
    if resolved_joint is None and adapter is not None:
        candidate = getattr(adapter, "joint_model", None)
        if candidate is not None:
            resolved_joint = candidate
    resolved_contract = contract or experiment.sampling_contract
    likelihood = getattr(adapter, "likelihood_model", None) if adapter is not None else None
    if callable(likelihood):
        likelihood = likelihood()
    eligibility = check_contract(
        resolved_contract,
        parameter_names=parameter_names,
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )

    node_ids = _node_ids(captures, adapter)
    center = captures.center_traces()
    has_distributions = any(trace.distribution is not None for trace in center)
    capability.add(
        CapabilityEntry(
            capability="distribution_comparison",
            status=MetricStatus.OK if has_distributions else MetricStatus.INSUFFICIENT_DATA,
            analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
            reason_code=None if has_distributions else "no_distribution_records",
            remedy=None
            if has_distributions
            else "run the experiment and capture at least one reported distribution",
        )
    )

    stencil_available = any(trace.stencil_role != "center" for trace in captures.traces)
    geometry_possible = bool(parameters) and stencil_available and has_distributions
    capability.add(
        CapabilityEntry(
            capability="node_geometry",
            status=MetricStatus.OK if geometry_possible else MetricStatus.INSUFFICIENT_DATA,
            analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
            reason_code=None
            if geometry_possible
            else (
                "no_parameters"
                if not parameters
                else ("missing_stencil_captures" if not stencil_available else "no_distributions")
            ),
            remedy=None
            if geometry_possible
            else "capture the finite-difference stencil points for each parameter",
        )
    )

    system_spec = adapter.describe() if adapter is not None else None
    composition_mode = (
        system_spec.composition_mode if system_spec is not None else CompositionMode.NODE_ONLY
    )
    if resolved_joint is not None:
        capability.add(
            CapabilityEntry(
                capability="system_information",
                status=MetricStatus.OK,
                analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                requires=["declared joint distribution"],
            )
        )
    elif composition_mode is CompositionMode.DECLARED_PRODUCT:
        capability.add(
            CapabilityEntry(
                capability="system_information",
                status=MetricStatus.CONDITIONAL,
                analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                reason_code="assumption_based",
                remedy="declared conditional independence: results are assumption based",
                requires=["declared conditional independence"],
            )
        )
    else:
        capability.add(
            CapabilityEntry(
                capability="system_information",
                status=MetricStatus.UNSUPPORTED,
                analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                reason_code="no_joint_model",
                remedy=(
                    "declare a joint model or conditional tree; marginals do not determine "
                    "system Fisher information"
                ),
            )
        )

    action_semantics = system_spec.routing_semantics if system_spec is not None else None
    action_available = action_semantics is RoutingSemantics.SAMPLED_OUTCOME
    capability.add(
        CapabilityEntry(
            capability="action_information",
            status=MetricStatus.OK if action_available else MetricStatus.UNSUPPORTED,
            analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
            reason_code=None if action_available else "deterministic_policy",
            remedy=None
            if action_available
            else (
                "trajectory and action Fisher are unavailable for a deterministic policy; "
                "declare a sampling model or a surrogate"
            ),
        )
    )

    if resolved_contract is not None and eligibility.eligible and likelihood is not None:
        capability.add(
            CapabilityEntry(
                capability="inference",
                status=MetricStatus.OK,
                analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
                requires=["complete sampling contract", "likelihood model"],
            )
        )
    else:
        reason = (
            "no_likelihood_model"
            if likelihood is None
            else (eligibility.reason_code or "incomplete_sampling_contract")
        )
        capability.add(
            CapabilityEntry(
                capability="inference",
                status=MetricStatus.UNSUPPORTED,
                analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
                reason_code=reason,
                remedy=eligibility.remedy
                or "declare a likelihood model and a complete sampling contract",
            )
        )

    node_analyses: list[NodeAnalysis] = []
    top_metrics: list[MetricResult] = []
    for node_id in node_ids:
        node_analyses.extend(
            _analyze_node(
                captures=captures,
                node_id=node_id,
                parameters=parameters,
                parameter_names=parameter_names,
                transform=transform,
                adapter=adapter,
                arrays=arrays,
                diagnostics=diagnostics,
            )
        )

    distances = _distribution_distances(captures)
    top_metrics.extend(distances)

    system_document = _analyze_system(
        captures=captures,
        adapter=adapter,
        node_ids=node_ids,
        joint_model=resolved_joint,
        aggregation=aggregation,
        arrays=arrays,
        diagnostics=diagnostics,
        composition_mode=composition_mode,
    )
    if resolved_joint is None and composition_mode is CompositionMode.DECLARED_PRODUCT:
        _add_declared_product_system(
            system_document,
            captures=captures,
            adapter=adapter,
            node_ids=node_ids,
            arrays=arrays,
        )

    if captures.incomplete:
        diagnostics.append(
            Diagnostic(
                code="incomplete_captures",
                message=(
                    "some captures failed; affected metrics are marked incomplete and "
                    "were not filled from other sources"
                ),
                severity="warning",
                details={"failed": len(captures.failures())},
            )
        )

    document = AnalysisDocument(
        run_id=captures.run_id,
        analysis_revision=0,
        created_utc=utc_now(),
        experiment_id=experiment.id,
        analysis_object=AnalysisObject.REPORTED_DISTRIBUTION
        if has_distributions
        else AnalysisObject.DECLARED_SYSTEM_MODEL,
        capability_matrix=capability,
        nodes=node_analyses,
        system=system_document,
        metrics=top_metrics,
        diagnostics=diagnostics,
        provenance={
            "provider_mode": captures.provider_mode,
            "live": captures.live,
            "experiment_hash": captures.plan_summary,
            "parameters": parameter_names,
        },
        warnings=list(captures.notes),
    )
    return Analysis(
        document=document,
        captures=captures,
        arrays=arrays,
        joint=resolved_joint.probabilities(experiment.center())
        if resolved_joint is not None
        else None,
        adapter=adapter,
        contract=resolved_contract,
        likelihood=likelihood,
        inference_eligibility=eligibility,
    )


def _node_ids(captures: Captures, adapter: SystemAdapter | None) -> list[str]:
    if adapter is not None:
        return adapter.describe().node_ids()
    return captures.node_ids()


def _analyze_node(
    *,
    captures: Captures,
    node_id: str,
    parameters: Sequence[Any],
    parameter_names: list[str],
    transform: CoordinateTransform,
    adapter: SystemAdapter | None,
    arrays: dict[str, FloatArray],
    diagnostics: list[Diagnostic],
) -> list[NodeAnalysis]:
    experiment = captures.experiment
    results: list[NodeAnalysis] = []
    for case in experiment.cases:
        for theta in experiment.theta_points:
            pid = _point_id(theta)
            repeat = 0
            traces = [
                trace
                for trace in captures.traces
                if trace.node_id == node_id and trace.case_id == case.id and trace.point_id == pid
            ]
            center_trace = next(
                (
                    trace
                    for trace in traces
                    if trace.stencil_role == "center" and trace.repeat == repeat
                ),
                None,
            )
            record = center_trace.distribution if center_trace is not None else None
            analysis_object = _distribution_object(record)
            metrics: list[MetricResult] = []
            geometry: GeometryResult | None = None
            status = MetricStatus.OK
            reason_code: str | None = None
            if record is not None:
                vector = np.asarray(record.probabilities(), dtype=np.float64)
                metrics.append(
                    _node_metric(
                        f"entropy:{node_id}:{case.id}:{pid}",
                        entropy(vector),
                        analysis_object=analysis_object,
                        coordinates=parameter_names,
                        provenance={"support": record.support, "point": pid},
                    )
                )
                if record.derived:
                    diagnostics.append(
                        Diagnostic(
                            code="working_simplex_derived",
                            message=(
                                f"node {node_id}: working simplex derived from raw total "
                                f"{record.raw_total!r}"
                            ),
                            severity="info",
                        )
                    )
            evaluator = TraceEvaluator(
                captures.traces, node_id=node_id, case_id=case.id, point_id=pid, repeat=repeat
            )
            computation = compute_jacobian(
                evaluator,
                node_id=node_id,
                case_id=case.id,
                point_id=pid,
                theta=theta,
                parameters=list(parameters),
                stencil=experiment.stencil,
            )
            if computation.values is not None and record is not None:
                geometry = _geometry_from_computation(
                    computation,
                    record=record,
                    transform=transform,
                    adapter=adapter,
                    node_id=node_id,
                    theta=theta,
                    parameter_names=parameter_names,
                    arrays=arrays,
                    diagnostics=diagnostics,
                )
                metrics.extend(geometry.metrics)
                status = geometry.status
                reason_code = geometry.reason_code
            else:
                status = computation.status
                reason_code = computation.reason_code
                metrics.append(
                    MetricResult.refusal(
                        f"fisher:{node_id}:{case.id}:{pid}",
                        status=computation.status,
                        analysis_object=analysis_object,
                        reason_code=computation.reason_code or "insufficient_data",
                        remedy=(
                            "capture the full finite-difference stencil for this node and point"
                        ),
                    )
                )
            results.append(
                NodeAnalysis(
                    node_id=node_id,
                    question_id=record.question_id if record is not None else node_id,
                    case_id=case.id,
                    point_id=pid,
                    theta=dict(theta),
                    support=record.support if record is not None else [],
                    distribution=record,
                    metrics=metrics,
                    geometry=geometry,
                    status=status,
                    reason_code=reason_code,
                )
            )
    return results


def _geometry_from_computation(
    computation: JacobianComputation,
    *,
    record: DistributionRecord,
    transform: CoordinateTransform,
    adapter: SystemAdapter | None,
    node_id: str,
    theta: Mapping[str, float],
    parameter_names: list[str],
    arrays: dict[str, FloatArray],
    diagnostics: list[Diagnostic],
) -> GeometryResult:
    assert computation.values is not None
    vector = np.asarray(record.probabilities(), dtype=np.float64)
    declared_fixed_zero = ()
    declared = getattr(adapter, "declared_fixed_zero", None)
    if callable(declared):
        declared_fixed_zero = tuple(declared(node_id))
    fisher = fisher_pullback(
        computation.values,
        vector,
        support=tuple(record.support),
        fixed_zero_outcomes=computation.fixed_zero_outcomes or declared_fixed_zero,
        declared_fixed_zero=bool(computation.fixed_zero_outcomes or declared_fixed_zero),
    )
    analysis_object = _distribution_object(record)
    metrics: list[MetricResult] = []
    jacobian_result = JacobianResult(
        node_id=node_id,
        case_id=record.case_id,
        point_id=record.point_id,
        parameter_names=parameter_names,
        support=list(record.support),
        values=[[float(value) for value in row] for row in computation.values],
        method=computation.method,
        stencil_kind=computation.stencil_kind,
        step_sizes=computation.step_sizes,
        status=computation.status,
        reason_code=computation.reason_code,
        row_sum_residuals=computation.row_sum_residuals,
        stability_relative=computation.stability_relative,
        stability_absolute=computation.stability_absolute,
        renderer_resolution_limited=computation.renderer_resolution_limited,
        diagnostics=computation.diagnostics,
    )
    if fisher.values is None:
        metrics.append(
            MetricResult.refusal(
                f"fisher:{node_id}:{record.case_id}:{record.point_id}",
                status=fisher.status,
                analysis_object=analysis_object,
                reason_code=fisher.reason_code or "fisher_undefined",
                remedy="inspect zero-probability outcomes or declare a fixed active support",
                assumptions=fisher.assumptions,
            )
        )
        return GeometryResult(
            node_id=node_id,
            case_id=record.case_id,
            point_id=record.point_id,
            theta=dict(theta),
            distribution=record,
            jacobian=jacobian_result,
            fisher=None,
            metrics=metrics,
            status=fisher.status,
            reason_code=fisher.reason_code,
        )
    assert fisher.matrix is not None
    standardized = transform.fisher(fisher.values)
    key = f"fisher__{node_id}__{record.case_id}__{record.point_id}"
    arrays[key] = fisher.values
    arrays[f"{key}__standardized"] = standardized
    fisher_geometry = FisherGeometry(
        node_id=node_id,
        case_id=record.case_id,
        point_id=record.point_id,
        parameter_names=parameter_names,
        coordinates="raw",
        values=[[float(value) for value in row] for row in fisher.values],
        sqrt_form_values=[[float(value) for value in row] for row in fisher.sqrt_form_values]
        if fisher.sqrt_form_values is not None
        else None,
        eigenvalues=[float(value) for value in fisher.matrix.eigenvalues],
        eigenvectors=[[float(value) for value in row] for row in fisher.matrix.eigenvectors],
        rank=fisher.matrix.rank,
        condition_number=fisher.matrix.condition_number,
        symmetry_residual=fisher.symmetry_residual,
        sqrt_form_residual=fisher.sqrt_form_residual,
        null_directions=[
            [float(value) for value in direction] for direction in fisher.matrix.null_directions
        ],
        status=fisher.status,
        reason_code=fisher.reason_code,
        assumptions=fisher.assumptions,
        tolerances={
            "rank_rtol": fisher.matrix.rank_rtol,
            "rank_atol": fisher.matrix.rank_atol,
        },
        diagnostics=fisher.diagnostics,
    )
    metrics.append(
        _node_metric(
            f"fisher_trace:{node_id}:{record.case_id}:{record.point_id}",
            float(np.trace(fisher.values)),
            analysis_object=analysis_object,
            units="nats",
            coordinates=parameter_names,
            tolerances={"rank_rtol": fisher.matrix.rank_rtol},
            numerical_method=computation.method,
            diagnostics={
                "rank": fisher.matrix.rank,
                "condition_number": fisher.matrix.condition_number,
                "stability_relative": computation.stability_relative,
            },
        )
    )
    metrics.append(
        _node_metric(
            f"fisher_trace_standardized:{node_id}:{record.case_id}:{record.point_id}",
            float(np.trace(standardized)),
            analysis_object=analysis_object,
            units="nats",
            coordinates=[f"{name}/scale" for name in parameter_names],
        )
    )
    for index, name in enumerate(parameter_names):
        metrics.append(
            _node_metric(
                f"fisher_diagonal:{node_id}:{record.case_id}:{record.point_id}:{name}",
                float(fisher.values[index, index]),
                analysis_object=analysis_object,
                units="nats",
                coordinates=[name],
            )
        )
    if fisher.matrix.rank < len(parameter_names):
        metrics.append(
            MetricResult.refusal(
                f"fisher_identifiability:{node_id}:{record.case_id}:{record.point_id}",
                status=MetricStatus.NOT_IDENTIFIABLE,
                analysis_object=analysis_object,
                reason_code="rank_deficient",
                remedy=(
                    f"rank {fisher.matrix.rank} of {len(parameter_names)}: some parameter "
                    "directions are not identifiable at this point"
                ),
                value_kind=ValueKind.NULL,
                diagnostics={
                    "null_directions": [
                        [float(value) for value in direction]
                        for direction in fisher.matrix.null_directions
                    ]
                },
            )
        )
    cross_check = _analytic_cross_check(
        adapter, node_id, theta, computation, diagnostics
    )
    if cross_check is not None:
        metrics.append(cross_check)
    geometry_status = computation.status
    if fisher.status is not MetricStatus.OK:
        geometry_status = fisher.status
    return GeometryResult(
        node_id=node_id,
        case_id=record.case_id,
        point_id=record.point_id,
        theta=dict(theta),
        distribution=record,
        jacobian=jacobian_result,
        fisher=fisher_geometry,
        metrics=metrics,
        status=geometry_status,
        reason_code=fisher.reason_code or computation.reason_code,
    )


def _analytic_cross_check(
    adapter: SystemAdapter | None,
    node_id: str,
    theta: Mapping[str, float],
    computation: JacobianComputation,
    diagnostics: list[Diagnostic],
) -> MetricResult | None:
    if adapter is None or computation.values is None:
        return None
    analytic = getattr(adapter, "analytic_jacobian", None)
    if not callable(analytic):
        return None
    expected = analytic(node_id, theta)
    if expected is None:
        return None
    expected_array = np.asarray(expected, dtype=np.float64)
    if expected_array.shape != computation.values.shape:
        return MetricResult.refusal(
            f"analytic_cross_check:{node_id}",
            status=MetricStatus.FAILED,
            analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
            reason_code="analytic_jacobian_shape_mismatch",
        )
    difference = float(np.max(np.abs(computation.values - expected_array)))
    scale = max(float(np.max(np.abs(expected_array))), 1e-12)
    relative = difference / scale
    return MetricResult.ok(
        f"analytic_cross_check:{node_id}",
        relative,
        analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
        units="relative",
        numerical_method="max-abs finite-difference vs analytic",
        tolerances={"relative": 1e-5},
        diagnostics={"max_absolute_difference": difference},
    )


def _distribution_distances(captures: Captures) -> list[MetricResult]:
    experiment = captures.experiment
    results: list[MetricResult] = []
    for node_id in captures.node_ids():
        for case in experiment.cases:
            records: list[tuple[str, DistributionRecord]] = []
            for theta in experiment.theta_points:
                pid = _point_id(theta)
                trace = next(
                    (
                        item
                        for item in captures.traces
                        if item.node_id == node_id
                        and item.case_id == case.id
                        and item.point_id == pid
                        and item.stencil_role == "center"
                        and item.distribution is not None
                    ),
                    None,
                )
                if trace is not None and trace.distribution is not None:
                    records.append((pid, trace.distribution))
            for index in range(len(records) - 1):
                left_id, left = records[index]
                right_id, right = records[index + 1]
                try:
                    left_vector = validate_probabilities(left.support, left.probabilities())
                    right_vector = validate_probabilities(right.support, right.probabilities())
                    left_values, right_values, _ = align_support(left_vector, right_vector)
                except Exception as error:  # noqa: BLE001 - reported as structured refusal
                    results.append(
                        MetricResult.refusal(
                            f"distance:{node_id}:{case.id}:{left_id}->{right_id}",
                            status=MetricStatus.UNSUPPORTED,
                            analysis_object=_distribution_object(left),
                            reason_code=getattr(error, "reason_code", "alignment_failed"),
                            remedy="provide a confirmed support mapping before comparing",
                        )
                    )
                    continue
                analysis_object = _distribution_object(left)
                prefix = f"distance:{node_id}:{case.id}:{left_id}->{right_id}"
                kl = kl_divergence(left_values, right_values)
                if kl.value is None:
                    results.append(
                        MetricResult.refusal(
                            f"{prefix}:kl",
                            status=kl.status,
                            analysis_object=analysis_object,
                            reason_code=kl.reason_code or "boundary_zero_probability",
                            remedy="the divergence is infinite; compare supports explicitly",
                        )
                    )
                else:
                    results.append(
                        _node_metric(f"{prefix}:kl", kl.value, analysis_object=analysis_object)
                    )
                js = js_divergence(left_values, right_values)
                if js.value is not None:
                    results.append(
                        _node_metric(f"{prefix}:js", js.value, analysis_object=analysis_object)
                    )
                hellinger = hellinger_distance(left_values, right_values)
                if hellinger.value is not None:
                    results.append(
                        _node_metric(
                            f"{prefix}:hellinger",
                            hellinger.value,
                            analysis_object=analysis_object,
                            units="hellinger",
                        )
                    )
                fisher_rao = fisher_rao_distance(left_values, right_values)
                if fisher_rao.value is not None:
                    results.append(
                        _node_metric(
                            f"{prefix}:fisher_rao",
                            fisher_rao.value,
                            analysis_object=analysis_object,
                            units="radians",
                            assumptions=[
                                "distance on the full categorical simplex, not an exact "
                                "geodesic inside a restricted parametric family"
                            ],
                        )
                    )
    return results


def _analyze_system(
    *,
    captures: Captures,
    adapter: SystemAdapter | None,
    node_ids: list[str],
    joint_model: JointModel | None,
    aggregation: AggregationMap | None,
    arrays: dict[str, FloatArray],
    diagnostics: list[Diagnostic],
    composition_mode: CompositionMode,
) -> SystemAnalysisDocument:
    experiment = captures.experiment
    system_spec = adapter.describe() if adapter is not None else None
    if joint_model is None:
        reason = (
            "declared_product_pending"
            if composition_mode is CompositionMode.DECLARED_PRODUCT
            else "no_joint_model"
        )
        return SystemAnalysisDocument(
            mode=composition_mode.value,
            node_order=node_ids,
            routing_semantics=system_spec.routing_semantics.value if system_spec else None,
            status=MetricStatus.UNSUPPORTED,
            reason_code=reason,
            assumptions=["system Fisher information requires a declared joint or product model"],
        )
    theta = experiment.center()
    try:
        joint = joint_model.probabilities(theta)
        jacobian = joint_model.jacobian(theta)
    except Exception as error:  # noqa: BLE001 - reported as structured refusal
        return SystemAnalysisDocument(
            mode=composition_mode.value,
            node_order=node_ids,
            status=MetricStatus.FAILED,
            reason_code="joint_evaluation_failed",
            diagnostics=[
                Diagnostic(
                    code="joint_evaluation_failed",
                    message=f"{type(error).__name__}: {error}",
                    severity="error",
                )
            ],
        )
    metrics: list[MetricResult] = []
    if jacobian is None:
        return SystemAnalysisDocument(
            mode=composition_mode.value,
            node_order=list(joint.node_order),
            routing_semantics=system_spec.routing_semantics.value if system_spec else None,
            status=MetricStatus.UNSUPPORTED,
            reason_code="missing_joint_jacobian",
            assumptions=list(joint.assumptions),
            metrics=metrics,
        )
    parameter_names = experiment.parameter_names()
    fisher = joint_fisher(
        np.asarray(joint.probabilities, dtype=np.float64),
        jacobian,
        node_order=joint.node_order,
        parameter_names=parameter_names,
    )
    arrays["joint__fisher"] = fisher.values if fisher.values is not None else np.zeros((0, 0))
    arrays["joint__probabilities"] = np.asarray(joint.probabilities, dtype=np.float64)
    if fisher.values is not None:
        metrics.append(
            _node_metric(
                "system_fisher_trace",
                float(np.trace(fisher.values)),
                analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                coordinates=parameter_names,
                assumptions=list(joint.assumptions),
            )
        )
        for left, right in _pairs(joint.node_order):
            information = mutual_information(joint, left, right)
            metrics.append(
                _node_metric(
                    f"mutual_information:{left}:{right}",
                    information,
                    analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                    assumptions=list(joint.assumptions),
                )
            )
    else:
        metrics.append(
            MetricResult.refusal(
                "system_fisher_trace",
                status=fisher.status,
                analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                reason_code=fisher.reason_code or "system_fisher_undefined",
                remedy="zero-probability outcomes prevent system Fisher information",
            )
        )
    if system_spec is not None and set(joint.node_order) <= set(system_spec.node_ids()):
        from jevometry.systems.graph import SystemGraph

        graph = SystemGraph.from_spec(system_spec)
        visits = graph.visit_probabilities(joint)
        for node_id, probability in sorted(visits.items()):
            metrics.append(
                _node_metric(
                    f"visit_probability:{node_id}",
                    float(probability),
                    analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                    units="probability",
                    assumptions=[
                        "visit probabilities under the declared model",
                        "not observed routing frequencies for a deterministic policy",
                    ],
                )
            )
    information_loss_metrics: list[MetricResult] = []
    if aggregation is not None and fisher.values is not None:
        loss = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=parameter_names,
        )
        if loss.difference is not None:
            arrays["information_loss__difference"] = loss.difference
            information_loss_metrics.append(
                _node_metric(
                    "information_loss_trace",
                    float(np.trace(loss.difference)),
                    analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                    assumptions=loss.assumptions,
                    diagnostics={"psd": loss.psd, "min_eigenvalue": loss.min_eigenvalue},
                )
            )
        else:
            information_loss_metrics.append(
                MetricResult.refusal(
                    "information_loss_trace",
                    status=loss.status,
                    analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                    reason_code=loss.reason_code or "information_loss_undefined",
                    remedy="fixed aggregation mapping and non-zero joint masses are required",
                )
            )
    redundancy: list[RedundancyCase] = []
    node_fisher = _node_fisher_by_name(captures, arrays)
    if fisher.values is not None and node_fisher:
        comparison = compare_independent_sum(
            name="independent_sum_vs_joint",
            description=(
                "sum of node Fisher traces (assumed-independent baseline) versus the "
                "declared joint Fisher trace"
            ),
            node_fisher=node_fisher,
            joint_probabilities=np.asarray(joint.probabilities, dtype=np.float64),
            joint_jacobian=jacobian,
        )
        redundancy.append(
            RedundancyCase(
                name=comparison.name,
                description=comparison.description,
                independent_sum_trace=comparison.independent_sum_trace,
                joint_trace=comparison.joint_trace,
                difference=comparison.difference,
                status=comparison.status,
                reason_code=comparison.reason_code,
                assumptions=comparison.assumptions,
            )
        )
    return SystemAnalysisDocument(
        mode=composition_mode.value,
        node_order=list(joint.node_order),
        routing_semantics=system_spec.routing_semantics.value if system_spec else None,
        action_distribution_available=(
            system_spec is not None
            and system_spec.routing_semantics is RoutingSemantics.SAMPLED_OUTCOME
        ),
        metrics=metrics,
        information_loss=information_loss_metrics,
        redundancy=redundancy,
        status=fisher.status,
        reason_code=fisher.reason_code,
        assumptions=list(joint.assumptions),
    )


def _add_declared_product_system(
    system_document: SystemAnalysisDocument,
    *,
    captures: Captures,
    adapter: SystemAdapter | None,
    node_ids: list[str],
    arrays: dict[str, FloatArray],
) -> None:
    from jevometry.systems.joint import product_joint_from_nodes, product_joint_jacobian

    experiment = captures.experiment
    theta = experiment.center()
    outcome_sets: dict[str, Sequence[str]] = {}
    distributions: dict[str, FloatArray] = {}
    jacobians: dict[str, FloatArray] = {}
    for node_id in node_ids:
        trace = next(
            (
                item
                for item in captures.center_traces()
                if item.node_id == node_id
                and item.case_id == experiment.cases[0].id
                and item.distribution is not None
            ),
            None,
        )
        if trace is None or trace.distribution is None:
            return
        record = trace.distribution
        outcome_sets[node_id] = record.support
        distributions[node_id] = np.asarray(record.probabilities(), dtype=np.float64)
        analytic = getattr(adapter, "analytic_jacobian", None)
        if not callable(analytic):
            return
        matrix = analytic(node_id, theta)
        if matrix is None:
            return
        jacobians[node_id] = np.asarray(matrix, dtype=np.float64)
    parameter_names = experiment.parameter_names()
    try:
        joint = product_joint_from_nodes(
            node_ids,
            outcome_sets,
            distributions,
            assumptions=["declared product model: conditional independence is an assumption"],
            theta=theta,
        )
        jacobian = product_joint_jacobian(
            node_ids, outcome_sets, distributions, jacobians, parameter_names
        )
    except ValueError:
        return
    fisher = joint_fisher(
        np.asarray(joint.probabilities, dtype=np.float64),
        jacobian,
        node_order=node_ids,
        parameter_names=parameter_names,
    )
    if fisher.values is None:
        return
    arrays["declared_product__fisher"] = fisher.values
    system_document.metrics.append(
        _node_metric(
            "declared_product_fisher_trace",
            float(np.trace(fisher.values)),
            analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
            coordinates=parameter_names,
            assumptions=["declared conditional independence and shared theta"],
        )
    )
    system_document.status = MetricStatus.CONDITIONAL
    system_document.reason_code = "assumption_based"
    system_document.assumptions = [
        "declared conditional independence across nodes",
        "shared external theta across nodes",
    ]


def _node_fisher_by_name(
    captures: Captures, arrays: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    center = captures.experiment.center()
    pid = _point_id(center)
    case_id = captures.experiment.cases[0].id
    for node_id in captures.node_ids():
        key = f"fisher__{node_id}__{case_id}__{pid}"
        if key in arrays:
            result[node_id] = arrays[key]
    return result


def _pairs(items: Sequence[str]) -> list[tuple[str, str]]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def _point_id(theta: Mapping[str, float]) -> str:
    from jevometry.experiments.design import point_id

    return point_id(theta)


def budget_tracker_summary(tracker: BudgetTracker) -> dict[str, Any]:
    return tracker.summary()
