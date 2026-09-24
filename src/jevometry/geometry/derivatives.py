"""Finite-difference derivatives of node output probabilities.

The engine is provider-agnostic: callers supply a deterministic evaluator
``theta -> NodeEvaluation``.  Every stencil point is checked for support,
semantic identity and model identity consistency before any derivative is
reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.schemas.common import Diagnostic, MetricStatus
from jevometry.schemas.parameters import ParameterSpec, StencilKind, StencilSpec

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NodeEvaluation:
    """One deterministic evaluation of a node at one theta point."""

    support: tuple[str, ...]
    probabilities: FloatArray
    rendered_fingerprint: str
    semantic_hash: str
    model_identity: str
    selected: str | None = None
    confidence: float | None = None
    raw_total: float = 1.0
    derived: bool = False
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None


NodeEvaluator = Callable[[Mapping[str, float]], NodeEvaluation]


@dataclass
class JacobianComputation:
    """Finite-difference Jacobian plus all numerical diagnostics."""

    node_id: str
    case_id: str
    point_id: str
    parameter_names: list[str]
    support: tuple[str, ...]
    values: FloatArray | None
    method: str
    stencil_kind: StencilKind
    step_sizes: dict[str, float] = field(default_factory=dict)
    stability_relative: dict[str, float] = field(default_factory=dict)
    stability_absolute: dict[str, float] = field(default_factory=dict)
    row_sum_residuals: dict[str, float] = field(default_factory=dict)
    renderer_resolution_limited: bool = False
    fixed_zero_outcomes: tuple[str, ...] = ()
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    center_probabilities: FloatArray | None = None
    theta: dict[str, float] = field(default_factory=dict)

    @property
    def active_support(self) -> tuple[str, ...]:
        """Support excluding outcomes declared and verified to be fixed zero."""
        if not self.fixed_zero_outcomes:
            return self.support
        zero = set(self.fixed_zero_outcomes)
        return tuple(outcome for outcome in self.support if outcome not in zero)


def _parameter_steps(
    parameters: Sequence[ParameterSpec], stencil: StencilSpec
) -> dict[str, float]:
    return {parameter.name: parameter.step * stencil.step_scales[0] for parameter in parameters}


def _within_bounds(parameter: ParameterSpec, value: float) -> bool:
    return parameter.contains(value)


def _check_consistency(
    center: NodeEvaluation,
    candidate: NodeEvaluation,
    *,
    parameter: str,
    offset: float,
    node_id: str,
) -> Diagnostic | None:
    if candidate.support != center.support:
        return Diagnostic(
            code="support_changed_with_theta",
            message=(
                f"node {node_id}: support changed when moving parameter {parameter} "
                f"by {offset}; refusing derivative on a moving support"
            ),
            severity="error",
        )
    if candidate.semantic_hash != center.semantic_hash:
        return Diagnostic(
            code="semantics_changed_with_theta",
            message=(
                f"node {node_id}: question semantics changed when moving parameter "
                f"{parameter} by {offset}"
            ),
            severity="error",
        )
    if candidate.model_identity != center.model_identity:
        return Diagnostic(
            code="model_identity_changed",
            message=(
                f"node {node_id}: resolved model identity changed from "
                f"{center.model_identity!r} to {candidate.model_identity!r}"
            ),
            severity="error",
        )
    if candidate.status is not MetricStatus.OK:
        return Diagnostic(
            code="stencil_point_failed",
            message=(
                f"node {node_id}: stencil point for {parameter}{offset:+g} failed with "
                f"{candidate.reason_code or candidate.status.value}"
            ),
            severity="error",
        )
    return None


def stencil_directions(
    parameter: ParameterSpec, theta: Mapping[str, float], kind: StencilKind
) -> tuple[float, ...]:
    """Return signed step multiples for the requested stencil orientation."""
    center = theta[parameter.name]
    step = parameter.step
    if kind is StencilKind.CENTRAL and _within_bounds(parameter, center - step) and (
        _within_bounds(parameter, center + step)
    ):
        return (-1.0, 1.0)
    if kind is StencilKind.BACKWARD:
        if _within_bounds(parameter, center - 2.0 * step):
            return (-2.0, -1.0)
        return ()
    if _within_bounds(parameter, center + 2.0 * step):
        return (1.0, 2.0)
    if kind is StencilKind.FORWARD:
        return ()
    if _within_bounds(parameter, center - 2.0 * step):
        return (-2.0, -1.0)
    return ()


@dataclass(frozen=True)
class StencilPoint:
    """One required off-center evaluation for the finite-difference stencil."""

    parameter: str
    scale: float
    multiple: float
    theta: dict[str, float]
    role: str


def stencil_points(
    theta: Mapping[str, float],
    parameters: Sequence[ParameterSpec],
    stencil: StencilSpec,
) -> list[StencilPoint]:
    """Enumerate every off-center evaluation the analysis will require.

    Providers use this to plan and capture exactly the points that the
    derivative engine will later consume.
    """
    points: list[StencilPoint] = []
    for parameter in parameters:
        for scale in stencil.step_scales:
            directions = stencil_directions(parameter, theta, stencil.kind)
            for multiple in directions:
                shifted = dict(theta)
                shifted[parameter.name] = theta[parameter.name] + multiple * parameter.step * scale
                points.append(
                    StencilPoint(
                        parameter=parameter.name,
                        scale=scale,
                        multiple=multiple,
                        theta=shifted,
                        role=f"{parameter.name}:{scale:g}:{multiple:+g}",
                    )
                )
    return points


def compute_jacobian(
    evaluator: NodeEvaluator,
    *,
    node_id: str,
    case_id: str,
    point_id: str,
    theta: Mapping[str, float],
    parameters: Sequence[ParameterSpec],
    stencil: StencilSpec,
) -> JacobianComputation:
    """Compute the Jacobian of ``q(theta)`` with h and h/2 stability checks."""
    center = evaluator(theta)
    names = [parameter.name for parameter in parameters]
    diagnostics: list[Diagnostic] = []
    if center.status is not MetricStatus.OK:
        return JacobianComputation(
            node_id=node_id,
            case_id=case_id,
            point_id=point_id,
            parameter_names=names,
            support=center.support,
            values=None,
            method="finite_difference",
            stencil_kind=stencil.kind,
            status=MetricStatus.INSUFFICIENT_DATA,
            reason_code=center.reason_code or "center_evaluation_failed",
            diagnostics=diagnostics,
            theta=dict(theta),
        )

    nominal_scale = stencil.step_scales[0]
    values = np.zeros((len(center.support), len(parameters)), dtype=np.float64)
    stability_relative: dict[str, float] = {}
    stability_absolute: dict[str, float] = {}
    renderer_resolution_limited = False
    fixed_zero_candidates: set[str] = set(center.support)

    def evaluate_offset(parameter: ParameterSpec, offset: float) -> NodeEvaluation | None:
        shifted = dict(theta)
        shifted[parameter.name] = theta[parameter.name] + offset
        candidate = evaluator(shifted)
        problem = _check_consistency(
            center, candidate, parameter=parameter.name, offset=offset, node_id=node_id
        )
        if problem is not None:
            diagnostics.append(problem)
            return None
        return candidate

    for axis, parameter in enumerate(parameters):
        step = parameter.step * nominal_scale
        directions = stencil_directions(parameter, theta, stencil.kind)
        if not directions:
            diagnostics.append(
                Diagnostic(
                    code="stencil_unavailable",
                    message=(
                        f"node {node_id}: no {stencil.kind.value} stencil inside bounds for "
                        f"parameter {parameter.name} at {theta[parameter.name]!r}"
                    ),
                    severity="warning",
                )
            )
            return JacobianComputation(
                node_id=node_id,
                case_id=case_id,
                point_id=point_id,
                parameter_names=names,
                support=center.support,
                values=None,
                method="finite_difference",
                stencil_kind=stencil.kind,
                status=MetricStatus.CONDITIONAL,
                reason_code="stencil_unavailable",
                diagnostics=diagnostics,
                center_probabilities=center.probabilities,
                theta=dict(theta),
            )
        points: dict[float, NodeEvaluation] = {}
        for multiple in directions:
            candidate = evaluate_offset(parameter, multiple * step)
            if candidate is None:
                return JacobianComputation(
                    node_id=node_id,
                    case_id=case_id,
                    point_id=point_id,
                    parameter_names=names,
                    support=center.support,
                    values=None,
                    method="finite_difference",
                    stencil_kind=stencil.kind,
                    status=MetricStatus.FAILED,
                    reason_code=diagnostics[-1].code,
                    diagnostics=diagnostics,
                    center_probabilities=center.probabilities,
                    theta=dict(theta),
                )
            points[multiple] = candidate
            if candidate.rendered_fingerprint == center.rendered_fingerprint:
                renderer_resolution_limited = True

        if stencil.kind is StencilKind.CENTRAL and -1.0 in points and 1.0 in points:
            plus = points[1.0].probabilities
            minus = points[-1.0].probabilities
            derivative = (plus - minus) / (2.0 * step)
        else:
            ordered = sorted(points)
            far = points[ordered[-1]].probabilities
            near = points[ordered[0]].probabilities
            sign = 1.0 if ordered[-1] > 0 else -1.0
            derivative = sign * (-3.0 * center.probabilities + 4.0 * near - far) / (2.0 * step)
        values[:, axis] = derivative

        for candidate in points.values():
            for index, outcome in enumerate(center.support):
                if candidate.probabilities[index] != 0.0:
                    fixed_zero_candidates.discard(outcome)

    step_sizes = {
        parameter.name: parameter.step * nominal_scale for parameter in parameters
    }
    row_sum_residuals = {
        outcome: float(abs(values[index, :].sum()))
        for index, outcome in enumerate(center.support)
    }

    if len(stencil.step_scales) > 1:
        coarse = values.copy()
        fine_scale = stencil.step_scales[1]
        fine = _jacobian_at_scale(
            evaluator,
            center=center,
            theta=theta,
            parameters=parameters,
            stencil=stencil,
            scale=fine_scale,
            diagnostics=diagnostics,
            node_id=node_id,
        )
        if fine is None:
            status = MetricStatus.UNSTABLE
            fine_reason: str | None = diagnostics[-1].code if diagnostics else "coarse_stencil_failed"
            return JacobianComputation(
                node_id=node_id,
                case_id=case_id,
                point_id=point_id,
                parameter_names=names,
                support=center.support,
                values=coarse,
                method="finite_difference",
                stencil_kind=stencil.kind,
                step_sizes=step_sizes,
                stability_relative=stability_relative,
                stability_absolute=stability_absolute,
                row_sum_residuals=row_sum_residuals,
                renderer_resolution_limited=renderer_resolution_limited,
                status=status,
                reason_code=fine_reason,
                diagnostics=diagnostics,
                center_probabilities=center.probabilities,
                theta=dict(theta),
            )
        difference = coarse - fine
        absolute = float(np.linalg.norm(difference, ord="fro"))
        denominator = max(
            float(np.linalg.norm(coarse, ord="fro")), stencil.matrix_relative_floor
        )
        stability_relative["matrix"] = absolute / denominator
        stability_absolute["matrix"] = absolute
        for axis, parameter in enumerate(parameters):
            column_difference = coarse[:, axis] - fine[:, axis]
            column_absolute = float(np.linalg.norm(column_difference))
            column_denominator = max(
                float(np.linalg.norm(coarse[:, axis])), stencil.matrix_relative_floor
            )
            stability_relative[parameter.name] = column_absolute / column_denominator
            stability_absolute[parameter.name] = column_absolute
        step_sizes = {
            parameter.name: parameter.step * nominal_scale for parameter in parameters
        }

    threshold = stencil.relative_stability_threshold
    unstable = [name for name, value in stability_relative.items() if value > threshold]
    if unstable:
        diagnostics.append(
            Diagnostic(
                code="step_stability_exceeded",
                message=(
                    "h and h/2 relative difference above threshold "
                    f"{threshold!r} for: {', '.join(sorted(unstable))}"
                ),
                severity="warning",
                details={"stability_relative": stability_relative},
            )
        )
    if renderer_resolution_limited:
        diagnostics.append(
            Diagnostic(
                code="resolution_limited",
                message=(
                    f"node {node_id}: renderer maps distinct theta values to the same input; "
                    "derivatives are resolution limited"
                ),
                severity="warning",
            )
        )

    status = MetricStatus.UNSTABLE if unstable else MetricStatus.OK
    reason_code: str | None = "step_stability_exceeded" if unstable else None
    if renderer_resolution_limited and status is MetricStatus.OK:
        status = MetricStatus.CONDITIONAL
        reason_code = "resolution_limited"

    return JacobianComputation(
        node_id=node_id,
        case_id=case_id,
        point_id=point_id,
        parameter_names=names,
        support=center.support,
        values=values,
        method="finite_difference",
        stencil_kind=stencil.kind,
        step_sizes=step_sizes,
        stability_relative=stability_relative,
        stability_absolute=stability_absolute,
        row_sum_residuals=row_sum_residuals,
        renderer_resolution_limited=renderer_resolution_limited,
        fixed_zero_outcomes=tuple(
            outcome for outcome in center.support if outcome in fixed_zero_candidates
        ),
        status=status,
        reason_code=reason_code,
        diagnostics=diagnostics,
        center_probabilities=center.probabilities,
        theta=dict(theta),
    )


def _jacobian_at_scale(
    evaluator: NodeEvaluator,
    *,
    center: NodeEvaluation,
    theta: Mapping[str, float],
    parameters: Sequence[ParameterSpec],
    stencil: StencilSpec,
    scale: float,
    diagnostics: list[Diagnostic],
    node_id: str,
) -> FloatArray | None:
    values = np.zeros((len(center.support), len(parameters)), dtype=np.float64)
    for axis, parameter in enumerate(parameters):
        step = parameter.step * scale
        directions = stencil_directions(parameter, theta, stencil.kind)
        if not directions:
            diagnostics.append(
                Diagnostic(
                    code="stencil_unavailable_at_fine_scale",
                    message=f"no stencil for {parameter.name} at scale {scale!r}",
                    severity="warning",
                )
            )
            return None
        points: dict[float, NodeEvaluation] = {}
        for multiple in directions:
            shifted = dict(theta)
            shifted[parameter.name] = theta[parameter.name] + multiple * step
            candidate = evaluator(shifted)
            problem = _check_consistency(
                center, candidate, parameter=parameter.name, offset=multiple * step, node_id=node_id
            )
            if problem is not None:
                diagnostics.append(problem)
                return None
            points[multiple] = candidate
        if -1.0 in points and 1.0 in points:
            values[:, axis] = (points[1.0].probabilities - points[-1.0].probabilities) / (
                2.0 * step
            )
        else:
            ordered = sorted(points)
            far = points[ordered[-1]].probabilities
            near = points[ordered[0]].probabilities
            sign = 1.0 if ordered[-1] > 0 else -1.0
            values[:, axis] = sign * (
                -3.0 * center.probabilities + 4.0 * near - far
            ) / (2.0 * step)
    return values


def relative_matrix_difference(
    left: FloatArray, right: FloatArray, *, floor: float = 1e-12
) -> tuple[float, float]:
    """Return (relative, absolute) Frobenius difference between two matrices."""
    difference = left - right
    absolute = float(np.linalg.norm(difference, ord="fro"))
    denominator = max(float(np.linalg.norm(left, ord="fro")), floor)
    return absolute / denominator, absolute


def analytic_cross_check(
    finite_difference: FloatArray,
    analytic: FloatArray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[bool, float]:
    """Compare finite-difference and analytic Jacobians."""
    if finite_difference.shape != analytic.shape:
        return False, math.inf
    difference = float(np.max(np.abs(finite_difference - analytic)))
    scale = max(float(np.max(np.abs(analytic))), atol)
    return bool(np.allclose(finite_difference, analytic, rtol=rtol, atol=atol)), difference / scale
