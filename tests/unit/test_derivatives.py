"""Finite-difference Jacobians against closed-form derivatives."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from jevometry.adapters.analytic import logistic_node, softmax_node
from jevometry.geometry.derivatives import (
    analytic_cross_check,
    compute_jacobian,
)
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.parameters import ParameterSpec, StencilKind, StencilSpec


def parameter(name: str, *, step: float = 1e-5, bounds: tuple[float, float] = (-20.0, 20.0)) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        role="task_relevant",
        unit="dimensionless",
        bounds=bounds,
        step=step,
    )


def test_logistic_derivative_matches_closed_form() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    theta = {"theta": 0.7}
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=theta,
        parameters=[parameter("theta")],
        stencil=StencilSpec(),
    )
    assert result.status is MetricStatus.OK
    assert result.values is not None
    p = float(expit(0.7))
    analytic = np.asarray([[-p * (1 - p)], [p * (1 - p)]])
    agree, residual = analytic_cross_check(result.values, analytic)
    assert agree, residual


def test_softmax_derivative_matches_closed_form() -> None:
    node = softmax_node(
        "y",
        outcomes=("a", "b", "c"),
        parameters=("theta1", "theta2"),
        weights=[[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
    )
    theta = {"theta1": 0.3, "theta2": -0.4}
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=theta,
        parameters=[parameter("theta1"), parameter("theta2")],
        stencil=StencilSpec(),
    )
    assert result.values is not None
    analytic = node.jacobian(theta)
    assert analytic is not None
    agree, residual = analytic_cross_check(result.values, analytic)
    assert agree, residual


def test_step_stability_reports_h_and_half() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter("theta")],
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
    )
    assert "matrix" in result.stability_relative
    assert "matrix" in result.stability_absolute
    assert result.stability_relative["matrix"] < 1e-6


def test_unstable_when_step_too_large() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter("theta", step=3.0)],
        stencil=StencilSpec(step_scales=[1.0, 0.5], relative_stability_threshold=1e-4),
    )
    assert result.status is MetricStatus.UNSTABLE
    assert result.reason_code == "step_stability_exceeded"


def test_one_sided_stencil_at_boundary() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta={"theta": 19.99999},
        parameters=[parameter("theta", step=1e-5, bounds=(0.0, 20.0))],
        stencil=StencilSpec(),
    )
    assert result.values is not None
    p = float(expit(19.99999))
    analytic = np.asarray([[-p * (1 - p)], [p * (1 - p)]])
    agree, residual = analytic_cross_check(result.values, analytic, rtol=1e-3, atol=1e-10)
    assert agree, residual


def test_renderer_resolution_limit_is_detected() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    original = node.evaluate

    def rounded(theta: dict[str, float]):
        evaluation = original(theta)
        rounded_theta = {key: round(value, 3) for key, value in theta.items()}
        return type(evaluation)(
            support=evaluation.support,
            probabilities=evaluation.probabilities,
            rendered_fingerprint=f"rounded:{rounded_theta['theta']}",
            semantic_hash=evaluation.semantic_hash,
            model_identity=evaluation.model_identity,
        )

    result = compute_jacobian(
        rounded,
        node_id="y",
        case_id="c",
        point_id="p",
        theta={"theta": 0.5},
        parameters=[parameter("theta", step=1e-4)],
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
    )
    assert result.renderer_resolution_limited is True
    assert result.status is MetricStatus.CONDITIONAL
    assert result.reason_code == "resolution_limited"


def test_support_change_with_theta_is_rejected() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")

    def changing(theta: dict[str, float]):
        evaluation = node.evaluate(theta)
        if theta["theta"] > 0.5:
            return type(evaluation)(
                support=("false", "true", "extra"),
                probabilities=np.asarray([*evaluation.probabilities, 0.0]),
                rendered_fingerprint="changed",
                semantic_hash=evaluation.semantic_hash,
                model_identity=evaluation.model_identity,
            )
        return evaluation

    result = compute_jacobian(
        changing,
        node_id="y",
        case_id="c",
        point_id="p",
        theta={"theta": 0.5},
        parameters=[parameter("theta", step=1.0)],
        stencil=StencilSpec(step_scales=[1.0]),
    )
    assert result.status is MetricStatus.FAILED
    assert result.reason_code == "support_changed_with_theta"


def test_stencil_unavailable_outside_bounds_is_conditional() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter("theta", step=1.0, bounds=(0.0, 0.5))],
        stencil=StencilSpec(kind=StencilKind.FORWARD),
    )
    assert result.status is MetricStatus.CONDITIONAL
    assert result.reason_code == "stencil_unavailable"


def test_fixed_zero_outcomes_are_tracked() -> None:
    def evaluator(theta: dict[str, float]):
        from jevometry.geometry.derivatives import NodeEvaluation

        p = float(theta["p"])
        return NodeEvaluation(
            support=("a", "b", "c"),
            probabilities=np.asarray([p, 1.0 - p, 0.0]),
            rendered_fingerprint=f"f{p}",
            semantic_hash="hash",
            model_identity="model",
        )

    result = compute_jacobian(
        evaluator,
        node_id="n",
        case_id="c",
        point_id="p",
        theta={"p": 0.4},
        parameters=[parameter("p", step=1e-6, bounds=(0.0, 1.0))],
        stencil=StencilSpec(),
    )
    assert result.fixed_zero_outcomes == ("c",)
