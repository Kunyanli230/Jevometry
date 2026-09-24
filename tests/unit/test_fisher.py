"""Fisher pullback metric: analytic values, coordinates, rank, boundary."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from jevometry.adapters.analytic import (
    logistic_node,
    rank_deficient_softmax_node,
    softmax_node,
)
from jevometry.geometry.coordinates import CoordinateTransform
from jevometry.geometry.derivatives import compute_jacobian
from jevometry.geometry.fisher import fisher_pullback, transform_coordinates
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec


def parameter(name: str, *, step: float = 1e-6, scale: float = 1.0) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        role="task_relevant",
        unit="dimensionless",
        bounds=(-20.0, 20.0),
        step=step,
        scale=scale,
    )


def test_logistic_fisher_in_logit_coordinates() -> None:
    node = logistic_node("y", parameter="theta", coordinate="logit")
    theta = {"theta": 0.8}
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=theta,
        parameters=[parameter("theta")],
        stencil=StencilSpec(),
    )
    assert result.values is not None
    fisher = fisher_pullback(result.values, node.probabilities(theta), support=node.support)
    assert fisher.values is not None
    p = float(expit(0.8))
    assert fisher.values[0, 0] == pytest.approx(p * (1 - p), rel=1e-4)
    assert fisher.sqrt_form_residual is not None
    assert fisher.sqrt_form_residual < 1e-10


def test_logistic_fisher_in_probability_coordinates() -> None:
    node = logistic_node("y", parameter="p", coordinate="probability")
    theta = {"p": 0.3}
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=theta,
        parameters=[parameter("p")],
        stencil=StencilSpec(),
    )
    assert result.values is not None
    fisher = fisher_pullback(result.values, node.probabilities(theta), support=node.support)
    assert fisher.values is not None
    assert fisher.values[0, 0] == pytest.approx(1.0 / (0.3 * 0.7), rel=1e-5)


def test_coordinate_change_matches_direct_computation() -> None:
    node = softmax_node(
        "y",
        outcomes=("a", "b", "c"),
        parameters=("theta1", "theta2"),
        weights=[[1.0, 0.2], [0.1, 1.0], [-0.5, -0.3]],
    )
    theta = {"theta1": 0.2, "theta2": -0.1}
    parameters = [parameter("theta1", scale=2.0), parameter("theta2", scale=0.5)]
    raw = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=theta,
        parameters=parameters,
        stencil=StencilSpec(),
    )
    assert raw.values is not None
    raw_fisher = fisher_pullback(raw.values, node.probabilities(theta), support=node.support)
    assert raw_fisher.values is not None
    transform = CoordinateTransform.from_parameters(parameters)
    transformed = transform.fisher(raw_fisher.values)

    def standardized_evaluator(z: dict[str, float]):
        return node.evaluate(
            {"theta1": z["theta1"] * 2.0, "theta2": z["theta2"] * 0.5}
        )

    z_point = transform.standardize(theta)
    z_parameters = [
        ParameterSpec(
            name="theta1",
            role="task_relevant",
            unit="standardized",
            bounds=(-40.0, 40.0),
            step=parameters[0].step,
        ),
        ParameterSpec(
            name="theta2",
            role="task_relevant",
            unit="standardized",
            bounds=(-40.0, 40.0),
            step=parameters[1].step,
        ),
    ]
    direct = compute_jacobian(
        standardized_evaluator,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=z_point,
        parameters=z_parameters,
        stencil=StencilSpec(),
    )
    assert direct.values is not None
    direct_fisher = fisher_pullback(
        direct.values, node.probabilities(theta), support=node.support
    )
    assert direct_fisher.values is not None
    assert np.allclose(transformed, direct_fisher.values, rtol=1e-4, atol=1e-8)


def test_rank_deficient_fisher_has_null_direction() -> None:
    node = rank_deficient_softmax_node("y")
    theta = {"theta1": 0.3, "theta2": -0.2}
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
    fisher = fisher_pullback(result.values, node.probabilities(theta), support=node.support)
    assert fisher.values is not None
    assert fisher.matrix is not None
    assert fisher.matrix.rank == 1
    assert len(fisher.matrix.null_directions) == 1
    null = fisher.matrix.null_directions[0]
    assert abs(abs(null[0]) - abs(null[1])) < 1e-6
    difference = fisher.values - fisher.values.T
    assert np.max(np.abs(difference)) < 1e-12


def test_fisher_is_psd() -> None:
    node = softmax_node(
        "y",
        outcomes=("a", "b"),
        parameters=("theta",),
        weights=[[1.0], [-1.0]],
    )
    theta = {"theta": 0.4}
    result = compute_jacobian(
        node.evaluate,
        node_id="y",
        case_id="c",
        point_id="p",
        theta=theta,
        parameters=[parameter("theta")],
        stencil=StencilSpec(),
    )
    assert result.values is not None
    fisher = fisher_pullback(result.values, node.probabilities(theta), support=node.support)
    assert fisher.values is not None
    assert fisher.matrix is not None
    assert not fisher.matrix.psd_violation


def test_zero_probability_refuses_fisher() -> None:
    jacobian = np.asarray([[-1.0], [1.0]])
    probabilities = np.asarray([1.0, 0.0])
    result = fisher_pullback(jacobian, probabilities, support=("a", "b"))
    assert result.values is None
    assert result.status is MetricStatus.UNDEFINED
    assert result.reason_code == "zero_probability_outcome"


def test_declared_fixed_zero_uses_active_support() -> None:
    jacobian = np.asarray([[-1.0], [1.0], [0.0]])
    probabilities = np.asarray([0.4, 0.6, 0.0])
    result = fisher_pullback(
        jacobian,
        probabilities,
        support=("a", "b", "c"),
        fixed_zero_outcomes=("c",),
        declared_fixed_zero=True,
    )
    assert result.values is not None
    assert result.active_support == ("a", "b")
    assert result.assumptions
    assert result.values[0, 0] == pytest.approx(1.0 / 0.4 + 1.0 / 0.6)


def test_declared_fixed_zero_with_gradient_is_rejected() -> None:
    jacobian = np.asarray([[-1.0], [1.0], [0.5]])
    probabilities = np.asarray([0.4, 0.6, 0.0])
    result = fisher_pullback(
        jacobian,
        probabilities,
        support=("a", "b", "c"),
        fixed_zero_outcomes=("c",),
        declared_fixed_zero=True,
    )
    assert result.status is MetricStatus.FAILED
    assert result.reason_code == "declared_zero_outcome_has_gradient"


def test_transform_coordinates_scaling() -> None:
    matrix = np.asarray([[2.0, 1.0], [1.0, 3.0]])
    scales = np.asarray([2.0, 0.5])
    transformed = transform_coordinates(matrix, scales)
    expected = np.asarray([[8.0, 1.0], [1.0, 0.75]])
    assert np.allclose(transformed, expected)


def test_coordinate_transform_rejects_wrong_scales() -> None:
    matrix = np.eye(2)
    with pytest.raises(ValueError):
        transform_coordinates(matrix, np.asarray([1.0]))


def test_parameter_set_requires_unique_names() -> None:
    with pytest.raises(ValueError):
        ParameterSet(parameters=[parameter("a"), parameter("a")])
