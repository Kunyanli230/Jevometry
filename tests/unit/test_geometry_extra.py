"""Additional coverage for distances, Fisher edge cases and derivatives."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.geometry.derivatives import (
    analytic_cross_check,
    compute_jacobian,
    relative_matrix_difference,
    stencil_points,
)
from jevometry.geometry.diagnostics import is_psd, matrix_diagnostics
from jevometry.geometry.distances import (
    js_distance,
    js_divergence,
    kl_divergence,
    kl_matrix,
)
from jevometry.geometry.fisher import fisher_pullback, transform_jacobian
from jevometry.geometry.simplex import validate_probabilities
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.parameters import ParameterSpec, StencilKind, StencilSpec


def parameter(name: str = "theta", step: float = 1e-4) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        role="task_relevant",
        unit="dimensionless",
        bounds=(-10.0, 10.0),
        step=step,
    )


def test_js_distance_is_square_root_of_divergence() -> None:
    p = [0.2, 0.8]
    q = [0.6, 0.4]
    divergence = js_divergence(p, q)
    distance = js_distance(p, q)
    assert divergence.value is not None and distance.value is not None
    assert distance.value == pytest.approx(np.sqrt(divergence.value))
    assert kl_matrix(p, q).value == pytest.approx(kl_divergence(p, q).value)


def test_kl_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1.0])


def test_matrix_diagnostics_reports_rank_and_nullspace() -> None:
    matrix = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    diagnostics = matrix_diagnostics(matrix)
    assert diagnostics.rank == 1
    assert len(diagnostics.null_directions) == 1
    assert diagnostics.condition_number is None
    assert not diagnostics.psd_violation
    assert diagnostics.diagnostics == []


def test_matrix_condition_number_uses_the_rank_threshold() -> None:
    nearly_singular = matrix_diagnostics(np.diag([1.0, 1e-11]))
    assert nearly_singular.rank == 1
    assert nearly_singular.condition_number is None

    full_rank = matrix_diagnostics(np.diag([1.0, 1e-5]))
    assert full_rank.rank == 2
    assert full_rank.condition_number == pytest.approx(1e5)


def test_matrix_diagnostics_flags_negative_eigenvalues() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, -1e-6]])
    diagnostics = matrix_diagnostics(matrix)
    assert diagnostics.psd_violation
    assert diagnostics.min_eigenvalue < 0.0
    assert not is_psd(matrix)


def test_matrix_diagnostics_requires_square() -> None:
    with pytest.raises(ValueError):
        matrix_diagnostics(np.zeros((2, 3)))


def test_stencil_points_enumeration() -> None:
    points = stencil_points(
        {"theta": 0.0}, [parameter()], StencilSpec(step_scales=[1.0, 0.5])
    )
    roles = sorted(point.role for point in points)
    assert set(roles) == {"theta:0.5:-1", "theta:0.5:+1", "theta:1:-1", "theta:1:+1"}
    assert all(point.theta["theta"] != 0.0 for point in points)


def test_forward_stencil_uses_second_order_formula() -> None:
    def evaluator(theta: dict[str, float]):
        from jevometry.geometry.derivatives import NodeEvaluation

        value = float(theta["theta"])
        return NodeEvaluation(
            support=("a", "b"),
            probabilities=np.asarray([0.5 + 0.1 * value, 0.5 - 0.1 * value]),
            rendered_fingerprint=f"f{value}",
            semantic_hash="h",
            model_identity="m",
        )

    result = compute_jacobian(
        evaluator,
        node_id="n",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter(step=1e-3)],
        stencil=StencilSpec(kind=StencilKind.FORWARD, step_scales=[1.0]),
    )
    assert result.values is not None
    assert result.values[0, 0] == pytest.approx(0.1, rel=1e-4)


def test_stencil_failure_marks_failed() -> None:
    def evaluator(theta: dict[str, float]):
        from jevometry.geometry.derivatives import NodeEvaluation

        if theta["theta"] > 0:
            return NodeEvaluation(
                support=("a", "b"),
                probabilities=np.asarray([0.5, 0.5]),
                rendered_fingerprint="x",
                semantic_hash="h",
                model_identity="different-model",
            )
        return NodeEvaluation(
            support=("a", "b"),
            probabilities=np.asarray([0.5, 0.5]),
            rendered_fingerprint="y",
            semantic_hash="h",
            model_identity="m",
        )

    result = compute_jacobian(
        evaluator,
        node_id="n",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter(step=1.0)],
        stencil=StencilSpec(step_scales=[1.0]),
    )
    assert result.status is MetricStatus.FAILED
    assert result.reason_code == "model_identity_changed"


def test_center_evaluation_failure_is_insufficient_data() -> None:
    from jevometry.geometry.derivatives import NodeEvaluation

    def evaluator(theta: dict[str, float]):
        return NodeEvaluation(
            support=("a",),
            probabilities=np.asarray([1.0]),
            rendered_fingerprint="x",
            semantic_hash="h",
            model_identity="m",
            status=MetricStatus.INSUFFICIENT_DATA,
            reason_code="missing_capture",
        )

    result = compute_jacobian(
        evaluator,
        node_id="n",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter()],
        stencil=StencilSpec(),
    )
    assert result.status is MetricStatus.INSUFFICIENT_DATA
    assert result.reason_code == "missing_capture"


def test_fine_scale_failure_is_unstable() -> None:
    calls = {"count": 0}

    def evaluator(theta: dict[str, float]):
        from jevometry.geometry.derivatives import NodeEvaluation

        calls["count"] += 1
        if calls["count"] > 3:
            return NodeEvaluation(
                support=("a", "b"),
                probabilities=np.asarray([0.5, 0.5]),
                rendered_fingerprint="late",
                semantic_hash="h",
                model_identity="different",
            )
        return NodeEvaluation(
            support=("a", "b"),
            probabilities=np.asarray([0.5 + 0.01 * theta["theta"], 0.5 - 0.01 * theta["theta"]]),
            rendered_fingerprint=f"f{theta['theta']}",
            semantic_hash="h",
            model_identity="m",
        )

    result = compute_jacobian(
        evaluator,
        node_id="n",
        case_id="c",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[parameter(step=0.5)],
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
    )
    assert result.status in {MetricStatus.UNSTABLE, MetricStatus.FAILED}
    assert result.reason_code is not None


def test_relative_matrix_difference_floor() -> None:
    left = np.zeros((2, 2))
    right = np.zeros((2, 2))
    relative, absolute = relative_matrix_difference(left, right, floor=1e-6)
    assert relative == 0.0 and absolute == 0.0
    relative, absolute = relative_matrix_difference(left, np.eye(2), floor=1e-6)
    assert absolute == pytest.approx(np.sqrt(2.0))
    assert relative > 0.0


def test_analytic_cross_check_shape_mismatch() -> None:
    agree, residual = analytic_cross_check(np.zeros((2, 1)), np.zeros((2, 2)))
    assert agree is False
    assert residual == float("inf")


def test_fisher_rejects_empty_active_support() -> None:
    jacobian = np.zeros((1, 1))
    probabilities = np.zeros(1)
    result = fisher_pullback(
        jacobian,
        probabilities,
        support=("a",),
        fixed_zero_outcomes=("a",),
        declared_fixed_zero=True,
    )
    assert result.status is MetricStatus.UNDEFINED
    assert result.reason_code == "empty_active_support"


def test_fisher_fixed_zero_mismatch_declaration() -> None:
    jacobian = np.asarray([[-1.0], [1.0]])
    probabilities = np.asarray([0.0, 1.0])
    result = fisher_pullback(
        jacobian,
        probabilities,
        support=("a", "b"),
        fixed_zero_outcomes=(),
        declared_fixed_zero=True,
    )
    assert result.status is MetricStatus.UNSUPPORTED
    assert result.reason_code == "fixed_zero_declaration_mismatch"


def test_transform_jacobian_validates_scales() -> None:
    with pytest.raises(ValueError):
        transform_jacobian(np.zeros((2, 2)), np.asarray([1.0]))


def test_validate_probabilities_shape_check() -> None:
    from jevometry.geometry.simplex import ProbabilityValidationError

    with pytest.raises(ProbabilityValidationError) as error:
        validate_probabilities(("a",), np.asarray([[0.5]]))
    assert error.value.reason_code == "shape"


def test_validated_vector_record_roundtrip() -> None:
    from jevometry.schemas.common import Primitive
    from jevometry.schemas.distribution import DistributionSource

    vector = validate_probabilities(("false", "true"), (0.4, 0.6))
    record = vector.as_record(
        node_id="n",
        question_id="q",
        primitive=Primitive.NOUL,
        source=DistributionSource.DECLARED,
        semantic_hash="sha256:x",
        case_id="c",
        point_id="p",
    )
    assert record.probabilities() == pytest.approx([0.4, 0.6])
    assert record.index_of("true") == 1
