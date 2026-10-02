"""Invalid support identity and unavailable derivatives must be refused."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.adapters.analytic import AnalyticAdapter, AnalyticNode, noul_question
from jevometry.analysis import analyze
from jevometry.experiments.acquisition import run_experiment
from jevometry.geometry.derivatives import NodeEvaluation, compute_jacobian
from jevometry.geometry.fisher import fisher_pullback
from jevometry.geometry.simplex import (
    ProbabilityValidationError,
    align_support,
    validate_probabilities,
)
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilKind, StencilSpec


def test_duplicate_outcome_ids_cannot_define_categorical_support() -> None:
    # The masses sum to one, but a repeated identity cannot identify two outcomes.
    with pytest.raises(ProbabilityValidationError) as error:
        validate_probabilities(("red", "red"), (0.3, 0.7))
    assert error.value.reason_code == "duplicate_support"


def test_support_mapping_cannot_merge_distinct_outcomes() -> None:
    left = validate_probabilities(("no", "yes"), (0.3, 0.7))
    right = validate_probabilities(("false", "true"), (0.3, 0.7))
    # Identifying both outcomes with "true" would lose the original event identity.
    with pytest.raises(ProbabilityValidationError) as error:
        align_support(left, right, mapping={"no": "true", "yes": "true"})
    assert error.value.reason_code == "support_mapping_not_injective"


def test_support_mapping_cannot_invent_an_unobserved_target() -> None:
    left = validate_probabilities(("no", "yes"), (0.3, 0.7))
    right = validate_probabilities(("false", "true"), (0.3, 0.7))
    with pytest.raises(ProbabilityValidationError) as error:
        align_support(left, right, mapping={"no": "false", "yes": "unknown"})
    assert error.value.reason_code == "support_mapping_unknown_target"


def test_failed_stencil_evaluation_refuses_even_with_matching_identity() -> None:
    def evaluator(theta: dict[str, float]) -> NodeEvaluation:
        x = theta["x"]
        return NodeEvaluation(
            support=("no", "yes"),
            probabilities=np.asarray([0.5 - 0.1 * x, 0.5 + 0.1 * x]),
            rendered_fingerprint=f"x:{x}",
            semantic_hash="fixed-question",
            model_identity="fixed-model",
            status=MetricStatus.FAILED if x > 0.0 else MetricStatus.OK,
            reason_code="provider_timeout" if x > 0.0 else None,
        )

    result = compute_jacobian(
        evaluator,
        node_id="Y",
        case_id="case",
        point_id="center",
        theta={"x": 0.0},
        parameters=[
            ParameterSpec(name="x", role="task_relevant", unit="u", bounds=(-1.0, 1.0), step=0.1)
        ],
        stencil=StencilSpec(),
    )
    assert result.status is MetricStatus.FAILED
    assert result.reason_code == "stencil_point_failed"
    assert result.values is None
    assert any("provider_timeout" in diagnostic.message for diagnostic in result.diagnostics)


def test_forward_stencil_refuses_when_second_point_is_out_of_bounds() -> None:
    def evaluator(theta: dict[str, float]) -> NodeEvaluation:
        x = theta["x"]
        assert 0.0 <= x <= 0.15
        return NodeEvaluation(
            support=("no", "yes"),
            probabilities=np.asarray([0.5 - 0.1 * x, 0.5 + 0.1 * x]),
            rendered_fingerprint=f"x:{x}",
            semantic_hash="fixed-question",
            model_identity="fixed-model",
        )

    result = compute_jacobian(
        evaluator,
        node_id="Y",
        case_id="case",
        point_id="boundary",
        theta={"x": 0.0},
        parameters=[
            ParameterSpec(name="x", role="task_relevant", unit="u", bounds=(0.0, 0.15), step=0.1)
        ],
        stencil=StencilSpec(kind=StencilKind.FORWARD),
    )
    assert result.status is MetricStatus.CONDITIONAL
    assert result.reason_code == "stencil_unavailable"
    assert result.values is None


@pytest.mark.parametrize("kind", [StencilKind.BACKWARD, StencilKind.CENTRAL])
@pytest.mark.parametrize("quadratic", [0.0, 0.0625], ids=["linear", "quadratic"])
@pytest.mark.parametrize("step", [0.125, 0.0625], ids=["h", "half-h"])
def test_backward_boundary_derivative_and_fisher_match_closed_form(
    kind: StencilKind, quadratic: float, step: float
) -> None:
    # p(x)=1/2 + x/8 + b*x^2. The second-order backward formula is exact
    # for both families; at x=1 the exact derivative is 1/8 + 2b.
    def evaluator(theta: dict[str, float]) -> NodeEvaluation:
        x = theta["x"]
        assert 0.0 <= x <= 1.0
        p = 0.5 + 0.125 * x + quadratic * x**2
        return NodeEvaluation(
            support=("no", "yes"),
            probabilities=np.asarray([1.0 - p, p]),
            rendered_fingerprint=f"x:{x}",
            semantic_hash="fixed-question",
            model_identity="polynomial-bernoulli",
        )

    result = compute_jacobian(
        evaluator,
        node_id="Y",
        case_id="case",
        point_id="upper-boundary",
        theta={"x": 1.0},
        parameters=[
            ParameterSpec(name="x", role="task_relevant", unit="u", bounds=(0.0, 1.0), step=step)
        ],
        stencil=StencilSpec(kind=kind, step_scales=[1.0, 0.5]),
    )
    derivative = 0.125 + 2.0 * quadratic
    assert result.status is MetricStatus.OK
    assert result.reason_code is None
    assert result.values == pytest.approx(np.asarray([[-derivative], [derivative]]))
    assert result.stability_absolute["matrix"] == 0.0
    assert result.stability_relative["matrix"] == 0.0
    assert result.active_support == ("no", "yes")
    assert result.values is not None
    p = 0.5 + 0.125 + quadratic
    fisher = fisher_pullback(result.values, np.asarray([1.0 - p, p]), support=result.support)
    assert fisher.status is MetricStatus.OK
    assert fisher.values == pytest.approx(np.asarray([[derivative**2 / (p * (1.0 - p))]]))


def test_enlarged_stencil_capture_and_derivative_stay_within_bounds() -> None:
    def probabilities(theta: dict[str, float]) -> np.ndarray:
        x = theta["x"]
        assert 0.0 <= x <= 1.0
        return np.asarray([1.0 - x**2, x**2])

    node = AnalyticNode(
        node_id="Y",
        question=noul_question("Y"),
        model_identity="squared-bernoulli",
        probability_function=probabilities,
        jacobian_function=lambda theta: np.asarray([[-2.0 * theta["x"]], [2.0 * theta["x"]]]),
    )
    adapter = AnalyticAdapter(system_id="large-scale", nodes={"Y": node})
    spec = ExperimentSpec(
        id="large-scale",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(name="x", role="task_relevant", unit="u", bounds=(0.0, 1.0), step=0.125)
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 2.0]),
        cases=[CaseSpec(id="case", state="fixed case")],
        theta_points=[{"x": 0.875}],
    )
    captures = run_experiment(spec, adapter)
    assert captures.failures() == []
    assert captures.plan_summary["expected_requests"] == len(captures.traces) == 5
    nominal = {trace.theta["x"] for trace in captures.traces if ":1:" in trace.stencil_role}
    enlarged = {trace.theta["x"] for trace in captures.traces if ":2:" in trace.stencil_role}
    assert nominal == {0.75, 1.0}
    assert enlarged == {0.375, 0.625}
    analysis = analyze(captures, adapter=adapter)
    geometry = analysis.document.nodes[0].geometry
    assert geometry is not None and geometry.jacobian is not None and geometry.fisher is not None
    assert geometry.status is MetricStatus.OK
    assert geometry.jacobian.values == pytest.approx(np.asarray([[-1.75], [1.75]]))
    assert geometry.jacobian.stability_absolute["matrix"] == 0.0
    assert geometry.jacobian.stability_relative["matrix"] == 0.0
    # For Bernoulli p=x^2 the exact Fisher is (2x)^2/[x^2(1-x^2)].
    assert geometry.fisher.values == pytest.approx(np.asarray([[4.0 / (1.0 - 0.875**2)]]))


@pytest.mark.parametrize(
    ("scales", "status", "reason"),
    [
        ([2.0, 1.0], MetricStatus.CONDITIONAL, "stencil_unavailable"),
        ([1.0, 2.0], MetricStatus.UNSTABLE, "stencil_unavailable_at_fine_scale"),
    ],
)
def test_unavailable_enlarged_stencil_is_qualified_without_out_of_bounds_capture(
    scales: list[float], status: MetricStatus, reason: str
) -> None:
    def probabilities(theta: dict[str, float]) -> np.ndarray:
        x = theta["x"]
        assert 0.0 <= x <= 1.0
        p = 0.5 + 0.1 * (x - 0.5)
        return np.asarray([1.0 - p, p])

    node = AnalyticNode(
        node_id="Y",
        question=noul_question("Y"),
        model_identity="bounded-bernoulli",
        probability_function=probabilities,
        jacobian_function=lambda theta: np.asarray([[-0.1], [0.1]]),
    )
    adapter = AnalyticAdapter(system_id="unavailable-scale", nodes={"Y": node})
    spec = ExperimentSpec(
        id="unavailable-scale",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(name="x", role="task_relevant", unit="u", bounds=(0.0, 1.0), step=0.3)
            ]
        ),
        stencil=StencilSpec(step_scales=scales),
        cases=[CaseSpec(id="case", state="fixed case")],
        theta_points=[{"x": 0.5}],
    )
    captures = run_experiment(spec, adapter)
    assert captures.failures() == []
    assert captures.plan_summary["expected_requests"] == len(captures.traces) == 3
    assert all(0.0 <= trace.theta["x"] <= 1.0 for trace in captures.traces)
    assert {trace.theta["x"] for trace in captures.traces} == {0.2, 0.5, 0.8}
    analysis = analyze(captures, adapter=adapter)
    result = analysis.document.nodes[0]
    assert result.status is status
    assert result.reason_code == reason
    if scales[0] == 2.0:
        assert result.geometry is None
        refusal = next(metric for metric in result.metrics if metric.name.startswith("fisher:"))
        assert refusal.value is None
    else:
        assert result.geometry is not None and result.geometry.fisher is not None
        assert result.geometry.fisher.status is status
        assert result.geometry.fisher.reason_code == reason
