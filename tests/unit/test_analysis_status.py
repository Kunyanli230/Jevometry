"""Analysis statuses follow numerical evidence, including downstream uses.

The cubic Bernoulli fixture has p(x) = 1/2 + ax + bx^3. At x = 0 its
central-difference derivative is a + bh^2, while its exact derivative is a.
These independent expressions distinguish diagnostic values from valid Fisher.
"""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.adapters.analytic import (
    AnalyticAdapter,
    AnalyticNode,
    bernoulli_node,
    noul_question,
    single_node_joint_model,
)
from jevometry.analysis import Analysis, analyze
from jevometry.experiments.acquisition import run_experiment
from jevometry.schemas.common import CapabilityEntry, MetricStatus
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.schemas.system import CompositionMode


def cubic_node(name: str = "Y", *, cubic: float = 1.0) -> AnalyticNode:
    def probabilities(theta: dict[str, float]) -> np.ndarray:
        x = theta["x"]
        p = 0.5 + 0.1 * x + cubic * x**3
        return np.asarray([1.0 - p, p])

    def jacobian(theta: dict[str, float]) -> np.ndarray:
        derivative = 0.1 + 3.0 * cubic * theta["x"] ** 2
        return np.asarray([[-derivative], [derivative]])

    return AnalyticNode(
        node_id=name,
        question=noul_question(name),
        model_identity="cubic-bernoulli",
        probability_function=probabilities,
        jacobian_function=jacobian,
    )


def cubic_spec() -> ExperimentSpec:
    return ExperimentSpec(
        id="status",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="x",
                    role="task_relevant",
                    unit="dimensionless",
                    bounds=(-0.5, 0.5),
                    step=0.2,
                    scale=2.0,
                    center=0.0,
                )
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 0.5], relative_stability_threshold=0.1),
        cases=[CaseSpec(id="case", state="fixed input")],
        theta_points=[{"x": 0.0}],
    )


def capability(analysis: Analysis, name: str) -> CapabilityEntry:
    return next(
        entry for entry in analysis.document.capability_matrix.entries if entry.capability == name
    )


def test_unstable_jacobian_does_not_produce_ok_fisher() -> None:
    adapter = AnalyticAdapter(system_id="cubic", nodes={"Y": cubic_node()})
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    node = analysis.document.nodes[0]
    geometry = node.geometry
    assert geometry is not None and geometry.jacobian is not None and geometry.fisher is not None
    coarse_derivative = 0.1 + 0.2**2
    fine_derivative = 0.1 + 0.1**2
    assert geometry.jacobian.values == pytest.approx(np.asarray([[-coarse_derivative], [coarse_derivative]]))
    assert geometry.jacobian.stability_relative["matrix"] == pytest.approx(
        (coarse_derivative - fine_derivative) / coarse_derivative
    )
    diagnostic_fisher = coarse_derivative**2 / (0.5 * 0.5)
    assert geometry.fisher.values == pytest.approx(np.asarray([[diagnostic_fisher]]))
    for result in (node, geometry, geometry.jacobian, geometry.fisher):
        assert result.status is MetricStatus.UNSTABLE
        assert result.reason_code == "step_stability_exceeded"
    fisher_metrics = [metric for metric in node.metrics if metric.name.startswith("fisher_")]
    assert len(fisher_metrics) == 3
    for metric in fisher_metrics:
        assert metric.status is MetricStatus.UNSTABLE
        assert metric.reason_code == "step_stability_exceeded"
        assert metric.diagnostics["diagnostic_only"] is True
        assert metric.value is not None
    standardized = next(metric for metric in fisher_metrics if "standardized" in metric.name)
    assert standardized.value == pytest.approx(4.0 * diagnostic_fisher)
    entry = capability(analysis, "node_geometry")
    assert entry.status is MetricStatus.UNSTABLE
    assert entry.reason_code == "step_stability_exceeded"
    # The comparison residual is computable, without certifying the source geometry.
    comparison = next(metric for metric in node.metrics if metric.name.startswith("analytic_cross_check:"))
    assert comparison.status is MetricStatus.OK
    assert comparison.value == pytest.approx((coarse_derivative - 0.1) / 0.1)
    assert comparison.diagnostics["jacobian_status"] == "unstable"
    assert comparison.diagnostics["within_tolerance"] is False
    assert comparison.assumptions


def test_renderer_resolution_status_propagates_to_fisher() -> None:
    adapter = AnalyticAdapter(system_id="affine", nodes={"Y": cubic_node(cubic=0.0)})
    captures = run_experiment(cubic_spec(), adapter)
    for trace in captures.traces:
        trace.rendered_fingerprint = "rounded-renderer:0"
    analysis = analyze(captures, adapter=adapter)
    geometry = analysis.document.nodes[0].geometry
    assert geometry is not None and geometry.fisher is not None and geometry.jacobian is not None
    assert geometry.fisher.values == pytest.approx(np.asarray([[0.1**2 / 0.25]]))
    for result in (geometry, geometry.fisher, geometry.jacobian):
        assert result.status is MetricStatus.CONDITIONAL
        assert result.reason_code == "resolution_limited"
    for metric in geometry.metrics:
        if metric.name.startswith("fisher_"):
            assert metric.status is MetricStatus.CONDITIONAL
            assert metric.reason_code == "resolution_limited"
    entry = capability(analysis, "node_geometry")
    assert entry.status is MetricStatus.CONDITIONAL
    assert entry.reason_code == "resolution_limited"


def test_resolution_limited_rank_does_not_certify_nonidentifiability() -> None:
    node = cubic_node(cubic=0.0)
    node.probability_function = lambda theta: np.asarray([0.5, 0.5])
    node.jacobian_function = lambda theta: np.asarray([[0.0], [0.0]])
    adapter = AnalyticAdapter(system_id="constant", nodes={"Y": node})
    captures = run_experiment(cubic_spec(), adapter)
    for trace in captures.traces:
        trace.rendered_fingerprint = "rounded-renderer:0"
    analysis = analyze(captures, adapter=adapter)
    metric = next(
        metric
        for metric in analysis.document.nodes[0].metrics
        if metric.name.startswith("fisher_identifiability:")
    )
    assert metric.status is MetricStatus.CONDITIONAL
    assert metric.reason_code == "resolution_limited"
    assert metric.value is None
    assert metric.diagnostics["rank"] == 0


@pytest.mark.parametrize("only_fine_missing", [False, True])
def test_missing_stencil_reports_missing_captures_not_instability(only_fine_missing: bool) -> None:
    adapter = AnalyticAdapter(system_id="cubic", nodes={"Y": cubic_node()})
    captures = run_experiment(cubic_spec(), adapter, include_stencil=only_fine_missing)
    if only_fine_missing:
        captures.traces = [trace for trace in captures.traces if ":0.5:" not in trace.stencil_role]
    analysis = analyze(captures, adapter=adapter)
    entry = capability(analysis, "node_geometry")
    assert entry.status is MetricStatus.INSUFFICIENT_DATA
    assert entry.reason_code == "missing_stencil_captures"
    node = analysis.document.nodes[0]
    assert node.geometry is None
    refusal = next(metric for metric in node.metrics if metric.name.startswith("fisher:"))
    assert refusal.status is MetricStatus.INSUFFICIENT_DATA
    assert refusal.reason_code == "missing_stencil_captures"
    assert refusal.value is None
    assert not any(key.startswith("fisher__") for key in analysis.arrays)


def test_support_change_reports_real_failure_with_complete_stencil() -> None:
    adapter = AnalyticAdapter(system_id="affine", nodes={"Y": cubic_node(cubic=0.0)})
    captures = run_experiment(cubic_spec(), adapter)
    shifted = next(trace for trace in captures.traces if trace.theta["x"] == 0.2)
    assert shifted.distribution is not None
    shifted.distribution.support = ["changed-false", "true"]
    analysis = analyze(captures, adapter=adapter)
    entry = capability(analysis, "node_geometry")
    assert entry.status is MetricStatus.FAILED
    assert entry.reason_code == "support_changed_with_theta"


def test_mixed_geometry_is_conditional_and_retains_per_node_reasons() -> None:
    adapter = AnalyticAdapter(
        system_id="mixed",
        nodes={"stable": cubic_node("stable", cubic=0.0), "unstable": cubic_node("unstable")},
    )
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    nodes = {node.node_id: node for node in analysis.document.nodes}
    assert nodes["stable"].status is MetricStatus.OK
    assert nodes["unstable"].status is MetricStatus.UNSTABLE
    entry = capability(analysis, "node_geometry")
    assert entry.status is MetricStatus.CONDITIONAL
    assert entry.reason_code == "mixed_node_geometry"


def test_independent_analytic_joint_stays_ok_but_unstable_baseline_is_refused() -> None:
    node = cubic_node()
    adapter = AnalyticAdapter(
        system_id="joint",
        nodes={"Y": node},
        joint_model=single_node_joint_model(node),
    )
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    system = analysis.document.system
    assert system is not None
    assert system.status is MetricStatus.OK
    system_trace = next(metric for metric in system.metrics if metric.name == "system_fisher_trace")
    assert system_trace.value == pytest.approx(0.1**2 / 0.25)
    assert system_trace.status is MetricStatus.OK
    assert capability(analysis, "system_information").status is MetricStatus.OK
    assert len(system.redundancy) == 1
    comparison = system.redundancy[0]
    assert comparison.status is MetricStatus.UNSTABLE
    assert comparison.reason_code == "step_stability_exceeded"
    assert comparison.independent_sum_trace is None
    assert comparison.difference is None
    assert comparison.joint_trace == pytest.approx(0.1**2 / 0.25)
    assert comparison.diagnostics[0].details["nodes"][0]["node_id"] == "Y"


def test_missing_geometry_cannot_form_partial_independent_sum() -> None:
    node = cubic_node("Y", cubic=0.0)
    adapter = AnalyticAdapter(
        system_id="partial",
        nodes={"Y": node, "Z": cubic_node("Z", cubic=0.0)},
        joint_model=single_node_joint_model(node),
    )
    captures = run_experiment(cubic_spec(), adapter)
    captures.traces = [
        trace for trace in captures.traces if trace.node_id != "Z" or trace.stencil_role == "center"
    ]
    analysis = analyze(captures, adapter=adapter)
    assert analysis.document.system is not None
    comparison = analysis.document.system.redundancy[0]
    assert comparison.status is MetricStatus.INSUFFICIENT_DATA
    assert comparison.reason_code == "missing_stencil_captures"
    assert comparison.independent_sum_trace is None and comparison.difference is None


def test_declared_product_analytic_value_has_assumption_status() -> None:
    adapter = AnalyticAdapter(
        system_id="product",
        nodes={"Y": cubic_node("Y"), "Z": cubic_node("Z")},
    )
    adapter.describe().composition_mode = CompositionMode.DECLARED_PRODUCT
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    system = analysis.document.system
    assert system is not None
    assert system.status is MetricStatus.CONDITIONAL
    assert system.reason_code == "assumption_based"
    trace = next(metric for metric in system.metrics if metric.name == "declared_product_fisher_trace")
    assert trace.value == pytest.approx(2.0 * 0.1**2 / 0.25)
    assert trace.status is MetricStatus.CONDITIONAL
    assert trace.reason_code == "assumption_based"
    assert capability(analysis, "system_information").reason_code == "assumption_based"


def test_missing_joint_jacobian_is_not_an_ok_system_capability() -> None:
    node = cubic_node(cubic=0.0)
    node.jacobian_function = None
    adapter = AnalyticAdapter(
        system_id="no-joint-derivative", nodes={"Y": node}, joint_model=single_node_joint_model(node)
    )
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    entry = capability(analysis, "system_information")
    assert entry.status is MetricStatus.UNSUPPORTED
    assert entry.reason_code == "missing_joint_jacobian"
    assert "joint__fisher" not in analysis.arrays


def test_joint_evaluation_failure_returns_structured_result() -> None:
    node = cubic_node(cubic=0.0)
    joint = single_node_joint_model(node)
    evaluations = 0

    def unavailable(theta: dict[str, float]) -> tuple[list[tuple[str, ...]], np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        raise ValueError("declared joint model is unavailable")

    joint.probability_function = unavailable
    adapter = AnalyticAdapter(system_id="unavailable-joint", nodes={"Y": node}, joint_model=joint)
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    entry = capability(analysis, "system_information")
    assert entry.status is MetricStatus.FAILED
    assert entry.reason_code == "joint_evaluation_failed"
    assert analysis.joint is None
    assert evaluations == 1


@pytest.mark.parametrize("declared_product", [False, True])
def test_undefined_joint_fisher_never_creates_a_zero_array(declared_product: bool) -> None:
    node = cubic_node(cubic=0.0)
    node.probability_function = lambda theta: np.asarray([1.0, 0.0])
    node.jacobian_function = lambda theta: np.asarray([[0.0], [0.0]])
    adapter = AnalyticAdapter(
        system_id="zero-joint",
        nodes={"Y": node},
        joint_model=None if declared_product else single_node_joint_model(node),
    )
    if declared_product:
        adapter.describe().composition_mode = CompositionMode.DECLARED_PRODUCT
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    entry = capability(analysis, "system_information")
    assert entry.status is MetricStatus.UNDEFINED
    assert entry.reason_code == "zero_probability_outcome"
    assert "joint__fisher" not in analysis.arrays
    assert "declared_product__fisher" not in analysis.arrays
    assert analysis.document.system is not None
    refusal = next(metric for metric in analysis.document.system.metrics if "fisher_trace" in metric.name)
    assert refusal.status is MetricStatus.UNDEFINED
    assert refusal.value is None


def test_declared_product_missing_distribution_reports_actual_reason() -> None:
    adapter = AnalyticAdapter(
        system_id="missing-product-node",
        nodes={"Y": cubic_node("Y", cubic=0.0), "Z": cubic_node("Z", cubic=0.0)},
    )
    adapter.describe().composition_mode = CompositionMode.DECLARED_PRODUCT
    captures = run_experiment(cubic_spec(), adapter)
    captures.traces = [trace for trace in captures.traces if trace.node_id != "Z"]
    analysis = analyze(captures, adapter=adapter)
    entry = capability(analysis, "system_information")
    assert entry.status is MetricStatus.INSUFFICIENT_DATA
    assert entry.reason_code == "missing_node_distribution"


def test_declared_product_missing_analytic_derivative_reports_actual_reason() -> None:
    node = cubic_node(cubic=0.0)
    node.jacobian_function = None
    adapter = AnalyticAdapter(system_id="missing-product-derivative", nodes={"Y": node})
    adapter.describe().composition_mode = CompositionMode.DECLARED_PRODUCT
    analysis = analyze(run_experiment(cubic_spec(), adapter), adapter=adapter)
    entry = capability(analysis, "system_information")
    assert entry.status is MetricStatus.UNSUPPORTED
    assert entry.reason_code == "missing_analytic_node_jacobian"


def test_invalid_declared_product_derivative_reports_evaluation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AnalyticAdapter(system_id="invalid-product", nodes={"Y": cubic_node(cubic=0.0)})
    adapter.describe().composition_mode = CompositionMode.DECLARED_PRODUCT
    captures = run_experiment(cubic_spec(), adapter)
    # The declared derivative has two coordinates while the experiment declares one.
    monkeypatch.setattr(adapter, "analytic_jacobian", lambda node, theta: np.ones((2, 2)))
    analysis = analyze(captures, adapter=adapter)
    entry = capability(analysis, "system_information")
    assert entry.status is MetricStatus.FAILED
    assert entry.reason_code == "declared_product_evaluation_failed"
    assert "declared_product__fisher" not in analysis.arrays


def test_stable_bernoulli_status_and_closed_form_are_unchanged() -> None:
    spec = cubic_spec()
    spec.parameter_set.parameters[0].name = "p"
    spec.parameter_set.parameters[0].bounds = (0.0, 1.0)
    spec.parameter_set.parameters[0].step = 1e-4
    spec.parameter_set.parameters[0].center = 0.3
    spec.theta_points = [{"p": 0.3}]
    node = bernoulli_node("Y", parameter="p")
    adapter = AnalyticAdapter(
        system_id="stable", nodes={"Y": node}, joint_model=single_node_joint_model(node)
    )
    analysis = analyze(run_experiment(spec, adapter), adapter=adapter)
    geometry = analysis.document.nodes[0].geometry
    assert geometry is not None and geometry.fisher is not None
    assert geometry.fisher.status is MetricStatus.OK
    assert geometry.fisher.values == pytest.approx(np.asarray([[1.0 / (0.3 * 0.7)]]))
    for metric in geometry.metrics:
        assert metric.status is MetricStatus.OK
    assert capability(analysis, "node_geometry").status is MetricStatus.OK
    assert analysis.document.system is not None
    assert analysis.document.system.redundancy[0].status is MetricStatus.OK
    assert analysis.document.system.redundancy[0].difference == pytest.approx(0.0, abs=1e-10)
