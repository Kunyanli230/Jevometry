"""Regression tests for the statistical and numerical contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jevometry.adapters.analytic import AnalyticAdapter, bernoulli_node
from jevometry.adapters.base import ExperimentPoint
from jevometry.analysis import TraceEvaluator, analyze
from jevometry.artifacts.integrity import hash_file, verify_checksums
from jevometry.artifacts.store import RunStore
from jevometry.experiments.acquisition import run_experiment
from jevometry.geometry.derivatives import NodeEvaluation, compute_jacobian
from jevometry.geometry.simplex import (
    ProbabilityValidationError,
    align_support,
    validate_probabilities,
)
from jevometry.pipeline import Experiment
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.distribution import DistributionRecord, DistributionSource
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.schemas.questions import OutcomeSpec, QuestionSpec
from jevometry.schemas.trace import EvaluationTrace, ProviderStatus
from jevometry.systems.joint import (
    joint_fisher,
    mutual_information,
    product_joint_from_nodes,
    product_joint_jacobian,
)


def bernoulli_spec(experiment_id: str = "regression") -> ExperimentSpec:
    return ExperimentSpec(
        id=experiment_id,
        model="analytic-bernoulli",
        adapter="analytic",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="p",
                    role="task_relevant",
                    unit="probability",
                    bounds=(0.0, 1.0),
                    step=1e-4,
                    center=0.3,
                )
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
        cases=[CaseSpec(id="case", state="fixture")],
        theta_points=[{"p": 0.3}],
    )


def test_support_alignment_is_reorder_invariant() -> None:
    left = validate_probabilities(("a", "b"), (0.4, 0.6))
    reordered = validate_probabilities(("b", "a"), (0.6, 0.4))
    values_a, values_b, support = align_support(
        left, reordered, mapping={"a": "a", "b": "b"}
    )
    assert support == ("a", "b")
    assert values_a == pytest.approx(values_b)
    with pytest.raises(ProbabilityValidationError):
        align_support(left, reordered)


def test_semantics_change_rejected_by_derivative_engine() -> None:
    node = bernoulli_node("Y", parameter="p")
    original = node.evaluate

    def changed(theta: dict[str, float]) -> NodeEvaluation:
        evaluation = original(theta)
        semantic = "different-rubric" if theta["p"] > 0.3 else evaluation.semantic_hash
        return NodeEvaluation(
            support=evaluation.support,
            probabilities=evaluation.probabilities,
            rendered_fingerprint=evaluation.rendered_fingerprint,
            semantic_hash=semantic,
            model_identity=evaluation.model_identity,
        )

    result = compute_jacobian(
        changed,
        node_id="Y",
        case_id="case",
        point_id="p",
        theta={"p": 0.3},
        parameters=[
            ParameterSpec(
                name="p", role="task_relevant", unit="u", bounds=(0, 1), step=1e-3
            )
        ],
        stencil=StencilSpec(step_scales=[1.0]),
    )
    assert result.status is MetricStatus.FAILED
    assert result.reason_code == "semantics_changed_with_theta"


def test_quantised_provider_output_is_resolution_limited() -> None:
    def evaluator(theta: dict[str, float]) -> NodeEvaluation:
        p = 0.5 + 0.1 * float(theta["theta"])
        quantised = round(p * 4.0) / 4.0
        return NodeEvaluation(
            support=("false", "true"),
            probabilities=np.asarray([1.0 - quantised, quantised]),
            rendered_fingerprint=f"quantised:{quantised}",
            semantic_hash="hash",
            model_identity="staircase",
        )

    result = compute_jacobian(
        evaluator,
        node_id="Y",
        case_id="case",
        point_id="p",
        theta={"theta": 0.0},
        parameters=[
            ParameterSpec(
                name="theta", role="task_relevant", unit="u", bounds=(-5, 5), step=0.001
            )
        ],
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
    )
    assert result.renderer_resolution_limited is True
    assert result.status is MetricStatus.CONDITIONAL
    assert result.reason_code == "resolution_limited"


def test_declared_product_joint_equals_sum_of_node_information() -> None:
    p = 0.3
    probabilities = np.asarray([1.0 - p, p])
    jacobian = np.asarray([[-1.0], [1.0]])
    outcome_sets = {"Y": ("0", "1"), "Z": ("0", "1")}
    distributions = {"Y": probabilities, "Z": probabilities}
    jacobians = {"Y": jacobian, "Z": jacobian}
    joint = product_joint_from_nodes(
        ["Y", "Z"],
        outcome_sets,
        distributions,
        assumptions=["declared independence"],
        theta={"p": p},
    )
    joint_jac = product_joint_jacobian(
        ["Y", "Z"], outcome_sets, distributions, jacobians, ["p"]
    )
    fisher = joint_fisher(
        np.asarray(joint.probabilities), joint_jac, node_order=["Y", "Z"], parameter_names=["p"]
    )
    assert fisher.values is not None
    single = float((jacobian.T @ np.diag(1.0 / probabilities) @ jacobian)[0, 0])
    assert fisher.values[0, 0] == pytest.approx(2.0 * single, rel=1e-10)


def test_node_only_system_refuses_information_sum() -> None:
    adapter = AnalyticAdapter(system_id="single", nodes={"Y": bernoulli_node("Y")})
    captures = run_experiment(bernoulli_spec(), adapter)
    analysis = analyze(captures, adapter=adapter)
    system = analysis.document.system
    assert system is not None
    assert system.status is MetricStatus.UNSUPPORTED
    assert system.reason_code == "no_joint_model"
    capability = {
        entry.capability: entry.status
        for entry in analysis.document.capability_matrix.entries
    }
    assert capability["system_information"] is MetricStatus.UNSUPPORTED


def test_same_marginals_different_joints_have_different_information() -> None:
    independent = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["X", "Y"],
        outcomes=[("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")],
        probabilities=[0.25, 0.25, 0.25, 0.25],
    )
    correlated = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["X", "Y"],
        outcomes=[("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")],
        probabilities=[0.4, 0.1, 0.1, 0.4],
    )
    assert independent.marginal("X") == correlated.marginal("X")
    assert independent.marginal("Y") == correlated.marginal("Y")
    assert mutual_information(independent, "X", "Y") == pytest.approx(0.0, abs=1e-12)
    expected = 0.8 * np.log(0.4 / 0.25) + 0.2 * np.log(0.1 / 0.25)
    assert mutual_information(correlated, "X", "Y") == pytest.approx(float(expected))


def test_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    adapter = AnalyticAdapter(system_id="single", nodes={"Y": bernoulli_node("Y")})
    experiment = Experiment.from_spec(bernoulli_spec())
    experiment.run(adapter, output=tmp_path / "run")
    assert verify_checksums(tmp_path / "run") == []
    target = tmp_path / "run" / "traces.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert "traces.jsonl" in verify_checksums(tmp_path / "run")


def test_manifest_hash_is_stable_across_reload(tmp_path: Path) -> None:
    adapter = AnalyticAdapter(system_id="single", nodes={"Y": bernoulli_node("Y")})
    experiment = Experiment.from_spec(bernoulli_spec())
    experiment.run(adapter, output=tmp_path / "run")
    first = RunStore.load(tmp_path / "run")
    second = RunStore.load(tmp_path / "run")
    assert first.manifest is not None and second.manifest is not None
    assert first.manifest.experiment_hash == second.manifest.experiment_hash
    assert hash_file(tmp_path / "run" / "manifest.json") == hash_file(
        tmp_path / "run" / "manifest.json"
    )


def test_report_escapes_external_text(tmp_path: Path) -> None:
    from jevometry.reporting import render_html

    question = QuestionSpec(
        id="q",
        primitive="choice",
        outcomes=[
            OutcomeSpec(id="<script>alert(1)</script>", description="<b>x</b>"),
            OutcomeSpec(id="safe", description="ok"),
        ],
    )
    record = DistributionRecord(
        node_id="Y",
        question_id="q",
        primitive="choice",
        support=["<script>alert(1)</script>", "safe"],
        raw=[0.5, 0.5],
        raw_total=1.0,
        source=DistributionSource.REPORTED,
        semantic_hash=question.semantic_hash(),
        case_id="case",
        point_id="p",
    )
    from jevometry.analysis import Analysis
    from jevometry.schemas.results import AnalysisDocument, NodeAnalysis

    document = AnalysisDocument(
        run_id="r",
        analysis_revision=1,
        created_utc="2026-09-24T00:00:00Z",
        experiment_id="e",
        nodes=[
            NodeAnalysis(
                node_id="Y",
                question_id="q",
                case_id="case",
                point_id="p",
                theta={"p": 0.3},
                support=record.support,
                distribution=record,
            )
        ],
    )
    captures = run_experiment(
        bernoulli_spec(), AnalyticAdapter(system_id="single", nodes={"Y": bernoulli_node("Y")})
    )
    analysis = Analysis(document=document, captures=captures)
    html = render_html(analysis)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html or "\\u003cscript\\u003e" in html


def test_missing_stencil_capture_is_not_reported_as_zero() -> None:
    adapter = AnalyticAdapter(system_id="single", nodes={"Y": bernoulli_node("Y")})
    experiment = Experiment.from_spec(bernoulli_spec())
    captures = experiment.run(adapter, include_stencil=False)
    analysis = analyze(captures, adapter=adapter)
    assert not any(
        metric.name.startswith("fisher_trace:")
        for node in analysis.document.nodes
        for metric in node.metrics
    )
    refusals = [
        metric
        for node in analysis.document.nodes
        for metric in node.metrics
        if metric.status is not MetricStatus.OK
    ]
    assert refusals
    for metric in refusals:
        assert metric.value is None
        assert metric.reason_code


def test_trace_evaluator_reports_missing_capture() -> None:
    evaluator = TraceEvaluator([], node_id="Y", case_id="case", point_id="p", repeat=0)
    evaluation = evaluator({"p": 0.3})
    assert evaluation.status is MetricStatus.INSUFFICIENT_DATA
    assert evaluation.reason_code == "missing_capture"


def test_reported_distribution_is_not_an_independent_draw() -> None:
    trace = EvaluationTrace(
        experiment_id="e",
        case_id="c",
        point_id="p",
        node_id="n",
        question_id="q",
        rendered_fingerprint="r",
        request_fingerprint="f",
        status=ProviderStatus(ok=True, model_resolved="m"),
        distribution=DistributionRecord(
            node_id="n",
            question_id="q",
            primitive="noul",
            support=["false", "true"],
            raw=[0.5, 0.5],
            raw_total=1.0,
            source=DistributionSource.REPORTED,
            semantic_hash="sha256:x",
            case_id="c",
            point_id="p",
        ),
    )
    point = ExperimentPoint(
        case=CaseSpec(id="c", state="x"), theta={"p": 0.5}, point_id="p"
    )
    assert trace.repeat == 0
    assert point.repeat == 0
    assert json.loads(trace.model_dump_json())["distribution"]["source"] == "reported"

def test_joint_model_reports_visit_probabilities() -> None:
    from jevometry.adapters.analytic import bernoulli_pair_joint_model

    adapter = AnalyticAdapter(
        system_id="pair",
        nodes={"Y": bernoulli_node("Y"), "Z": bernoulli_node("Z")},
        joint_model=bernoulli_pair_joint_model("p", mode="deterministic_copy"),
    )
    captures = run_experiment(bernoulli_spec("visit-probability"), adapter)
    analysis = analyze(captures, adapter=adapter)
    assert analysis.document.system is not None
    visits = {
        metric.name: metric.value
        for metric in analysis.document.system.metrics
        if metric.name.startswith("visit_probability:")
    }
    assert visits["visit_probability:Y"] == pytest.approx(1.0)
    assert visits["visit_probability:Z"] == pytest.approx(1.0)
