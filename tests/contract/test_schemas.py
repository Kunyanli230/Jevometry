"""Schema round-trips, JSON safety and metric refusal contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jevometry.schemas.common import (
    AnalysisObject,
    MetricResult,
    MetricStatus,
    ValueKind,
)
from jevometry.schemas.distribution import DistributionRecord, DistributionSource
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.schemas.questions import OutcomeSpec, QuestionSpec
from jevometry.schemas.results import AnalysisDocument, RunManifest
from jevometry.schemas.system import NodeSpec, SystemSpec
from jevometry.schemas.trace import EvaluationTrace, ProviderStatus


def test_metric_result_roundtrip() -> None:
    metric = MetricResult.ok(
        "fisher_trace",
        1.5,
        analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
        units="nats",
        coordinates=["theta"],
        value_kind=ValueKind.SCALAR,
        tolerances={"rank_rtol": 1e-8},
    )
    payload = metric.model_dump(mode="json")
    restored = MetricResult.model_validate(payload)
    assert restored == metric
    assert json.dumps(payload)


def test_metric_result_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        MetricResult(
            name="x",
            status=MetricStatus.OK,
            analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
            value=float("inf"),
        )


def test_refusal_requires_reason_and_status() -> None:
    with pytest.raises(ValueError):
        MetricResult.refusal(
            "x",
            status=MetricStatus.OK,
            analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
            reason_code="should_not_be_ok",
        )
    refusal = MetricResult.refusal(
        "x",
        status=MetricStatus.UNDEFINED,
        analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
        reason_code="zero_probability_outcome",
        remedy="fix the source",
    )
    assert refusal.value is None
    assert refusal.reason_code == "zero_probability_outcome"


def test_distribution_record_rejects_unknown_selected() -> None:
    with pytest.raises(ValidationError):
        DistributionRecord(
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
            selected="maybe",
        )


def test_question_semantic_hash_changes_with_rubric() -> None:
    base = QuestionSpec(
        id="q",
        primitive="choice",
        outcomes=[OutcomeSpec(id="a"), OutcomeSpec(id="b")],
        instructions="pick",
    )
    changed = QuestionSpec(
        id="q",
        primitive="choice",
        outcomes=[OutcomeSpec(id="a"), OutcomeSpec(id="b")],
        instructions="pick",
        rubric="different rubric",
    )
    assert base.semantic_hash() != changed.semantic_hash()
    assert base.semantic_hash() == base.model_copy().semantic_hash()


def test_noul_support_is_enforced() -> None:
    with pytest.raises(ValidationError):
        QuestionSpec(
            id="q",
            primitive="noul",
            outcomes=[OutcomeSpec(id="yes"), OutcomeSpec(id="no")],
        )


def test_experiment_spec_requires_all_parameters() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec(
            id="e",
            parameter_set=ParameterSet(
                parameters=[
                    ParameterSpec(
                        name="a", role="task_relevant", unit="u", bounds=(0, 1), step=0.1
                    ),
                    ParameterSpec(
                        name="b", role="nuisance", unit="u", bounds=(0, 1), step=0.1
                    ),
                ]
            ),
            cases=[CaseSpec(id="c", state="x")],
            theta_points=[{"a": 0.5}],
        )


def test_joint_distribution_requires_unique_paths() -> None:
    with pytest.raises(ValidationError):
        JointDistribution(
            mode=ConstructionMode.EXPLICIT,
            node_order=["a"],
            outcomes=[("x",), ("x",)],
            probabilities=[0.5, 0.5],
        )


def test_joint_distribution_marginal() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["a", "b"],
        outcomes=[("0", "0"), ("0", "1"), ("1", "0")],
        probabilities=[0.25, 0.25, 0.5],
    )
    assert joint.marginal("a") == {"0": 0.5, "1": 0.5}
    assert joint.marginal_support("b") == ["0", "1"]


def test_analysis_document_roundtrip_is_json_safe() -> None:
    document = AnalysisDocument(
        run_id="r",
        analysis_revision=1,
        created_utc="2026-09-24T00:00:00Z",
        experiment_id="e",
    )
    payload = document.model_dump(mode="json")
    restored = AnalysisDocument.model_validate(json.loads(json.dumps(payload)))
    assert restored.run_id == "r"
    assert restored.capability_matrix.entries == []


def test_run_manifest_roundtrip() -> None:
    manifest = RunManifest(
        run_id="r",
        created_utc="2026-09-24T00:00:00Z",
        experiment_id="e",
        experiment_hash="sha256:x",
        provider_mode="offline",
    )
    restored = RunManifest.model_validate(json.loads(manifest.model_dump_json()))
    assert restored.live is False


def test_trace_schema_roundtrip() -> None:
    trace = EvaluationTrace(
        experiment_id="e",
        case_id="c",
        point_id="p",
        node_id="n",
        question_id="q",
        rendered_fingerprint="r",
        request_fingerprint="f",
        status=ProviderStatus(ok=True),
    )
    restored = EvaluationTrace.model_validate(json.loads(trace.model_dump_json()))
    assert restored.stencil_role == "center"


def test_system_spec_rejects_unknown_edges() -> None:
    with pytest.raises(ValidationError):
        SystemSpec(
            id="s",
            nodes=[NodeSpec(id="a", question_id="q")],
            edges=[{"source": "a", "target": "b"}],
        )


def test_stencil_spec_requires_nominal_step() -> None:
    with pytest.raises(ValidationError):
        StencilSpec(step_scales=[0.5, 0.25])
