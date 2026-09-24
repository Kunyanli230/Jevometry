"""Schema validators and helper methods."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jevometry.schemas.common import (
    AnalysisObject,
    CapabilityEntry,
    CapabilityMatrix,
    MetricResult,
    MetricStatus,
    ValueKind,
)
from jevometry.schemas.distribution import DistributionRecord, DistributionSource
from jevometry.schemas.experiment import (
    Budget,
    CaseSpec,
    ExperimentSpec,
    SamplingContract,
    SamplingKind,
)
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.schemas.questions import (
    OutcomeSpec,
    QuestionSet,
    QuestionSpec,
    SupportMapping,
)
from jevometry.schemas.results import JacobianResult
from jevometry.schemas.system import (
    Capabilities,
    CompositionAssumptions,
    CompositionMode,
    NodeSpec,
    SystemSpec,
)


def parameter(name: str = "theta", **overrides: object) -> ParameterSpec:
    values: dict[str, object] = {
        "name": name,
        "role": "task_relevant",
        "unit": "u",
        "bounds": (0.0, 1.0),
        "step": 0.1,
    }
    values.update(overrides)
    return ParameterSpec(**values)  # type: ignore[arg-type]


def test_parameter_validators() -> None:
    with pytest.raises(ValidationError):
        parameter(bounds=(1.0, 1.0))
    with pytest.raises(ValidationError):
        parameter(bounds=(float("-inf"), 1.0))
    with pytest.raises(ValidationError):
        parameter(scale=0.0)
    with pytest.raises(ValidationError):
        parameter(step=-1.0)
    with pytest.raises(ValidationError):
        parameter(center=5.0)
    spec = parameter(center=0.5, scale=2.0)
    assert spec.contains(0.5)
    assert not spec.contains(1.5)
    assert spec.standardize(1.0) == 0.5


def test_parameter_set_lookup_and_scales() -> None:
    parameter_set = ParameterSet(parameters=[parameter("a"), parameter("b")])
    assert parameter_set.names == ["a", "b"]
    assert parameter_set.by_name("b").name == "b"
    assert parameter_set.scales() == [1.0, 1.0]
    with pytest.raises(KeyError):
        parameter_set.by_name("c")
    with pytest.raises(ValidationError):
        ParameterSet(parameters=[])


def test_stencil_validators() -> None:
    with pytest.raises(ValidationError):
        StencilSpec(step_scales=[])
    with pytest.raises(ValidationError):
        StencilSpec(step_scales=[1.0, -0.5])
    with pytest.raises(ValidationError):
        StencilSpec(relative_stability_threshold=0.0)
    with pytest.raises(ValidationError):
        StencilSpec(step_scales=[2.0])


def test_question_validators_and_legend() -> None:
    with pytest.raises(ValidationError):
        QuestionSpec(id="q", primitive="choice", outcomes=[])
    with pytest.raises(ValidationError):
        QuestionSpec(
            id="q",
            primitive="choice",
            outcomes=[OutcomeSpec(id="a"), OutcomeSpec(id="a")],
        )
    with pytest.raises(ValidationError):
        QuestionSpec(
            id="q",
            primitive="choice",
            outcomes=[OutcomeSpec(id="a")],
            legend={"b": "unknown"},
        )
    question = QuestionSpec(
        id="q",
        primitive="score",
        outcomes=[OutcomeSpec(id="0", numeric=0.0), OutcomeSpec(id="1", numeric=1.0)],
        legend={"0": "low", "1": "high"},
    )
    payload = question.semantic_payload()
    assert payload["id"] == "q"
    assert question.numeric_encoding() == {"0": 0.0, "1": 1.0}
    partial = QuestionSpec(
        id="q2",
        primitive="score",
        outcomes=[OutcomeSpec(id="0", numeric=0.0), OutcomeSpec(id="1")],
    )
    assert partial.numeric_encoding() is None


def test_support_mapping_requires_confirmation() -> None:
    with pytest.raises(ValidationError):
        SupportMapping(left_question="a", right_question="b", mapping={"x": "y"})
    with pytest.raises(ValidationError):
        SupportMapping(
            left_question="a", right_question="b", mapping={"x": "y", "z": "y"}, confirmed=True
        )
    mapping = SupportMapping(
        left_question="a", right_question="b", mapping={"x": "y"}, confirmed=True
    )
    assert mapping.confirmed


def test_question_set_uniqueness() -> None:
    question = QuestionSpec(id="q", primitive="noul", outcomes=[OutcomeSpec(id="false"), OutcomeSpec(id="true")])
    with pytest.raises(ValidationError):
        QuestionSet(questions=[question, question])
    question_set = QuestionSet(questions=[question])
    assert question_set.by_id("q").id == "q"
    with pytest.raises(KeyError):
        question_set.by_id("other")


def test_system_spec_validators() -> None:
    with pytest.raises(ValidationError):
        NodeSpec(id="a", question_id="q", parents=["a"])
    with pytest.raises(ValidationError):
        SystemSpec(
            id="s",
            nodes=[NodeSpec(id="a", question_id="q"), NodeSpec(id="a", question_id="q")],
        )
    with pytest.raises(ValidationError):
        SystemSpec(
            id="s",
            nodes=[NodeSpec(id="a", question_id="q", parents=["missing"])],
        )
    with pytest.raises(ValidationError):
        SystemSpec(id="s", composition_mode=CompositionMode.EXPLICIT_JOINT)
    spec = SystemSpec(
        id="s",
        nodes=[NodeSpec(id="a", question_id="q")],
        composition_mode=CompositionMode.DECLARED_PRODUCT,
        capabilities=Capabilities(joint_model=True),
        composition_assumptions=CompositionAssumptions(
            conditional_independence=True, shared_theta=True
        ),
    )
    assert spec.node_ids() == ["a"]
    assert spec.by_id("a").question_id == "q"
    with pytest.raises(KeyError):
        spec.by_id("b")


def test_experiment_spec_validators() -> None:
    parameter_set = ParameterSet(parameters=[parameter("a")])
    with pytest.raises(ValidationError):
        ExperimentSpec(id="", parameter_set=parameter_set)
    with pytest.raises(ValidationError):
        ExperimentSpec(id="e", parameter_set=parameter_set, cases=[])
    with pytest.raises(ValidationError):
        ExperimentSpec(
            id="e",
            parameter_set=parameter_set,
            cases=[CaseSpec(id="c", state="x")],
            theta_points=[],
        )
    with pytest.raises(ValidationError):
        ExperimentSpec(
            id="e",
            parameter_set=parameter_set,
            cases=[CaseSpec(id="c", state="x")],
            theta_points=[{"other": 1.0}],
        )
    with pytest.raises(ValidationError):
        ExperimentSpec(
            id="e",
            parameter_set=ParameterSet(parameters=[parameter("a", continuous=False)]),
            cases=[CaseSpec(id="c", state="x")],
            theta_points=[{"a": 0.5}],
        )
    with pytest.raises(ValidationError):
        CaseSpec(id="c", state=None)


def test_budget_and_contract_helpers() -> None:
    with pytest.raises(ValidationError):
        Budget(max_attempts=0)
    contract = SamplingContract()
    assert contract.is_complete() is False
    assert "observable" in contract.missing_fields()
    assert contract.sampling is SamplingKind.NONE
    with pytest.raises(ValidationError):
        SamplingContract(sample_size=0)


def test_distribution_record_validation() -> None:
    with pytest.raises(ValidationError):
        DistributionRecord(
            node_id="n",
            question_id="q",
            primitive="noul",
            support=["false", "true"],
            raw=[0.5],
            raw_total=0.5,
            source=DistributionSource.REPORTED,
            semantic_hash="h",
            case_id="c",
            point_id="p",
        )
    with pytest.raises(ValidationError):
        DistributionRecord(
            node_id="n",
            question_id="q",
            primitive="noul",
            support=["false", "false"],
            raw=[0.5, 0.5],
            raw_total=1.0,
            source=DistributionSource.REPORTED,
            semantic_hash="h",
            case_id="c",
            point_id="p",
        )
    record = DistributionRecord(
        node_id="n",
        question_id="q",
        primitive="noul",
        support=["false", "true"],
        raw=[0.5, 0.5],
        working=[0.5, 0.5],
        raw_total=1.0,
        source=DistributionSource.DECLARED,
        semantic_hash="h",
        case_id="c",
        point_id="p",
    )
    assert record.analysis_object is AnalysisObject.DECLARED_SYSTEM_MODEL
    assert record.probabilities() == [0.5, 0.5]


def test_joint_distribution_validators() -> None:
    with pytest.raises(ValidationError):
        JointDistribution(
            mode=ConstructionMode.EXPLICIT, node_order=[], outcomes=[("a",)], probabilities=[1.0]
        )
    with pytest.raises(ValidationError):
        JointDistribution(
            mode=ConstructionMode.EXPLICIT, node_order=["a"], outcomes=[], probabilities=[]
        )
    with pytest.raises(ValidationError):
        JointDistribution(
            mode=ConstructionMode.EXPLICIT,
            node_order=["a"],
            outcomes=[("a", "b")],
            probabilities=[1.0],
        )
    with pytest.raises(ValidationError):
        JointDistribution(
            mode=ConstructionMode.EXPLICIT,
            node_order=["a"],
            outcomes=[("a",)],
            probabilities=[1.0],
            history_provenance=[{}, {}],
        )
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["a"],
        outcomes=[("a",)],
        probabilities=[1.0],
    )
    assert joint.total() == 1.0
    assert joint.index_of(("a",)) == 0
    assert joint.marginal_vector("a") == (["a"], [1.0])


def test_jacobian_result_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        JacobianResult(
            node_id="n",
            case_id="c",
            point_id="p",
            parameter_names=["t"],
            support=["a"],
            values=[[float("nan")]],
            method="finite_difference",
            stencil_kind="central",
        )


def test_capability_matrix_helpers() -> None:
    matrix = CapabilityMatrix()
    matrix.add(
        CapabilityEntry(
            capability="node_geometry",
            status=MetricStatus.OK,
            analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
        )
    )
    assert matrix.status_of("node_geometry") is MetricStatus.OK
    assert matrix.status_of("missing") is None
    assert matrix.entries[0].requires == []


def test_metric_result_helpers() -> None:
    ok = MetricResult.ok(
        "x", 1.0, analysis_object=AnalysisObject.REPORTED_DISTRIBUTION
    )
    assert ok.computed is True
    assert ok.to_public_dict()["value"] == 1.0
    refusal = MetricResult.refusal(
        "y",
        status=MetricStatus.UNSUPPORTED,
        analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
        reason_code="no_joint_model",
    )
    assert refusal.computed is False
    assert refusal.value_kind is ValueKind.NULL

def test_resource_caps_are_enforced_with_explicit_messages() -> None:
    with pytest.raises(ValidationError) as error:
        ParameterSet(
            parameters=[
                parameter(f"p{index}") for index in range(9)
            ]
        )
    assert "v0.1 limit" in str(error.value)

    with pytest.raises(ValidationError) as error:
        QuestionSpec(
            id="q",
            primitive="choice",
            outcomes=[OutcomeSpec(id=f"o{index}") for index in range(33)],
        )
    assert "v0.1 limit" in str(error.value)

    with pytest.raises(ValidationError) as error:
        SystemSpec(
            id="s",
            nodes=[NodeSpec(id=f"n{index}", question_id="q") for index in range(65)],
        )
    assert "v0.1 limit" in str(error.value)

    with pytest.raises(ValidationError) as error:
        JointDistribution(
            mode=ConstructionMode.EXPLICIT,
            node_order=["a"],
            outcomes=[(str(index),) for index in range(4097)],
            probabilities=[1.0 / 4097] * 4097,
        )
    assert "enumeration limit" in str(error.value)
