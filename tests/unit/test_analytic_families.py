"""Analytic family constructors, joint models and likelihood helpers."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.adapters.analytic import (
    AnalyticAdapter,
    AnalyticJointModel,
    AnalyticLikelihoodModel,
    AnalyticNode,
    bernoulli_likelihood,
    bernoulli_node,
    bernoulli_pair_joint_model,
    choice_question,
    conditional_tree_joint_model,
    logistic_node,
    noul_question,
    rank_deficient_softmax_node,
    sample_case,
    score_question,
    single_node_joint_model,
    softmax_node,
)
from jevometry.schemas.common import MetricStatus, Primitive
from jevometry.schemas.joint import ConstructionMode
from jevometry.schemas.system import CompositionMode
from jevometry.systems.tree import ConditionalNode, ConditionalTree


def test_question_helpers_declare_support() -> None:
    choice = choice_question("c", {"a": "A", "b": "B"})
    assert choice.primitive is Primitive.CHOICE
    assert choice.support == ("a", "b")
    score = score_question("s", ["low", "high"], numerics=[0.0, 1.0])
    assert score.numeric_encoding() == {"0": 0.0, "1": 1.0}
    with pytest.raises(ValueError):
        score_question("s", ["low", "high"], numerics=[0.0])
    noul = noul_question("n")
    assert noul.support == ("false", "true")
    assert sample_case("case", "state").state == "state"


def test_logistic_coordinate_validation() -> None:
    with pytest.raises(ValueError):
        logistic_node("n", parameter="theta", coordinate="log")


def test_softmax_shape_validation() -> None:
    with pytest.raises(ValueError):
        softmax_node("n", outcomes=("a", "b"), parameters=("t",), weights=[[1.0]])
    with pytest.raises(ValueError):
        softmax_node(
            "n", outcomes=("a", "b"), parameters=("t",), weights=[[1.0], [1.0]], bias=[1.0]
        )


def test_rank_deficient_shape_validation() -> None:
    with pytest.raises(ValueError):
        rank_deficient_softmax_node("n", outcomes=("a", "b"), parameters=("t1", "t2"))


def test_analytic_node_reports_validation_failure() -> None:
    node = AnalyticNode(
        node_id="bad",
        question=noul_question("bad"),
        model_identity="bad",
        probability_function=lambda theta: np.asarray([0.5, 0.9]),
    )
    evaluation = node.evaluate({})
    assert evaluation.status is MetricStatus.FAILED
    assert evaluation.reason_code == "sum_out_of_tolerance"


def test_analytic_node_rejects_wrong_width() -> None:
    node = AnalyticNode(
        node_id="bad",
        question=noul_question("bad"),
        model_identity="bad",
        probability_function=lambda theta: np.asarray([1.0]),
    )
    with pytest.raises(ValueError):
        node.probabilities({})


def test_bernoulli_pair_joint_modes() -> None:
    copy = bernoulli_pair_joint_model("p", mode="deterministic_copy")
    joint = copy.probabilities({"p": 0.3})
    assert joint.outcomes == [("0", "0"), ("1", "1")]
    assert copy.jacobian({"p": 0.3}) == pytest.approx(np.asarray([[-1.0], [1.0]]))
    independent = bernoulli_pair_joint_model("p", mode="independent", draws=2)
    joint = independent.probabilities({"p": 0.3})
    assert len(joint.outcomes) == 4
    assert sum(joint.probabilities) == pytest.approx(1.0)
    jacobian = independent.jacobian({"p": 0.3})
    assert jacobian is not None and jacobian.shape == (4, 1)
    with pytest.raises(ValueError):
        bernoulli_pair_joint_model("p", mode="other")


def test_single_node_joint_model_roundtrip() -> None:
    node = bernoulli_node("Y", parameter="p")
    model = single_node_joint_model(node)
    joint = model.probabilities({"p": 0.4})
    assert joint.node_order == ["Y"]
    assert model.jacobian({"p": 0.4}) == pytest.approx(node.jacobian({"p": 0.4}))


def test_conditional_tree_joint_model_matches_enumeration() -> None:
    tree = ConditionalTree(
        tree_id="t",
        nodes=[ConditionalNode("Y", ("a", "b"), lambda theta, history: {"a": 0.25, "b": 0.75})],
    )
    model = conditional_tree_joint_model(tree)
    joint = model.probabilities({"p": 0.5})
    assert joint.probabilities == pytest.approx([0.25, 0.75])
    assert model.jacobian({"p": 0.5}) is None


def test_analytic_joint_model_without_jacobian() -> None:
    model = AnalyticJointModel(
        node_order=["Y"],
        parameter_names=["p"],
        probability_function=lambda theta: ([("a",), ("b",)], np.asarray([0.4, 0.6])),
        jacobian_function=None,
        model_identity="no-jacobian",
    )
    assert model.jacobian({"p": 0.5}) is None


def test_likelihood_model_boundary_and_sampling() -> None:
    model = AnalyticLikelihoodModel(
        ("0", "1"),
        lambda theta: np.asarray([1.0, 0.0]),
    )
    assert model.log_prob([1], {}) == float("-inf")
    assert model.log_prob([0, 0], {}) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        model.log_prob([2], {})
    rng = np.random.default_rng(0)
    samples = model.sample({}, 5, rng)
    assert set(samples.tolist()) == {0}


def test_adapter_helpers_expose_families() -> None:
    node = bernoulli_node("Y", parameter="p")
    adapter = AnalyticAdapter(
        system_id="s",
        nodes={"Y": node},
        likelihood=bernoulli_likelihood("p"),
    )
    assert adapter.analytic_jacobian("Y", {"p": 0.4}) is not None
    assert adapter.declared_fixed_zero("Y") == ()
    assert adapter.likelihood_model("Y") is not None
    assert adapter.likelihood_model() is not None
    assert adapter.questions()["Y"].id == "Y"
    assert adapter.describe().capabilities.analytic_jacobian is True


def test_adapter_without_likelihood_returns_none() -> None:
    adapter = AnalyticAdapter(system_id="s", nodes={"Y": bernoulli_node("Y")})
    assert adapter.likelihood_model("Y") is None
    with pytest.raises(ValueError):
        AnalyticAdapter(system_id="s", nodes={})


@pytest.mark.parametrize(
    ("construction", "expected"),
    [
        (ConstructionMode.EXPLICIT, CompositionMode.EXPLICIT_JOINT),
        (ConstructionMode.DECLARED_PRODUCT, CompositionMode.DECLARED_PRODUCT),
        (ConstructionMode.CONDITIONAL_TREE, CompositionMode.CONDITIONAL_TREE),
        (ConstructionMode.ESTIMATED, CompositionMode.EXPLICIT_JOINT),
    ],
)
def test_adapter_preserves_joint_construction_mode(
    construction: ConstructionMode, expected: CompositionMode
) -> None:
    model = single_node_joint_model(bernoulli_node("Y"))
    model.construction_mode = construction
    adapter = AnalyticAdapter(system_id="s", nodes={"Y": bernoulli_node("Y")}, joint_model=model)
    spec = adapter.describe()
    assert spec.composition_mode is expected
    assert spec.composition_assumptions.conditional_independence is (
        construction is ConstructionMode.DECLARED_PRODUCT
    )
