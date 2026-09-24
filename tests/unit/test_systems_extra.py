"""Additional coverage for system composition, graphs and policies."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from jevometry.schemas.common import MetricStatus
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.schemas.system import (
    CompositionMode,
    EdgeSpec,
    NodeSpec,
    RoutingSemantics,
    SystemSpec,
)
from jevometry.systems.graph import SystemGraph
from jevometry.systems.joint import (
    conditional_mutual_information,
    marginalize,
    mutual_information,
)
from jevometry.systems.policy import analyse_policy
from jevometry.systems.redundancy import (
    deterministic_copy_jacobian,
    deterministic_copy_joint,
    independent_draw_jacobian,
    independent_draw_joint,
)
from jevometry.systems.tree import ConditionalNode, ConditionalTree, enumerate_tree


def test_graph_topological_order_and_visit_probabilities() -> None:
    spec = SystemSpec(
        id="s",
        nodes=[
            NodeSpec(id="a", question_id="qa"),
            NodeSpec(id="b", question_id="qb", parents=["a"]),
            NodeSpec(id="c", question_id="qc", parents=["a"]),
        ],
        edges=[EdgeSpec(source="a", target="b"), EdgeSpec(source="a", target="c")],
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        composition_mode=CompositionMode.CONDITIONAL_TREE,
    )
    graph = SystemGraph.from_spec(spec)
    assert graph.topological_order == ["a", "b", "c"]
    assert graph.children("a") == ["b", "c"]
    assert graph.parents("b") == ["a"]
    joint = JointDistribution(
        mode=ConstructionMode.CONDITIONAL_TREE,
        node_order=["a", "b", "c"],
        outcomes=[("x", "y", "z"), ("x", "<stop>", "z"), ("w", "<stop>", "<stop>")],
        probabilities=[0.5, 0.2, 0.3],
    )
    visits = graph.visit_probabilities(joint)
    assert visits["a"] == pytest.approx(1.0)
    assert visits["b"] == pytest.approx(0.5)
    assert visits["c"] == pytest.approx(0.7)


def test_graph_rejects_cycles() -> None:
    spec = SystemSpec(
        id="s",
        nodes=[NodeSpec(id="a", question_id="qa"), NodeSpec(id="b", question_id="qb")],
        edges=[EdgeSpec(source="a", target="b"), EdgeSpec(source="b", target="a")],
    )
    with pytest.raises(ValueError):
        SystemGraph.from_spec(spec)


def test_marginalize_keeps_declared_nodes() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["a", "b"],
        outcomes=[("0", "0"), ("0", "1"), ("1", "0")],
        probabilities=[0.25, 0.25, 0.5],
    )
    marginal = marginalize(joint, ["b"])
    assert marginal.node_order == ["b"]
    masses = dict(zip(marginal.outcomes, marginal.probabilities, strict=True))
    assert masses[("0",)] == pytest.approx(0.75)
    assert masses[("1",)] == pytest.approx(0.25)
    with pytest.raises(KeyError):
        marginalize(joint, ["unknown"])


def test_mutual_information_self_is_entropy() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["a"],
        outcomes=[("0",), ("1",), ("2",)],
        probabilities=[0.5, 0.25, 0.25],
    )
    expected = -(0.5 * math.log(0.5) + 0.25 * math.log(0.25) + 0.25 * math.log(0.25))
    assert mutual_information(joint, "a", "a") == pytest.approx(expected)


def test_conditional_mutual_information_analytic() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["x", "y", "z"],
        outcomes=[
            ("0", "0", "0"),
            ("1", "1", "0"),
            ("0", "0", "1"),
            ("0", "1", "1"),
            ("1", "0", "1"),
            ("1", "1", "1"),
        ],
        probabilities=[0.25, 0.25, 0.125, 0.125, 0.125, 0.125],
    )
    value = conditional_mutual_information(joint, "x", "y", "z")
    assert value == pytest.approx(0.5 * math.log(2.0), rel=1e-9)
    with pytest.raises(KeyError):
        conditional_mutual_information(joint, "x", "y", "missing")


def test_redundancy_helpers_align_shapes() -> None:
    probabilities = np.asarray([0.4, 0.6])
    jacobian = np.asarray([[-1.0], [1.0]])
    outcomes, masses = deterministic_copy_joint(["0", "1"], probabilities)
    assert outcomes == [("0", "0"), ("1", "1")]
    assert masses == pytest.approx(probabilities)
    assert deterministic_copy_jacobian(jacobian) == pytest.approx(jacobian)
    grid, joint_masses = independent_draw_joint(["0", "1"], probabilities, draws=3)
    assert len(grid) == 8
    assert joint_masses.sum() == pytest.approx(1.0)
    joint_jacobian = independent_draw_jacobian(
        ["0", "1"], probabilities, jacobian, draws=3
    )
    assert joint_jacobian.shape == (8, 1)


def test_policy_analysis_detects_flips_and_counts() -> None:
    points = [{"theta": 0.0}, {"theta": 1.0}, {"theta": 2.0}]
    probabilities = [
        {"signal": 0.1},
        {"signal": 0.9},
        {"signal": 0.9},
    ]

    def policy(theta: dict[str, float], distribution: dict[str, float]) -> str:
        return "act" if distribution["signal"] > 0.5 else "wait"

    result = analyse_policy(
        policy, policy_name="threshold", points=points, probabilities=probabilities
    )
    assert result.action_counts == {"wait": 1, "act": 2}
    assert len(result.flips) == 1
    assert result.flips[0]["from_action"] == "wait"
    assert result.flips[0]["to_action"] == "act"
    assert len(result.to_metrics()) == 3
    assert result.status is MetricStatus.OK


def test_tree_inactive_node_history_is_recorded() -> None:
    tree = ConditionalTree(
        tree_id="stop-only",
        nodes=[
            ConditionalNode("Y", ("a",), lambda theta, history: {"a": 1.0}),
            ConditionalNode(
                "Z",
                ("0", "1"),
                lambda theta, history: {"0": 0.5, "1": 0.5},
                active=lambda history: False,
            ),
        ],
    )
    enumeration = enumerate_tree(tree, {})
    assert enumeration.outcomes == [("a", "<stop>")]
    assert enumeration.visit_probabilities()["Z"] == pytest.approx(0.0)


def test_tree_rejects_unknown_outcomes() -> None:
    tree = ConditionalTree(
        tree_id="bad",
        nodes=[
            ConditionalNode("Y", ("a", "b"), lambda theta, history: {"a": 1.0}),
        ],
    )
    with pytest.raises(ValueError):
        enumerate_tree(tree, {})

def test_tree_enumeration_cap_is_refused_not_truncated() -> None:
    node = ConditionalNode(
        "Y",
        tuple(f"o{index}" for index in range(40)),
        lambda theta, history: {f"o{index}": 1.0 / 40 for index in range(40)},
    )
    tree = ConditionalTree(
        tree_id="wide",
        nodes=[
            node,
            replace(node, node_id="Z"),
            replace(node, node_id="W"),
        ],
    )
    enumeration = enumerate_tree(tree, {})
    assert enumeration.status is MetricStatus.UNSUPPORTED
    assert enumeration.reason_code == "enumeration_limit_exceeded"
    assert enumeration.outcomes == []
    assert "64000" in enumeration.assumptions[0]
