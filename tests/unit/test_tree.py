"""Conditional tree enumeration and the conditional information identity."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.systems.tree import (
    ConditionalNode,
    ConditionalTree,
    conditional_fisher_identity,
    enumerate_tree,
    tree_joint,
    tree_path_jacobian,
)
from tests.conftest import two_layer_tree


def test_exact_enumeration_sums_to_one() -> None:
    tree = two_layer_tree()
    enumeration = enumerate_tree(tree, {"p": 0.3})
    assert enumeration.total == pytest.approx(1.0)
    assert len(enumeration.outcomes) == 4


def test_tree_joint_matches_analytic_product() -> None:
    tree = two_layer_tree()
    p = 0.3
    joint = tree_joint(tree, {"p": p})
    masses = dict(zip(joint.outcomes, joint.probabilities, strict=True))
    from scipy.special import expit

    z0 = float(expit(p))
    z1 = float(expit(p + 1.0))
    assert masses[("a", "0")] == pytest.approx((1 - p) * (1 - z0))
    assert masses[("a", "1")] == pytest.approx((1 - p) * z0)
    assert masses[("b", "0")] == pytest.approx(p * (1 - z1))
    assert masses[("b", "1")] == pytest.approx(p * z1)


def test_stop_branches_are_part_of_the_model() -> None:
    def active(history: dict[str, str]) -> bool:
        return history["Y"] == "a"

    tree = ConditionalTree(
        tree_id="stop",
        nodes=[
            ConditionalNode("Y", ("a", "b"), lambda theta, history: {"a": 0.5, "b": 0.5}),
            ConditionalNode(
                "Z",
                ("0", "1"),
                lambda theta, history: {"0": 0.5, "1": 0.5},
                active=active,
            ),
        ],
    )
    enumeration = enumerate_tree(tree, {})
    assert enumeration.total == pytest.approx(1.0)
    assert ("b", "<stop>") in enumeration.outcomes
    visits = enumeration.visit_probabilities()
    assert visits["Y"] == pytest.approx(1.0)
    assert visits["Z"] == pytest.approx(0.5)


def test_conditional_information_identity_holds() -> None:
    tree = two_layer_tree()
    theta = {"p": 0.35}
    result = conditional_fisher_identity(tree, theta, parameter_names=["p"])
    assert result.total is not None
    assert result.joint_fisher is not None
    assert result.residual is not None
    assert result.residual < 1e-10
    assert np.allclose(result.total, result.joint_fisher, rtol=1e-8)
    expected = _joint_fisher_by_finite_difference(tree, theta)
    assert result.joint_fisher[0, 0] == pytest.approx(expected, rel=1e-6)
    assert result.joint_fisher[0, 0] > 1.0 / (0.35 * 0.65)


def test_conditional_identity_reduces_to_first_node_when_children_ignore_theta() -> None:
    constant_child = ConditionalTree(
        tree_id="constant-child",
        nodes=[
            ConditionalNode(
                "Y",
                ("a", "b"),
                lambda theta, history: {"a": 1.0 - theta["p"], "b": theta["p"]},
                lambda theta, history: {"a": {"p": -1.0}, "b": {"p": 1.0}},
            ),
            ConditionalNode(
                "Z",
                ("0", "1"),
                lambda theta, history: {"0": 0.5, "1": 0.5},
                lambda theta, history: {"0": {"p": 0.0}, "1": {"p": 0.0}},
            ),
        ],
    )
    result = conditional_fisher_identity(constant_child, {"p": 0.35}, parameter_names=["p"])
    assert result.joint_fisher is not None
    assert result.joint_fisher[0, 0] == pytest.approx(1.0 / (0.35 * 0.65), rel=1e-8)
    assert result.per_node["Z"][0, 0] == pytest.approx(0.0, abs=1e-12)


def _joint_fisher_by_finite_difference(tree: ConditionalTree, theta: dict[str, float]) -> float:
    step = 1e-6
    plus = dict(theta)
    minus = dict(theta)
    plus["p"] += step
    minus["p"] -= step
    base = enumerate_tree(tree, theta)
    upper = enumerate_tree(tree, plus)
    lower = enumerate_tree(tree, minus)
    information = 0.0
    for index, probability in enumerate(base.probabilities):
        if probability <= 0.0:
            continue
        derivative = (upper.probabilities[index] - lower.probabilities[index]) / (2 * step)
        information += derivative**2 / probability
    return float(information)


def test_conditional_contributions_are_history_weighted() -> None:
    tree = two_layer_tree()
    result = conditional_fisher_identity(tree, {"p": 0.2}, parameter_names=["p"])
    assert set(result.per_node) == {"Y", "Z"}
    assert result.per_node["Y"][0, 0] == pytest.approx(1.0 / (0.2 * 0.8), rel=1e-8)
    assert result.per_node["Z"][0, 0] > 0.0


def test_path_jacobian_is_exact() -> None:
    tree = two_layer_tree()
    theta = {"p": 0.4}
    outcomes, jacobian = tree_path_jacobian(tree, theta)
    assert jacobian is not None
    joint = tree_joint(tree, theta)
    assert list(joint.outcomes) == list(outcomes)
    step = 1e-6
    plus = dict(theta)
    minus = dict(theta)
    plus["p"] += step
    minus["p"] -= step
    masses_plus = dict(
        zip(*_outcomes_and_masses(tree, plus), strict=True)
    )
    masses_minus = dict(
        zip(*_outcomes_and_masses(tree, minus), strict=True)
    )
    for row, outcome in enumerate(outcomes):
        expected = (masses_plus[outcome] - masses_minus[outcome]) / (2 * step)
        assert jacobian[row, 0] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def _outcomes_and_masses(tree: ConditionalTree, theta: dict[str, float]):
    enumeration = enumerate_tree(tree, theta)
    return enumeration.outcomes, enumeration.probabilities
