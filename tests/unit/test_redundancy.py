"""Deterministic replication adds no information; independent draws do."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.systems.redundancy import (
    compare_independent_sum,
    independent_draw_jacobian,
    independent_draw_joint,
)


def test_deterministic_copy_has_same_information_as_one_node() -> None:
    p = 0.3
    probabilities = np.asarray([1.0 - p, p])
    jacobian = np.asarray([[-1.0], [1.0]])
    single = jacobian.T @ np.diag(1.0 / probabilities) @ jacobian
    copy_probabilities = probabilities.copy()
    copy_jacobian = jacobian.copy()
    comparison = compare_independent_sum(
        name="copy",
        description="deterministic copy of the same output",
        node_fisher={"Y": single, "Z": single},
        joint_probabilities=copy_probabilities,
        joint_jacobian=copy_jacobian,
    )
    assert comparison.independent_sum_trace is not None
    assert comparison.joint_trace is not None
    assert comparison.joint_trace == pytest.approx(1.0 / (p * (1 - p)), rel=1e-10)
    assert comparison.difference == pytest.approx(1.0 / (p * (1 - p)), rel=1e-10)


def test_independent_draws_double_information() -> None:
    p = 0.4
    probabilities = np.asarray([1.0 - p, p])
    jacobian = np.asarray([[-1.0], [1.0]])
    _, joint_probabilities = independent_draw_joint(["0", "1"], probabilities)
    joint_jacobian = independent_draw_jacobian(["0", "1"], probabilities, jacobian)
    assert joint_probabilities.sum() == pytest.approx(1.0)
    single = float((jacobian.T @ np.diag(1.0 / probabilities) @ jacobian)[0, 0])
    comparison = compare_independent_sum(
        name="independent",
        description="two independent draws",
        node_fisher={"Y1": np.asarray([[single]]), "Y2": np.asarray([[single]])},
        joint_probabilities=joint_probabilities,
        joint_jacobian=joint_jacobian,
    )
    assert comparison.joint_trace == pytest.approx(2.0 / (p * (1 - p)), rel=1e-10)
    assert comparison.difference == pytest.approx(0.0, abs=1e-10)


def test_independent_joint_jacobian_matches_finite_difference() -> None:
    p = 0.37
    probabilities = np.asarray([1.0 - p, p])
    jacobian = np.asarray([[-1.0], [1.0]])
    outcomes, joint_probabilities = independent_draw_joint(["0", "1"], probabilities, draws=3)
    joint_jacobian = independent_draw_jacobian(
        ["0", "1"], probabilities, jacobian, draws=3
    )
    step = 1e-6
    plus_outcomes, plus = independent_draw_joint(
        ["0", "1"], np.asarray([1.0 - (p + step), p + step]), draws=3
    )
    minus_outcomes, minus = independent_draw_joint(
        ["0", "1"], np.asarray([1.0 - (p - step), p - step]), draws=3
    )
    assert plus_outcomes == minus_outcomes == outcomes
    expected = (plus - minus) / (2 * step)
    assert np.allclose(joint_jacobian[:, 0], expected, rtol=1e-6, atol=1e-9)
