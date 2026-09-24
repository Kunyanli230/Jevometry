"""Declared finite joint models, marginals and information measures.

System-level Fisher information only exists once a joint distribution (or an
explicitly declared combination model) is available.  Nothing here recovers a
joint from marginals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jevometry.schemas.common import MetricStatus
from jevometry.schemas.joint import ConstructionMode, JointDistribution

FloatArray = NDArray[np.float64]


@dataclass
class JointFisherResult:
    """Fisher information of a declared joint family."""

    values: FloatArray | None
    node_order: list[str]
    parameter_names: list[str]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = field(default_factory=list)


def joint_fisher(
    probabilities: FloatArray,
    jacobian: FloatArray,
    *,
    node_order: Sequence[str],
    parameter_names: Sequence[str],
    zero_atol: float = 1e-15,
) -> JointFisherResult:
    """I(theta) = J^T diag(1/P) J for a declared joint distribution."""
    values = np.asarray(probabilities, dtype=np.float64)
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.shape[0] != values.shape[0]:
        raise ValueError("joint Jacobian rows must match outcome count")
    if jac.shape[1] != len(parameter_names):
        raise ValueError("joint Jacobian columns must match parameter count")
    if np.any(values < -zero_atol):
        return JointFisherResult(
            values=None,
            node_order=list(node_order),
            parameter_names=list(parameter_names),
            status=MetricStatus.FAILED,
            reason_code="negative_joint_probability",
        )
    if np.any(values <= zero_atol):
        return JointFisherResult(
            values=None,
            node_order=list(node_order),
            parameter_names=list(parameter_names),
            status=MetricStatus.UNDEFINED,
            reason_code="zero_probability_outcome",
            assumptions=["no epsilon smoothing applied"],
        )
    weighted = jac / np.sqrt(values)[:, None]
    return JointFisherResult(
        values=weighted.T @ weighted,
        node_order=list(node_order),
        parameter_names=list(parameter_names),
    )


def marginalize(joint: JointDistribution, keep: Sequence[str]) -> JointDistribution:
    """Marginalise a joint onto a subset of nodes."""
    keep_list = list(keep)
    unknown = set(keep_list) - set(joint.node_order)
    if unknown:
        raise KeyError(f"unknown nodes: {sorted(unknown)}")
    axes = [joint.node_order.index(node) for node in keep_list]
    masses: dict[tuple[str, ...], float] = {}
    for outcome, probability in zip(joint.outcomes, joint.probabilities, strict=True):
        key = tuple(outcome[axis] for axis in axes)
        masses[key] = masses.get(key, 0.0) + probability
    outcomes = sorted(masses)
    return JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=keep_list,
        outcomes=outcomes,
        probabilities=[masses[key] for key in outcomes],
        assumptions=[*joint.assumptions, "marginalised from a declared joint"],
    )


def _entropy_from_masses(masses: Mapping[Any, float]) -> float:
    total = 0.0
    for probability in masses.values():
        if probability > 0.0:
            total -= probability * math.log(probability)
    return total


def mutual_information(joint: JointDistribution, left: str, right: str) -> float:
    """Mutual information I(X;Y) from a declared joint."""
    if left == right:
        return float(_entropy_from_masses(joint.marginal(left)))
    joint_masses: dict[tuple[str, str], float] = {}
    for outcome, probability in zip(joint.outcomes, joint.probabilities, strict=True):
        key = (outcome[joint.node_order.index(left)], outcome[joint.node_order.index(right)])
        joint_masses[key] = joint_masses.get(key, 0.0) + probability
    left_marginal = joint.marginal(left)
    right_marginal = joint.marginal(right)
    information = 0.0
    for (x_value, y_value), probability in joint_masses.items():
        if probability <= 0.0:
            continue
        denominator = left_marginal[x_value] * right_marginal[y_value]
        information += probability * math.log(probability / denominator)
    return float(information)


def conditional_mutual_information(
    joint: JointDistribution, left: str, right: str, given: str
) -> float:
    """Conditional mutual information I(X;Y|Z) from a declared joint."""
    for node in (left, right, given):
        if node not in joint.node_order:
            raise KeyError(node)
    masses: dict[tuple[str, str, str], float] = {}
    for outcome, probability in zip(joint.outcomes, joint.probabilities, strict=True):
        key = (
            outcome[joint.node_order.index(left)],
            outcome[joint.node_order.index(right)],
            outcome[joint.node_order.index(given)],
        )
        masses[key] = masses.get(key, 0.0) + probability
    given_marginal = joint.marginal(given)
    information = 0.0
    for z_value, p_z in given_marginal.items():
        if p_z <= 0.0:
            continue
        per_z = {
            (x_value, y_value): probability
            for (x_value, y_value, key_z), probability in masses.items()
            if key_z == z_value
        }
        x_marginal: dict[str, float] = {}
        y_marginal: dict[str, float] = {}
        for (x_value, y_value), probability in per_z.items():
            x_marginal[x_value] = x_marginal.get(x_value, 0.0) + probability
            y_marginal[y_value] = y_marginal.get(y_value, 0.0) + probability
        contribution = 0.0
        for (x_value, y_value), probability in per_z.items():
            if probability <= 0.0:
                continue
            denominator = x_marginal[x_value] * y_marginal[y_value]
            if denominator <= 0.0:
                continue
            contribution += probability * math.log(probability * p_z / denominator)
        information += contribution
    return float(information)


DECLARED_PRODUCT_ASSUMPTIONS: tuple[str, ...] = (
    "declared conditional independence across nodes",
    "shared external theta across nodes",
)


def product_joint_from_nodes(
    node_order: Sequence[str],
    outcome_sets: Mapping[str, Sequence[str]],
    distributions: Mapping[str, FloatArray],
    *,
    assumptions: Sequence[str],
    theta: Mapping[str, float],
) -> JointDistribution:
    """Build P(z) = prod_i q_i(z_i) from declared node vectors."""
    grid: list[tuple[str, ...]] = [()]
    masses: list[float] = [1.0]
    for node in node_order:
        outcomes = list(outcome_sets[node])
        vector = np.asarray(distributions[node], dtype=np.float64)
        if vector.shape[0] != len(outcomes):
            raise ValueError(f"node {node!r} distribution width does not match its outcomes")
        new_grid: list[tuple[str, ...]] = []
        new_masses: list[float] = []
        for prefix, mass in zip(grid, masses, strict=True):
            for outcome, probability in zip(outcomes, vector, strict=True):
                new_grid.append((*prefix, outcome))
                new_masses.append(mass * float(probability))
        grid = new_grid
        masses = new_masses
    return JointDistribution(
        mode=ConstructionMode.DECLARED_PRODUCT,
        node_order=list(node_order),
        outcomes=grid,
        probabilities=masses,
        assumptions=list(assumptions),
        theta=dict(theta),
    )


def product_joint_jacobian(
    node_order: Sequence[str],
    outcome_sets: Mapping[str, Sequence[str]],
    distributions: Mapping[str, FloatArray],
    jacobians: Mapping[str, FloatArray],
    parameter_names: Sequence[str],
) -> FloatArray:
    """Exact Jacobian of a declared product joint via the product rule."""
    rows: list[tuple[str, ...]] = [()]
    for node in node_order:
        rows = [(*prefix, outcome) for prefix in rows for outcome in outcome_sets[node]]
    jacobian = np.zeros((len(rows), len(parameter_names)), dtype=np.float64)
    for index, row in enumerate(rows):
        product = 1.0
        for node, outcome in zip(node_order, row, strict=True):
            outcomes = list(outcome_sets[node])
            product *= float(distributions[node][outcomes.index(outcome)])
        derivative = np.zeros(len(parameter_names), dtype=np.float64)
        for node, outcome in zip(node_order, row, strict=True):
            outcomes = list(outcome_sets[node])
            position = outcomes.index(outcome)
            conditional = float(distributions[node][position])
            if conditional == 0.0:
                other_product = 1.0
                for other, other_outcome in zip(node_order, row, strict=True):
                    if other == node:
                        continue
                    other_outcomes = list(outcome_sets[other])
                    other_product *= float(
                        distributions[other][other_outcomes.index(other_outcome)]
                    )
                derivative += other_product * np.asarray(
                    jacobians[node][position, :], dtype=np.float64
                )
            else:
                derivative += (product / conditional) * np.asarray(
                    jacobians[node][position, :], dtype=np.float64
                )
        jacobian[index, :] = derivative
    return jacobian
