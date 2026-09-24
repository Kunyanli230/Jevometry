"""Redundancy: independent-sum versus correct-joint information.

The central identity demonstrated here is that deterministically copying a
probability output adds no information, while independently re-sampling the
same declared distribution doubles the information of the *declared sampling
experiment* without creating new external evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.schemas.common import MetricStatus

FloatArray = NDArray[np.float64]


@dataclass
class RedundancyComparison:
    """Independent-sum versus declared-joint information."""

    name: str
    description: str
    independent_sum_trace: float | None
    joint_trace: float | None
    difference: float | None
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = field(default_factory=list)
    node_traces: dict[str, float] = field(default_factory=dict)


def _trace(matrix: FloatArray) -> float:
    return float(np.trace(matrix))


def compare_independent_sum(
    *,
    name: str,
    description: str,
    node_fisher: Mapping[str, FloatArray],
    joint_probabilities: FloatArray,
    joint_jacobian: FloatArray,
    zero_atol: float = 1e-15,
) -> RedundancyComparison:
    """Compare sum of node Fisher traces against the declared joint's trace."""
    values = np.asarray(joint_probabilities, dtype=np.float64)
    jacobian = np.asarray(joint_jacobian, dtype=np.float64)
    node_traces = {node: _trace(np.asarray(matrix, dtype=np.float64)) for node, matrix in node_fisher.items()}
    independent_sum = float(sum(node_traces.values()))
    if np.any(values <= zero_atol):
        return RedundancyComparison(
            name=name,
            description=description,
            independent_sum_trace=independent_sum,
            joint_trace=None,
            difference=None,
            status=MetricStatus.UNDEFINED,
            reason_code="zero_probability_outcome",
            assumptions=["no epsilon smoothing applied"],
            node_traces=node_traces,
        )
    weighted = jacobian / np.sqrt(values)[:, None]
    joint_information = weighted.T @ weighted
    joint_trace = _trace(joint_information)
    return RedundancyComparison(
        name=name,
        description=description,
        independent_sum_trace=independent_sum,
        joint_trace=joint_trace,
        difference=independent_sum - joint_trace,
        assumptions=[
            "node Fisher traces are summed only as an explicitly assumed baseline",
            "joint trace comes from the declared joint distribution",
        ],
        node_traces=node_traces,
    )


def deterministic_copy_joint(
    outcome_values: Sequence[str], probabilities: FloatArray
) -> tuple[list[tuple[str, ...]], FloatArray]:
    """Joint of (Y, Z) with Z = Y: support is the diagonal only."""
    values = np.asarray(probabilities, dtype=np.float64)
    outcomes: list[tuple[str, ...]] = [(outcome, outcome) for outcome in outcome_values]
    return outcomes, values.copy()


def deterministic_copy_jacobian(jacobian: FloatArray) -> FloatArray:
    """Jacobian of the diagonal joint equals the node Jacobian."""
    return np.asarray(jacobian, dtype=np.float64).copy()


def independent_draw_joint(
    outcome_values: Sequence[str], probabilities: FloatArray, *, draws: int = 2
) -> tuple[list[tuple[str, ...]], FloatArray]:
    """Joint of ``draws`` independent categorical draws from the same vector."""
    values = np.asarray(probabilities, dtype=np.float64)
    outcomes: list[tuple[str, ...]] = [()]
    masses = np.asarray([1.0])
    for _ in range(draws):
        outcomes = [(*prefix, outcome) for prefix in outcomes for outcome in outcome_values]
        masses = np.concatenate([masses * probability for probability in values])
    return outcomes, masses


def independent_draw_jacobian(
    outcome_values: Sequence[str], probabilities: FloatArray, jacobian: FloatArray, *, draws: int = 2
) -> FloatArray:
    """Exact Jacobian of the independent-draws joint via the product rule."""
    values = np.asarray(probabilities, dtype=np.float64)
    jac = np.asarray(jacobian, dtype=np.float64)
    outcomes: list[tuple[int, ...]] = [()]
    for _ in range(draws):
        outcomes = [(*prefix, index) for prefix in outcomes for index in range(len(values))]
    result = np.zeros((len(outcomes), jac.shape[1]), dtype=np.float64)
    for row, outcome in enumerate(outcomes):
        product = 1.0
        for index in outcome:
            product *= float(values[index])
        derivative = np.zeros(jac.shape[1], dtype=np.float64)
        for position, index in enumerate(outcome):
            others = 1.0
            for other_position, other_index in enumerate(outcome):
                if other_position == position:
                    continue
                others *= float(values[other_index])
            derivative += others * jac[index, :]
        result[row, :] = derivative
    return result
