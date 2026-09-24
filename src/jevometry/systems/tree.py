"""Finite conditional decision trees with exact path enumeration.

A tree is a fixed, finite sequence of conditional nodes.  Each node's
distribution may depend on the observed history; nodes may be inactive for a
given history, in which case the path records the stop outcome.  Nothing is
truncated: the enumerated path probabilities sum to one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.schemas.common import MetricStatus
from jevometry.schemas.joint import ConstructionMode, JointDistribution

FloatArray = NDArray[np.float64]

DistributionFunction = Callable[[Mapping[str, float], Mapping[str, str]], Mapping[str, float]]
JacobianFunction = Callable[
    [Mapping[str, float], Mapping[str, str]], Mapping[str, Mapping[str, float]]
]
ActiveFunction = Callable[[Mapping[str, str]], bool]

STOP_OUTCOME = "<stop>"


@dataclass(frozen=True)
class ConditionalNode:
    """One conditional decision node in a tree."""

    node_id: str
    outcomes: tuple[str, ...]
    distribution: DistributionFunction
    jacobian: JacobianFunction | None = None
    active: ActiveFunction | None = None


@dataclass
class TreeEnumeration:
    """Exact enumeration of all paths through a conditional tree."""

    tree_id: str
    node_order: list[str]
    outcomes: list[tuple[str, ...]]
    probabilities: FloatArray
    history_before: list[list[dict[str, str]]]
    active_flags: list[list[bool]]
    total: float
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = field(default_factory=list)

    def visit_probabilities(self) -> dict[str, float]:
        """Probability that each node is actually visited."""
        visits = {node: 0.0 for node in self.node_order}
        for _path, probability, flags in zip(
            self.outcomes, self.probabilities, self.active_flags, strict=True
        ):
            for node, active in zip(self.node_order, flags, strict=True):
                if active:
                    visits[node] += float(probability)
        return visits

    def history_groups(
        self, node_index: int
    ) -> dict[tuple[tuple[str, str], ...], float]:
        """Group path mass by the history prefix before ``node_index``."""
        groups: dict[tuple[tuple[str, str], ...], float] = {}
        for probability, history, flags in zip(
            self.probabilities, self.history_before, self.active_flags, strict=True
        ):
            if not flags[node_index]:
                continue
            key = tuple(sorted(history[node_index].items()))
            groups[key] = groups.get(key, 0.0) + float(probability)
        return groups


@dataclass
class ConditionalTree:
    """A finite conditional tree declaration."""

    tree_id: str
    nodes: Sequence[ConditionalNode]

    @property
    def node_order(self) -> list[str]:
        return [node.node_id for node in self.nodes]

    def conditional_distribution(
        self, node_index: int, theta: Mapping[str, float], history: Mapping[str, str]
    ) -> dict[str, float]:
        node = self.nodes[node_index]
        values = node.distribution(theta, history)
        missing = set(node.outcomes) - set(values)
        extra = set(values) - set(node.outcomes)
        if missing or extra:
            raise ValueError(
                f"node {node.node_id!r} returned outcomes outside its declaration: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        return {outcome: float(values[outcome]) for outcome in node.outcomes}

    def is_active(self, node_index: int, history: Mapping[str, str]) -> bool:
        node = self.nodes[node_index]
        if node.active is None:
            return True
        return bool(node.active(history))


MAX_TREE_PATHS = 4096


def _worst_case_paths(tree: ConditionalTree) -> int:
    count = 1
    for node in tree.nodes:
        count *= max(len(node.outcomes), 1)
        if count > MAX_TREE_PATHS:
            return count
    return count


def enumerate_tree(
    tree: ConditionalTree,
    theta: Mapping[str, float],
    *,
    max_paths: int = MAX_TREE_PATHS,
) -> TreeEnumeration:
    """Enumerate every path with exact probability (no truncation).

    When the worst-case path count exceeds the declared cap the enumeration is
    refused with the estimated requirement instead of being truncated.
    """
    estimated = _worst_case_paths(tree)
    if estimated > max_paths:
        return TreeEnumeration(
            tree_id=tree.tree_id,
            node_order=tree.node_order,
            outcomes=[],
            probabilities=np.zeros(0, dtype=np.float64),
            history_before=[],
            active_flags=[],
            total=0.0,
            status=MetricStatus.UNSUPPORTED,
            reason_code="enumeration_limit_exceeded",
            assumptions=[
                f"worst-case path count {estimated} exceeds the exact enumeration limit "
                f"{max_paths}; the tree was not truncated"
            ],
        )
    outcomes: list[tuple[str, ...]] = []
    probabilities: list[float] = []
    histories: list[list[dict[str, str]]] = []
    active_flags: list[list[bool]] = []

    def walk(
        index: int,
        history: dict[str, str],
        weight: float,
        path: list[str],
        history_chain: list[dict[str, str]],
        flags: list[bool],
    ) -> None:
        if index == len(tree.nodes):
            outcomes.append(tuple(path))
            probabilities.append(weight)
            histories.append(history_chain)
            active_flags.append(flags)
            return
        node = tree.nodes[index]
        active = tree.is_active(index, history)
        history_chain = [*history_chain, dict(history)]
        flags = [*flags, active]
        if not active:
            walk(index + 1, history, weight, [*path, STOP_OUTCOME], history_chain, flags)
            return
        distribution = tree.conditional_distribution(index, theta, history)
        for outcome in node.outcomes:
            probability = distribution[outcome]
            new_history = {**history, node.node_id: outcome}
            walk(
                index + 1,
                new_history,
                weight * probability,
                [*path, outcome],
                history_chain,
                flags,
            )

    walk(0, {}, 1.0, [], [], [])
    values = np.asarray(probabilities, dtype=np.float64)
    return TreeEnumeration(
        tree_id=tree.tree_id,
        node_order=tree.node_order,
        outcomes=outcomes,
        probabilities=values,
        history_before=histories,
        active_flags=active_flags,
        total=float(values.sum()),
        assumptions=["exact enumeration of a finite declared conditional tree"],
    )


def tree_joint(tree: ConditionalTree, theta: Mapping[str, float]) -> JointDistribution:
    """Exact joint distribution over tree paths."""
    enumeration = enumerate_tree(tree, theta)
    if enumeration.status is not MetricStatus.OK:
        raise ValueError(
            f"tree enumeration refused: {enumeration.reason_code}; "
            f"{' '.join(enumeration.assumptions)}"
        )
    return JointDistribution(
        mode=ConstructionMode.CONDITIONAL_TREE,
        node_order=enumeration.node_order,
        outcomes=enumeration.outcomes,
        probabilities=[float(value) for value in enumeration.probabilities],
        assumptions=list(enumeration.assumptions),
        theta=dict(theta),
    )


def tree_path_jacobian(
    tree: ConditionalTree, theta: Mapping[str, float]
) -> tuple[list[tuple[str, ...]], FloatArray | None]:
    """Exact path-probability Jacobian via the product rule.

    Uses each node's analytic conditional Jacobian when provided.  Returns
    ``None`` when any visited node lacks an analytic Jacobian, so that callers
    can fall back to finite differences explicitly.
    """
    parameter_names = _parameter_names(tree, theta)
    enumeration = enumerate_tree(tree, theta)
    jacobian = np.zeros((len(enumeration.outcomes), len(parameter_names)), dtype=np.float64)
    for path_index, (history_chain, flags) in enumerate(
        zip(enumeration.history_before, enumeration.active_flags, strict=True)
    ):
        factors: list[tuple[int, str, float]] = []
        for index, node in enumerate(tree.nodes):
            if not flags[index]:
                continue
            if node.jacobian is None:
                return enumeration.outcomes, None
            distribution = tree.conditional_distribution(index, theta, history_chain[index])
            factors.append((index, enumeration.outcomes[path_index][index], distribution[enumeration.outcomes[path_index][index]]))
        prefix = [1.0]
        for _, _, probability in factors:
            prefix.append(prefix[-1] * probability)
        suffix = [1.0]
        for _, _, probability in reversed(factors):
            suffix.append(suffix[-1] * probability)
        suffix.reverse()
        for position, (index, outcome, _) in enumerate(factors):
            node = tree.nodes[index]
            assert node.jacobian is not None
            derivatives = node.jacobian(theta, history_chain[index])
            for axis, parameter in enumerate(parameter_names):
                derivative = float(derivatives[outcome][parameter])
                jacobian[path_index, axis] += derivative * prefix[position] * suffix[position + 1]
    return enumeration.outcomes, jacobian


def _parameter_names(tree: ConditionalTree, theta: Mapping[str, float]) -> list[str]:
    if theta:
        return list(theta)
    return []


@dataclass
class ConditionalFisherIdentity:
    """Trajectory Fisher from conditional Fisher information.

    ``I_trajectory = sum_i E_history[I_i(theta | history)]`` holds only under the
    declared conditional distributions and standard regularity conditions.  It
    is never the sum of node metrics along one realised trajectory.
    """

    per_node: dict[str, FloatArray]
    total: FloatArray | None
    joint_fisher: FloatArray | None
    parameter_names: list[str]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    residual: float | None = None
    assumptions: list[str] = field(
        default_factory=lambda: [
            "declared conditional distributions with a shared theta",
            "regularity: conditional Fisher finite and differentiable",
        ]
    )
    diagnostics: list[str] = field(default_factory=list)


def conditional_fisher_identity(
    tree: ConditionalTree,
    theta: Mapping[str, float],
    *,
    parameter_names: Sequence[str],
    parameter_steps: Mapping[str, float] | None = None,
    relative_tolerance: float = 1e-8,
) -> ConditionalFisherIdentity:
    """Compare exact joint Fisher with the history-weighted conditional sum."""
    names = list(parameter_names)
    enumeration = enumerate_tree(tree, theta)
    per_node: dict[str, FloatArray] = {}
    total = np.zeros((len(names), len(names)), dtype=np.float64)
    diagnostics: list[str] = []
    zero_found = False
    for index, node in enumerate(tree.nodes):
        contribution = np.zeros((len(names), len(names)), dtype=np.float64)
        groups = enumeration.history_groups(index)
        for history_key, weight in groups.items():
            history = dict(history_key)
            distribution = tree.conditional_distribution(index, theta, history)
            vector = np.asarray([distribution[outcome] for outcome in node.outcomes], dtype=np.float64)
            if np.any(vector <= 0.0):
                zero_found = True
                diagnostics.append(
                    f"node {node.node_id!r}: zero conditional probability for history {history!r}"
                )
                continue
            jacobian = _conditional_jacobian(
                tree, index, theta, history, node.outcomes, names, parameter_steps
            )
            if jacobian is None:
                return ConditionalFisherIdentity(
                    per_node=per_node,
                    total=None,
                    joint_fisher=None,
                    parameter_names=names,
                    status=MetricStatus.UNSUPPORTED,
                    reason_code="missing_conditional_jacobian",
                    diagnostics=diagnostics,
                )
            weighted = jacobian / np.sqrt(vector)[:, None]
            contribution += weight * (weighted.T @ weighted)
        per_node[node.node_id] = contribution
        total += contribution
    outcomes, joint_jacobian = tree_path_jacobian(tree, theta)
    if joint_jacobian is None:
        return ConditionalFisherIdentity(
            per_node=per_node,
            total=total,
            joint_fisher=None,
            parameter_names=names,
            status=MetricStatus.CONDITIONAL,
            reason_code="missing_path_jacobian",
            diagnostics=diagnostics,
        )
    path_values = np.asarray(enumeration.probabilities, dtype=np.float64)
    if np.any(path_values <= 0.0):
        zero_found = True
    if zero_found:
        return ConditionalFisherIdentity(
            per_node=per_node,
            total=total,
            joint_fisher=None,
            parameter_names=names,
            status=MetricStatus.UNDEFINED,
            reason_code="zero_probability_outcome",
            diagnostics=diagnostics,
        )
    path_weighted = joint_jacobian / np.sqrt(path_values)[:, None]
    joint_fisher = path_weighted.T @ path_weighted
    denominator = max(float(np.linalg.norm(joint_fisher, ord="fro")), 1e-12)
    residual = float(np.linalg.norm(total - joint_fisher, ord="fro") / denominator)
    status = MetricStatus.OK
    reason_code: str | None = None
    if residual > relative_tolerance:
        status = MetricStatus.UNSTABLE
        reason_code = "conditional_identity_residual"
        diagnostics.append(f"conditional identity residual {residual:.3e}")
    return ConditionalFisherIdentity(
        per_node=per_node,
        total=total,
        joint_fisher=joint_fisher,
        parameter_names=names,
        status=status,
        reason_code=reason_code,
        residual=residual,
        diagnostics=diagnostics,
    )


def _conditional_jacobian(
    tree: ConditionalTree,
    index: int,
    theta: Mapping[str, float],
    history: Mapping[str, str],
    outcomes: Sequence[str],
    parameter_names: Sequence[str],
    parameter_steps: Mapping[str, float] | None,
) -> FloatArray | None:
    node = tree.nodes[index]
    if node.jacobian is not None:
        derivatives = node.jacobian(theta, history)
        return np.asarray(
            [
                [float(derivatives[outcome][parameter]) for parameter in parameter_names]
                for outcome in outcomes
            ],
            dtype=np.float64,
        )
    if parameter_steps is None:
        return None
    jacobian = np.zeros((len(outcomes), len(parameter_names)), dtype=np.float64)
    for axis, parameter in enumerate(parameter_names):
        step = parameter_steps[parameter]
        plus = dict(theta)
        minus = dict(theta)
        plus[parameter] = theta[parameter] + step
        minus[parameter] = theta[parameter] - step
        plus_values = tree.conditional_distribution(index, plus, history)
        minus_values = tree.conditional_distribution(index, minus, history)
        jacobian[:, axis] = [
            (plus_values[outcome] - minus_values[outcome]) / (2.0 * step) for outcome in outcomes
        ]
    return jacobian
