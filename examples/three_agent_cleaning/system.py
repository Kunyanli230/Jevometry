"""Three declared agent distributions and interchangeable cleaning decision rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from jevometry.adapters.analytic import (
    AnalyticJointModel,
    AnalyticNode,
    choice_question,
    noul_question,
    score_question,
)
from jevometry.schemas.joint import ConstructionMode
from jevometry.schemas.questions import QuestionSpec
from jevometry.systems.joint import product_joint_from_nodes, product_joint_jacobian
from jevometry.systems.pushforward import AggregationMap

FloatArray = NDArray[np.float64]
CENTER = np.asarray([0.7, 0.3], dtype=np.float64)
PARAMETERS = ("evidence", "ambiguity")
RuleFn = Callable[[tuple[str, str, str]], str]


def _agent(
    name: str,
    question: QuestionSpec,
    center_probabilities: list[float],
    weights: list[list[float]],
) -> AnalyticNode:
    """A local softmax family with a transparent centre and two input slopes."""
    baseline = np.asarray(center_probabilities, dtype=np.float64)
    slopes = np.asarray(weights, dtype=np.float64)

    def probabilities(theta: Mapping[str, float]) -> FloatArray:
        point = np.asarray([theta[name] for name in PARAMETERS], dtype=np.float64)
        return np.asarray(softmax(np.log(baseline) + slopes @ (point - CENTER)), dtype=np.float64)

    def jacobian(theta: Mapping[str, float]) -> FloatArray:
        values = probabilities(theta)
        return np.asarray(values[:, None] * (slopes - values @ slopes), dtype=np.float64)

    return AnalyticNode(
        node_id=name,
        question=question,
        model_identity=f"three-agent-cleaning:{name}:v1",
        probability_function=probabilities,
        jacobian_function=jacobian,
        description=f"Declared analytic model for {name}",
    )


def final_action(outcome: tuple[str, str, str]) -> str:
    """Baseline coordinator: reject / auto_fix / human_review."""
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_fix"
    return "human_review"


def final_action_stricter(outcome: tuple[str, str, str]) -> str:
    """Stricter: mid-risk also rejects (in addition to baseline rejects)."""
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk in {"1", "2"}:
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_fix"
    return "human_review"


def final_action_wider(outcome: tuple[str, str, str]) -> str:
    """Wider: impute+low-risk can also auto-apply, with a distinct label.

    Reject conditions match baseline. Using separate auto_normalize /
    auto_impute labels avoids merging those paths into one coarse action
    (or into human_review), so Fisher retention is typically higher.
    """
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_normalize"
    if repair == "impute" and risk == "0":
        return "auto_impute"
    return "human_review"


def final_action_near_identity(outcome: tuple[str, str, str]) -> str:
    """Near-identity upper bound: one label per joint outcome tuple."""
    applicable, repair, risk = outcome
    return f"{applicable}|{repair}|{risk}"


def final_action_quarantine_review(outcome: tuple[str, str, str]) -> str:
    """Softer: quarantine goes to human_review instead of reject.

    Reject only on inapplicable or high risk. Quarantine is treated as a
    reviewable decision rather than an automatic hard reject.
    """
    applicable, repair, risk = outcome
    if applicable == "false" or risk == "2":
        return "reject"
    if repair == "quarantine":
        return "human_review"
    if repair == "normalize" and risk == "0":
        return "auto_fix"
    return "human_review"


def final_action_coarse_binary(outcome: tuple[str, str, str]) -> str:
    """Coarse binary gate: reject vs proceed (merges auto_fix and review)."""
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    return "proceed"


def build_aggregation_map(
    name: str,
    rule: RuleFn,
    node_order: Sequence[str],
    outcomes: Mapping[str, Sequence[str]],
    *,
    description: str | None = None,
) -> AggregationMap:
    """Build a theta-independent AggregationMap covering the Cartesian product."""
    mapping = {
        path: rule(path)  # type: ignore[arg-type]
        for path in product(*(outcomes[name] for name in node_order))
    }
    return AggregationMap(
        name=name,
        mapping=mapping,
        theta_dependent=False,
        description=description,
    )


def _build_nodes() -> dict[str, AnalyticNode]:
    return {
        "applicability_agent": _agent(
            "applicability_agent",
            noul_question("applicability_agent"),
            [0.2, 0.8],
            [[0.0, 0.0], [3.6, -2.5]],
        ),
        "repair_agent": _agent(
            "repair_agent",
            choice_question(
                "repair_agent",
                {
                    "normalize": "parse the thousands separator",
                    "impute": "infer a missing value",
                    "quarantine": "defer the record",
                },
            ),
            [0.5, 0.4, 0.1],
            [[1.86, 0.0], [0.0, 0.93], [-0.93, -1.86]],
        ),
        "risk_agent": _agent(
            "risk_agent",
            score_question("risk_agent", ["low", "medium", "high"]),
            [0.59, 0.146, 0.264],
            [[1.526, -0.763], [0.0, 0.0], [-0.763, 1.526]],
        ),
    }


def _build_joint(nodes: Mapping[str, AnalyticNode]) -> AnalyticJointModel:
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}

    def probabilities(theta: Mapping[str, float]) -> tuple[list[tuple[str, ...]], FloatArray]:
        joint = product_joint_from_nodes(
            node_order,
            outcomes,
            {name: node.probabilities(theta) for name, node in nodes.items()},
            assumptions=["agents are conditionally independent given evidence and ambiguity"],
            theta=theta,
        )
        return joint.outcomes, np.asarray(joint.probabilities, dtype=np.float64)

    def jacobian(theta: Mapping[str, float]) -> FloatArray:
        return product_joint_jacobian(
            node_order,
            outcomes,
            {name: node.probabilities(theta) for name, node in nodes.items()},
            {name: node.jacobian(theta) for name, node in nodes.items()},
            PARAMETERS,
        )

    return AnalyticJointModel(
        node_order=node_order,
        parameter_names=list(PARAMETERS),
        probability_function=probabilities,
        jacobian_function=jacobian,
        model_identity="three-agent-cleaning:declared-product:v1",
        construction_mode=ConstructionMode.DECLARED_PRODUCT,
        assumptions=["agents are conditionally independent given evidence and ambiguity"],
    )


RULE_SPECS: list[tuple[str, RuleFn, str]] = [
    ("baseline", final_action, "Fixed coordinator: reject, auto_fix or human_review"),
    ("stricter", final_action_stricter, "Mid-risk also rejects"),
    ("wider", final_action_wider, "Impute+low-risk can auto-apply (distinct labels)"),
    (
        "quarantine_review",
        final_action_quarantine_review,
        "Quarantine routes to human_review instead of reject",
    ),
    (
        "coarse_binary",
        final_action_coarse_binary,
        "Binary reject vs proceed (auto_fix and review merged)",
    ),
    (
        "near_identity",
        final_action_near_identity,
        "One label per joint outcome (Fisher retention upper bound)",
    ),
]


def build_rule_maps(
    nodes: Mapping[str, AnalyticNode] | None = None,
) -> list[tuple[str, AggregationMap]]:
    """Return named AggregationMaps for baseline and alternative rules."""
    if nodes is None:
        nodes = _build_nodes()
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}
    return [
        (
            name,
            build_aggregation_map(name, rule, node_order, outcomes, description=description),
        )
        for name, rule, description in RULE_SPECS
    ]


def build_system() -> tuple[dict[str, AnalyticNode], AnalyticJointModel, AggregationMap]:
    """Return the agents, declared product law and baseline coordinator."""
    nodes = _build_nodes()
    joint = _build_joint(nodes)
    baseline = dict(build_rule_maps(nodes))["baseline"]
    # Keep the historical AggregationMap name so run.py / reports stay stable.
    policy = AggregationMap(
        name="final_action",
        mapping=baseline.mapping,
        theta_dependent=False,
        description=baseline.description,
    )
    return nodes, joint, policy
