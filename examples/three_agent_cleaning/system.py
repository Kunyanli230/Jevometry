"""Three declared agent distributions and a fixed cleaning decision rule."""

from __future__ import annotations

from collections.abc import Mapping
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
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_fix"
    return "human_review"


def build_system() -> tuple[dict[str, AnalyticNode], AnalyticJointModel, AggregationMap]:
    """Return the agents, declared product law and deterministic coordinator."""
    nodes = {
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

    joint = AnalyticJointModel(
        node_order=node_order,
        parameter_names=list(PARAMETERS),
        probability_function=probabilities,
        jacobian_function=jacobian,
        model_identity="three-agent-cleaning:declared-product:v1",
        construction_mode=ConstructionMode.DECLARED_PRODUCT,
        assumptions=["agents are conditionally independent given evidence and ambiguity"],
    )
    mapping = {
        path: final_action(path) for path in product(*(outcomes[name] for name in node_order))
    }
    policy = AggregationMap(
        name="final_action",
        mapping=mapping,
        description="Fixed coordinator: reject, auto_fix or human_review",
    )
    return nodes, joint, policy
