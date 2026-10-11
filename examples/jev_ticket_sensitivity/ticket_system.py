"""A side-effect-free synthetic ticket-triage system for Example C.

The system has three fixed Jev questions over a rendered ticket state:

* ``request_type``  Choice: billing / technical / refund
* ``needs_escalation``  Noul: yes/no
* ``impact``  Score: low / medium / high

Probabilities depend on the two external parameters ``amount`` (USD) and
``wait_hours``.  A deterministic policy maps the reported probabilities to a
local action label.  Nothing here performs a side effect; the module is a
drop-in adapter for the offline analysis pipeline and the same shape can be
served by the live TypeSafe provider.

``build_adapter(config)`` is the module entrypoint used by ``jevometry run``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

from itertools import product

from jevometry.adapters.analytic import (
    AnalyticAdapter,
    AnalyticJointModel,
    AnalyticNode,
    choice_question,
    noul_question,
    score_question,
)
from jevometry.experiments.renderer import StructuredRenderer
from jevometry.schemas.common import RoutingSemantics
from jevometry.schemas.joint import ConstructionMode
from jevometry.systems.joint import product_joint_from_nodes, product_joint_jacobian
from jevometry.systems.pushforward import AggregationMap

ACTIONS = ("standard_queue", "priority_queue", "refund_review", "escalate")

STATE_TEMPLATE = {
    "amount_usd": "{amount:.2f}",
    "wait_hours": "{wait_hours:.2f}",
}
STATIC_STATE = {
    "ticket": "Synthetic ticket fixture; no customer data is present.",
    "channel": "email",
}


def renderer() -> StructuredRenderer:
    return StructuredRenderer(STATE_TEMPLATE, static=STATIC_STATE, version="ticket-v1")


LOGIT_SCALE = 40.0



def _logits(amount: float, wait: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax/logit scores scaled by LOGIT_SCALE.

    ParameterSpec uses raw amount/wait units, so unscaled slopes yield tiny Fisher
    traces; LOGIT_SCALE (default 40) restores usable curvature without changing the
    qualitative dependence on amount and wait_hours.
    """
    scaled_amount = amount / 500.0
    scaled_wait = wait / 24.0
    request = LOGIT_SCALE * np.asarray(
        [0.9 * scaled_wait, 1.1 * scaled_amount - 0.4 * scaled_wait, 0.6 * scaled_amount],
        dtype=np.float64,
    )
    escalation = LOGIT_SCALE * np.asarray(
        [1.6 * scaled_amount - 0.9 * scaled_wait - 0.6], dtype=np.float64
    )
    impact = LOGIT_SCALE * np.asarray(
        [
            -0.4 * scaled_amount + 0.2 * scaled_wait + 0.2,
            0.4 * scaled_amount - 0.3 * scaled_wait,
            1.0 * scaled_amount + 0.1 * scaled_wait - 0.4,
        ],
        dtype=np.float64,
    )
    return request, escalation, impact


def request_type_node() -> AnalyticNode:
    question = choice_question(
        "request_type",
        {"billing": "Billing question", "technical": "Technical issue", "refund": "Refund request"},
        instructions="What kind of request is this?",
    )

    def probabilities(theta: Mapping[str, float]) -> np.ndarray:
        request, _, _ = _logits(float(theta["amount"]), float(theta["wait_hours"]))
        shifted = request - request.max()
        exponentials = np.exp(shifted)
        return np.asarray(exponentials / exponentials.sum(), dtype=np.float64)

    def jacobian(theta: Mapping[str, float]) -> np.ndarray:
        amount = float(theta["amount"])
        wait = float(theta["wait_hours"])
        request, _, _ = _logits(amount, wait)
        shifted = request - request.max()
        q = np.exp(shifted) / np.exp(shifted).sum()
        # d logits / d theta
        derivative = LOGIT_SCALE * np.asarray(
            [
                [0.0, 0.9 / 24.0],
                [1.1 / 500.0, -0.4 / 24.0],
                [0.6 / 500.0, 0.0],
            ],
            dtype=np.float64,
        )
        return np.asarray(q[:, None] * (derivative - q @ derivative), dtype=np.float64)

    return AnalyticNode(
        node_id="request_type",
        question=question,
        model_identity="ticket:request_type",
        probability_function=probabilities,
        jacobian_function=jacobian,
    )


def needs_escalation_node() -> AnalyticNode:
    question = noul_question(
        "needs_escalation", instructions="Does this ticket need escalation?"
    )

    def probability(theta: Mapping[str, float]) -> float:
        amount = float(theta["amount"])
        wait = float(theta["wait_hours"])
        _, escalation, _ = _logits(amount, wait)
        return float(expit(float(escalation[0])))

    def probabilities(theta: Mapping[str, float]) -> np.ndarray:
        p = probability(theta)
        return np.asarray([1.0 - p, p], dtype=np.float64)

    def jacobian(theta: Mapping[str, float]) -> np.ndarray:
        p = probability(theta)
        derivative = p * (1.0 - p)
        # d logit / d theta scaled by LOGIT_SCALE
        d_amount = derivative * LOGIT_SCALE * (1.6 / 500.0)
        d_wait = derivative * LOGIT_SCALE * (-0.9 / 24.0)
        return np.asarray([[-d_amount, -d_wait], [d_amount, d_wait]], dtype=np.float64)

    return AnalyticNode(
        node_id="needs_escalation",
        question=question,
        model_identity="ticket:needs_escalation",
        probability_function=probabilities,
        jacobian_function=jacobian,
    )


def impact_node() -> AnalyticNode:
    question = score_question(
        "impact",
        ["low", "medium", "high"],
        numerics=[0.0, 1.0, 2.0],
        instructions="How severe is the impact?",
    )

    def probabilities(theta: Mapping[str, float]) -> np.ndarray:
        _, _, impact = _logits(float(theta["amount"]), float(theta["wait_hours"]))
        shifted = impact - impact.max()
        exponentials = np.exp(shifted)
        return np.asarray(exponentials / exponentials.sum(), dtype=np.float64)

    def jacobian(theta: Mapping[str, float]) -> np.ndarray:
        amount = float(theta["amount"])
        wait = float(theta["wait_hours"])
        _, _, impact = _logits(amount, wait)
        shifted = impact - impact.max()
        q = np.exp(shifted) / np.exp(shifted).sum()
        derivative = LOGIT_SCALE * np.asarray(
            [
                [-0.4 / 500.0, 0.2 / 24.0],
                [0.4 / 500.0, -0.3 / 24.0],
                [1.0 / 500.0, 0.1 / 24.0],
            ],
            dtype=np.float64,
        )
        return np.asarray(q[:, None] * (derivative - q @ derivative), dtype=np.float64)

    return AnalyticNode(
        node_id="impact",
        question=question,
        model_identity="ticket:impact",
        probability_function=probabilities,
        jacobian_function=jacobian,
    )


def build_nodes() -> dict[str, AnalyticNode]:
    return {
        "request_type": request_type_node(),
        "needs_escalation": needs_escalation_node(),
        "impact": impact_node(),
    }


@dataclass(frozen=True)
class TicketPolicy:
    """Deterministic policy over the reported probability vectors."""

    escalation_threshold: float = 0.6
    impact_threshold: float = 1.5

    def action(self, distribution: Mapping[str, float]) -> str:
        if distribution["needs_escalation.true"] >= self.escalation_threshold:
            return "escalate"
        if distribution["request_type.refund"] > max(
            distribution["request_type.billing"], distribution["request_type.technical"]
        ):
            return "refund_review"
        expected_impact = (
            distribution["impact.0"] * 0.0
            + distribution["impact.1"] * 1.0
            + distribution["impact.2"] * 2.0
        )
        if expected_impact >= self.impact_threshold:
            return "priority_queue"
        return "standard_queue"

    def __call__(self, theta: Mapping[str, float], distribution: Mapping[str, float]) -> str:
        del theta
        return self.action(distribution)


PARAMETERS = ("amount", "wait_hours")


def outcome_to_surrogate_action(
    outcome: tuple[str, ...], *, policy: "TicketPolicy | None" = None
) -> str:
    """Baseline hard-outcome map: matches ``TicketPolicy.action`` on one-hots."""
    rule = policy or TicketPolicy()
    request, escalation, impact = outcome
    distribution = {
        "request_type.billing": 1.0 if request == "billing" else 0.0,
        "request_type.technical": 1.0 if request == "technical" else 0.0,
        "request_type.refund": 1.0 if request == "refund" else 0.0,
        "needs_escalation.false": 1.0 if escalation == "false" else 0.0,
        "needs_escalation.true": 1.0 if escalation == "true" else 0.0,
        "impact.0": 1.0 if impact == "0" else 0.0,
        "impact.1": 1.0 if impact == "1" else 0.0,
        "impact.2": 1.0 if impact == "2" else 0.0,
    }
    return rule.action(distribution)


def rule_baseline(outcome: tuple[str, ...]) -> str:
    return outcome_to_surrogate_action(outcome)


def rule_stricter(outcome: tuple[str, ...]) -> str:
    """Escalate on true; mid or high impact goes to priority; refund still reviewed."""
    request, escalation, impact = outcome
    if escalation == "true":
        return "escalate"
    if request == "refund":
        return "refund_review"
    if impact in ("1", "2"):
        return "priority_queue"
    return "standard_queue"


def rule_coarse_binary(outcome: tuple[str, ...]) -> str:
    """Binary gate: escalate vs proceed."""
    _, escalation, _ = outcome
    return "escalate" if escalation == "true" else "proceed"


def rule_near_identity(outcome: tuple[str, ...]) -> str:
    """Upper-bound reference: one label per joint outcome tuple."""
    return "|".join(outcome)


def build_declared_joint(
    nodes: Mapping[str, AnalyticNode],
) -> AnalyticJointModel:
    """Shared DECLARED_PRODUCT joint (independent of aggregation rule)."""
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}
    assumptions = [
        "surrogate: nodes are conditionally independent given amount and wait_hours",
        "surrogate AggregationMap is theta-independent; not production routing",
    ]

    def probabilities(theta: Mapping[str, float]):
        joint = product_joint_from_nodes(
            node_order,
            outcomes,
            {name: node.probabilities(theta) for name, node in nodes.items()},
            assumptions=assumptions,
            theta=theta,
        )
        return joint.outcomes, np.asarray(joint.probabilities, dtype=np.float64)

    def jacobian(theta: Mapping[str, float]):
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
        model_identity="ticket:declared-product-surrogate:v1",
        construction_mode=ConstructionMode.DECLARED_PRODUCT,
        assumptions=assumptions,
    )


def build_aggregation_map(
    nodes: Mapping[str, AnalyticNode],
    rule,
    *,
    name: str,
    description: str,
) -> AggregationMap:
    node_order = list(nodes)
    outcomes = {name_: node.support for name_, node in nodes.items()}
    mapping = {
        path: rule(path) for path in product(*(outcomes[name_] for name_ in node_order))
    }
    return AggregationMap(name=name, mapping=mapping, description=description)


def build_rule_maps(
    nodes: Mapping[str, AnalyticNode] | None = None,
) -> list[tuple[str, AggregationMap]]:
    """Four theta-independent surrogate coordinators for information-loss comparison."""
    nodes = nodes or build_nodes()
    return [
        (
            "baseline",
            build_aggregation_map(
                nodes,
                rule_baseline,
                name="surrogate_baseline",
                description="Baseline hard-outcome TicketPolicy map onto ACTIONS",
            ),
        ),
        (
            "stricter",
            build_aggregation_map(
                nodes,
                rule_stricter,
                name="surrogate_stricter",
                description="Stricter: mid-impact also priority_queue",
            ),
        ),
        (
            "coarse_binary",
            build_aggregation_map(
                nodes,
                rule_coarse_binary,
                name="surrogate_coarse_binary",
                description="Binary escalate vs proceed",
            ),
        ),
        (
            "near_identity",
            build_aggregation_map(
                nodes,
                rule_near_identity,
                name="surrogate_near_identity",
                description="Near-identity: one label per joint outcome",
            ),
        ),
    ]


def build_declared_product_surrogate(
    nodes: Mapping[str, AnalyticNode],
    *,
    policy: "TicketPolicy | None" = None,
) -> tuple[AnalyticJointModel, AggregationMap]:
    """Declared product joint + baseline AggregationMap (adapter default)."""
    del policy  # baseline hard map mirrors default TicketPolicy thresholds
    joint = build_declared_joint(nodes)
    aggregation = build_aggregation_map(
        nodes,
        rule_baseline,
        name="surrogate_ticket_action",
        description=(
            "Declared-product surrogate coordinator onto ACTIONS; "
            "not the production soft-probability TicketPolicy routing"
        ),
    )
    return joint, aggregation


def node_probability_map(
    nodes: Mapping[str, AnalyticNode], theta: Mapping[str, float]
) -> dict[str, float]:
    """Flatten declared node probabilities into a single policy input vector."""
    flattened: dict[str, float] = {}
    for node_id, node in nodes.items():
        vector = node.probabilities(theta)
        for outcome, probability in zip(node.support, vector, strict=True):
            flattened[f"{node_id}.{outcome}"] = float(probability)
    return flattened


class TicketSystemAdapter(AnalyticAdapter):
    """Analytic ticket system with a declared product surrogate and soft policy."""

    def __init__(self, *, system_id: str, policy: TicketPolicy | None = None) -> None:
        nodes = build_nodes()
        self.policy = policy or TicketPolicy()
        joint, aggregation = build_declared_product_surrogate(nodes, policy=self.policy)
        super().__init__(
            system_id=system_id,
            nodes=nodes,
            joint_model=joint,
            renderer=renderer(),
            routing_semantics=RoutingSemantics.DETERMINISTIC_POLICY,
            description=(
                "Synthetic ticket triage: reported node distributions with a deterministic "
                "soft policy; DECLARED_PRODUCT joint + AggregationMap are a surrogate for "
                "information-loss analysis, not production routing frequencies"
            ),
        )
        self.surrogate_aggregation = aggregation

    def action(self, theta: Mapping[str, float]) -> str:
        return self.policy.action(node_probability_map(self.nodes, theta))

    def surrogate_action_distribution(
        self, theta: Mapping[str, float]
    ) -> dict[str, float]:
        """Action distribution under a declared product surrogate.

        This assumes conditional independence across the three nodes and applies
        the policy to realised outcomes.  It is a surrogate, not the production
        routing law, and must be reported separately.
        """
        node_ids = list(self.nodes)
        grid: list[tuple[str, ...]] = [()]
        masses: list[float] = [1.0]
        for node_id in node_ids:
            node = self.nodes[node_id]
            vector = node.probabilities(theta)
            new_grid: list[tuple[str, ...]] = []
            new_masses: list[float] = []
            for prefix, mass in zip(grid, masses, strict=True):
                for outcome, probability in zip(node.support, vector, strict=True):
                    new_grid.append((*prefix, outcome))
                    new_masses.append(mass * float(probability))
            grid = new_grid
            masses = new_masses
        action_masses = {action: 0.0 for action in ACTIONS}
        for outcomes, mass in zip(grid, masses, strict=True):
            distribution: dict[str, float] = {}
            for node_id, outcome in zip(node_ids, outcomes, strict=True):
                node = self.nodes[node_id]
                for candidate in node.support:
                    distribution[f"{node_id}.{candidate}"] = (
                        1.0 if candidate == outcome else 0.0
                    )
            action_masses[self.policy.action(distribution)] += mass
        return action_masses


def build_adapter(config: Any) -> TicketSystemAdapter:
    """Module entrypoint for ``provider.kind: module``."""
    return TicketSystemAdapter(system_id=getattr(config, "id", "ticket-system"))


def policy_action_sequence(
    adapter: TicketSystemAdapter, points: Sequence[Mapping[str, float]]
) -> list[str]:
    return [adapter.action(theta) for theta in points]
