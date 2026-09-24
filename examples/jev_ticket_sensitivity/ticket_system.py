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

from jevometry.adapters.analytic import (
    AnalyticAdapter,
    AnalyticNode,
    choice_question,
    noul_question,
    score_question,
)
from jevometry.experiments.renderer import StructuredRenderer
from jevometry.schemas.common import RoutingSemantics
from jevometry.schemas.system import CompositionMode

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


def _logits(amount: float, wait: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled_amount = amount / 500.0
    scaled_wait = wait / 24.0
    request = np.asarray(
        [0.9 * scaled_wait, 1.1 * scaled_amount - 0.4 * scaled_wait, 0.6 * scaled_amount],
        dtype=np.float64,
    )
    escalation = np.asarray([1.6 * scaled_amount - 0.9 * scaled_wait - 0.6], dtype=np.float64)
    impact = np.asarray(
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
        derivative = np.asarray(
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
        # d logit / d theta = (1.6/500, -0.9/24)
        d_amount = derivative * (1.6 / 500.0)
        d_wait = derivative * (-0.9 / 24.0)
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
        derivative = np.asarray(
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
    """Analytic ticket system with a declared deterministic policy."""

    def __init__(self, *, system_id: str, policy: TicketPolicy | None = None) -> None:
        nodes = build_nodes()
        super().__init__(
            system_id=system_id,
            nodes=nodes,
            renderer=renderer(),
            routing_semantics=RoutingSemantics.DETERMINISTIC_POLICY,
            description=(
                "Synthetic ticket triage with a deterministic policy; probabilities are "
                "reported distributions, not action frequencies"
            ),
        )
        self.policy = policy or TicketPolicy()
        self._system = self._system.model_copy(
            update={"composition_mode": CompositionMode.NODE_ONLY}
        )

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
