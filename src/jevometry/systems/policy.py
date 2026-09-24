"""Deterministic policy sensitivity.

For a deterministic policy the probability values are *not* observed routing
frequencies.  What can be reported is the action the policy selects, where it
flips as theta moves, and how close the decision is to a boundary.  Action or
trajectory Fisher information requires a declared sampling model and is
reported separately as a surrogate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from jevometry.schemas.common import MetricStatus

ActionFunction = Callable[[Mapping[str, float], Mapping[str, float]], str]


@dataclass
class PolicyAnalysis:
    """Action labels and flip locations across a declared design grid."""

    policy_name: str
    actions: list[dict[str, object]]
    flips: list[dict[str, object]]
    action_counts: dict[str, int]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = field(
        default_factory=lambda: [
            "deterministic policy: probabilities are not routing frequencies",
            "action frequencies below are design-point counts, not observed frequencies",
        ]
    )

    def to_metrics(self) -> list[dict[str, object]]:
        return [
            {
                "name": f"policy_action:{self.policy_name}:{entry['point_id']}",
                "value": entry["action"],
                "status": MetricStatus.OK.value,
            }
            for entry in self.actions
        ]


def analyse_policy(
    policy: ActionFunction,
    *,
    policy_name: str,
    points: Sequence[Mapping[str, float]],
    probabilities: Sequence[Mapping[str, float]],
) -> PolicyAnalysis:
    """Evaluate a deterministic policy over the design grid and find flips."""
    actions: list[dict[str, object]] = []
    for theta, distribution in zip(points, probabilities, strict=True):
        action = str(policy(theta, distribution))
        actions.append({"point_id": _point_key(theta), "theta": dict(theta), "action": action})
    flips: list[dict[str, object]] = []
    for index in range(len(actions) - 1):
        left = actions[index]
        right = actions[index + 1]
        if left["action"] != right["action"]:
            flips.append(
                {
                    "from_point": left["point_id"],
                    "to_point": right["point_id"],
                    "from_action": left["action"],
                    "to_action": right["action"],
                    "theta_from": left["theta"],
                    "theta_to": right["theta"],
                }
            )
    counts: dict[str, int] = {}
    for entry in actions:
        key = str(entry["action"])
        counts[key] = counts.get(key, 0) + 1
    return PolicyAnalysis(
        policy_name=policy_name,
        actions=actions,
        flips=flips,
        action_counts=counts,
    )


def _point_key(theta: Mapping[str, float]) -> str:
    from jevometry.experiments.design import point_id

    return point_id(theta)
