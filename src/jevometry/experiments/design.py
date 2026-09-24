"""Experiment point identity, stencil expansion and acquisition planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from jevometry.schemas.experiment import Budget, ExperimentSpec
from jevometry.schemas.parameters import StencilKind, StencilSpec


def point_id(theta: Mapping[str, float]) -> str:
    """Stable identifier for one theta point."""
    payload = json.dumps(
        {name: round(float(value), 12) for name, value in sorted(theta.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "p_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def theta_vector(theta: Mapping[str, float], names: Sequence[str]) -> list[float]:
    return [float(theta[name]) for name in names]


def stencil_evaluations_per_parameter(
    stencil: StencilSpec, *, kind: StencilKind | None = None
) -> int:
    """Number of extra evaluations each parameter needs for the stencil."""
    orientation = kind or stencil.kind
    per_scale = 2 if orientation is StencilKind.CENTRAL else 2
    return per_scale * len(stencil.step_scales)


@dataclass
class AcquisitionPlan:
    """A complete, inspectable plan of provider requests before execution."""

    experiment_id: str
    live: bool
    repeats: int
    case_ids: list[str]
    point_ids: list[str]
    node_ids: list[str]
    center_requests: int
    stencil_requests: int
    expected_requests: int
    budget: Budget
    within_budget: bool
    notes: list[str] = field(default_factory=list)
    stencil_kind: StencilKind = StencilKind.CENTRAL
    parameters: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "live": self.live,
            "repeats": self.repeats,
            "cases": len(self.case_ids),
            "theta_points": len(self.point_ids),
            "nodes": len(self.node_ids),
            "parameters": self.parameters,
            "stencil_kind": self.stencil_kind.value,
            "center_requests": self.center_requests,
            "stencil_requests": self.stencil_requests,
            "expected_requests": self.expected_requests,
            "budget_max_attempts": self.budget.max_attempts,
            "budget_max_input_tokens": self.budget.max_input_tokens,
            "within_budget": self.within_budget,
            "notes": self.notes,
        }


def build_acquisition_plan(
    experiment: ExperimentSpec,
    node_ids: Sequence[str],
    *,
    live: bool = False,
    repeats: int | None = None,
    include_stencil: bool = True,
    requests_per_point: int = 1,
) -> AcquisitionPlan:
    """Plan every request the run will make, before any request is made.

    ``requests_per_point`` is 1 when a provider batches all node questions into
    one call and ``len(node_ids)`` when it does not.
    """
    effective_repeats = repeats if repeats is not None else experiment.budget.repeats
    cases = len(experiment.cases)
    points = len(experiment.theta_points)
    parameters = experiment.parameter_names()
    center_requests = cases * points * requests_per_point * effective_repeats
    per_parameter = stencil_evaluations_per_parameter(experiment.stencil)
    stencil_requests = (
        cases * points * requests_per_point * len(parameters) * per_parameter
        if include_stencil
        else 0
    )
    expected = center_requests + stencil_requests
    notes: list[str] = []
    within_budget = expected <= experiment.budget.max_attempts
    if not within_budget:
        notes.append(
            f"plan requires {expected} attempts but max_attempts is "
            f"{experiment.budget.max_attempts}"
        )
    if live:
        notes.append("live mode: requests count against provider budget and API key")
    else:
        notes.append("offline mode: analytic/replay providers only")
    return AcquisitionPlan(
        experiment_id=experiment.id,
        live=live,
        repeats=effective_repeats,
        case_ids=[case.id for case in experiment.cases],
        point_ids=[point_id(point) for point in experiment.theta_points],
        node_ids=list(node_ids),
        center_requests=center_requests,
        stencil_requests=stencil_requests,
        expected_requests=expected,
        budget=experiment.budget,
        within_budget=within_budget,
        notes=notes,
        stencil_kind=experiment.stencil.kind,
        parameters=parameters,
    )
