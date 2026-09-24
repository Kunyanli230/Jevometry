"""Module entrypoint fixture whose provider always fails."""

from __future__ import annotations

from typing import Any

import numpy as np

from jevometry.adapters.base import ExperimentPoint
from jevometry.geometry.derivatives import NodeEvaluation
from jevometry.schemas.common import MetricStatus, RoutingSemantics
from jevometry.schemas.system import Capabilities, CompositionMode, NodeSpec, SystemSpec
from jevometry.schemas.trace import EvaluationTrace, ProviderStatus


class FailingAdapter:
    """Reports every node evaluation as a provider failure."""

    is_live = False
    provider_name = "failing-fixture"

    def __init__(self, system_id: str = "failing-fixture") -> None:
        self.system_id = system_id

    def describe(self) -> SystemSpec:
        return SystemSpec(
            id=self.system_id,
            nodes=[NodeSpec(id="Y", question_id="Y")],
            composition_mode=CompositionMode.NODE_ONLY,
            routing_semantics=RoutingSemantics.NOT_APPLICABLE,
            capabilities=Capabilities(),
        )

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]:
        return [
            EvaluationTrace(
                experiment_id=self.system_id,
                case_id=point.case.id,
                point_id=point.point_id,
                repeat=point.repeat,
                node_id="Y",
                question_id="Y",
                theta=dict(point.theta),
                rendered_fingerprint="failed",
                request_fingerprint="failed",
                stencil_role=point.stencil_role,
                status=ProviderStatus(
                    ok=False,
                    error_type="simulated_failure",
                    error_message="simulated provider failure",
                    reason_code="simulated_failure",
                    incomplete=True,
                ),
                provenance={"provider": "failing-fixture"},
            )
        ]

    def node_evaluator(self, node_id: str) -> Any:
        del node_id

        def evaluator(theta: dict[str, float]) -> NodeEvaluation:
            del theta
            return NodeEvaluation(
                support=(),
                probabilities=np.zeros(0, dtype=np.float64),
                rendered_fingerprint="failed",
                semantic_hash="",
                model_identity="failing-fixture",
                status=MetricStatus.INSUFFICIENT_DATA,
                reason_code="simulated_failure",
            )

        return evaluator


def build_adapter(config: Any) -> FailingAdapter:
    return FailingAdapter(system_id=getattr(config, "id", "failing-fixture"))
