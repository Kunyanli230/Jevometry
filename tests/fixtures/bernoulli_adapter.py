"""Module entrypoint fixture: a Bernoulli system with a likelihood model."""

from __future__ import annotations

from typing import Any

from jevometry.adapters.analytic import (
    AnalyticAdapter,
    bernoulli_likelihood,
    bernoulli_node,
)
from jevometry.experiments.renderer import TextTemplateRenderer
from jevometry.schemas.common import RoutingSemantics


def build_adapter(config: Any) -> AnalyticAdapter:
    node = bernoulli_node("Y", parameter="p")
    return AnalyticAdapter(
        system_id=getattr(config, "id", "bernoulli-fixture"),
        nodes={"Y": node},
        renderer=TextTemplateRenderer("p={p:.6f}"),
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        likelihood=bernoulli_likelihood("p"),
        description="fixture Bernoulli system with an observation model",
    )
