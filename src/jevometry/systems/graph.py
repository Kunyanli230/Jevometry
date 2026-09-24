"""System graph representation and visit probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from jevometry.schemas.joint import JointDistribution
from jevometry.schemas.system import CompositionMode, SystemSpec


@dataclass
class SystemGraph:
    """A DAG view of the declared system."""

    node_ids: list[str]
    edges: list[tuple[str, str]]
    composition_mode: CompositionMode
    routing_semantics: str
    topological_order: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: SystemSpec) -> SystemGraph:
        node_ids = spec.node_ids()
        edges = [(edge.source, edge.target) for edge in spec.edges]
        return cls(
            node_ids=node_ids,
            edges=edges,
            composition_mode=spec.composition_mode,
            routing_semantics=spec.routing_semantics.value,
            topological_order=_topological_order(node_ids, edges),
        )

    def children(self, node_id: str) -> list[str]:
        return [target for source, target in self.edges if source == node_id]

    def parents(self, node_id: str) -> list[str]:
        return [source for source, target in self.edges if target == node_id]

    def visit_probabilities(self, joint: JointDistribution) -> dict[str, float]:
        """Node visit frequencies under a declared joint model.

        Only meaningful when the joint describes actual routing.  For a
        deterministic policy these are model visit probabilities, not observed
        action frequencies.
        """
        masses: dict[str, float] = {node: 0.0 for node in self.node_ids}
        stop = "<stop>"
        for outcome, probability in zip(joint.outcomes, joint.probabilities, strict=True):
            for node, value in zip(joint.node_order, outcome, strict=True):
                if node in masses and value != stop:
                    masses[node] += probability
        return masses


def _topological_order(node_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    incoming = {node: 0 for node in node_ids}
    adjacency: dict[str, list[str]] = {node: [] for node in node_ids}
    for source, target in edges:
        adjacency[source].append(target)
        incoming[target] += 1
    queue = [node for node in node_ids if incoming[node] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in adjacency[node]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if len(order) != len(node_ids):
        raise ValueError("system graph contains a cycle; only DAGs are supported in v0.1")
    return order
